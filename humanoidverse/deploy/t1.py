# Unified Robot Control Interface for HumanoidVerse
# Weiji Xie @ 2025.03.04

import mujoco, mujoco_viewer, mujoco.viewer
import glfw 
import os
import time
import signal
import sys
from pathlib import Path

import torch
from humanoidverse.deploy import URCIRobot
from scipy.spatial.transform import Rotation as R
import logging
from utils.config_utils import *  # noqa: E402, F403
# add argparse arguments

from typing import Dict, Optional, Tuple
from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from loguru import logger
from humanoidverse.utils.helpers import np2torch, torch2np
from humanoidverse.utils.real.rotation_helper import *
from humanoidverse.utils.noise_tool import noise_process_dict, RadialPerturbation
from description.robots.dtype import RobotExitException

import numpy as np

import threading
from copy import deepcopy
from booster_robotics_sdk_python import ChannelFactory, B1LowCmdPublisher, LowCmd, LowCmdType, MotorCmd, B1JointCnt, B1JointIndex
from booster_robotics_sdk_python import (
    ChannelFactory,
    B1LocoClient,
    B1LowCmdPublisher,
    B1LowStateSubscriber,
    LowCmd,
    LowState,
    B1JointCnt,
    RobotMode,
    GetModeResponse
)

import pickle


# STIFFNESS = [
#     10, 10,
#     10, 10, 10, 10,
#     10, 10, 10, 10,
#     50,
#     100, 100, 100,
#     150, 40, 40,
#     100, 100, 100,
#     150, 40, 40
# ]
STIFFNESS = [
    20, 20,
    20, 20, 20, 20,
    20, 20, 20, 20,
    200,
    200, 200, 200, 200, 50, 50,
    200, 200, 200, 200, 50, 50
]
# DAMPING = [
#     1.0, 1.0, 
#     2.0, 2.0, 2.0, 2.0, 2.0, 
#     2.0, 2.0, 2.0, 2.0, 2.0, 
#     5.0, 
#     2.0, 2.0, 2.0,
#     4.0, 2.0, 2.0,
#     2.0, 2.0, 2.0,
#     4.0, 2.0, 2.0]
DAMPING = [
    0.2, 0.2,
    0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5,
    5,
    5, 5, 5, 5, 3, 3,
    5, 5, 5, 5, 3, 3
]
DOF_NAMES = [
    'AAHead_yaw', 'Head_pitch', 
    'Left_Shoulder_Pitch', 'Left_Shoulder_Roll', 'Left_Elbow_Pitch', 'Left_Elbow_Yaw', 
    'Right_Shoulder_Pitch', 'Right_Shoulder_Roll', 'Right_Elbow_Pitch', 'Right_Elbow_Yaw', 
    'Waist', 
    'Left_Hip_Pitch', 'Left_Hip_Roll', 'Left_Hip_Yaw', 
    'Left_Knee_Pitch', 'Left_Ankle_Pitch', 'Left_Ankle_Roll', 
    'Right_Hip_Pitch', 'Right_Hip_Roll', 'Right_Hip_Yaw', 
    'Right_Knee_Pitch', 'Right_Ankle_Pitch', 'Right_Ankle_Roll']

UPPER_BODY_MASK = [
    True, True,
    True, True, True, True,
    True, True, True, True,
    # True,
    False,
    False, False, False,
    False, False, False,
    False, False, False,
    False, False, False,
]

LOWER_BODY_MASK = [
    False, False,
    False, False, False, False,
    False, False, False, False,
    # False,
    True,
    True, True, True,
    True, True, True,
    True, True, True,
    True, True, True,
]

def prepare_low_cmd(
    low_cmd: LowCmd,
    *,
    q=None,
    dq=None,
    tau=None,
    stiffness=None,
    damping=None
):
    low_cmd.cmd_type = LowCmdType.SERIAL
    motorCmds = [MotorCmd() for _ in range(B1JointCnt)]
    low_cmd.motor_cmd = motorCmds

    for i in range(B1JointCnt):
        if q is not None:
            low_cmd.motor_cmd[i].q = q[i]
        if dq is not None:
            low_cmd.motor_cmd[i].dq = dq[i]
        if tau is not None:
            low_cmd.motor_cmd[i].tau = tau[i]
        if stiffness is not None:
            low_cmd.motor_cmd[i].kp = stiffness[i]
        if damping is not None:
            low_cmd.motor_cmd[i].kd = damping[i]


def rotate_vector_inverse_rpy(roll, pitch, yaw, vector):
    """
    Rotate a vector by the inverse of the given roll, pitch, and yaw angles.

    Parameters:
    roll (float): The roll angle in radians.
    pitch (float): The pitch angle in radians.
    yaw (float): The yaw angle in radians.
    vector (np.ndarray): The 3D vector to be rotated.

    Returns:
    np.ndarray: The rotated 3D vector.
    """
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return (R_z @ R_y @ R_x).T @ vector


class MujocoRobot(URCIRobot):
    high_damping = [
        0.2, 0.2,
        5, 5, 5, 5,
        5, 5, 5, 5,
        10,
        10, 10, 10, 10, 6, 6,
        10, 10, 10, 10, 6, 6
    ]
    low_stiffness = [
        10, 10,
        10, 10, 10, 10,
        10, 10, 10, 10,
        200,
        150, 150, 150, 150, 50, 50,
        150, 150, 150, 150, 50, 50
    ]
    REAL=True
    
    def __init__(self, cfg):
        super().__init__(cfg)

        self.cfg = cfg
        self.cnt = -1    # Update compute heavy states when cnt < timer
        
        self._init_communication()
        self._init_custom_mode()

        self.act_log = []
        self.obs_log = []

        self.Reset()

    def _init_communication(self) -> None:
        try:
            ChannelFactory.Instance().Init(0)
            self.low_cmd = LowCmd()
            self.low_state_subscriber = B1LowStateSubscriber(self._low_state_handler)
            self.low_cmd_publisher = B1LowCmdPublisher()
            self.client = B1LocoClient()

            self.low_state_subscriber.InitChannel()
            self.low_cmd_publisher.InitChannel()
            self.client.Init()

        except Exception as e:
            logger.error(f"Failed to initialize communication: {e}")
            raise

    def _init_custom_mode(self):
        prepare_low_cmd(
            self.low_cmd,
            q=self.dof_init_pose,
            stiffness=self.low_stiffness,
            damping=self.high_damping
        )
        self.low_cmd_publisher.Write(self.low_cmd)

        self.client.ChangeMode(RobotMode.kCustom)

        # Wait until near target_q.
        self._get_state()
        while np.linalg.norm(self.q - self.dof_init_pose) > 1.0:
            time.sleep(0.01)
            self._get_state()

    def Reset(self):
        super().Reset()
        self.act[:] = self.q


    def _reset(self):
        # def routing must be called prior to _reset. Otherwise do nothing.
        # self.motion_lib is set at def SetObsCfg. Manually set motion_lib.
        if not self.motion_libs:
            return
        else:
            ref_pid = 0 if self._ref_pid == -2 else self._ref_pid
            self.motion_lib = self.motion_libs[ref_pid]

        # High damp to move joint safely.
        self.motion_res = self._kick_motion_res()
        prepare_low_cmd(
            self.low_cmd,
            q=self.motion_res["dof_pos"][0].numpy(),
            dq=self.motion_res["dof_vel"][0].numpy(),
            stiffness=self.low_stiffness,
            damping=self.high_damping
        )
        self.low_cmd_publisher.Write(self.low_cmd)


        # Wait until near target_q.
        self._get_state()
        while np.linalg.norm(self.q - self.motion_res["dof_pos"][0].numpy()) > 1.0:
            time.sleep(0.01)
            self._get_state()

    def _get_state(self):
        low_state_msg = self.low_state_msg

        for i, motor in enumerate(low_state_msg.motor_state_serial):
            self.q[i] = motor.q
            self.dq[i] = motor.dq

        self.gvec = rotate_vector_inverse_rpy(
            low_state_msg.imu_state.rpy[0],
            low_state_msg.imu_state.rpy[1],
            low_state_msg.imu_state.rpy[2],
            np.array([0.0, 0.0, -1.0]))

        self.rpy = np.array(low_state_msg.imu_state.rpy)
        self.quat = rpy_to_quaternion_array(self.rpy)
        self.omega = np.array(low_state_msg.imu_state.gyro)

    def _apply_action(self, target_q):

        stiffness = []
        damping = []
        for dof_name in self.dof_names:
            for name in self.cfg.robot.control.stiffness:
                if name in dof_name:
                    stiffness.append(self.cfg.robot.control.stiffness[name])
                    damping.append(self.cfg.robot.control.damping[name])

        # masked_target_q = target_q
        # masked_target_q *= np.array(UPPER_BODY_MASK)
        # masked_target_q += self.motion_res["dof_pos"][0].numpy() * np.array(LOWER_BODY_MASK)

        prepare_low_cmd(
            self.low_cmd,
            q=target_q,
            # q=masked_target_q,
            stiffness=stiffness,
            damping=damping,
            # damping=self.high_damping,  # NOTE: For debug purpose.
            # stiffness=self.low_stiffness
        )
        
        self.low_cmd_publisher.Write(self.low_cmd)
        
        # self.act_log.append(target_q)
        # import pickle
        # with open("tmp_dump_act.pkl", "wb") as f:
        #     pickle.dump(self.act_log, f)
        
    def _low_state_handler(self, low_state_msg: LowState):
        self.low_state_msg = low_state_msg

    def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, "low_cmd_publisher"):
            self.low_cmd_publisher.CloseChannel()
        if hasattr(self, "low_state_subscriber"):
            self.low_state_subscriber.CloseChannel()

    def _get_motion_to_save_np(self)->Tuple[float, Dict[str, np.ndarray]]:
        
        from scipy.spatial.transform import Rotation as sRot
        
        motion_time = (self.timer) * self.dt 

        # Odometer
        root_trans = np.zeros(3).astype(np.double)
        root_lin_vel = np.zeros(3).astype(np.double)
        root_rot = self.quat # XYZW
        root_rot_vec = np.array(sRot.from_quat(root_rot).as_rotvec(), dtype=np.float32) # type: ignore
        dof = self.q
        # T, num_env, J, 3
        # print(self._motion_lib.mesh_parsers.dof_axis)
        pose_aa = np.concatenate([root_rot_vec[..., None, :], 
                                np.array(self._dof_axis * dof[..., None]), 
                                np.zeros((self.num_augment_joint, 3))], axis = 0) # type: ignore
        
        return motion_time, {
            'root_trans_offset': (root_trans).copy(),
            'pose_aa': pose_aa,
            'dof': (dof).copy(),
            'root_rot': (root_rot).copy(), # 统一save xyzw
            'actor_obs': (self.obs_buf_dict['actor_obs']),
            'action': (self.act).copy(),
            'terminate': np.zeros((1,)),
            'root_lin_vel': (root_lin_vel).copy(),
            'root_ang_vel': (self.omega).copy(),
            'dof_vel': (self.dq).copy(),
            
            # 'clock_time': (np.array([time.time()])),
            # 'tau': (self.data.ctrl).copy(),
            # 'cmd': (self.cmd).copy()
        }
        
    _get_motion_to_save = _get_motion_to_save_np


    def ApplyAction(self, action):
        with open("tmp_dump_act.pkl", "wb") as f:
            pickle.dump(self.act_log, f)
        return super().ApplyAction(action)

    def Obs(self)->Dict[str, np.ndarray]:
        obs = super().Obs()
        with open("tmp_dump_obs.pkl", "wb") as f:
            pickle.dump(self.obs_log, f)
        return obs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()


    def routing(self, cfg_policies):
        """
            Usage: Input a list of Policy, and the robot can switch between them.
            
            - Policies are indexed by integers (Pid), 0 to len(cfg_policies)-1.
            - special pid: 
                - -2: Reset the robot.
                - 0: Default policy, should be stationary. 
                    - The Robot will switch to this policy once the motion tracking is Done or being Reset.
            - Switching Mechanism:
                - The instance (MuJoCo or Real) should implement the Pid control logic. It can be changed at any time.
                - When the instance want to Reset the robot, it should set the pid to -2.
        """
        self._pid_size = len(cfg_policies)
        self._make_motionlib(cfg_policies)
        self._check_init()
        self.cmd[3]=self.rpy[2]
        cur_pid = -1

        try: 

            obs_list = []
            act_list = []

            while True:
                t1 = time.time()
                
                if cur_pid != self._ref_pid or self._ref_pid == -2:
                    if self._ref_pid == -2:
                        self.Reset()
                        self._ref_pid = 0
                        t1 = time.time()
                        ...
                    
                    
                    self._ref_pid %= self._pid_size
                    assert self._ref_pid >= 0 and self._ref_pid < self._pid_size, f"Invalid policy id: {self._ref_pid}"
                    self.TrySaveMotionFile(pid=cur_pid)       
                    logger.info(f"Switch to the policy {self._ref_pid}")

                    
                    cur_pid = self._ref_pid
                    self.SetObsCfg(cfg_policies[cur_pid][0])
                    policy_fn = cfg_policies[cur_pid][1]
                    if self.SWITCH_EMA:
                        self.old_act = self.act.copy()
                    # print('Debug: ',self.Obs()['actor_obs'])
                    # breakpoint()

                    
                    # breakpoint()
                
                self.UpdateObs()
                
                action = policy_fn(self.Obs())[0]
                
                if self.BYPASS_ACT: action = np.zeros_like(action)
                
                if self.SWITCH_EMA and self.timer <10:
                    self.old_act = self.old_act * 0.9 + action * 0.1
                    action = self.old_act
                    
                
                self.ApplyAction(action)
                
                obs_list.append(self.obs_buf_dict_raw)
                act_list.append(action)

                # if self.timer > 100:
                #     import pickle
                #     with open('real_obs_list.pkl', 'wb') as f:
                #         pickle.dump(obs_list, f)
                #     with open('real_act_list.pkl', 'wb') as f:
                #         pickle.dump(act_list, f)
                #     break
                
                self.TrySaveMotionStep()
                
                if self.motion_len > 0 and self.ref_motion_phase > 1.0:
                    # self.Reset()
                    if self._ref_pid == 0:
                        self._ref_pid = -2
                    else:
                        self._ref_pid = 0
                    self.TrySaveMotionFile(pid=cur_pid)
                    logger.info("Motion End. Switch to the Default Policy")
                    break
                t2 = time.time()
                
                # print(f"t2-t1 = {(t2-t1)*1e3} ms")
                if self.REAL:
                # if True:
                    # print(f"t2-t1 = {(t2-t1)*1e3} ms")
                    remain_dt = self.dt - (t2-t1)
                    if remain_dt > 0:
                        time.sleep(remain_dt)
                    else:
                        logger.warning(f"Warning! delay = {t2-t1} longer than policy_dt = {self.dt} , skip sleeping")
        except RobotExitException as e:
            self.TrySaveMotionFile(pid=cur_pid)
            self.cleanup()
            raise e