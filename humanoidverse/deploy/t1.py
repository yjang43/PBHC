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

def prepare_low_cmd(low_cmd: LowCmd):
    low_cmd.cmd_type = LowCmdType.SERIAL
    motorCmds = [MotorCmd() for _ in range(B1JointCnt)]
    low_cmd.motor_cmd = motorCmds

    for i in range(B1JointCnt):
        low_cmd.motor_cmd[i].q = 0.0
        low_cmd.motor_cmd[i].dq = 0.0
        low_cmd.motor_cmd[i].tau = 0.0
        low_cmd.motor_cmd[i].kp = STIFFNESS[i]
        low_cmd.motor_cmd[i].kd = DAMPING[i]



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
    REAL=True
    
    def __init__(self, cfg):
        super().__init__(cfg)

        self.cfg = cfg
        self.cnt = -1    # Update compute heavy states when cnt < timer
        
        self._init_communication()
        self.publish_runner = None
        self.running = True

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

            # Change to custom mode
            self._enter_custom_mode()
        except Exception as e:
            logger.error(f"Failed to initialize communication: {e}")
            raise

    def _reset(self):
        # default_joint_angles offset is applied by ApplyAction.
        self.ApplyAction(np.zeros(self.num_dofs))

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
        low_cmd = LowCmd()
        prepare_low_cmd(low_cmd)
        for i in range(B1JointCnt):
            low_cmd.motor_cmd[i].q = target_q[i]
        self.low_cmd_publisher.Write(low_cmd)

    def _enter_custom_mode(self):
        low_cmd = LowCmd()
        prepare_low_cmd(low_cmd)
        for i in range(B1JointCnt):
            low_cmd.motor_cmd[i].q = self.dof_init_pose[i]
        self.low_cmd_publisher.Write(low_cmd)

        self.client.ChangeMode(RobotMode.kCustom)
        
    def _low_state_handler(self, low_state_msg: LowState):
        self.low_state_msg = low_state_msg

    def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, "low_cmd_publisher"):
            self.low_cmd_publisher.CloseChannel()
        if hasattr(self, "low_state_subscriber"):
            self.low_state_subscriber.CloseChannel()



if __name__ == "__main__":
    robot = MujocoRobot(cfg)
    robot.Reset()
    robot.ApplyAction(np.zeros(robot.num_dofs))