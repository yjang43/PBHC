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
import pickle
from hydra.utils import instantiate

from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking
from humanoidverse.utils.torch_utils import quat_rotate_inverse, get_euler_xyz


class IsaacGymRobot(URCIRobot):
    REAL=False
    
    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        pt_checkpoint = cfg.checkpoint.parent.parent / (cfg.checkpoint.stem + ".pt")
        override_cfg = OmegaConf.create({
            "checkpoint": pt_checkpoint,
            "device": "cuda:0",
            "env": {"config": {"enforce_randomize_motion_start_eval": False}},
            "num_envs": 1
        })
        cfg = OmegaConf.merge(cfg, override_cfg)
        env = instantiate(cfg.env, device=cfg.device)
        env.is_evaluating = True
        # algo = instantiate(cfg.algo, env=env, device=cfg.device, log_dir=None)
        # algo.setup()
        # algo.load(cfg.checkpoint)
        # algo.evaluate_policy()
        # self.algo = algo


        self.env: LeggedRobotMotionTracking = env
        self.save_motion = False

        # Add gravity vector for coordinate transformations
        self.gvec = np.array([0., 0., -1.])  # Gravity in world frame

        # # Set quaternion once during initialization (for inference, this stays constant)
        # self.base_quat = self.env.simulator.robot_root_states[0, 3:7].cpu().numpy()  # XYZW

        # self.Reset()


    # def Obs(self)->Dict[str, np.ndarray]:
    #     return {'actor_obs': torch2np(self.obs_buf_dict['actor_obs']).reshape(1, -1)}

    def _reset(self):
        self.env.reset_all()

    def _get_state(self):
        # Get observation from environment using function from self.env.
        # Get observation from environment manually.
        # breakpoint()
        # self.q = self.env._get_obs_dof_pos().squeeze(0).cpu().numpy() + self.dof_init_pose
        # self.dq = self.env._get_obs_dof_vel().squeeze(0).cpu().numpy()
        # self.omega = self.env._get_obs_base_ang_vel().squeeze(0).cpu().numpy()
        # self.gvec = self.env._get_obs_projected_gravity().squeeze(0).cpu().numpy()

        # self.q = torch2np(self.env.simulator.dof_pos.squeeze(0)) - self.dof_init_pose
        # self.omega = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 10:13])
        # self.gvec = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.q = torch2np(self.env.simulator.dof_pos.squeeze(0))
        self.dq = torch2np(self.env.simulator.dof_vel.squeeze(0))
        self.omega = torch2np(self.env.simulator.robot_root_states[:, 10:13].squeeze(0))
        self.rpy = torch2np(torch.concat(get_euler_xyz(self.env.simulator.base_quat), dim=-1).squeeze(0))
        self.quat = torch2np(self.env.simulator.base_quat.squeeze(0))

        # Store torch versions for quat_rotate_inverse compatibility
        self.omega_torch = self.env.simulator.robot_root_states[:, 10:13].cpu()
        self.quat_torch = self.env.simulator.base_quat.cpu()
        self.gvec_torch = torch.tensor([[0., 0., -1.]], device=self.env.device, dtype=torch.float32).cpu()

        # self.base_quat[:] = self.simulator.base_quat[:]
        # self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        # self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 7:10])
        # self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 10:13])
        # self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)


        # # breakpoint()
        # # Get raw angular velocity (world frame) from root states
        # raw_omega = self.env.simulator.robot_root_states[0, 10:13].cpu().numpy()  # World frame

        # # Transform to local frame using fixed quaternion (set once at initialization)
        # quat_torch = torch.from_numpy(self.fixed_quat).unsqueeze(0).float()
        # omega_torch = torch.from_numpy(raw_omega).unsqueeze(0).float()
        # local_omega = quat_rotate_inverse(quat_torch, omega_torch)
        # self.omega = local_omega.squeeze(0).cpu().numpy()

        # # Transform gravity to local frame using fixed quaternion
        # gravity_torch = torch.from_numpy(self.gravity_vec).unsqueeze(0).float()
        # projected_gravity = quat_rotate_inverse(quat_torch, gravity_torch)
        # self.gvec = projected_gravity.squeeze(0).cpu().numpy()

        # # Set quaternion and rpy (fixed values)
        # self.quat = self.fixed_quat

        # self.quat = np.full(4, np.nan)
        # self.rpy = np.full(3, np.nan)  # You can compute from quat if needed

        # self.ref_motion_phase = self.env._get_obs_ref_motion_phase()


    def Obs(self)->Dict[str, np.ndarray]:
        # # Compare environment vs URCIRobot observations
        env_obs = self.env.obs_buf_dict_raw["actor_obs"]
        urci_obs = self.obs_buf_dict_raw["actor_obs"]
        # urci_obs["ref_motion_phase"] = env_obs["ref_motion_phase"].cpu().reshape(urci_obs["ref_motion_phase"].shape)
        # urci_obs["history_actor"] = env_obs["history_actor"].cpu().reshape(urci_obs["history_actor"].shape)

        # breakpoint()
        different_keys = []
        self.obs_buf_dict = dict()
        
        for obs_key, obs_config in self._obs_cfg_obs.obs_dict.items():
            if not obs_key=='actor_obs': continue
            obs_keys = sorted(obs_config)
            # (Pdb) sorted(obs_config)
            # ['actions', 'base_ang_vel', 'base_lin_vel', 'dif_local_rigid_body_pos', 'dof_pos', 'dof_vel', 'dr_base_com', 'dr_ctrl_delay', 'dr_friction', 'dr_kd', 'dr_kp', 'dr_link_mass', 'history_critic', 'local_ref_rigid_body_pos', 'projected_gravity', 'ref_motion_phase']
            
            # print("obs_keys", obs_keys, self.obs_buf_dict_raw[obs_key])            
            # print("obs_keys:", obs_keys)
            # print("obs shape:", {key: self.obs_buf_dict_raw[obs_key][key].shape for key in obs_keys})            
            self.obs_buf_dict[obs_key] = torch.cat([self.obs_buf_dict_raw[obs_key][key] for key in obs_keys], dim=-1)
            
        clip_obs = self.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():
            if not obs_key=='actor_obs': continue
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs)

        for key in env_obs.keys():
            if key in urci_obs:
                env_val = env_obs[key].cpu().numpy() if hasattr(env_obs[key], 'cpu') else env_obs[key]
                urci_val = urci_obs[key].cpu().numpy() if hasattr(urci_obs[key], 'cpu') else urci_obs[key]

                diff = np.abs(env_val - urci_val).max()
                if diff > 1e-5:
                    different_keys.append(key)

        print("Keys with different values:", different_keys)
        # breakpoint()
        # def tree_map(func, tree):
        #     if isinstance(tree, dict):
        #         return {k: tree_map(func, v) for k, v in tree.items()}
        #     else:
        #         return func(tree)
        
        # obs_buf_dict = deepcopy(self.env.obs_buf_dict)
        # self.obs_buf_dict = tree_map(lambda x: torch2np(x).astype(np.float32).reshape(1, -1), obs_buf_dict)
        return {'actor_obs': torch2np(self.obs_buf_dict['actor_obs']).reshape(1, -1)}

    # def UpdateObs(self):
    #     self.GetState()
    #     self.KickMotionLib()
    #     self.obs_buf_dict_raw["actor_obs"] = self.env.obs_buf_dict_raw["actor_obs"]
    #     self.UpdateObsWoHistory()
    #     self.UpdateObsForHistory()
    # def UpdateObs(self):
    #     super().UpdateObs()
    #     # Overwrite with obs_buf_dict coming from env.
        
    #     def tree_map(func, tree):
    #         if isinstance(tree, dict):
    #             return {k: tree_map(func, v) for k, v in tree.items()}
    #         else:
    #             return func(tree)
        
    #     obs_buf_dict = deepcopy(self.env.obs_buf_dict)
    #     self.obs_buf_dict = tree_map(lambda x: torch2np(x).astype(np.float32).reshape(1, -1), obs_buf_dict)

    def _apply_action(self, target_q):
        # self.env.step({"actions": torch.tensor(target_q, device=self.env.device)})
        # def tree_map(func, tree):
        #     if isinstance(tree, dict):
        #         return {k: tree_map(func, v) for k, v in tree.items()}
        #     else:
        #         return func(tree)
        
        # obs_buf_dict = deepcopy(self.env.obs_buf_dict)
        # self.obs_buf_dict = tree_map(lambda x: torch2np(x).astype(np.float32).reshape(1, -1), obs_buf_dict)

        # NOTE: LeggedRobotMotionTracking environment takes normalized action, so unnormalize.
        target_q = (target_q - self.dof_init_pose) / self.action_scale
        self.env.step({"actions": torch.tensor(target_q[None, :], device=self.env.device)})

    # def routing(self, cfg_policies):
    #     """
    #         Usage: Input a list of Policy, and the robot can switch between them.
            
    #         - Policies are indexed by integers (Pid), 0 to len(cfg_policies)-1.
    #         - special pid: 
    #             - -2: Reset the robot.
    #             - 0: Default policy, should be stationary. 
    #                 - The Robot will switch to this policy once the motion tracking is Done or being Reset.
    #         - Switching Mechanism:
    #             - The instance (MuJoCo or Real) should implement the Pid control logic. It can be changed at any time.
    #             - When the instance want to Reset the robot, it should set the pid to -2.
    #     """

    #     def tree_map(func, tree):
    #         if isinstance(tree, dict):
    #             return {k: tree_map(func, v) for k, v in tree.items()}
    #         else:
    #             return func(tree)
        

    #     # policy_fn = cfg_policies[0][1]
    #     # self.env.reset_all()
    #     # for _ in range(1000):
    #     #     obs_buf_dict = deepcopy(self.env.obs_buf_dict)
    #     #     self.obs_buf_dict = tree_map(lambda x: torch2np(x).astype(np.float32).reshape(1, -1), obs_buf_dict)
    #     #     obs = {'actor_obs': torch2np(self.obs_buf_dict['actor_obs']).reshape(1, -1)}
    #     #     action = policy_fn(obs)[0]
    #     #     self.env.step({"actions": torch.tensor(action[None, :], device=self.env.device)})

    #     self._pid_size = len(cfg_policies)
    #     self._make_motionlib(cfg_policies)
    #     self._check_init()
    #     # self.cmd[3]=self.rpy[2]
    #     cur_pid = -1

    #     try: 

    #         while True:
    #             t1 = time.time()
                
    #             # breakpoint()
    #             if cur_pid != self._ref_pid or self._ref_pid == -2:
    #                 if self._ref_pid == -2:
    #                     self.Reset()
    #                     self._ref_pid = 0
    #                     t1 = time.time()
    #                     ...
                    
                    
    #                 self._ref_pid %= self._pid_size
    #                 assert self._ref_pid >= 0 and self._ref_pid < self._pid_size, f"Invalid policy id: {self._ref_pid}"
    #                 self.TrySaveMotionFile(pid=cur_pid)       
    #                 logger.info(f"Switch to the policy {self._ref_pid}")

                    
    #                 cur_pid = self._ref_pid
    #                 self.SetObsCfg(cfg_policies[cur_pid][0])
    #                 policy_fn = cfg_policies[cur_pid][1]
    #                 if self.SWITCH_EMA:
    #                     self.old_act = self.act.copy()
    #                 # print('Debug: ',self.Obs()['actor_obs'])
    #                 # breakpoint()

                    
    #                 # breakpoint()
                
    #             self.UpdateObs()
                
    #             action = policy_fn(self.Obs())[0]
                
    #             # breakpoint()
    #             if self.BYPASS_ACT: action = np.zeros_like(action)
                
    #             if self.SWITCH_EMA and self.timer <10:
    #                 self.old_act = self.old_act * 0.9 + action * 0.1
    #                 action = self.old_act
                    
                
    #             # self.env.step({"actions": torch.tensor(action[None, :], device=self.env.device)})
    #             self.ApplyAction(action)
                
    #             self.TrySaveMotionStep()
                
    #             if self.motion_len > 0 and self.ref_motion_phase > 1.0:
    #                 # self.Reset()
    #                 if self._ref_pid == 0:
    #                     self._ref_pid = -2
    #                 else:
    #                     self._ref_pid = 0
    #                 self.TrySaveMotionFile(pid=cur_pid)
    #                 logger.info("Motion End. Switch to the Default Policy")
    #                 break
    #             t2 = time.time()
                
    #             # print(f"t2-t1 = {(t2-t1)*1e3} ms")
    #             if self.REAL:
    #             # if True:
    #                 # print(f"t2-t1 = {(t2-t1)*1e3} ms")
    #                 remain_dt = self.dt - (t2-t1)
    #                 if remain_dt > 0:
    #                     time.sleep(remain_dt)
    #                 else:
    #                     logger.warning(f"Warning! delay = {t2-t1} longer than policy_dt = {self.dt} , skip sleeping")
    #     except RobotExitException as e:
    #         self.TrySaveMotionFile(pid=cur_pid)
    #         self.cleanup()
    #         raise e

    def _get_obs_actions(self,):
        return np2torch(self.act)
    
    def _get_obs_base_ang_vel(self,):
        return quat_rotate_inverse(self.quat_torch, self.omega_torch).squeeze(0)

    def _get_obs_projected_gravity(self,):
        return quat_rotate_inverse(self.quat_torch, self.gvec_torch).squeeze(0)
    
    def _get_obs_dof_pos(self,):
        return np2torch(self.q - self.dof_init_pose)
    
    def _get_obs_dof_vel(self,):
        # print(f"dof_vel: mean:{self.dq.mean()}, std:{self.dq.std()}")
        return np2torch(self.dq)
    
        # self.base_quat[:] = self.simulator.base_quat[:]
        # self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        # self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 7:10])
        # self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 10:13])
        # self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)