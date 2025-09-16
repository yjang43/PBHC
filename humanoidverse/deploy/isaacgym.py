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
        # algo = instantiate(cfg.algo, env=env, device=cfg.device, log_dir=None)
        # algo.setup()
        # algo.load(cfg.checkpoint)
        # algo.evaluate_policy()
        # self.algo = algo

        self.env = env
        self.save_motion = False


    def Obs(self)->Dict[str, np.ndarray]:
        return {'actor_obs': torch2np(self.obs_buf_dict['actor_obs']).reshape(1, -1)}

    def _reset(self):
        self.env.reset_all()

    def _get_state(self):
        # Get observation from environment using function from self.env.
        # Get observation from environment manually.
        self.q = ...
        self.dq = ...
        self.gvec = ...
        self.rpy = ...
        self.quat = ...
        self.omega = ...

    def ApplyAction(self, action): 
        self._apply_action(action)

    def Obs(self)->Dict[str, np.ndarray]:
        return {'actor_obs': torch2np(self.obs_buf_dict['actor_obs']).reshape(1, -1)}

    def UpdateObs(self):
        # super().UpdateObs()
        # Overwrite with obs_buf_dict coming from env.
        
        def tree_map(func, tree):
            if isinstance(tree, dict):
                return {k: tree_map(func, v) for k, v in tree.items()}
            else:
                return func(tree)
        
        obs_buf_dict = deepcopy(self.env.obs_buf_dict)
        self.obs_buf_dict = tree_map(lambda x: torch2np(x).astype(np.float32).reshape(1, -1), obs_buf_dict)

    def _apply_action(self, target_q):
        # self.env.step({"actions": torch.tensor(target_q, device=self.env.device)})
        # def tree_map(func, tree):
        #     if isinstance(tree, dict):
        #         return {k: tree_map(func, v) for k, v in tree.items()}
        #     else:
        #         return func(tree)
        
        # obs_buf_dict = deepcopy(self.env.obs_buf_dict)
        # self.obs_buf_dict = tree_map(lambda x: torch2np(x).astype(np.float32).reshape(1, -1), obs_buf_dict)
        
        self.env.step({"actions": torch.tensor(target_q[None, :], device=self.env.device)})

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

        def tree_map(func, tree):
            if isinstance(tree, dict):
                return {k: tree_map(func, v) for k, v in tree.items()}
            else:
                return func(tree)
        

        # policy_fn = cfg_policies[0][1]
        # self.env.reset_all()
        # for _ in range(1000):
        #     obs_buf_dict = deepcopy(self.env.obs_buf_dict)
        #     self.obs_buf_dict = tree_map(lambda x: torch2np(x).astype(np.float32).reshape(1, -1), obs_buf_dict)
        #     obs = {'actor_obs': torch2np(self.obs_buf_dict['actor_obs']).reshape(1, -1)}
        #     action = policy_fn(obs)[0]
        #     self.env.step({"actions": torch.tensor(action[None, :], device=self.env.device)})

        self._pid_size = len(cfg_policies)
        self._make_motionlib(cfg_policies)
        self._check_init()
        self.cmd[3]=self.rpy[2]
        cur_pid = -1

        try: 

            while True:
                t1 = time.time()
                
                # breakpoint()
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
                
                # breakpoint()
                if self.BYPASS_ACT: action = np.zeros_like(action)
                
                if self.SWITCH_EMA and self.timer <10:
                    self.old_act = self.old_act * 0.9 + action * 0.1
                    action = self.old_act
                    
                
                self.ApplyAction(action)
                
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