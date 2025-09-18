from typing import Dict, List, Tuple, Callable, Optional, Any, Union, TypeVar
import numpy as np
import time
import yaml
import threading


from omegaconf import OmegaConf, DictConfig, ListConfig
from humanoidverse.utils.helpers import parse_observation, np2torch, torch2np
import torch

from booster_robotics_sdk_python import (
    ChannelFactory,
    B1LocoClient,
    B1LowCmdPublisher,
    B1LowStateSubscriber,
    LowCmd,
    LowState,
    B1JointCnt,
    RobotMode,
)

from utils.command import create_prepare_cmd, create_first_frame_rl_cmd
from utils.rotate import rotate_vector_inverse_rpy
from utils.timer import TimerConfig, Timer

from humanoidverse.utils.real.rotation_helper import rpy_to_quaternion_array
from humanoidverse.deploy.urcirobot import URCIRobot

URCIRobotType = TypeVar('URCIRobotType', bound='URCIRobot')
ObsCfg = Union[DictConfig, Callable[[URCIRobotType],np.ndarray]]
URCIPolicyObs = Tuple[ObsCfg, Callable]

class T1Robot(URCIRobot):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        ChannelFactory.Instance().Init(0)
        time.sleep(2)  # Wait for channels to initialize
        print("Initialization complete.")

        self._init_timer()
        self._init_low_state_values()
        self._init_communication()
        self.publish_runner = None
        self.running = True

        self.publish_lock = threading.Lock()

    def _init_timer(self):
        self.real_timer = Timer(TimerConfig(self.dt))
        self.next_publish_time = self.real_timer.get_time()
        self.next_inference_time = self.real_timer.get_time()

    def _init_low_state_values(self):
        self.q = np.zeros(B1JointCnt, dtype=np.float32)
        self.dq = np.zeros(B1JointCnt, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.omega = np.zeros(3, dtype=np.float32)
        self.gvec = np.zeros(3, dtype=np.float32)
        self.rpy = np.zeros(3, dtype=np.float32)

        self.target_q = np.zeros(B1JointCnt, dtype=np.float32)
        self.filtered_target_q = np.zeros(B1JointCnt, dtype=np.float32)
        self.q_latest = np.zeros(B1JointCnt, dtype=np.float32)

    def _init_communication(self) -> None:
        try:
            self.low_cmd = LowCmd()
            self.low_state_subscriber = B1LowStateSubscriber(self._low_state_handler)
            self.low_cmd_publisher = B1LowCmdPublisher()
            self.client = B1LocoClient()

            self.low_state_subscriber.InitChannel()
            self.low_cmd_publisher.InitChannel()
            self.client.Init()
        except Exception as e:
            print(f"Failed to initialize communication: {e}")
            raise

    def _low_state_handler(self, low_state_msg: LowState):
        if abs(low_state_msg.imu_state.rpy[0]) > 1.0 or abs(low_state_msg.imu_state.rpy[1]) > 1.0:
            print("IMU base rpy values are too large: {}".format(low_state_msg.imu_state.rpy))
            self.running = False
        self.real_timer.tick_timer_if_sim()
        time_now = self.real_timer.get_time()
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            self.q_latest[i] = motor.q
        if time_now >= self.next_inference_time:
            self.gvec[:] = rotate_vector_inverse_rpy(
                low_state_msg.imu_state.rpy[0],
                low_state_msg.imu_state.rpy[1],
                low_state_msg.imu_state.rpy[2],
                np.array([0.0, 0.0, -1.0]),
            )
            self.omega[:] = rotate_vector_inverse_rpy(
                low_state_msg.imu_state.rpy[0],
                low_state_msg.imu_state.rpy[1],
                low_state_msg.imu_state.rpy[2],
                np.array(low_state_msg.imu_state.gyro)
            )
            for i, motor in enumerate(low_state_msg.motor_state_serial):
                self.q[i] = motor.q
                self.dq[i] = motor.dq

            self.rpy[:] = low_state_msg.imu_state.rpy
            self.quat[:] = rpy_to_quaternion_array(self.rpy)

            if self.is_motion_tracking:
                self.motion_time = (self.timer + 1) * self.dt 
                self.ref_motion_phase = self.motion_time / self.motion_len

            self._compute_observations()

    def _send_cmd(self, cmd: LowCmd):
        self.low_cmd_publisher.Write(cmd)

    def _start_custom_mode(self):
        start_time = time.perf_counter()
        create_prepare_cmd(self.low_cmd, self.cfg)
        for i in range(B1JointCnt):
            self.dof_target[i] = self.dof_init_pose[i]
            self.filtered_dof_target[i] = self.dof_init_pose[i]
        self._send_cmd(self.low_cmd)
        send_time = time.perf_counter()
        self.client.ChangeMode(RobotMode.kCustom)
        end_time = time.perf_counter()

    def routing(self, cfg_policies: List[URCIPolicyObs]):
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
        self._make_motionlib(cfg_policies)
        self._check_init()

        self.act[:] = 0
        self.history_handler.reset([0])
        self.timer: int = 0
        if self.SWITCH_EMA:
            self.old_act = self.act

        try: 
            self._start_custom_mode()
            policy_fn = cfg_policies[0][1]
            while self.running:
                time_now = self.real_timer.get_time()
                if time_now < self.next_inference_time:
                    time.sleep(0.001)
                    return
                self.next_inference_time += self.policy.get_policy_interval()
                start_time = time.perf_counter()

                action = policy_fn({'actor_obs': torch2np(self.obs_buf_dict['actor_obs']).reshape(1, -1)})[0]

                if self.SWITCH_EMA and self.timer < 10:
                    self.old_act = self.old_act * 0.9 + action * 0.1
                    action = self.old_act

                self.timer += 1
                self.act = action.copy()
                self.target_q[:] = np.clip(action, -self.clip_action_limit, self.clip_action_limit) * self.action_scale + self.dof_init_pose

                inference_time = time.perf_counter()
                time.sleep(0.001)
            self.client.ChangeMode(RobotMode.kDamping)
        
        except:
            self.cleanup()

    def _compute_observations(self):
        self.obs_buf_dict_raw = {}
        self.hist_obs_dict = {}

        for obs_key, obs_config in self.cfg.obs.obs_dict.items():
            if not obs_key=='actor_obs': continue
            self.obs_buf_dict_raw[obs_key] = dict()

            parse_observation(self, obs_config, self.obs_buf_dict_raw[obs_key], self.cfg.obs.obs_scales, self.cfg.obs.noise_scales, 0)
        
        # Compute history observations
        history_obs_list = self.history_handler.history.keys()
        parse_observation(self, history_obs_list, self.hist_obs_dict, self.cfg.obs.obs_scales, self.cfg.obs.noise_scales, 0)

        self.obs_buf_dict = dict()
        
        for obs_key, obs_config in self.cfg.obs.obs_dict.items():
            if not obs_key=='actor_obs': continue
            obs_keys = sorted(obs_config)
            # print("obs_keys", obs_keys)            
            self.obs_buf_dict[obs_key] = torch.cat([self.obs_buf_dict_raw[obs_key][key] for key in obs_keys], dim=-1)

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs)

        for key in self.history_handler.history.keys():
            self.history_handler.add(key, self.hist_obs_dict[key])


    def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, "low_cmd_publisher"):
            self.low_cmd_publisher.CloseChannel()
        if hasattr(self, "low_state_subscriber"):
            self.low_state_subscriber.CloseChannel()
        if hasattr(self, "publish_runner") and getattr(self, "publish_runner") != None:
            self.publish_runner.join(timeout=1.0)

    def _publish_cmd(self):
        while self.running:
            time_now = self.real_timer.get_time()
            if time_now < self.next_publish_time:
                time.sleep(0.001)
                continue
            self.next_publish_time += self.dt

            self.filtered_target_q = self.filtered_target_q * 0.8 + self.target_q * 0.2

            for i in range(B1JointCnt):
                self.low_cmd.motor_cmd[i].q = self.filtered_target_q[i]

            # Use series-parallel conversion for torque to avoid non-linearity
            for i in self.cfg["mech"]["parallel_mech_indexes"]:
                self.low_cmd.motor_cmd[i].q = self.q_latest[i]
                self.low_cmd.motor_cmd[i].tau = np.clip(
                    (self.filtered_target_q[i] - self.q_latest[i]) * self.kp[i],
                    -self.tau_limit[i],
                    self.tau_limit[i],
                )
                self.low_cmd.motor_cmd[i].kp = 0.0

            start_time = time.perf_counter()
            self._send_cmd(self.low_cmd)
            publish_time = time.perf_counter()
            time.sleep(0.001)

    def _get_obs_actions(self,):
        return np2torch(self.act)
    
    def _get_obs_base_ang_vel(self,):
        assert self.omega.ndim == 1
        return np2torch(self.omega)

    def _get_obs_projected_gravity(self,):
        return np2torch(self.gvec)
    
    def _get_obs_dof_pos(self,):
        return np2torch(self.q - self.dof_init_pose)
    
    def _get_obs_dof_vel(self,):
        # print(f"dof_vel: mean:{self.dq.mean()}, std:{self.dq.std()}")
        return np2torch(self.dq)

    def _get_obs_ref_motion_phase(self):
        return torch.tensor(self.ref_motion_phase).reshape(1,)

    def __enter__(self) -> "T1Robot":
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()
        
