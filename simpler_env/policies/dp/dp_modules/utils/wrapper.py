from typing import Any, SupportsFloat, Tuple, Dict, Optional
import gymnasium as gym
from collections import deque
from gymnasium.core import Env
import numpy as np
from homebot_sapien.utils.math import (
    quat2euler,
    get_pose_from_rot_pos,
    rot6d2mat,
    euler2mat,
    mat2euler,
)
from transforms3d.quaternions import qmult, qconjugate, quat2mat, mat2quat
from homebot_sapien.algorithm.imitation.dataset import OBS_NORMALIZE_PARAMS


class DoneOnSuccessWrapper(gym.Wrapper):
    """
    Reset on success and offsets the reward.
    Useful for GoalEnv.
    """

    def __init__(self, env):
        super(DoneOnSuccessWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        done = done or info.get("is_success", False)
        return obs, reward, done, truncated, info


class IOFromConfigWrapper(gym.Wrapper):
    """
    Reorder, normalize, stack images, etc. according the config file
    """

    def __init__(self, env: Env, config: dict, obs_normalize_params: Optional[dict]):
        super().__init__(env)
        assert isinstance(self.env.observation_space, gym.spaces.Dict)
        self.config = config
        if obs_normalize_params is None:
            obs_normalize_params = OBS_NORMALIZE_PARAMS
        self.obs_normalize_params = obs_normalize_params
        self._image_history = deque(maxlen=config["data_config"]["n_images"])
        self._inference_freq = config["diffusion_config"].get("inference_horizon", 10)
        self._elapsed_step = 0
        # Overwrite observation space and action space
        _history = (
            config["data_config"]["n_images"]
            if config["data_config"]["image_wrist_or_head"] != "both"
            else 2 * config["data_config"]["n_images"]
        )
        _image_shape = self.env.observation_space["third-rgb"].shape
        _n_robot_state = 0
        for key in self.config["data_config"]["robot_state_keys"]:
            if key == "pose":
                _n_robot_state += 9
            elif key == "gripper_width":
                _n_robot_state += 1
            elif key == "joint":
                _n_robot_state += 7
        _n_action = 0
        for key in self.config["data_config"]["action_keys"]:
            if key == "gripper":
                _n_action += 1
            elif key == "pose":
                _n_action += 9
        self._pose_at_obs = None
        self.observation_space = gym.spaces.Dict(
            dict(
                image=gym.spaces.Box(
                    0,
                    255,
                    shape=(_history, _image_shape[2], _image_shape[0], _image_shape[1]),
                    dtype=np.uint8,
                ),
                robot_state=gym.spaces.Box(
                    -100, 100, shape=(_n_robot_state,), dtype=np.float32
                ),
                lang=gym.spaces.Text(max_length=1000),
                privileged_obs=gym.spaces.Box(
                    -100,
                    100,
                    shape=(self.observation_space["privileged_obs"].shape[0] + 1,),
                    dtype=np.float32,
                ),
            )
        )
        for key in self.env.observation_space:
            self.observation_space[key] = self.env.observation_space[key]
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(_n_action,), dtype=np.float32
        )

    def _update_obs(self, obs_env: dict):
        # Maintain the history, and reorganize the obs according to config
        wrist_rgb = np.transpose(
            obs_env["wrist-rgb"], (2, 0, 1)
        )  # get rid of vec dimension
        head_rgb = np.transpose(obs_env["third-rgb"], (2, 0, 1))
        if len(self._image_history) == 0:
            append_num = self._image_history.maxlen
        else:
            append_num = 1
        for _ in range(append_num):
            if self.config["data_config"]["image_wrist_or_head"] == "wrist":
                self._image_history.append(wrist_rgb)
            elif self.config["data_config"]["image_wrist_or_head"] == "head":
                self._image_history.append(head_rgb)
            elif self.config["data_config"]["image_wrist_or_head"] == "both":
                self._image_history.append(np.stack([wrist_rgb, head_rgb], axis=0))
            else:
                raise NotImplementedError
        images = np.concatenate(self._image_history, axis=0)
        images = images.astype(np.float32)

        robot_states = np.zeros(0)
        if len(self.config["data_config"]["robot_state_keys"]):
            for key in self.config["data_config"]["robot_state_keys"]:
                if key == "pose":
                    # tcp_pose = np.concatenate(
                    #     [obs_env["tcp_pose"][:3], euler2mat(quat2euler(obs_env["tcp_pose"][3:]))[:, :2].reshape(-1)]
                    # )
                    obs_pose = np.eye(4)
                    if self._elapsed_step % self._inference_freq == 0:
                        self._pose_at_obs = get_pose_from_rot_pos(
                            euler2mat(quat2euler(obs_env["tcp_pose"][3:])),
                            obs_env["tcp_pose"][:3],
                        )
                    robot_states = np.concatenate(
                        [robot_states, obs_pose[:3, 3], obs_pose[:3, :2].reshape(-1)]
                    )
                elif key == "gripper_width":
                    gripper_width = np.array([obs_env["gripper_width"]])
                    robot_states = np.concatenate([robot_states, gripper_width])
                elif key == "joint":
                    joint = obs_env["robot_joints"]
                    robot_states = np.concatenate([robot_states, joint])
            state_mean = np.concatenate(
                [
                    self.obs_normalize_params[key]["mean"]
                    for key in self.config["data_config"]["robot_state_keys"]
                ]
            )
            state_scale = np.concatenate(
                [
                    self.obs_normalize_params[key]["scale"]
                    for key in self.config["data_config"]["robot_state_keys"]
                ]
            )
            robot_states = (robot_states - state_mean) / state_scale
        robot_states = robot_states.astype(np.float32)

        lang = " "
        obs_learn = dict(
            image=images,
            robot_state=robot_states,
            lang=lang,
            # privileged_obs=np.concatenate(
            #     [
            #         obs_env["privileged_obs"],
            #         [self._elapsed_step / self.spec.max_episode_steps],
            #     ]
            # ),
        )
        obs_learn.update(obs_env)
        return obs_learn

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        obs_env, info = self.env.reset(**kwargs)
        self._elapsed_step = 0
        obs_learn = self._update_obs(obs_env)
        return obs_learn, info

    def step(
            self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        # action: numpy array predicted from the configured learning algorithm
        assert len(action.shape) == 1
        _start_idx = 0
        base_action = np.zeros((2,))
        converted_action = np.zeros((0,))
        for key in self.config["data_config"]["action_keys"]:
            if key == "gripper":
                gripper_action = (
                        action[_start_idx: _start_idx + 1]
                        * self.obs_normalize_params["gripper_width"]["scale"]
                        + self.obs_normalize_params["gripper_width"]["mean"]
                )
                converted_action = np.concatenate([converted_action, gripper_action])
                _start_idx += 1
            elif key == "pose":
                scaled_action = (
                        action[_start_idx: _start_idx + 9]
                        * self.obs_normalize_params["pose"]["scale"]
                        + self.obs_normalize_params["pose"]["mean"]
                )
                _pos = scaled_action[:3]
                _rot = rot6d2mat(scaled_action[3:])
                init_to_desired_pose = self._pose_at_obs @ get_pose_from_rot_pos(
                    _rot, _pos
                )
                pose_action = np.concatenate(
                    [
                        init_to_desired_pose[:3, 3],
                        mat2euler(init_to_desired_pose[:3, :3]),
                    ]
                )
                converted_action = np.concatenate([converted_action, pose_action])
                _start_idx += 9
            elif key == "joint":
                joint_action = (
                        action[_start_idx: _start_idx + 7]
                        * self.obs_normalize_params["joint"]["scale"]
                        + self.obs_normalize_params["joint"]["mean"]
                )
                converted_action = np.concatenate([converted_action, joint_action])
                _start_idx += 7
            else:
                raise NotImplementedError
        converted_action = np.concatenate([base_action, converted_action], axis=-1)
        obs_env, reward, done, truncated, info = self.env.step(converted_action)
        self._elapsed_step += 1
        # Port obs_env to the learning algorithm
        obs_learn = self._update_obs(obs_env)
        if done or truncated:
            self._image_history.clear()
        return obs_learn, reward, done, truncated, info


class SeqActionFromConfigWrapper(IOFromConfigWrapper):
    """
    convert predicted actions to environment spec,
    send (part) of action chunk to the env,
    make the observation suitable for learning algo
    """

    def __init__(
            self,
            env: Env,
            config: dict,
            obs_normalize_params: dict,
            gamma: float,
            return_middle_obs: bool,
    ):
        super().__init__(env, config, obs_normalize_params)
        self.gamma = gamma
        self.return_middle_obs = return_middle_obs
        _n_action = 0
        for key in self.config["data_config"]["action_keys"]:
            if key == "gripper":
                _n_action += 1
            elif key == "pose":
                _n_action += 9
            elif key == "joint":
                _n_action += 7
        self.action_space = gym.spaces.Box(
            -1.0,
            1.0,
            shape=(
                self.config["diffusion_config"]["prediction_horizon"],
                _n_action,
            ),
            dtype=np.float32,
        )

    def step(
            self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        action_seq = action.reshape(
            (self.config["diffusion_config"]["prediction_horizon"], -1)
        )
        _env_done = False
        is_pad = np.zeros(
            (self.config["diffusion_config"]["prediction_horizon"],), dtype=bool
        )
        is_pad[self._inference_freq:] = True
        macro_reward = 0
        obs_seq = []
        desired_pose_seq = []
        desired_joint_seq = []
        desired_gripper_width_seq = []
        for i in range(self._inference_freq):
            if not _env_done:
                obs_learn, reward, done, truncated, info = super().step(action_seq[i])
                macro_reward += np.power(self.gamma, i) * reward
                _env_done = done or truncated
            else:
                is_pad[i] = True
            if self.return_middle_obs:
                obs_seq.append(obs_learn)
                desired_pose_seq.append(info.get("desired_relative_pose"))
                desired_joint_seq.append(info.get("desired_joints"))
                desired_gripper_width_seq.append(info.get("desired_gripper_width"))
        info["is_pad"] = is_pad
        if self.return_middle_obs:
            info["obs_seq"] = np.array(obs_seq)
            if desired_pose_seq[0] is not None:
                info["desired_relative_pose_seq"] = np.array(desired_pose_seq)
            if desired_joint_seq[0] is not None:
                info["desired_joints_seq"] = np.array(desired_joint_seq)
            if desired_gripper_width_seq[0] is not None:
                info["desired_gripper_width_seq"] = np.array(desired_gripper_width_seq)
        return obs_learn, macro_reward, done, truncated, info


class TimeLimit(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        max_episode_steps: int = 1000,
    ):
        super().__init__(env)

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):

        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):

        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class StateObservationWrapper(gym.Wrapper):

    def __init__(self, env, seed=0, action_format="mat_6"):
        super().__init__(env)

        self.ep_reward = 0
        self.ep_len = 0

        self.action_format = action_format
        assert action_format in ["mat_6", "euler"]

        self.random_state = np.random.RandomState(seed=seed)

        _tcp_pose_dim = 9
        _gripper_dim = 1
        _obs_dim = _tcp_pose_dim + _gripper_dim  # 10

        self.observation_space = gym.spaces.Box(
            -10,
            10,
            shape=(_obs_dim,),
            dtype=np.float32,
        )

        _action_dim = 10 if action_format == "mat_6" else 7

        self.action_space = gym.spaces.Box(
            -1.0,
            1.0,
            shape=(_action_dim,),
            dtype=np.float32,
        )

        if action_format == "mat_6":
            _pose_gripper_max = np.array([0.01, 0.01, 0.01, 1., 0.04, 0.04, 1., 0.04, 0.04, 0.04])
            _pose_gripper_min = np.array([-0.01, -0.01, -0.01, 0.99, -0.04, -0.04, 0.99, -0.04, -0.04, 0.])
        else:
            _pose_gripper_max = np.array([0.01, 0.01, 0.01, 0.04, 0.04, 0.04,  0.04])
            _pose_gripper_min = np.array([-0.01, -0.01, -0.01, -0.04, -0.04, -0.04, 0.])
        self.pose_gripper_mean = (_pose_gripper_max + _pose_gripper_min) / 2
        self.pose_gripper_scale = (_pose_gripper_max - _pose_gripper_min) / 2

        _obs_max = np.array([
            0.55, 0.35, 0.42,
            1., 1., 1., 1., 1., 1., 0.04,
        ])
        _obs_min = np.array([
            0.25, -0.35, -0.01,
            -1., -1., -1., -1., -1., -1., 0.,
        ])
        self.obs_mean = (_obs_max + _obs_min) / 2
        self.obs_scale = (_obs_max - _obs_min) / 2

    def process_obs(self, obs):
        tcp_pose = obs["tcp_pose"]
        tcp_p = tcp_pose[:3]
        tcp_mat_6 = quat2mat(tcp_pose[3:])[:, :2].reshape(-1)

        gripper_width = obs["gripper_width"]

        full_obs = np.concatenate([
            tcp_p,
            tcp_mat_6,
            [gripper_width],
        ], axis=0)

        # full_obs = (full_obs - self.obs_mean) / self.obs_scale

        return full_obs

    def process_action(self, action):
        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
        action = action * self.pose_gripper_scale + self.pose_gripper_mean

        if self.action_format == "mat_6":
            mat_6 = action[3:9].reshape(3, 2)
            mat_6[:, 0] = mat_6[:, 0] / (np.linalg.norm(mat_6[:, 0]) + 1e-8)
            mat_6[:, 1] = mat_6[:, 1] / (np.linalg.norm(mat_6[:, 1]) + 1e-8)
            z_vec = np.cross(mat_6[:, 0], mat_6[:, 1])
            mat = np.c_[mat_6, z_vec]

            pos = action[:3]
            gripper_width = action[-1]

            pose_action = np.concatenate(
                [
                    pos,
                    mat2euler(mat),
                    [gripper_width]
                ]
            )
        else:
            pose_action = action

        return pose_action

    def reset(self):
        # print(type(self.env))
        seed = self.random_state.randint(low=0, high=int(1e8))
        obs, info = self.env.reset(seed=seed)

        obs = self.process_obs(obs)

        self.ep_reward = 0
        self.ep_len = 0

        return obs

    def step(self, action):

        pose_action = self.process_action(action)

        # print(pose_action)
        raw_obs, reward, done, truncated, info = self.env.step(pose_action)

        obs = self.process_obs(raw_obs)
        info.update(dict(raw_obs=raw_obs, raw_action=pose_action))

        done = done or info.get("is_success", False)
        reward = reward * 2.

        self.ep_reward += reward
        self.ep_len += 1

        if done or truncated:
            info.update(dict(episode=dict(r=self.ep_reward, l=self.ep_len)))
            # self.reset()
            # done = True

        return obs, reward, done, truncated, info

    def render(self, **kwargs):
        images = self.env.capture_images_new()
        image = images["third-rgb"]
        return image


class StateObservationObjectWrapper(gym.Wrapper):

    def __init__(self, env, seed=0, action_format="mat_6"):
        super().__init__(env)

        self.ep_reward = 0
        self.ep_len = 0

        self.action_format = action_format
        assert action_format in ["mat_6", "euler"]

        self.random_state = np.random.RandomState(seed=seed)

        _tcp_pose_dim = 9
        _gripper_dim = 1
        _obj_pose_dim = 9
        _obj_size_dim = 3
        _obs_dim = _tcp_pose_dim + _gripper_dim + _obj_pose_dim + _obj_size_dim  # 22

        self.observation_space = gym.spaces.Box(
            -10,
            10,
            shape=(_obs_dim,),
            dtype=np.float32,
        )

        _action_dim = 10 if action_format == "mat_6" else 7

        self.action_space = gym.spaces.Box(
            -1.0,
            1.0,
            shape=(_action_dim,),
            dtype=np.float32,
        )

        if action_format == "mat_6":
            _pose_gripper_max = np.array([0.01, 0.01, 0.01, 1., 0.04, 0.04, 1., 0.04, 0.04, 0.04])
            _pose_gripper_min = np.array([-0.01, -0.01, -0.01, 0.99, -0.04, -0.04, 0.99, -0.04, -0.04, 0.])
        else:
            _pose_gripper_max = np.array([0.01, 0.01, 0.01, 0.04, 0.04, 0.04,  0.04])
            _pose_gripper_min = np.array([-0.01, -0.01, -0.01, -0.04, -0.04, -0.04, 0.])
        self.pose_gripper_mean = (_pose_gripper_max + _pose_gripper_min) / 2
        self.pose_gripper_scale = (_pose_gripper_max - _pose_gripper_min) / 2

        _obs_max = np.array([
            0.55, 0.35, 0.42,
            1., 1., 1., 1., 1., 1., 0.04,
            0.55, 0.35, 0.20,
            1., 1., 1., 1., 1., 1.,
            0.05, 0.05, 0.05,
        ])
        _obs_min = np.array([
            0.25, -0.35, -0.01,
            -1., -1., -1., -1., -1., -1., 0.,
            0.25, -0.35, 0.,
            -1., -1., -1., -1., -1., -1.,
            0.01, 0.01, 0.01
        ])
        self.obs_mean = (_obs_max + _obs_min) / 2
        self.obs_scale = (_obs_max - _obs_min) / 2

    def process_obs(self, obs):
        tcp_pose = obs["tcp_pose"]
        tcp_p = tcp_pose[:3]
        tcp_mat_6 = quat2mat(tcp_pose[3:])[:, :2].reshape(-1)

        gripper_width = obs["gripper_width"]

        obj_state = obs["obj_states"][0]
        obj_p, obj_q, obj_size = obj_state[:3], obj_state[3:7], obj_state[7:]
        obj_mat_6 = quat2mat(obj_q)[:, :2].reshape(-1)

        full_obs = np.concatenate([
            tcp_p,
            tcp_mat_6,
            [gripper_width],
            obj_p,
            obj_mat_6,
            obj_size
        ], axis=0)

        full_obs = (full_obs - self.obs_mean) / self.obs_scale

        return full_obs

    def process_action(self, action):
        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
        action = action * self.pose_gripper_scale + self.pose_gripper_mean

        if self.action_format == "mat_6":
            mat_6 = action[3:9].reshape(3, 2)
            mat_6[:, 0] = mat_6[:, 0] / (np.linalg.norm(mat_6[:, 0]) + 1e-8)
            mat_6[:, 1] = mat_6[:, 1] / (np.linalg.norm(mat_6[:, 1]) + 1e-8)
            z_vec = np.cross(mat_6[:, 0], mat_6[:, 1])
            mat = np.c_[mat_6, z_vec]

            pos = action[:3]
            gripper_width = action[-1]

            pose_action = np.concatenate(
                [
                    pos,
                    mat2euler(mat),
                    [gripper_width]
                ]
            )
        else:
            pose_action = action

        return pose_action

    def reset(self):
        # print(type(self.env))
        seed = self.random_state.randint(low=0, high=int(1e8))
        obs, info = self.env.reset(seed=seed)

        obs = self.process_obs(obs)

        self.ep_reward = 0
        self.ep_len = 0

        return obs

    def step(self, action):

        pose_action = self.process_action(action)

        # print(pose_action)
        raw_obs, reward, done, truncated, info = self.env.step(pose_action)

        obs = self.process_obs(raw_obs)
        info.update(dict(raw_obs=raw_obs, raw_action=pose_action))

        done = done or info.get("is_success", False)
        reward = reward * 2.

        self.ep_reward += reward
        self.ep_len += 1

        if done or truncated:
            info.update(dict(episode=dict(r=self.ep_reward, l=self.ep_len)))
            # self.reset()
            # done = True

        return obs, reward, done, truncated, info

    def render(self, **kwargs):
        images = self.env.capture_images_new()
        image = images["third-rgb"]
        return image
