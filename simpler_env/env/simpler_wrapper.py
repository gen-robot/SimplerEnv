import gymnasium as gym
import torch
from mani_skill.envs.sapien_env import BaseEnv

def get_robot_control_mode(robot: str):
    if "google_robot_static" in robot:
        return "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
    elif "widowx" in robot:
        return "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
    else:
        raise NotImplementedError(f"Robot {robot} not supported")


class SimlerWrapper:
    def __init__(self, env_id, num_envs, policy_setup, seed):
        self.seed = seed
        robot_control_mode = get_robot_control_mode(policy_setup)

        env_config = dict(
            id=env_id,
            num_envs=num_envs,
            obs_mode="rgb+segmentation",
            control_mode=robot_control_mode,
            sim_backend="gpu",
            sim_config={
                "sim_freq": 500,
                "control_freq": 5,
            },
            max_episode_steps=100,
            sensor_configs={"shader_pack": "default"},
        )
        self.env: BaseEnv = gym.make(**env_config)

    def reset(self):
        obs, info = self.env.reset(seed=self.seed)
        obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
        instruction = self.env.unwrapped.get_language_instruction()

        # todo: add instruction support

        obs_image = obs_image.reshape(obs_image.shape[0], -1)

        return obs_image, info


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
        reward = reward.reshape(-1, 1)
        terminated = terminated.reshape(-1, 1)
        truncated = truncated.reshape(-1, 1)

        obs_image = obs_image.reshape(obs_image.shape[0], -1)

        # todo: auto reset

        return obs_image, reward, terminated, truncated, info

