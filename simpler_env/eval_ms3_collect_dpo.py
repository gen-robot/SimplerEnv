from collections import defaultdict
import json
import os
import signal
import time
import numpy as np
from typing import Annotated, Optional

import torch
import tree
from mani_skill.utils import common
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video

signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c

import gymnasium as gym
import numpy as np
from PIL import Image
from mani_skill.envs.sapien_env import BaseEnv
import tyro
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Args:
    """
    This is a script to evaluate policies on real2sim environments. Example command to run:

    XLA_PYTHON_CLIENT_PREALLOCATE=false python real2sim_eval_maniskill3.py \
        --model="octo-small" -e "PutEggplantInBasketScene-v1" -s 0 --num-episodes 192 --num-envs 64
    """

    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PutCarrotOnPlateInScene-v1"
    """The environment ID of the task you want to simulate. Can be one of
    PutCarrotOnPlateInScene-v1, PutSpoonOnTableClothInScene-v1, StackGreenCubeOnYellowCubeBakedTexInScene-v1, PutEggplantInBasketScene-v1"""

    shader: str = "default"  # default, rt

    num_envs: int = 5
    """Number of environments to run. With more than 1 environment the environment will use the GPU backend 
    which runs faster enabling faster large-scale evaluations. Note that the overall behavior of the simulation
    will be slightly different between CPU and GPU backends."""

    num_episodes: int = 80
    """Number of episodes to run and record evaluation metrics over"""

    max_episode_len: int = 80
    """Max episode length"""

    num_trails: int = 5
    """Number of trails per episode"""

    record_dir: str = "videos"
    """The directory to save videos and results"""

    ckpt_path: str = ""
    """Checkpoint path for models. Only used for RT models"""

    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Seed the model and environment. Default seed is 0"""

    info_on_video: bool = False
    """Whether to write info text onto the video"""

    save_video: bool = True
    """Whether to save videos"""

    debug: bool = False

    # openvla specific
    num_train_carrots: int = 16
    unnorm_key: str = "bridge_orig"


def get_robot_control_mode(robot: str):
    if "google_robot_static" in robot:
        return "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
    elif "widowx" in robot:
        return "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
    else:
        raise NotImplementedError(f"Robot {robot} not supported")


def main():
    args = tyro.cli(Args)
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Setup up the policy inference model
    policy_setup = "widowx_bridge"

    env: BaseEnv = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        obs_mode="rgb+segmentation",
        control_mode=get_robot_control_mode(policy_setup),
        sim_backend="gpu",
        sim_config={
            "sim_freq": 500,
            "control_freq": 5,
        },
        max_episode_steps=args.max_episode_len,
        sensor_configs={"shader_pack": args.shader},
    )
    sim_backend = 'gpu' if env.device.type == 'cuda' else 'cpu'

    from simpler_env.policies.openvla.openvla_infer import OpenVLAInference
    model = OpenVLAInference(saved_model_path=args.ckpt_path, policy_setup=policy_setup, action_scale=1.0,
                             unnorm_key=args.unnorm_key)

    model_name = Path(args.ckpt_path).name if args.ckpt_path else "random"
    exp_dir = Path(args.record_dir) / f"dpo/{model_name}_{args.env_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    eval_metrics = defaultdict(list)

    print(f"Using {args.num_envs} environments on the {sim_backend} simulation backend")

    timers = {"env.step+inference": 0, "env.step": 0, "inference": 0, "total": 0}
    total_start_time = time.time()

    for idx_episode in range(args.num_episodes):
        ep_id = torch.randint(1000000000, size=(1,), device=env.device).repeat(args.num_envs)

        # data dump
        datas = [{
            "image": [],  # obs_t: [0, T-1]
            "instruction": "",
            "action": [],  # a_t: [0, T-1]
            "info": [],  # info after executing a_t: [1, T]
        } for idx in range(args.num_envs)]

        # env and policy reset
        options = {
            "episode_id": ep_id,
            "obj_set": "train", # train, test, all
            "num_train_carrots": args.num_train_carrots
        }
        obs, info = env.reset(options=options)
        obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
        instruction = env.unwrapped.get_language_instruction()
        assert all([ins == instruction[0] for ins in instruction])
        model.reset(instruction)

        print("instruction[0]:", instruction[0])

        # data dump: instruction
        for idx in range(args.num_envs):
            datas[idx]["instruction"] = instruction[idx]

        elapsed_steps = 0
        predicted_terminated, truncated = False, False
        while not (predicted_terminated or truncated):
            # inference
            start_time = time.time()

            raw_action, action = model.step(obs_image, instruction)
            action = torch.cat([action["world_vector"], action["rot_axangle"], action["gripper"]], dim=1)
            # action = env.action_space.sample() # random

            timers["inference"] += time.time() - start_time

            # step
            start_time = time.time()

            obs, reward, terminated, truncated, info = env.step(action)
            obs_image_new = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
            info = {k: v.cpu().numpy() for k, v in info.items()}
            truncated = bool(truncated.any())  # note that all envs truncate and terminate at the same time.

            timers["env.step"] += time.time() - start_time

            # print info
            info_dict = {k: v.mean().tolist() for k, v in info.items()}
            print(f"step {elapsed_steps}: {info_dict}")

            # data dump: image, action, info
            for i in range(args.num_envs):
                log_image = obs_image[i].cpu().numpy()
                log_action = action[i].cpu().numpy().tolist()
                log_info = {k: v[i].tolist() for k, v in info.items()}
                datas[i]["image"].append(log_image)
                datas[i]["action"].append(log_action)
                datas[i]["info"].append(log_info)

            # add count
            obs_image = obs_image_new
            elapsed_steps += 1

        # data dump: last image
        for i in range(args.num_envs):
            log_image = obs_image[i].cpu().numpy()
            datas[i]["image"].append(log_image)

        # save data
        for i in range(args.num_envs):
            is_grasp = datas[i]["info"][-1]["is_src_obj_grasped"]
            cons_grasp = datas[i]["info"][-1]["consecutive_grasp"]
            success = datas[i]["info"][-1]["success"]

            reward = 0
            reward += is_grasp * 0.1
            reward += cons_grasp * 0.1
            reward += (success & is_grasp) * 1.0

            folder = exp_dir / f"episode_{idx_episode:0>3d}"
            folder.mkdir(parents=True, exist_ok=True)
            path_name = folder / f"trail_{i:0>4d}-g_{is_grasp}-cg_{cons_grasp}-s_{success}-reward_{reward:.1f}.npy"

            res = datas[i].copy()
            res["image"] = [Image.fromarray(im).convert("RGB") for im in res["image"]]
            np.save(path_name, res)

        # metrics log and print
        for k, v in info.items():
            eval_metrics[k].append(v.flatten())
            print(f"{k}: {np.mean(eval_metrics[k])}")


    # Print timing information
    timers["total"] = time.time() - total_start_time
    timers["env.step+inference"] = timers["env.step"] + timers["inference"]

    print("\nTiming Info:")
    for key, value in timers.items():
        print(f"{key}: {value:.2f} seconds")


if __name__ == "__main__":
    main()
