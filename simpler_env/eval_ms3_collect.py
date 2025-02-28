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

    num_envs: int = 1
    """Number of environments to run. With more than 1 environment the environment will use the GPU backend 
    which runs faster enabling faster large-scale evaluations. Note that the overall behavior of the simulation
    will be slightly different between CPU and GPU backends."""

    num_episodes: int = 100
    """Number of episodes to run and record evaluation metrics over"""

    record_dir: str = "videos"
    """The directory to save videos and results"""

    model: Optional[str] = None
    """The model to evaluate on the given environment. Can be one of octo-base, octo-small, rt-1x. If not given, random actions are sampled."""

    ckpt_path: str = ""
    """Checkpoint path for models. Only used for RT models"""

    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Seed the model and environment. Default seed is 0"""

    reset_by_episode_id: bool = True
    """Whether to reset by fixed episode ids instead of random sampling initial states."""

    info_on_video: bool = False
    """Whether to write info text onto the video"""

    save_video: bool = True
    """Whether to save videos"""

    debug: bool = False

    # openvla specific
    openvla_unnorm_key: str = None


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

    # Setup up the policy inference model
    print(f"model is {args.model}")
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
        max_episode_steps=100,
        sensor_configs={"shader_pack": args.shader},
    )
    sim_backend = 'gpu' if env.device.type == 'cuda' else 'cpu'

    if args.model == "octo-base" or args.model == "octo-small":
        from simpler_env.policies.octo.octo_model import OctoInference
        model = OctoInference(model_type=args.model, policy_setup=policy_setup, init_rng=args.seed, action_scale=1)
    elif args.model == "rt-1x":
        from simpler_env.policies.rt1.rt1_model import RT1Inference
        model = RT1Inference(saved_model_path=args.ckpt_path, policy_setup=policy_setup, action_scale=1)
    elif args.model == "openvla":
        from simpler_env.policies.openvla.openvla_model_ms3 import OpenVLAInference
        model = OpenVLAInference(saved_model_path=args.ckpt_path, policy_setup=policy_setup, action_scale=1.,
                                 unnorm_key=args.openvla_unnorm_key)
    elif args.model == "cogact":
        from simpler_env.policies.sim_cogact import CogACTInference
        model = CogACTInference(
            saved_model_path=args.ckpt_path,  # e.g., CogACT/CogACT-Base
            policy_setup=policy_setup,
            action_scale=1.0,
            action_model_type='DiT-L',
            cfg_scale=1.5  # cfg from 1.5 to 7 also performs well
        )
    elif args.model == "spatialvla":
        from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference
        model = SpatialVLAInference(saved_model_path=args.ckpt_path, policy_setup=policy_setup, action_scale=1.0, )
    else:
        raise NotImplementedError

    model_name = Path(args.ckpt_path).name if args.ckpt_path else "random"
    exp_dir = Path(args.record_dir) / f"collect/{model_name}_{args.env_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    eval_metrics = defaultdict(list)
    eps_count = 0

    print(f"Running Real2Sim Evaluation of model {args.model} on environment {args.env_id}")
    print(f"Using {args.num_envs} environments on the {sim_backend} simulation backend")

    timers = {"env.step+inference": 0, "env.step": 0, "inference": 0, "total": 0}
    total_start_time = time.time()

    while eps_count < args.num_episodes:
        seed = args.seed + eps_count

        env_reset_options = {
            "episode_id": torch.arange(args.num_envs) + eps_count
        }
        obs, _ = env.reset(seed=seed, options=env_reset_options)
        obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
        instruction = env.unwrapped.get_language_instruction()
        model.reset(instruction)

        print("instruction[0]:", instruction[0])

        datas = [{
            "image": [],
            "instruction": instruction[idx],
            "action": [],
            "info": [],

        } for idx in range(args.num_envs)]

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
            obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
            info = {k: v.cpu().numpy() for k, v in info.items()}
            elapsed_steps += 1
            truncated = bool(truncated.any())  # note that all envs truncate and terminate at the same time.

            timers["env.step"] += time.time() - start_time

            info_dict = {k: v.mean().tolist() for k, v in info.items()}
            print(f"step {elapsed_steps}: {info_dict}")

            # save data
            for i in range(args.num_envs):
                log_image = Image.fromarray(obs_image[i].cpu().numpy()).convert("RGB")
                log_action = action[i].cpu().numpy().tolist()
                log_info = {k: v[i].tolist() for k, v in info.items()}
                datas[i]["image"].append(log_image)
                datas[i]["action"].append(log_action)
                datas[i]["info"].append(log_info)

        # save data
        for i in range(args.num_envs):
            if np.sum([d["success"] for d in datas[i]["info"]]) < 6:
                continue
            np.save(exp_dir / f"{eps_count + i:0>4d}.npy", datas[i])


        for k, v in info.items():
            eval_metrics[k].append(v.flatten())
            print(f"{k}: {np.mean(eval_metrics[k])}")

        eps_count += args.num_envs


    # Print timing information
    timers["total"] = time.time() - total_start_time
    timers["env.step+inference"] = timers["env.step"] + timers["inference"]

    print("\nTiming Info:")
    for key, value in timers.items():
        print(f"{key}: {value:.2f} seconds")


if __name__ == "__main__":
    main()
