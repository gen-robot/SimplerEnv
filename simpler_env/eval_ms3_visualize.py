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
from simpler_env.utils.env.observation_utils import get_image_from_maniskill3_obs_dict

import gymnasium as gym
import numpy as np
from sapien.core import Pose
from transforms3d.euler import euler2quat
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

    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PutEggplantInBasketScene-v1" 
    """The environment ID of the task you want to simulate. Can be one of
    PutCarrotOnPlateInScene-v1, PutSpoonOnTableClothInScene-v1, StackGreenCubeOnYellowCubeBakedTexInScene-v1, PutEggplantInBasketScene-v1 and

    PandaPutCarrotOnPlateInScene-v1, PandaPutSpoonOnTableClothInScene-v1, PandaStackGreenCubeOnYellowCubeBakedTexInScene-v1, PandaPutEggplantInBasketScene-v1"""

    shader: str = "default" # default, rt

    num_envs: int = 1
    """Number of environments to run. With more than 1 environment the environment will use the GPU backend 
    which runs faster enabling faster large-scale evaluations. Note that the overall behavior of the simulation
    will be slightly different between CPU and GPU backends."""

    num_episodes: int = 100
    """Number of episodes to run and record evaluation metrics over"""

    record_dir: str = "videos"
    """The directory to save videos and results"""

    model: Optional[str] = None # 'rt-1x' rdt octo-base octo-small
    """The model to evaluate on the given environment. Can be one of octo-base, octo-small, rt-1x. If not given, random actions are sampled."""

    ckpt_path: str = ""
    """Checkpoint path for models. Used for RT and RDT models"""

    policy_setup: str = "widowx_bridge" # widowx_bridge, franka

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
    openvla_unnorm_key: Optional[str] = None

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
    policy_setup = args.policy_setup

    env: BaseEnv = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        sensor_configs={"shader_pack": args.shader},
        obs_mode="rgb+segmentation",
        control_mode=get_robot_control_mode(policy_setup), # "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos", # In BaseBridgeEnv and WidowX250SBridgeDatasetFlatTable
        sim_backend = 'gpu',
        sim_config={
            "sim_freq": 500,
            "control_freq": 5,
        },
        max_episode_steps=100,
    )
    sim_backend = 'gpu' if env.device.type == 'cuda' else 'cpu'

    verbose = True
    if verbose:
        print("---------------------------------------------")
        print("Observation space", env.observation_space)
        print("\nAction space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)
        print("---------------------------------------------")

    # Setup up the policy inference model
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
    elif args.model == "rdt":
        from third_party.rdt.constants import RDT1B_FT_PATH, RDT1B_PATH
        from simpler_env.policies.rdt.rdt_model import RDTInference
        model = RDTInference(ctrl_freq = 25, action_scale=1, robot_name=policy_setup, dtype=torch.bfloat16, action_horizon=1,
            env = env, enable_eef_obs = True, enable_qvel_obs = False, pretrained_checkpoint=args.ckpt_path, # RDT1B_FT_PATH， RDT1B_PATH
        )
    else:
        model = None
        raise ValueError(f"Model {args.model} does not exist / is not supported.")


    # model_name = args.model if args.model is not None else "random"
    model_name = Path(args.ckpt_path).name if args.ckpt_path else "random"
    timestamp  = f"{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    exp_dir = os.path.join(args.record_dir, f"real2sim_eval/{model_name}_{args.env_id}",timestamp)
    if model_name == "random":
        print("Using random actions.")
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    eval_metrics = defaultdict(list)
    eps_count = 0

    print(f"Running Real2Sim Evaluation of model {args.model} on environment {args.env_id}")
    print(f"Using {args.num_envs} environments on the {sim_backend} simulation backend")

    timers = {"env.step+inference": 0, "env.step": 0, "inference": 0, "total": 0}
    total_start_time = time.time()
    print("max_num_episodes: ", args.num_episodes)

    while eps_count < args.num_episodes:
        seed = args.seed + eps_count

        env_reset_options = {
            "episode_id": torch.arange(args.num_envs) + eps_count
        }
        obs, _ = env.reset(seed=seed, options=env_reset_options)
        instruction = env.unwrapped.get_language_instruction()
        print("instruction:", instruction[0])
        if model is not None:
            model.reset(instruction[0])
        images = []
        predicted_terminated, truncated = False, False
        images.append(get_image_from_maniskill3_obs_dict(env, obs))
        elapsed_steps = 0
        while not (predicted_terminated or truncated):
            if model is not None:
                start_time = time.time()

                if not args.model == 'rdt':
                    raw_action, action = model.step(images[-1], instruction[0]) # 0?
                    action = torch.cat([action["world_vector"], action["rot_axangle"], action["gripper"]], dim=1) # 0 ?
                else:
                    raw_action, action = model.step(obs, instruction[0])
                    # # x- > front, y -> left, z -> up, rot_1, rot_2, rot_3, gripper[-1 -> close, 1 -> open] -> 7 dimension 
                    action = torch.cat([torch.as_tensor(action["world_vector"]), torch.as_tensor(action["rot_axangle"]), 
                                        torch.as_tensor(action["gripper"])], dim=0).to(dtype=torch.float32)
                    # action = torch.cat([torch.as_tensor([0,0,0]), torch.as_tensor([0,0,0]),
                    #                     torch.as_tensor([0.5])], dim=0).to(dtype=torch.float32)

                timers["inference"] += time.time() - start_time
            else:
                action = env.action_space.sample()

            if elapsed_steps > 0:
                if args.save_video and args.info_on_video:
                    for i in range(len(images[-1])):
                        images[-1][i] = visualization.put_info_on_image(images[-1][i],
                                                                        tree.map_structure(lambda x: x[i], info))

            start_time = time.time()
            print("step:", elapsed_steps, "    ", "delta_action_pose:", action)
            obs, reward, terminated, truncated, info = env.step(action)
            timers["env.step"] += time.time() - start_time
            elapsed_steps += 1
            info = common.to_numpy(info)

            truncated = bool(truncated.any())  # note that all envs truncate and terminate at the same time.
            images.append(get_image_from_maniskill3_obs_dict(env, obs))

        for k, v in info.items():
            eval_metrics[k].append(v.flatten())
        if args.save_video:
            for i in range(len(images[-1])):
                images_to_video([img[i].cpu().numpy() for img in images], exp_dir,
                                f"{sim_backend}_eval_{seed + i}_success={info['success'][i].item()}", fps=10,
                                verbose=True)
        eps_count += args.num_envs
        if args.num_envs == 1:
            print(f"Evaluated episode {eps_count}. Seed {seed}. Results after {eps_count} episodes:")
        else:
            print(
                f"Evaluated {args.num_envs} episodes, seeds {seed} to {eps_count}. Results after {eps_count} episodes:")
        for k, v in eval_metrics.items():
            print(f"{k}: {np.mean(v)}")
    # Print timing information
    timers["total"] = time.time() - total_start_time
    timers["env.step+inference"] = timers["env.step"] + timers["inference"]
    mean_metrics = {k: np.mean(v) for k, v in eval_metrics.items()}
    mean_metrics["total_episodes"] = eps_count
    mean_metrics["time/episodes_per_second"] = eps_count / timers["total"]
    print("Timing Info:")
    for key, value in timers.items():
        mean_metrics[f"time/{key}"] = value
        print(f"{key}: {value:.2f} seconds")
    metrics_path = os.path.join(exp_dir, f"{sim_backend}_eval_metrics.json")
    if sim_backend == "gpu":
        metrics_path = metrics_path.replace("gpu", f"gpu_{args.num_envs}_envs")
    with open(metrics_path, "w") as f:
        json.dump(mean_metrics, f, indent=4)
    print(f"Evaluation complete. Results saved to {exp_dir}. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
