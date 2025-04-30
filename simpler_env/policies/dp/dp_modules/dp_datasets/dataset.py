import numpy as np
import torch
import os
import pickle
from time import time
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import colorsys
from tqdm import tqdm
import random
import time
import imageio
from transforms3d.quaternions import qmult, qconjugate, quat2mat, mat2quat
import sys
sys.path.append('/home/zhouzhiting/Projects')

from diffusion_policy.utils.math_utils import wrap_to_pi, euler2quat, quat2euler, get_pose_from_rot_pos
from tqdm import tqdm
import copy


class Sim2SimEpisodeDataset(Dataset):
    def __init__(
            self,
            data_roots,
            num_seeds,
            chunk_size,
            split="train",
            norm_stats_path=None,
            augment_images=True,
            use_desired_action=True,
            **kwargs
    ):
        super(Sim2SimEpisodeDataset).__init__()
        self.data_roots = data_roots
        self.camera_names = kwargs.get("camera_names", ["third", "wrist"])

        self.chunk_size = chunk_size
        self.augment_images = augment_images
        self.use_desired_action = use_desired_action

        self.transformations = None

        self.split = split

        for data_root in self.data_roots:
            assert os.path.exists(data_root), f"Path {data_root} does not exist"
        assert self.split == "train" or self.split == "val"

        episode_list = []
        num_steps_list = []

        for data_idx, data_root in enumerate(self.data_roots):
            for s in range(num_seeds):
                seed_path = os.path.join(data_root, f"seed_{s}")
                ep_info = pickle.load(open(os.path.join(seed_path, "info.pkl"), "rb"))
                for ep_id, suc, num_steps in ep_info:
                    if suc == "s" and num_steps > 1:
                        item_index = (data_idx, s, ep_id)
                        episode_list.append(item_index)
                        num_steps_list.append(num_steps)

        # print(len(index_list))
        if split == "train":
            self.episode_list = episode_list[:int(0.99 * len(episode_list))]
            self.num_steps_list = num_steps_list[:int(0.99 * len(episode_list))]
        else:
            self.episode_list = episode_list[int(0.99 * len(episode_list)):]
            self.num_steps_list = num_steps_list[int(0.99 * len(episode_list)):]

        assert len(self.episode_list) == len(self.num_steps_list)
        self.cum_steps_list = np.cumsum(self.num_steps_list)

        print(split, len(self.episode_list), self.cum_steps_list[-1])

        if norm_stats_path is None:
            stats = self.compute_normalize_stats()
        else:
            stats = pickle.load(open(norm_stats_path, "rb"))
        print(stats)
        self.update_obs_normalize_params(stats)
        # exit()

        self.__getitem__(0)  # initialize self.transformations

    def __len__(self):
        return self.cum_steps_list[-1]

    def __getitem__(self, index: int):
        result = self.get_unnormalized_item(index)

        robot_state = result["robot_state"]
        proprio_state = result["proprio_state"]
        is_pad = result["is_pad"]
        action_chunk = result["action"]
        robot_state = (
                              robot_state - self.pose_gripper_mean
                      ) / self.pose_gripper_scale
        action_chunk[~is_pad] = (
                                        action_chunk[~is_pad] - np.expand_dims(self.pose_gripper_mean, axis=0)
                                ) / np.expand_dims(self.pose_gripper_scale, axis=0)
        proprio_state = (
                                proprio_state - self.proprio_gripper_mean
                        ) / self.proprio_gripper_scale

        result["robot_state"] = torch.from_numpy(robot_state)
        result["proprio_state"] = torch.from_numpy(proprio_state)
        result["action"] = torch.from_numpy(action_chunk)
        result["is_pad"] = torch.from_numpy(is_pad)

        images = torch.from_numpy(result["images"])
        images = torch.einsum('k h w c -> k c h w', images)

        if self.transformations is None:
            print('Initializing transformations')
            original_size = images.shape[2:]
            ratio = 0.95
            self.transformations = [
                transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                transforms.Resize((224, 224), antialias=True),
            ]

        if self.augment_images:
            for transform in self.transformations:
                images = transform(images)

        images = images / 255.0
        result["images"] = images

        return result

    def _locate(self, index: int):
        assert index < len(self)
        traj_idx = np.where(self.cum_steps_list > index)[0][0]
        steps_before = self.cum_steps_list[traj_idx - 1] if traj_idx > 0 else 0
        start_ts = index - steps_before
        return traj_idx, start_ts

    def update_obs_normalize_params(self, obs_normalize_params):
        self.OBS_NORMALIZE_PARAMS = copy.deepcopy(obs_normalize_params)
        pickle.dump(obs_normalize_params, open(os.path.join(self.data_roots[0], f"norm_stats_{len(self.data_roots)}.pkl"), "wb"))
        # self.joint_gripper_mean = np.concatenate(
        #     [
        #         self.OBS_NORMALIZE_PARAMS[key]["mean"]
        #         for key in ["joint", "gripper_width"]
        #     ]
        # )
        # self.joint_gripper_scale = np.concatenate(
        #     [
        #         self.OBS_NORMALIZE_PARAMS[key]["scale"]
        #         for key in ["joint", "gripper_width"]
        #     ]
        # )
        self.pose_gripper_mean = np.concatenate(
            [
                self.OBS_NORMALIZE_PARAMS[key]["mean"]
                for key in ["pose", "gripper_width"]
            ]
        )
        self.pose_gripper_scale = np.concatenate(
            [
                self.OBS_NORMALIZE_PARAMS[key]["scale"]
                for key in ["pose", "gripper_width"]
            ]
        )

        self.proprio_gripper_mean = np.concatenate(
            [
                self.OBS_NORMALIZE_PARAMS[key]["mean"]
                for key in ["proprio_state", "gripper_width"]
            ]
        )
        self.proprio_gripper_scale = np.concatenate(
            [
                self.OBS_NORMALIZE_PARAMS[key]["scale"]
                for key in ["proprio_state", "gripper_width"]
            ]
        )

    def get_unnormalized_item(self, index):
        result_dict = {}
        result_dict["lang"] = " "
        traj_idx, start_ts = self._locate(index)
        # image: start_ts
        # pose: [start_ts, start_ts + chunk_size]
        end_ts = min(self.num_steps_list[traj_idx], start_ts + self.chunk_size + 1)
        is_pad = np.zeros((self.chunk_size,), dtype=bool)
        if end_ts < start_ts + self.chunk_size + 1:
            is_pad[-(start_ts + self.chunk_size + 1 - end_ts):] = True
        result_dict["is_pad"] = is_pad

        data_idx, s, ep_id = self.episode_list[traj_idx]
        data_root = self.data_roots[data_idx]

        # image
        images = []
        for cam in self.camera_names:
            image_path = os.path.join(data_root, f"seed_{s}", f"ep_{ep_id}", f"step_{start_ts}_cam_{cam}.jpg")
            image = imageio.imread(image_path)
            images.append(image)
        images = np.stack(images, axis=0)
        result_dict["images"] = images

        # pose
        action_chunk = np.zeros((self.chunk_size, 10), dtype=np.float32)
        pose_at_obs = None
        pose_chunk = []
        gripper_width_chunk = []
        proprio_state = np.zeros((10,), dtype=np.float32)
        robot_state = np.zeros((10,), dtype=np.float32)

        prev_pose = None

        for step_idx in range(start_ts, end_ts):
            data_path = os.path.join(data_root, f"seed_{s}", f"ep_{ep_id}", f"step_{step_idx}.pkl")
            data = pickle.load(open(data_path, "rb"))
            tcp_pose = data["tcp_pose"]
            pose_p, pose_q = tcp_pose[:3], tcp_pose[3:]
            pose_mat = quat2mat(pose_q)
            pose = get_pose_from_rot_pos(
                pose_mat, pose_p
            )

            if step_idx == start_ts:
                pose_at_obs = pose
                pose_mat_6 = pose_mat[:, :2].reshape(-1)
                proprio_state[:] = np.concatenate(
                    [
                        pose_p,
                        pose_mat_6,
                        np.array([data["gripper_width"]]),
                    ]
                )
                robot_state[-1] = data["gripper_width"]

            elif step_idx > start_ts:
                if self.use_desired_action:
                    desired_action = data["action"]
                    # desired_dp = desired_action[2:5]
                    desired_dp = desired_action[:3]
                    # desired_dq = euler2quat(desired_action[5:8])
                    desired_dq = euler2quat(desired_action[3:6])
                    desired_gripper_width = desired_action[-1]

                    desired_d_pose = get_pose_from_rot_pos(
                        quat2mat(desired_dq), desired_dp
                    )
                    desired_pose = prev_pose @ desired_d_pose

                    pose_chunk.append(desired_pose)
                    gripper_width_chunk.append(np.array([desired_gripper_width]))
                else:
                    pose_chunk.append(pose)
                    gripper_width_chunk.append(np.array([data["gripper_width"]]))

            prev_pose = pose

        # make relative
        _pose_relative = np.eye(4)
        robot_state[:9] = np.concatenate(
            [_pose_relative[:3, 3], _pose_relative[:3, :2].reshape(-1)]
        )
        for i in range(end_ts - start_ts - 1):
            _pose_relative = np.linalg.inv(pose_at_obs) @ pose_chunk[i]
            action_chunk[i] = np.concatenate(
                [
                    _pose_relative[:3, 3],
                    _pose_relative[:3, :2].reshape(-1),
                    gripper_width_chunk[i],
                ]
            )

        result_dict["robot_state"] = robot_state
        result_dict["proprio_state"] = proprio_state
        result_dict["action"] = action_chunk
        return result_dict

    def compute_normalize_stats(self, scale_eps=0.03):
        print("compute normalize stats...")
        # min and max scale
        joint_min, joint_max = None, None
        gripper_width_min, gripper_width_max = None, None
        pose_min, pose_max = None, None
        proprio_min, proprio_max = None, None

        def safe_minimum(a: np.ndarray, b: np.ndarray):
            if a is None:
                return b
            if b is None:
                return a
            return np.minimum(a, b)

        def safe_maximum(a: np.ndarray, b: np.ndarray):
            if a is None:
                return b
            if b is None:
                return a
            return np.maximum(a, b)

        def safe_min(a: np.ndarray, axis: int):
            if a.shape[axis] == 0:
                return None
            return np.min(a, axis=axis)

        def safe_max(a: np.ndarray, axis: int):
            if a.shape[axis] == 0:
                return None
            return np.max(a, axis=axis)

        for i in tqdm(range(len(self))):
            item_dict = self.get_unnormalized_item(i)
            pose = item_dict["robot_state"][:9]
            action_pose = item_dict["action"][~item_dict["is_pad"]][:, :9]
            gripper_width = item_dict["robot_state"][9:10]
            action_gripper_width = item_dict["action"][~item_dict["is_pad"]][:, 9:10]
            proprio_pose = item_dict["proprio_state"][:9]

            pose_min = safe_minimum(
                safe_minimum(pose_min, pose), safe_min(action_pose, axis=0)
            )
            pose_max = safe_maximum(
                safe_maximum(pose_max, pose), safe_max(action_pose, axis=0)
            )
            gripper_width_min = safe_minimum(
                safe_minimum(gripper_width_min, gripper_width),
                safe_min(action_gripper_width, axis=0),
            )
            gripper_width_max = safe_maximum(
                safe_maximum(gripper_width_max, gripper_width),
                safe_max(action_gripper_width, axis=0),
            )
            proprio_min = safe_minimum(
                proprio_min, proprio_pose
            )
            proprio_max = safe_maximum(
                proprio_max, proprio_pose
            )

        params = {}
        params["pose"] = {
            "mean": (pose_min + pose_max) / 2,
            "scale": np.maximum((pose_max - pose_min) / 2, scale_eps),
        }
        params["gripper_width"] = {
            "mean": (gripper_width_min + gripper_width_max) / 2,
            "scale": np.maximum((gripper_width_max - gripper_width_min) / 2, scale_eps),
        }
        params["proprio_state"] = {
            "mean": (proprio_min + proprio_max) / 2,
            "scale": np.maximum((proprio_max - proprio_min) / 2, scale_eps),
        }
        return params


def step_collate_fn(samples):
    batch = {}
    for key in samples[0].keys():
        if key != "lang":
            # print(key, samples[0][key].shape)
            batched_array = torch.stack([sample[key] for sample in samples], dim=0)
            batch[key] = batched_array
    batch["lang"] = [sample["lang"] for sample in samples]
    return batch


def load_sim2sim_data(data_roots, num_seeds, train_batch_size, val_batch_size, chunk_size, **kwargs):
    # construct dataset and dataloader
    train_dataset = Sim2SimEpisodeDataset(
        data_roots,
        num_seeds,
        split="train",
        chunk_size=chunk_size,
        norm_stats_path=os.path.join(data_roots[0], f"norm_stats_{len(data_roots)}.pkl"),
        **kwargs
    )
    val_dataset = Sim2SimEpisodeDataset(
        data_roots,
        num_seeds,
        split="val",
        chunk_size=chunk_size,
        norm_stats_path=os.path.join(data_roots[0], f"norm_stats_{len(data_roots)}.pkl"),
        **kwargs
    )
    train_num_workers = 32 #8
    val_num_workers = 32 #8
    print(
        f'Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=True,
        collate_fn=step_collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=True,
        collate_fn=step_collate_fn,
    )
    return train_dataloader, val_dataloader, train_dataset.OBS_NORMALIZE_PARAMS, val_dataset.OBS_NORMALIZE_PARAMS
