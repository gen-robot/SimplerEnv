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
from transforms3d.quaternions import qmult, qconjugate, quat2mat, mat2quat
import sys
sys.path.append('/home/zhouzhiting/Projects')
from collections import defaultdict
import numpy as np
import threading

from diffusion_policy.utils.math_utils import wrap_to_pi, euler2quat, quat2euler, get_pose_from_rot_pos
import copy
from turbojpeg import TurboJPEG
from turbojpeg import TJPF_RGB


timing_stats = defaultdict(list)
timing_lock = threading.Lock()  
batch_counter = 0
print_interval = 10 

class TimingContext:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        with timing_lock:
            timing_stats[self.name].append(elapsed)


def print_timing_stats():
    print("\n===== 数据加载时间统计 =====")
    for operation, times in sorted(timing_stats.items()):
        if times:
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            print(f"{operation:<30} - 平均: {avg_time*1000:.2f}ms, 最大: {max_time*1000:.2f}ms, 最小: {min_time*1000:.2f}ms, 调用次数: {len(times)}")
    print("=========================\n")



class Sim2SimEpisodeDatasetEff(Dataset):
    def __init__(
            self,
            data_roots,
            num_seeds,
            chunk_size,
            split="train",
            norm_stats_path=None,
            augment_images=True,
            use_desired_action=True,
            use_pre_img=False,   # for debug, compare with pre_img
            **kwargs
    ):
        super().__init__()
        with TimingContext("dataset_initialization"):  #time
            
            self.jpeg = TurboJPEG()

            self.data_roots = data_roots
            self.camera_names = kwargs.get("camera_names", ["third", "wrist"])
            self.chunk_size = chunk_size
            self.augment_images = augment_images
            self.use_desired_action = use_desired_action
            self.transformations = None
            self.split = split
            self.use_pre_img = use_pre_img 
            
            self.episode_data = {}
            episode_list = []
            num_steps_list = []

            with TimingContext("total_steps loading"): #time
                for data_idx, data_root in enumerate(self.data_roots):
                    for s in range(num_seeds):
                        seed_path = os.path.join(data_root, f"seed_{s}")
                        total_steps_path = os.path.join(seed_path, "ep_0", "total_steps.npz")  # total_steps_new.npz
                        if not os.path.exists(total_steps_path):
                            continue
                        ep_info = pickle.load(open(os.path.join(seed_path, "info.pkl"), "rb"))
                        for ep_id, suc, num_steps in ep_info:
                            if suc == "f" or num_steps <= 1:
                                continue
                            else:
                                data = np.load(total_steps_path, allow_pickle=True)
                                item_index = (data_idx, s, 0)  # ep_id 始终为0
                                episode_list.append(item_index)
                                num_steps_list.append(num_steps)
                    
                                self.episode_data[item_index] = {
                                    'tcp_pose': data['tcp_pose'],
                                    'gripper_width': data['gripper_width'],
                                    'robot_joints': data['robot_joints'],
                                    'privileged_obs': data['privileged_obs'],
                                    'action': data['action'],
                                    'desired_grasp_pose': data['desired_grasp_pose'],
                                    'desired_gripper_width': data['desired_gripper_width']
                                }
            print("episode_list", len(episode_list))                           
            if split == "train":
                self.episode_list = episode_list[:int(0.99 * len(episode_list))]
                self.num_steps_list = num_steps_list[:int(0.99 * len(episode_list))]
            else:
                self.episode_list = episode_list[int(0.99 * len(episode_list)):]
                self.num_steps_list = num_steps_list[int(0.99 * len(episode_list)):]

            self.cum_steps_list = np.cumsum(self.num_steps_list)
            print(split, len(self.episode_list), self.cum_steps_list[-1])

            if norm_stats_path is None:
                stats = self.compute_normalize_stats()
            else:
                with TimingContext("load_norm_stats"):
                    stats = pickle.load(open(norm_stats_path, "rb"))
            self.update_obs_normalize_params(stats)

            result_0 = self.__getitem__(0) 
            # for debug
            # with open("dataset_array_result_0.txt", "w") as f:
            #     f.write(str(result_0))
            # # Save images from result_0 for debugging
            # images = result_0["images"]
            # img = images[-1]
            # img_path = f"dataset_array_result_0_image1.png"
            # transforms.ToPILImage()(img).save(img_path)

    def __len__(self):
        return self.cum_steps_list[-1]

    def __getitem__(self, index: int):
        with TimingContext("getitem_total"):  
            with TimingContext("get_unnormalized_item"):    
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
            # for debug 
            # result["pose_chunk"] = torch.from_numpy(result["pose_chunk"]) #
            

            images = torch.from_numpy(result["images"])
            images = torch.einsum('k h w c -> k c h w', images)

            if self.use_pre_img == False:
                with TimingContext("image_processing"):    
                    if self.transformations is None:
                        # print('Initializing transformations')
                        original_size = images.shape[2:]
                        ratio = 0.95
                        self.transformations = [
                            transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                            transforms.Resize((224, 224), antialias=True),
                        ]

                    if self.augment_images:
                        for transform in self.transformations:
                            images = transform(images)
            else:
                assert images.shape[2] == 224 and images.shape[3] == 224, "processed image size should be 224x224"

            images = images / 255.0
            result["images"] = images

            return result

    def _locate(self, index):
        assert index < len(self)
        traj_idx = np.where(self.cum_steps_list > index)[0][0]
        steps_before = self.cum_steps_list[traj_idx - 1] if traj_idx > 0 else 0
        start_ts = index - steps_before
        return traj_idx, start_ts

    def update_obs_normalize_params(self, obs_normalize_params):
        self.OBS_NORMALIZE_PARAMS = copy.deepcopy(obs_normalize_params)
        pickle.dump(obs_normalize_params, open(os.path.join(self.data_roots[0], f"norm_stats_{len(self.data_roots)}.pkl"), "wb"))

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
        
        with TimingContext("locate_trajectory"):
            traj_idx, start_ts = self._locate(index)
            end_ts = min(self.num_steps_list[traj_idx], start_ts + self.chunk_size + 1)
            is_pad = np.zeros((self.chunk_size,), dtype=bool)
            if end_ts < start_ts + self.chunk_size + 1:
                is_pad[-(start_ts + self.chunk_size + 1 - end_ts):] = True
            result_dict["is_pad"] = is_pad

            data_idx, s, ep_id = self.episode_list[traj_idx]
            data_root = self.data_roots[data_idx]

        # image
        with TimingContext("image_loading"):
            images = []
            for cam in self.camera_names:
                # image_path = os.path.join(data_root, f"seed_{s}", f"ep_{ep_id}", f"step_{start_ts}_cam_{cam}.jpg")
                if self.use_pre_img:
                    # _processed.jpg
                    image_filename = f"step_{start_ts}_cam_{cam}_processed.jpg"
                else:
                    image_filename = f"step_{start_ts}_cam_{cam}.jpg"
                image_path = os.path.join(data_root, f"seed_{s}", f"ep_{ep_id}", image_filename)
                # image = imageio.imread(image_path)
                with open(image_path, 'rb') as f:
                    jpeg_data = f.read()
                    try:
                        image = self.jpeg.decode(jpeg_data, pixel_format=TJPF_RGB)  #0
                    except Exception as e:
                        import wandb
                        print(f"Error decoding image {image_path}: {e}")
                        wandb.log({"error": f"Error decoding image {image_path}: {e}"})

                images.append(image)
            images = np.stack(images, axis=0)
            result_dict["images"] = images

        with TimingContext("data_fetching"):
            episode_data = self.episode_data[(data_idx, s, ep_id)]

        with TimingContext("pose_processing"):
            action_chunk = np.zeros((self.chunk_size, 10), dtype=np.float32)
            pose_at_obs = None
            pose_chunk = []
            gripper_width_chunk = []
            proprio_state = np.zeros((10,), dtype=np.float32)
            robot_state = np.zeros((10,), dtype=np.float32)

            prev_pose = None

            for step_idx in range(start_ts, end_ts):
                tcp_pose = episode_data["tcp_pose"][step_idx]
                pose_p, pose_q = tcp_pose[:3], tcp_pose[3:]
                pose_mat = quat2mat(pose_q)
                pose = get_pose_from_rot_pos(pose_mat, pose_p)
            
                if step_idx == start_ts:
                    pose_at_obs = pose
                    pose_mat_6 = pose_mat[:, :2].reshape(-1)
                    proprio_state[:] = np.concatenate([
                        pose_p,
                        pose_mat_6,
                        np.array([episode_data["gripper_width"][step_idx]]),
                    ])
                    robot_state[-1] = episode_data["gripper_width"][step_idx]

                elif step_idx > start_ts:
                    if self.use_desired_action:
                        desired_action = episode_data["action"][step_idx]
                        desired_dp = desired_action[:3]
                        desired_dq = euler2quat(desired_action[3:6])
                        desired_gripper_width = desired_action[-1]

                        desired_d_pose = get_pose_from_rot_pos(
                            quat2mat(desired_dq), desired_dp
                        )
                        desired_pose = desired_d_pose # for abs ee_pose, modified by bingwen
                        pose_chunk.append(desired_pose)
                        gripper_width_chunk.append(np.array([desired_gripper_width]))
                    else:
                        pose_chunk.append(pose)
                        gripper_width_chunk.append(np.array([episode_data["gripper_width"][step_idx]]))
                
                prev_pose = pose

            # compute the relative pose
            _pose_relative = np.eye(4)
            robot_state[:9] = np.concatenate(
                [_pose_relative[:3, 3], _pose_relative[:3, :2].reshape(-1)]
            )
            for i in range(end_ts - start_ts - 1):
                _pose_relative = pose_chunk[i] # abs_ee_pose modified by bingwen
                action_chunk[i] = np.concatenate([
                    _pose_relative[:3, 3],
                    _pose_relative[:3, :2].reshape(-1),
                    gripper_width_chunk[i],
                ]) 

            result_dict["robot_state"] = robot_state
            result_dict["proprio_state"] = proprio_state
            result_dict["action"] = action_chunk
            # for debug 
            # if len(pose_chunk) == 0:
            #     result_dict["pose_chunk"]  = np.zeros((4, 4), dtype=np.float32)
            # else:
            #     result_dict["pose_chunk"] = np.array(pose_chunk)

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
    train_dataset = Sim2SimEpisodeDatasetEff(
        data_roots,
        num_seeds,
        split="train",
        chunk_size=chunk_size,
        # norm_stats_path=os.path.join(data_roots[0], f"norm_stats_{len(data_roots)}.pkl"), 
        norm_stats_path=None, 
        **kwargs
    )
    val_dataset = Sim2SimEpisodeDatasetEff(
        data_roots,
        num_seeds,
        split="val",
        chunk_size=chunk_size,
        norm_stats_path=os.path.join(data_roots[0], f"norm_stats_{len(data_roots)}.pkl"), 
        **kwargs
    )
    train_num_workers = 16 #8 
    val_num_workers = 16 #8 
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


if __name__ == "__main__":
    print("Testing timing functionality...")
    data_roots =["/home/zhouzhiting/Data/panda_data/cano_policy_pd_3"]
    
    try:
        train_loader, _, _, _ = load_sim2sim_data(
            data_roots=data_roots,
            num_seeds= 5000,
            train_batch_size=128,
            val_batch_size=128,
            chunk_size=20,
            usages = ["obs"],
            camera_names=["third"]  #, "wrist"
        )

        for i, batch in enumerate(train_loader):
            print(f"Processed batch {i}")
            if i >= 2:  
                break
                
        print_timing_stats()
        
    except Exception as e:
        print(f"Error testing timing: {e}")
        import traceback
        traceback.print_exc()

