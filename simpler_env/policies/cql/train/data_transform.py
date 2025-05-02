import numpy as np
import d3rlpy
from d3rlpy.dataset import MDPDataset
import glob
import os
import logging
import time
import traceback
import sys
from multiprocessing import Pool, cpu_count
import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_single_file(file_info):
    """
    处理单个数据文件
    
    参数:
        file_info: 包含文件路径和索引的元组 (file_idx, file_path)
    
    返回:
        包含处理结果的字典
    """
    file_idx, file_path = file_info
    try:
        # 准备数据收集列表
        observations = []
        actions = []
        rewards = []
        terminals = []
        timeouts = []
        
        # 加载数据
        data = np.load(file_path, allow_pickle=True)["arr_0"].tolist()
        
        # 基本数据验证
        required_keys = ['action', 'timestamp', 'observation', 'success']
        if not all(key in data for key in required_keys):
            return {
                'error': f"文件 {file_path} 缺少必要字段",
                'success': False
            }
            
        # 获取轨迹长度
        traj_length = len(data['timestamp'])
        if traj_length <= 1:
            return {
                'error': f"文件 {file_path} 轨迹长度过短 ({traj_length})",
                'success': False
            }
            
        # 记录轨迹是否成功
        is_success = data.get('success', False)
        
        # 预先计算一些常量
        last_step_idx = traj_length - 2
        
        # 对每个时间步处理数据
        for i in range(traj_length-1):  # 减1是因为最后一步没有下一个状态
            try:
                # 观察值处理 - RGB数据处理
                if "rgb" in data["observation"]:
                    rgb_data = data["observation"]["rgb"][i]
                    # 高效的降采样方法
                    if len(rgb_data.shape) >= 3 and rgb_data.shape[-3:] == (480, 640, 3):
                        # 使用切片进行更高效的降采样
                        rgb_data = rgb_data[::4, ::4, :]  # 每4个像素取1个
                        obs = rgb_data.flatten()  # 展平为一维向量
                
                # 动作处理 - 提取必要的部分
                try:
                    # 直接从数据中提取动作分量
                    end_position = data['action']['end']['position'][i].flatten()
                    end_orientation = data['action']['end']['orientation'][i].flatten()
                    
                    # 提取gripper信息
                    gripper_pos = data['action']['effector']['position_gripper'][i].flatten()
                    
                    # 高效连接动作分量
                    action = np.concatenate([end_position, end_orientation, gripper_pos])
                except (KeyError, IndexError) as e:
                    continue  # 跳过这个样本
                
                # 奖励计算 - 使用预计算的常量
                is_last_step = (i == last_step_idx)
                reward = 1.0 if (is_last_step and is_success) else (0.01 if is_success else 0.0)
                
                # 添加到收集列表
                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                terminals.append(is_last_step)
                timeouts.append(False)
                
            except Exception as e:
                # 记录错误但继续处理
                pass
                
        # 返回处理结果
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'terminals': terminals,
            'timeouts': timeouts,
            'is_success': is_success,
            'success': True
        }
        
    except Exception as e:
        return {
            'error': f"{str(e)} (文件: {file_path})",
            'success': False
        }

def load_data_sequential(data_files):
    """顺序处理数据文件（不使用多进程）"""
    all_observations = []
    all_actions = []
    all_rewards = []
    all_terminals = []
    all_timeouts = []
    
    total_transitions = 0
    successful_trajectories = 0
    
    for file_idx, file_path in enumerate(tqdm.tqdm(data_files, desc="处理文件")):
        result = process_single_file((file_idx, file_path))
        if result['success']:
            # 添加转换数据
            if 'observations' in result:
                all_observations.extend(result['observations'])
                all_actions.extend(result['actions'])
                all_rewards.extend(result['rewards'])
                all_terminals.extend(result['terminals'])
                all_timeouts.extend(result['timeouts'])
                
                # 更新统计信息
                total_transitions += len(result['observations'])
                if result.get('is_success', False):
                    successful_trajectories += 1
        else:
            logger.warning(f"处理文件失败: {result.get('error', '未知错误')}")
    
    return {
        'observations': all_observations,
        'actions': all_actions,
        'rewards': all_rewards,
        'terminals': all_terminals,
        'timeouts': all_timeouts,
        'total_transitions': total_transitions,
        'successful_trajectories': successful_trajectories
    }

def load_and_preprocess_data(data_dir, n_processes=None, use_multiprocessing=True):
    """
    加载并预处理数据集中的所有文件
    
    参数:
        data_dir: 数据文件所在的目录
        n_processes: 使用的进程数, 默认为CPU核心数95%
        use_multiprocessing: 是否使用多进程处理
    
    返回:
        预处理后的数据集和转换计数的元组
    """
    start_time = time.time()
    
    # 获取所有数据文件
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录 {data_dir} 不存在")
    
    data_files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not data_files:
        raise ValueError(f"在 {data_dir} 中没有找到.npz文件")
    
    file_count = len(data_files)
    logger.info(f"找到 {file_count} 个数据文件")
    
    # 决定是否使用多进程
    if use_multiprocessing:
        # 设置进程数
        if n_processes is None:
            n_processes = max(1, int(cpu_count() * 0.95))  # 使用95%的CPU核心
        n_processes = min(n_processes, file_count)  # 进程数不超过文件数
        
        logger.info(f"使用 {n_processes} 个进程并行处理文件")
        
        try:
            # 准备文件信息列表
            file_infos = list(enumerate(data_files))
            
            # 初始化结果变量
            all_observations = []
            all_actions = []
            all_rewards = []
            all_terminals = []
            all_timeouts = []
            total_transitions = 0
            successful_trajectories = 0
            
            # 使用多进程处理文件
            with Pool(processes=n_processes) as pool:
                # 使用tqdm显示进度
                results = []
                for result in tqdm.tqdm(pool.imap(process_single_file, file_infos), total=file_count, desc="处理文件"):
                    results.append(result)
            
            # 处理结果
            for result in results:
                if result['success']:
                    # 添加转换数据
                    if 'observations' in result:
                        all_observations.extend(result['observations'])
                        all_actions.extend(result['actions'])
                        all_rewards.extend(result['rewards'])
                        all_terminals.extend(result['terminals'])
                        all_timeouts.extend(result['timeouts'])
                        
                        # 更新统计信息
                        total_transitions += len(result['observations'])
                        if result.get('is_success', False):
                            successful_trajectories += 1
                else:
                    logger.warning(f"处理文件失败: {result.get('error', '未知错误')}")
        
        except Exception as e:
            logger.error(f"多进程处理出错: {str(e)}")
            logger.error(traceback.format_exc())
            logger.info("尝试使用顺序处理...")
            
            # 如果多进程处理失败，回退到顺序处理
            result_data = load_data_sequential(data_files)
            all_observations = result_data['observations']
            all_actions = result_data['actions']
            all_rewards = result_data['rewards']
            all_terminals = result_data['terminals']
            all_timeouts = result_data['timeouts']
            total_transitions = result_data['total_transitions']
            successful_trajectories = result_data['successful_trajectories']
    else:
        # 顺序处理
        logger.info("使用顺序处理文件")
        result_data = load_data_sequential(data_files)
        all_observations = result_data['observations']
        all_actions = result_data['actions']
        all_rewards = result_data['rewards']
        all_terminals = result_data['terminals']
        all_timeouts = result_data['timeouts']
        total_transitions = result_data['total_transitions']
        successful_trajectories = result_data['successful_trajectories']
    
    if not all_observations:
        raise ValueError("没有有效的转换数据，无法创建数据集")
    
    # 转换为numpy数组 - 预先计算形状以提高效率
    obs_count = len(all_observations)
    
    logger.info(f"转换数据为numpy数组...")
    all_observations = np.array(all_observations)
    all_actions = np.array(all_actions)
    all_rewards = np.array(all_rewards)
    all_terminals = np.array(all_terminals)
    all_timeouts = np.array(all_timeouts)
    
    # 打印数据统计信息
    processing_time = time.time() - start_time
    logger.info(f"数据处理完成，耗时 {processing_time:.2f} 秒")
    logger.info(f"数据集统计:")
    logger.info(f"- 总转换数: {total_transitions}")
    logger.info(f"- 成功轨迹数: {successful_trajectories}/{file_count} ({successful_trajectories/file_count*100:.1f}%)")
    logger.info(f"- 观察值形状: {all_observations.shape}")
    logger.info(f"- 动作值形状: {all_actions.shape}")
    logger.info(f"- 处理速度: {total_transitions/processing_time:.2f} 样本/秒")
    
    # 归一化处理开始前计时
    norm_start_time = time.time()
    logger.info("开始数据归一化...")
    
    # 对观察值和动作值进行归一化
    observation_mean = np.mean(all_observations, axis=0)
    observation_std = np.std(all_observations, axis=0) + 1e-6  # 避免除以零
    normalized_observations = (all_observations - observation_mean) / observation_std
    
    action_mean = np.mean(all_actions, axis=0)
    action_std = np.std(all_actions, axis=0) + 1e-6
    normalized_actions = (all_actions - action_mean) / action_std
    
    norm_time = time.time() - norm_start_time
    logger.info(f"归一化完成，耗时 {norm_time:.2f} 秒")
    
    # 创建数据集
    dataset_start_time = time.time()
    logger.info("创建MDPDataset...")
    
    try:
        # 创建并返回数据集对象
        dataset = MDPDataset(
            observations=normalized_observations,
            actions=normalized_actions,
            rewards=all_rewards,
            terminals=all_terminals
        )
        
        dataset_time = time.time() - dataset_start_time
        logger.info(f"数据集创建完成，耗时 {dataset_time:.2f} 秒")
        
        # 保存归一化参数，以便在使用模型时可以应用相同的归一化
        normalization_params = {
            'observation_mean': observation_mean,
            'observation_std': observation_std,
            'action_mean': action_mean,
            'action_std': action_std
        }
        np.savez('normalization_params.npz', **normalization_params)
        logger.info("归一化参数已保存到 normalization_params.npz")
        
        # 总处理时间
        total_time = time.time() - start_time
        logger.info(f"总处理时间: {total_time:.2f} 秒")
        
        return dataset, total_transitions
    
    except Exception as e:
        logger.error(f"创建数据集时出错: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    # 数据目录
    data_dir = "/home/chenyinuo/data/worldModel/world_model_rl_dataset/tomato_pick_and_place/success/"
    
    # 可以指定使用的进程数，默认为CPU核心数的95%
    n_processes = None  # 或者设置为具体数值，如 4
    
    # 是否使用多进程
    use_multiprocessing = True
    
    try:
        # 记录开始时间
        total_start_time = time.time()
        
        # 加载和预处理数据
        dataset, transitions_count = load_and_preprocess_data(
            data_dir, 
            n_processes=n_processes,
            use_multiprocessing=use_multiprocessing
        )
        logger.info(f"数据集创建完成，共有 {transitions_count} 个转换")
        
        # 提供一些数据集的基本信息
        logger.info(f"数据集类型: {type(dataset)}")
        logger.info(f"数据集属性: {dir(dataset)}")
        
        # 保存数据集（可选）
        save_dataset = input("是否保存处理后的数据集? (y/n): ").strip().lower() == 'y'
        if save_dataset:
            output_file = 'processed_dataset.h5'
            logger.info(f"保存数据集到 {output_file}...")
            dataset.dump(output_file)
            logger.info(f"数据集已保存")
        
        # 显示总耗时
        total_time = time.time() - total_start_time
        logger.info(f"全部流程完成，总耗时: {total_time:.2f} 秒")
        
    except KeyboardInterrupt:
        logger.info("用户中断处理")
    except Exception as e:
        logger.error(f"处理数据时出错: {str(e)}")
        logger.error(traceback.format_exc())