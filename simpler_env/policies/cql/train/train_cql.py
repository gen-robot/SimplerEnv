import d3rlpy
from d3rlpy.dataset import MDPDataset
import os
import numpy as np
import argparse
import logging
import time
import wandb  # 导入wandb库


# 导入data_transform.py中的方法
from simpler_env.policies.cql.train.data_transform import load_and_preprocess_data

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义一个函数来捕获并记录评估结果
def log_evaluation_metrics(algorithm, dataset):
    """评估算法并将结果记录到wandb"""
    try:
        # 尝试评估TD错误
        td_error = d3rlpy.metrics.td_error(algorithm, dataset)
        wandb.log({"evaluation/td_error": float(td_error)})
        logger.info(f"TD错误: {td_error}")
        
        # 尝试评估值估计
        value_estimation = d3rlpy.metrics.average_value_estimation(algorithm, dataset)
        wandb.log({"evaluation/average_value": float(value_estimation)})
        logger.info(f"平均值估计: {value_estimation}")
        
        # 如果有其他评估指标，也可以添加
    except Exception as e:
        logger.warning(f"评估指标计算失败: {str(e)}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练CQL模型')
    parser.add_argument('--raw_data_dir', type=str, 
                        default="/home/chenyinuo/data/worldModel/world_model_rl_dataset/tomato_pick_and_place/success/",
                        help='原始数据目录路径')
    parser.add_argument('--n_processes', type=int, default=None, 
                        help='处理数据的进程数，默认为CPU核心数的95%')
    parser.add_argument('--use_multiprocessing', action='store_true', default=True,
                        help='是否使用多进程处理数据')
    parser.add_argument('--n_steps', type=int, default=100000, 
                        help='训练步数')
    parser.add_argument('--n_steps_per_epoch', type=int, default=1000, 
                        help='每个epoch的步数')
    parser.add_argument('--device', type=str, default='cuda:1', 
                        help='训练设备')
    parser.add_argument('--project_name', type=str, default='cql-training', 
                        help='wandb项目名称')
    parser.add_argument('--experiment_name', type=str, default=None, 
                        help='wandb实验名称, 默认为自动生成')
    args = parser.parse_args()
    
    # 初始化wandb
    experiment_name = args.experiment_name or f"cql-{time.strftime('%Y%m%d-%H%M%S')}"
    wandb.init(
        project=args.project_name,
        name=experiment_name,
        config={
            "raw_data_dir": args.raw_data_dir,
            "n_processes": args.n_processes,
            "use_multiprocessing": args.use_multiprocessing,
            "n_steps": args.n_steps,
            "n_steps_per_epoch": args.n_steps_per_epoch,
            "device": args.device,
            "algorithm": "CQL",
            "framework": "d3rlpy",
            "python_version": os.environ.get("PYTHONVERSION", "unknown"),
        },
        tags=["offline-rl", "cql", "robotics"],
        notes="使用d3rlpy库训练CQL模型用于机器人控制",
    )
    logger.info(f"初始化wandb, 项目名: {args.project_name}，实验名: {experiment_name}")
    
    # 确定数据集
    dataset = None
    
    logger.info(f"处理原始数据集: {args.raw_data_dir}")
    # 记录开始时间
    start_time = time.time()
    
    # 使用data_transform.py中的方法处理数据
    dataset, transitions_count = load_and_preprocess_data(
        args.raw_data_dir, 
        n_processes=args.n_processes,
        use_multiprocessing=args.use_multiprocessing
    )
    
    logger.info(f"数据集创建完成，共有 {transitions_count} 个转换")
    wandb.log({"transitions_count": transitions_count})
    
    # 显示处理时间
    processing_time = time.time() - start_time
    logger.info(f"数据处理完成，耗时 {processing_time:.2f} 秒")
    wandb.log({"data_processing_time": processing_time})
    
    # 可选：添加断点，用于调试
    # breakpoint()
    
    # 使用连续动作空间的CQL算法
    logger.info(f"创建CQL算法, 使用设备: {args.device}")
    cql = d3rlpy.algos.CQLConfig().create(device=args.device)
    
    # 创建wandb回调函数，用于记录训练过程中的指标
    def wandb_callback(algorithm, epoch, total_step):
        # 检查算法中是否有最新的训练指标
        if hasattr(algorithm, '_impl') and hasattr(algorithm._impl, 'logger'):
            metrics = algorithm._impl.logger.get_metrics()
            if metrics:
                for key, val in metrics.items():
                    wandb.log({f"train/{key}": val}, step=total_step)
        
        # 记录当前epoch
        wandb.log({"epoch": epoch}, step=total_step)
        
        # 可以添加额外的评估指标
        if hasattr(algorithm, '_eval_results'):
            for name, val in algorithm._eval_results.items():
                wandb.log({f"eval/{name}": val}, step=total_step)
            
    # 开始训练，使用离线评估指标
    logger.info(f"开始训练，总步数: {args.n_steps}，每轮步数: {args.n_steps_per_epoch}")
    cql.fit(
        dataset,
        n_steps=args.n_steps,
        n_steps_per_epoch=args.n_steps_per_epoch,
        evaluators={
            'td_error': d3rlpy.metrics.TDErrorEvaluator(),
            'value_scale': d3rlpy.metrics.AverageValueEstimationEvaluator(),
        },
        callback=wandb_callback,  # 添加wandb回调
    )
    
    # 训练结束后评估模型并记录到wandb
    logger.info("训练完成，开始最终评估...")
    log_evaluation_metrics(cql, dataset)
    
    # 保存训练好的模型
    model_path = 'cql_model.pt'
    logger.info(f"保存模型到 {model_path}")
    cql.save_model(model_path)
    
    # 上传模型到wandb
    wandb.save(model_path)
    logger.info(f"模型已上传到wandb")
    
    # 结束wandb会话
    wandb.finish()
    logger.info("训练完成")

if __name__ == "__main__":
    main()