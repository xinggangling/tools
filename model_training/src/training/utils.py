import torch
import os
import json
from typing import Dict, Any, Optional


def setup_device() -> torch.device:
    """
    设置训练设备
    优先使用MPS（Mac M系列），其次是CUDA，最后是CPU
    Returns:
        设备对象
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用MPS设备（Mac M系列GPU）")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用CUDA设备: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("使用CPU设备")

    return device


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                    epoch: int,
                    global_step: int,
                    metrics: Dict[str, float],
                    config: Dict[str, Any],
                    filepath: str,
                    scaler: Optional[torch.cuda.amp.GradScaler] = None):
    """
    保存模型检查点
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前epoch
        global_step: 全局步数
        metrics: 评估指标
        config: 配置信息
        filepath: 保存路径
        scaler: 混合精度训练的scaler
    """
    # 创建目录
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'metrics': metrics,
        'config': config
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    torch.save(checkpoint, filepath)
    print(f"检查点已保存: {filepath}")


def load_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                    filepath: str,
                    device: torch.device,
                    scaler: Optional[torch.cuda.amp.GradScaler] = None) -> Dict[str, Any]:
    """
    加载模型检查点
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        filepath: 检查点路径
        device: 设备
        scaler: 混合精度训练的scaler
    Returns:
        加载的信息字典
    """
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    print(f"检查点已加载: {filepath}")

    return {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {})
    }


def save_config(config: Dict[str, Any], filepath: str):
    """
    保存配置到文件
    Args:
        config: 配置字典
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"配置已保存: {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """
    从文件加载配置
    Args:
        filepath: 配置文件路径
    Returns:
        配置字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print(f"配置已加载: {filepath}")
    return config


def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型参数数量
    Args:
        model: 模型
    Returns:
        参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    计算模型大小（MB）
    Args:
        model: 模型
    Returns:
        模型大小（MB）
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def print_model_info(model: torch.nn.Module):
    """
    打印模型信息
    Args:
        model: 模型
    """
    print("=" * 50)
    print("模型信息")
    print("=" * 50)
    print(f"参数数量: {count_parameters(model):,}")
    print(f"模型大小: {get_model_size_mb(model):.2f} MB")
    print("=" * 50)


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """
    创建实验目录
    Args:
        base_dir: 基础目录
        experiment_name: 实验名称
    Returns:
        实验目录路径
    """
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")

    os.makedirs(experiment_dir, exist_ok=True)
    print(f"实验目录已创建: {experiment_dir}")

    return experiment_dir


def setup_logging(log_dir: str, experiment_name: str):
    """
    设置日志记录
    Args:
        log_dir: 日志目录
        experiment_name: 实验名称
    """
    import logging

    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 设置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 创建文件处理器
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))

    # 配置根日志记录器
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )

    print(f"日志记录已设置: {log_file}")


def set_random_seed(seed: int):
    """
    设置随机种子以确保可重现性
    Args:
        seed: 随机种子
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"随机种子已设置为: {seed}")
