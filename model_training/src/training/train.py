#!/usr/bin/env python3
"""
翻译模型训练主脚本
支持从配置文件加载参数，完整的训练流程
"""

from src.training.utils import (
    setup_device, save_config, load_config,
    create_experiment_dir, setup_logging, set_random_seed,
    print_model_info
)
from src.training import Trainer
from src.data.download_data import (
    download_ted_talks, download_europarl, create_sample_data
)
from src.data import (
    TranslationDataset, create_dataloaders, split_dataset,
    build_bilingual_tokenizer, load_custom_data
)
from src.models import create_translation_model
import argparse
import yaml
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_yaml_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: dict):
    """准备训练数据"""
    data_config = config['data']
    dataset_type = data_config['dataset_type']

    print(f"准备数据集: {dataset_type}")

    if dataset_type == "sample":
        # 创建示例数据
        src_texts, tgt_texts = create_sample_data()

    elif dataset_type == "ted_talks":
        # 下载TED Talks数据
        language_pair = data_config['language_pair']
        max_samples = data_config.get('max_samples')
        src_texts, tgt_texts = download_ted_talks(
            language_pair=language_pair,
            max_samples=max_samples
        )

    elif dataset_type == "europarl":
        # 下载Europarl数据
        language_pair = data_config['language_pair']
        max_samples = data_config.get('max_samples')
        src_texts, tgt_texts = download_europarl(
            language_pair=language_pair,
            max_samples=max_samples
        )

    elif dataset_type == "custom":
        # 加载自定义数据
        data_path = data_config['custom_data_path']
        src_texts, tgt_texts = load_custom_data(data_path)

    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")

    if not src_texts or not tgt_texts:
        raise ValueError("数据加载失败，请检查数据源")

    print(f"数据加载完成，共 {len(src_texts)} 个样本")

    return src_texts, tgt_texts


def build_tokenizers(src_texts: list, tgt_texts: list, config: dict):
    """构建分词器"""
    vocab_config = config['data']['vocab']

    print("构建分词器...")
    src_tokenizer, tgt_tokenizer = build_bilingual_tokenizer(
        src_texts=src_texts,
        tgt_texts=tgt_texts,
        min_freq=vocab_config['min_freq'],
        max_vocab_size=vocab_config['max_vocab_size']
    )

    print(f"源语言词汇表大小: {src_tokenizer.vocab_size}")
    print(f"目标语言词汇表大小: {tgt_tokenizer.vocab_size}")

    return src_tokenizer, tgt_tokenizer


def create_datasets(src_texts: list, tgt_texts: list,
                    src_tokenizer, tgt_tokenizer, config: dict):
    """创建数据集"""
    data_config = config['data']

    print("创建数据集...")

    # 创建完整数据集
    full_dataset = TranslationDataset(
        src_texts=src_texts,
        tgt_texts=tgt_texts,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_length=data_config['max_length']
    )

    # 分割数据集
    train_ratio = data_config['train_ratio']
    val_ratio = data_config['val_ratio']

    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset=full_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def create_model(src_tokenizer, tgt_tokenizer, config: dict):
    """创建模型"""
    model_config = config['model']

    print("创建模型...")

    model = create_translation_model(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        config=model_config
    )

    print_model_info(model)

    return model


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="翻译模型训练")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments",
        help="输出目录"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="从检查点恢复训练"
    )

    args = parser.parse_args()

    # 加载配置
    print("加载配置...")
    config = load_yaml_config(args.config)

    # 设置随机种子
    set_random_seed(config['training']['random_seed'])

    # 创建实验目录
    experiment_name = config['experiment']['name']
    experiment_dir = create_experiment_dir(args.output_dir, experiment_name)

    # 设置日志
    log_dir = os.path.join(experiment_dir, "logs")
    setup_logging(log_dir, experiment_name)

    # 保存配置
    config_path = os.path.join(experiment_dir, "config.yaml")
    save_config(config, config_path)

    # 准备数据
    src_texts, tgt_texts = prepare_data(config)

    # 构建分词器
    src_tokenizer, tgt_tokenizer = build_tokenizers(
        src_texts, tgt_texts, config)

    # 保存分词器
    tokenizer_dir = os.path.join(experiment_dir, "tokenizers")
    os.makedirs(tokenizer_dir, exist_ok=True)
    src_tokenizer.save(os.path.join(tokenizer_dir, "src_tokenizer.json"))
    tgt_tokenizer.save(os.path.join(tokenizer_dir, "tgt_tokenizer.json"))

    # 创建数据集
    train_dataset, val_dataset, test_dataset = create_datasets(
        src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, config
    )

    # 创建数据加载器
    training_config = config['training']
    device_config = config['device']

    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=training_config['batch_size'],
        num_workers=device_config['num_workers']
    )

    # 创建模型
    model = create_model(src_tokenizer, tgt_tokenizer, config)

    # 设置设备
    device = setup_device()
    if device_config.get('force_cpu', False):
        device = torch.device('cpu')
        print("强制使用CPU")

    # 准备训练配置
    trainer_config = {
        'learning_rate': training_config['learning_rate'],
        'weight_decay': training_config['weight_decay'],
        'gradient_accumulation_steps': training_config['gradient_accumulation_steps'],
        'max_grad_norm': training_config['max_grad_norm'],
        'use_amp': training_config['use_amp'],
        'log_interval': config['logging']['log_interval'],
        'save_interval': config['logging']['save_interval'],
        'eval_interval': config['logging']['eval_interval'],
        'output_dir': os.path.join(experiment_dir, "checkpoints"),
        'use_wandb': config['logging']['use_wandb'],
        'wandb_project': config['logging']['wandb_project']
    }

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config
    )

    # 从检查点恢复（如果指定）
    if args.resume:
        print(f"从检查点恢复: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 开始训练
    num_epochs = training_config['num_epochs']
    trainer.train(num_epochs)

    print("训练完成！")
    print(f"实验结果保存在: {experiment_dir}")


if __name__ == "__main__":
    main()
