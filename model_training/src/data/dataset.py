import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import numpy as np
from .tokenizer import Tokenizer


class TranslationDataset(Dataset):
    """
    翻译数据集类
    支持平行语料的加载和预处理
    """

    def __init__(self, src_texts: List[str], tgt_texts: List[str],
                 src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer,
                 max_length: int = 128):
        """
        初始化数据集
        Args:
            src_texts: 源语言文本列表
            tgt_texts: 目标语言文本列表
            src_tokenizer: 源语言分词器
            tgt_tokenizer: 目标语言分词器
            max_length: 最大序列长度
        """
        assert len(src_texts) == len(tgt_texts), "源语言和目标语言文本数量必须相同"

        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length

        # 预处理数据
        self.processed_data = self._preprocess_data()

    def _preprocess_data(self) -> List[Dict[str, torch.Tensor]]:
        """预处理数据，将文本转换为token ID"""
        processed_data = []

        for src_text, tgt_text in zip(self.src_texts, self.tgt_texts):
            # 编码文本
            src_ids = self.src_tokenizer.encode(src_text)
            tgt_ids = self.tgt_tokenizer.encode(tgt_text)

            # 过滤过长的序列
            if len(src_ids) > self.max_length or len(tgt_ids) > self.max_length:
                continue

            # 创建数据项
            data_item = {
                'src_ids': torch.tensor(src_ids, dtype=torch.long),
                'tgt_ids': torch.tensor(tgt_ids, dtype=torch.long),
                'src_length': len(src_ids),
                'tgt_length': len(tgt_ids)
            }

            processed_data.append(data_item)

        return processed_data

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.processed_data[idx]


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    数据批处理函数
    将不同长度的序列填充到相同长度
    Args:
        batch: 批次数据
    Returns:
        处理后的批次数据
    """
    # 获取批次中的最大长度
    max_src_len = max(item['src_length'] for item in batch)
    max_tgt_len = max(item['tgt_length'] for item in batch)

    # 初始化批次张量
    batch_size = len(batch)
    src_ids = torch.full((batch_size, max_src_len), 0,
                         dtype=torch.long)  # pad_token_id = 0
    tgt_ids = torch.full((batch_size, max_tgt_len), 0, dtype=torch.long)
    src_mask = torch.zeros((batch_size, max_src_len), dtype=torch.bool)
    tgt_mask = torch.zeros((batch_size, max_tgt_len), dtype=torch.bool)

    # 填充数据
    for i, item in enumerate(batch):
        src_len = item['src_length']
        tgt_len = item['tgt_length']

        # 源序列
        src_ids[i, :src_len] = item['src_ids']
        src_mask[i, :src_len] = True

        # 目标序列
        tgt_ids[i, :tgt_len] = item['tgt_ids']
        tgt_mask[i, :tgt_len] = True

    return {
        'src_ids': src_ids,
        'tgt_ids': tgt_ids,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
        'src_lengths': torch.tensor([item['src_length'] for item in batch]),
        'tgt_lengths': torch.tensor([item['tgt_length'] for item in batch])
    }


def create_dataloaders(train_dataset: TranslationDataset,
                       val_dataset: Optional[TranslationDataset] = None,
                       batch_size: int = 32,
                       num_workers: int = 0,
                       shuffle: bool = True) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    创建数据加载器
    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        batch_size: 批次大小
        num_workers: 工作进程数
        shuffle: 是否打乱数据
    Returns:
        训练和验证数据加载器
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

    return train_loader, val_loader


def split_dataset(dataset: TranslationDataset,
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  random_seed: int = 42) -> Tuple[TranslationDataset, TranslationDataset, TranslationDataset]:
    """
    分割数据集为训练、验证和测试集
    Args:
        dataset: 原始数据集
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        random_seed: 随机种子
    Returns:
        训练、验证和测试数据集
    """
    assert train_ratio + val_ratio < 1.0, "训练集和验证集比例之和必须小于1"

    # 设置随机种子
    np.random.seed(random_seed)

    # 计算分割点
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # 随机打乱索引
    indices = np.random.permutation(total_size)

    # 分割索引
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # 创建子数据集
    train_dataset = _create_subset(dataset, train_indices)
    val_dataset = _create_subset(dataset, val_indices)
    test_dataset = _create_subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def _create_subset(dataset: TranslationDataset, indices: np.ndarray) -> TranslationDataset:
    """创建数据集的子集"""
    src_texts = [dataset.src_texts[i] for i in indices]
    tgt_texts = [dataset.tgt_texts[i] for i in indices]

    return TranslationDataset(
        src_texts=src_texts,
        tgt_texts=tgt_texts,
        src_tokenizer=dataset.src_tokenizer,
        tgt_tokenizer=dataset.tgt_tokenizer,
        max_length=dataset.max_length
    )
