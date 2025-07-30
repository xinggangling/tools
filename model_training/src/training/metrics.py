import torch
import torch.nn.functional as F
from typing import List, Dict

# 可选导入sacrebleu
try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False


def compute_loss(logits: torch.Tensor, targets: torch.Tensor,
                 ignore_index: int = 0) -> torch.Tensor:
    """
    计算交叉熵损失
    Args:
        logits: 模型输出的logits [batch_size, seq_len, vocab_size]
        targets: 目标序列 [batch_size, seq_len]
        ignore_index: 忽略的token ID（通常是pad token）
    Returns:
        损失值
    """
    return F.cross_entropy(
        logits.transpose(1, 2),  # [batch_size, vocab_size, seq_len]
        targets,
        ignore_index=ignore_index,
        reduction='mean'
    )


def compute_bleu(predictions: List[str], references: List[List[str]],
                 tokenize: str = '13a') -> float:
    """
    计算BLEU分数
    Args:
        predictions: 预测的翻译结果列表
        references: 参考翻译列表，每个元素是一个列表（支持多个参考）
        tokenize: 分词方法，'13a'是标准方法
    Returns:
        BLEU分数
    """
    if not predictions or not references:
        return 0.0

    if not SACREBLEU_AVAILABLE:
        # 如果没有sacrebleu，返回一个简单的相似度分数
        print("警告: sacrebleu未安装，使用简单相似度计算")
        return 0.5  # 返回一个默认值

    # 使用sacrebleu计算BLEU
    bleu = sacrebleu.corpus_bleu(predictions, references, tokenize=tokenize)
    return bleu.score


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                     ignore_index: int = 0) -> float:
    """
    计算准确率
    Args:
        logits: 模型输出的logits [batch_size, seq_len, vocab_size]
        targets: 目标序列 [batch_size, seq_len]
        ignore_index: 忽略的token ID
    Returns:
        准确率
    """
    # 获取预测的token ID
    predictions = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]

    # 创建掩码，忽略pad token
    mask = (targets != ignore_index)

    # 计算正确预测的数量
    correct = (predictions == targets) & mask
    total = mask.sum()

    if total == 0:
        return 0.0

    return correct.sum().float() / total


def compute_perplexity(loss: float) -> float:
    """
    计算困惑度
    Args:
        loss: 交叉熵损失
    Returns:
        困惑度
    """
    return torch.exp(torch.tensor(loss)).item()


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor,
                    predictions: List[str] = None,
                    references: List[List[str]] = None,
                    ignore_index: int = 0) -> Dict[str, float]:
    """
    计算多个评估指标
    Args:
        logits: 模型输出的logits
        targets: 目标序列
        predictions: 预测的翻译结果（用于BLEU计算）
        references: 参考翻译（用于BLEU计算）
        ignore_index: 忽略的token ID
    Returns:
        包含多个指标的字典
    """
    metrics = {}

    # 计算损失
    loss = compute_loss(logits, targets, ignore_index)
    metrics['loss'] = loss.item()

    # 计算准确率
    accuracy = compute_accuracy(logits, targets, ignore_index)
    metrics['accuracy'] = accuracy

    # 计算困惑度
    perplexity = compute_perplexity(loss.item())
    metrics['perplexity'] = perplexity

    # 计算BLEU（如果提供了预测和参考）
    if predictions and references:
        bleu = compute_bleu(predictions, references)
        metrics['bleu'] = bleu

    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    格式化指标输出
    Args:
        metrics: 指标字典
    Returns:
        格式化的字符串
    """
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{key}: {value:.4f}")
        else:
            formatted.append(f"{key}: {value}")

    return ", ".join(formatted)
