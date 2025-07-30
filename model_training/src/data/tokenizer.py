import re
import json
from collections import Counter
from typing import List, Dict, Tuple, Optional


class Tokenizer:
    """
    简单的分词器实现
    支持词汇表构建、文本编码和解码
    """

    def __init__(self, vocab: Dict[str, int], special_tokens: Optional[Dict[str, int]] = None):
        """
        初始化分词器
        Args:
            vocab: 词汇表 {token: id}
            special_tokens: 特殊标记 {token_name: id}
        """
        self.vocab = vocab
        self.id_to_token = {id: token for token, id in vocab.items()}

        # 默认特殊标记
        if special_tokens is None:
            special_tokens = {
                '<pad>': 0,
                '<unk>': 1,
                '<sos>': 2,
                '<eos>': 3
            }

        self.special_tokens = special_tokens
        self.pad_token_id = special_tokens['<pad>']
        self.unk_token_id = special_tokens['<unk>']
        self.sos_token_id = special_tokens['<sos>']
        self.eos_token_id = special_tokens['<eos>']

        # 词汇表大小
        self.vocab_size = len(vocab)

    def tokenize(self, text: str) -> List[str]:
        """
        将文本分词
        Args:
            text: 输入文本
        Returns:
            分词后的token列表
        """
        # 简单的基于空格和标点的分词
        text = text.lower().strip()
        # 在标点符号前后添加空格
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        # 分割并过滤空字符串
        tokens = [token for token in text.split() if token]
        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        将文本编码为token ID序列
        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊标记
        Returns:
            token ID列表
        """
        tokens = self.tokenize(text)
        ids = []

        if add_special_tokens:
            ids.append(self.sos_token_id)

        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.unk_token_id)

        if add_special_tokens:
            ids.append(self.eos_token_id)

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        将token ID序列解码为文本
        Args:
            ids: token ID列表
            skip_special_tokens: 是否跳过特殊标记
        Returns:
            解码后的文本
        """
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                token = self.id_to_token[id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)

        # 简单的文本重建
        text = ' '.join(tokens)
        # 修复标点符号
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text.strip()

    def save(self, filepath: str):
        """保存分词器到文件"""
        data = {
            'vocab': self.vocab,
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'Tokenizer':
        """从文件加载分词器"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls(
            vocab=data['vocab'],
            special_tokens=data['special_tokens']
        )


def build_tokenizer(texts: List[str], min_freq: int = 2, max_vocab_size: int = 30000) -> Tokenizer:
    """
    从文本列表构建分词器
    Args:
        texts: 文本列表
        min_freq: 最小词频
        max_vocab_size: 最大词汇表大小
    Returns:
        构建好的分词器
    """
    # 统计词频
    counter = Counter()
    for text in texts:
        tokens = Tokenizer({}).tokenize(text)  # 使用空词汇表进行分词
        counter.update(tokens)

    # 过滤低频词
    counter = {token: freq for token,
               freq in counter.items() if freq >= min_freq}

    # 按频率排序并限制词汇表大小
    sorted_tokens = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    # 特殊标记
    special_tokens = {
        '<pad>': 0,
        '<unk>': 1,
        '<sos>': 2,
        '<eos>': 3
    }

    # 构建词汇表
    vocab = special_tokens.copy()
    for token, _ in sorted_tokens[:max_vocab_size - len(special_tokens)]:
        vocab[token] = len(vocab)

    return Tokenizer(vocab, special_tokens)


def build_bilingual_tokenizer(src_texts: List[str], tgt_texts: List[str],
                              min_freq: int = 2, max_vocab_size: int = 30000) -> Tuple[Tokenizer, Tokenizer]:
    """
    构建双语分词器
    Args:
        src_texts: 源语言文本列表
        tgt_texts: 目标语言文本列表
        min_freq: 最小词频
        max_vocab_size: 最大词汇表大小
    Returns:
        源语言和目标语言分词器
    """
    src_tokenizer = build_tokenizer(src_texts, min_freq, max_vocab_size)
    tgt_tokenizer = build_tokenizer(tgt_texts, min_freq, max_vocab_size)

    return src_tokenizer, tgt_tokenizer
