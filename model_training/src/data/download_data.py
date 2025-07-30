import os
import requests
from datasets import load_dataset
from typing import List, Tuple, Optional
import pandas as pd
from tqdm import tqdm


def download_ted_talks(language_pair: str = "en-fr",
                       max_samples: Optional[int] = None,
                       output_dir: str = "data/ted_talks") -> Tuple[List[str], List[str]]:
    """
    下载TED Talks数据集
    Args:
        language_pair: 语言对，如"en-fr"表示英语到法语
        max_samples: 最大样本数，None表示下载全部
        output_dir: 输出目录
    Returns:
        源语言和目标语言文本列表
    """
    print(f"正在下载TED Talks数据集 ({language_pair})...")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 从Hugging Face加载数据集
        dataset = load_dataset("ted_talks_iwslt", language_pair)

        # 提取训练集
        train_data = dataset['train']

        # 限制样本数
        if max_samples is not None:
            train_data = train_data.select(
                range(min(max_samples, len(train_data))))

        # 提取文本
        src_texts = []
        tgt_texts = []

        for item in tqdm(train_data, desc="处理TED Talks数据"):
            src_texts.append(item['translation'][language_pair.split('-')[0]])
            tgt_texts.append(item['translation'][language_pair.split('-')[1]])

        # 保存到文件
        output_file = os.path.join(
            output_dir, f"ted_talks_{language_pair}.csv")
        df = pd.DataFrame({
            'src': src_texts,
            'tgt': tgt_texts
        })
        df.to_csv(output_file, index=False)

        print(f"TED Talks数据集已保存到: {output_file}")
        print(f"数据集大小: {len(src_texts)} 个样本")

        return src_texts, tgt_texts

    except Exception as e:
        print(f"下载TED Talks数据集失败: {e}")
        print("尝试使用备用方法...")
        return _download_ted_talks_fallback(language_pair, max_samples, output_dir)


def _download_ted_talks_fallback(language_pair: str, max_samples: Optional[int], output_dir: str):
    """备用下载方法"""
    # 这里可以添加其他下载源或使用本地数据
    print("备用下载方法暂未实现，请手动下载数据")
    return [], []


def download_europarl(language_pair: str = "en-fr",
                      max_samples: Optional[int] = None,
                      output_dir: str = "data/europarl") -> Tuple[List[str], List[str]]:
    """
    下载Europarl数据集
    Args:
        language_pair: 语言对，如"en-fr"表示英语到法语
        max_samples: 最大样本数，None表示下载全部
        output_dir: 输出目录
    Returns:
        源语言和目标语言文本列表
    """
    print(f"正在下载Europarl数据集 ({language_pair})...")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 从Hugging Face加载数据集
        dataset = load_dataset("opus_europarl", language_pair)

        # 提取训练集
        train_data = dataset['train']

        # 限制样本数
        if max_samples is not None:
            train_data = train_data.select(
                range(min(max_samples, len(train_data))))

        # 提取文本
        src_texts = []
        tgt_texts = []

        for item in tqdm(train_data, desc="处理Europarl数据"):
            src_texts.append(item['translation'][language_pair.split('-')[0]])
            tgt_texts.append(item['translation'][language_pair.split('-')[1]])

        # 保存到文件
        output_file = os.path.join(output_dir, f"europarl_{language_pair}.csv")
        df = pd.DataFrame({
            'src': src_texts,
            'tgt': tgt_texts
        })
        df.to_csv(output_file, index=False)

        print(f"Europarl数据集已保存到: {output_file}")
        print(f"数据集大小: {len(src_texts)} 个样本")

        return src_texts, tgt_texts

    except Exception as e:
        print(f"下载Europarl数据集失败: {e}")
        print("尝试使用备用方法...")
        return _download_europarl_fallback(language_pair, max_samples, output_dir)


def _download_europarl_fallback(language_pair: str, max_samples: Optional[int], output_dir: str):
    """备用下载方法"""
    # 这里可以添加其他下载源或使用本地数据
    print("备用下载方法暂未实现，请手动下载数据")
    return [], []


def load_custom_data(file_path: str, src_col: str = "src", tgt_col: str = "tgt") -> Tuple[List[str], List[str]]:
    """
    加载自定义数据集
    Args:
        file_path: 数据文件路径
        src_col: 源语言列名
        tgt_col: 目标语言列名
    Returns:
        源语言和目标语言文本列表
    """
    print(f"正在加载自定义数据集: {file_path}")

    try:
        # 根据文件扩展名选择加载方法
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t')
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")

        # 检查列是否存在
        if src_col not in df.columns or tgt_col not in df.columns:
            raise ValueError(f"数据文件中缺少必要的列: {src_col} 或 {tgt_col}")

        # 提取文本
        src_texts = df[src_col].tolist()
        tgt_texts = df[tgt_col].tolist()

        # 过滤空值
        valid_pairs = [(src, tgt) for src, tgt in zip(src_texts, tgt_texts)
                       if pd.notna(src) and pd.notna(tgt) and str(src).strip() and str(tgt).strip()]

        src_texts, tgt_texts = zip(*valid_pairs)

        print(f"自定义数据集加载完成，大小: {len(src_texts)} 个样本")

        return list(src_texts), list(tgt_texts)

    except Exception as e:
        print(f"加载自定义数据集失败: {e}")
        return [], []


def create_sample_data(output_dir: str = "data/sample") -> Tuple[List[str], List[str]]:
    """
    创建示例数据用于测试
    Args:
        output_dir: 输出目录
    Returns:
        源语言和目标语言文本列表
    """
    print("正在创建示例数据...")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 示例英语到中文的翻译数据
    sample_data = [
        ("Hello, how are you?", "你好，你好吗？"),
        ("I love this movie.", "我喜欢这部电影。"),
        ("The weather is nice today.", "今天天气很好。"),
        ("What time is it?", "现在几点了？"),
        ("Thank you very much.", "非常感谢。"),
        ("Where is the nearest restaurant?", "最近的餐厅在哪里？"),
        ("I don't understand.", "我不明白。"),
        ("Can you help me?", "你能帮助我吗？"),
        ("This is very interesting.", "这很有趣。"),
        ("I want to learn Chinese.", "我想学习中文。"),
        ("The food is delicious.", "食物很美味。"),
        ("I'm tired.", "我累了。"),
        ("Let's go for a walk.", "我们去散步吧。"),
        ("What's your name?", "你叫什么名字？"),
        ("Nice to meet you.", "很高兴认识你。"),
        ("How old are you?", "你多大了？"),
        ("I live in Beijing.", "我住在北京。"),
        ("Do you speak English?", "你会说英语吗？"),
        ("I'm sorry.", "对不起。"),
        ("You're welcome.", "不客气。")
    ]

    src_texts, tgt_texts = zip(*sample_data)

    # 保存到文件
    output_file = os.path.join(output_dir, "sample_en_zh.csv")
    df = pd.DataFrame({
        'src': src_texts,
        'tgt': tgt_texts
    })
    df.to_csv(output_file, index=False)

    print(f"示例数据已保存到: {output_file}")
    print(f"数据集大小: {len(src_texts)} 个样本")

    return list(src_texts), list(tgt_texts)


if __name__ == "__main__":
    # 示例用法
    print("数据下载工具")
    print("=" * 50)

    # 创建示例数据
    src_texts, tgt_texts = create_sample_data()

    # 下载TED Talks数据集（小规模用于测试）
    # src_texts, tgt_texts = download_ted_talks("en-fr", max_samples=1000)

    # 下载Europarl数据集（小规模用于测试）
    # src_texts, tgt_texts = download_europarl("en-fr", max_samples=10000)

    print("数据下载完成！")
