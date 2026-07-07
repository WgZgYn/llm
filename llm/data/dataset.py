"""数据加载模块。

两种数据模式:
1. MemMapDataProvider:  用于大规模 LM（Shakespeare、OpenWebText）
    数据预编码为 .bin 文件，通过 np.memmap 零拷贝读取
2. DatasetProvider:     用于小规模/教学数据集（FashionMNIST、AdditionDataset）
    包装标准 PyTorch Dataset + DataLoader

抽象接口 DataProvider 让 Trainer 的数据获取与具体格式解耦。
"""

import os
import random
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════════════════════
# 抽象接口
# ═══════════════════════════════════════════════════════════════════════════════

class DataProvider(ABC):
    """训练数据的抽象接口。

    设计要点:
    - get_batch() 返回 CPU tensor（Trainer 负责移动到 GPU）
    - 支持异步预取（backward 期间加载下一 batch）
    """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """词汇表大小（用于初始化模型）。"""
        ...

    @abstractmethod
    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (input_ids, target_ids)，shape 均为 [batch_size, seq_len]。
        split: 'train' 或 'val'
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# MemMapDataProvider: 大规模 LM 数据（nanoGPT 风格）
# ═══════════════════════════════════════════════════════════════════════════════

class MemMapDataProvider(DataProvider):
    """从预编码的 .bin 文件加载数据。

    数据格式: 连续的 uint16 token ids，train.bin + val.bin。
    每次 get_batch 随机采样 block_size 长度的片段。

    DDP 数据分片: 传入 rank/world_size 后，每 GPU 仅从自己分片内采样，
    避免多进程争抢同一文件。

    用法:
        data = MemMapDataProvider("data/shakespeare_char", batch_size=12,
                                  block_size=1024, rank=0, world_size=4)
        X, Y = data.get_batch("train")  # 返回 CPU tensor
    """

    def __init__(self, data_dir: str, batch_size: int, block_size: int,
                 rank: int = 0, world_size: int = 1):
        import pickle
        import warnings

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self._rank = rank
        self._world_size = world_size

        # 从 meta.pkl 读取 vocab_size
        meta_path = os.path.join(data_dir, 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self._vocab_size = meta['vocab_size']
            print(f"found vocab_size = {self._vocab_size} (inside {meta_path})")
        else:
            print(f"no meta.pkl found, defaulting vocab_size to 50304")
            self._vocab_size = 50304

        # ── 缓存 memmap（只 open 一次，不复读）──
        self._cached_data: dict[str, np.memmap] = {}
        self._cached_tensor: dict[str, torch.Tensor] = {}  # torch view（零拷贝）
        self._shard_ranges: dict[str, tuple[int, int]] = {}

        for split_name, filename in [('train', 'train.bin'), ('val', 'val.bin')]:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                mmap = np.memmap(filepath, dtype=np.uint16, mode='r')
                self._cached_data[split_name] = mmap
                # torch.from_numpy(mmap) 共享 mmap 底层 buffer，零拷贝
                # mmap mode='r' 只读，但我们只读取不写入，抑制 writable 警告
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore",
                        message=".*non-writable.*")
                    self._cached_tensor[split_name] = torch.from_numpy(mmap)

                # DDP 数据分片：每 GPU 有独立的采样区间
                total = len(mmap)
                if split_name == 'train' and world_size > 1:
                    shard_size = total // world_size
                    start = rank * shard_size
                    end = (rank + 1) * shard_size if rank < world_size - 1 else total
                else:
                    start, end = 0, total
                self._shard_ranges[split_name] = (start, end)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """随机采样一个 batch（torch 索引，一次调用完成）。

        使用 torch.from_numpy(mmap) 的零拷贝视图 → torch 高级索引 → .long()，
        避免 numpy list comprehension 的逐条拷贝开销。
        """
        raw = self._cached_tensor[split]                  # torch view（零拷贝）
        shard_start, shard_end = self._shard_ranges[split]

        max_start = shard_end - self.block_size - 1
        starts = torch.randint(shard_start, max_start, (self.batch_size,))

        # 向量化索引: [B] → [B, T]
        offsets = torch.arange(self.block_size, dtype=torch.long)
        idx_x = starts.unsqueeze(1) + offsets.unsqueeze(0)       # [B, T]
        idx_y = idx_x + 1                                        # [B, T] 右移一位

        # torch 高级索引 → .long() 转换 dtype（仅复制 [B,T] 切片，非全量）
        x = raw[idx_x].long()
        y = raw[idx_y].long()

        return x, y  # CPU tensor; Trainer 负责移动


# ═══════════════════════════════════════════════════════════════════════════════
# DatasetProvider: PyTorch Dataset/DataLoader 包装
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetProvider(DataProvider):
    """包装 PyTorch Dataset + DataLoader，提供统一的 DataProvider 接口。

    用法:
        dataset = AdditionDataset(tokenizer, train=True)
        data = DatasetProvider(dataset, batch_size=64, collate_fn=collate_fn)
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        vocab_size: int,
        collate_fn=None,
        num_workers: int = 4,
    ):
        self._vocab_size = vocab_size
        self.batch_size = batch_size
        self.collate_fn = collate_fn

        self.train_dataset = dataset
        self.train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=True, drop_last=True,
        )
        self._train_iter = None

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """从 DataLoader 迭代器获取一批数据。自动循环。"""
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)

        try:
            return next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            return next(self._train_iter)


# ═══════════════════════════════════════════════════════════════════════════════
# Character Tokenizer（用于字符级 LM）
# ═══════════════════════════════════════════════════════════════════════════════

class CharTokenizer:
    """字符级 tokenizer —— 最简单的 tokenizer 实现。

    每个字符映射为一个独立的 token id。
    用于 shakespeare_char 等小型数据集。

    用法:
        tokenizer = CharTokenizer("abcdefg#_")
        ids = tokenizer.encode("abc")  # [0, 1, 2]
        text = tokenizer.decode([0, 1, 2])  # "abc"
    """

    def __init__(self, vocab: list[str] | None = None):
        if vocab is None:
            # 默认: 加法任务词汇表
            vocab = [
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '+', '=', '#',  # EOS
                '_',              # PAD
            ]
        self.vocab = vocab
        self.stoi = {s: i for i, s in enumerate(vocab)}   # string → id
        self.itos = {i: s for i, s in enumerate(vocab)}   # id → string

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.stoi.get('_', 0)

    @property
    def eos_id(self) -> int:
        return self.stoi.get('#', 0)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[s] for s in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


# ═══════════════════════════════════════════════════════════════════════════════
# 加法任务数据集（来自 week4 的 FlexibleAdditionDataset）
# ═══════════════════════════════════════════════════════════════════════════════

class AdditionDataset(Dataset):
    """两位数加法算术任务数据集。

    生成 "a+b=c#" 格式的样本，用于测试模型学习算术规律的能力。
    支持可变位数、前导零增强。
    """

    def __init__(
        self,
        tokenizer: CharTokenizer,
        train: bool = True,
        train_rate: float = 0.8,
        num_samples: int = 100000,
        max_digits: int = 5,
        leading_zero_prob: float = 0.1,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_digits = max_digits
        self.leading_zero_prob = leading_zero_prob

        rng = random.Random(seed)

        # 位数分布：较短的数更常见
        digit_probs = [0.35, 0.25, 0.20, 0.12, 0.08][:max_digits]
        s = sum(digit_probs)
        digit_probs = [p / s for p in digit_probs]

        full_data = []
        for _ in range(num_samples):
            digits_a = rng.choices(range(1, max_digits + 1), weights=digit_probs, k=1)[0]
            digits_b = rng.choices(range(1, max_digits + 1), weights=digit_probs, k=1)[0]
            a = self._random_number(rng, digits_a)
            b = self._random_number(rng, digits_b)
            a_str = self._maybe_add_leading_zero(rng, a)
            b_str = self._maybe_add_leading_zero(rng, b)
            sample = f"{a_str}+{b_str}={a + b}#"
            tokens = tokenizer.encode(sample)
            full_data.append((
                torch.tensor(tokens[:-1], dtype=torch.long),
                torch.tensor(tokens[1:], dtype=torch.long),
            ))

        rng.shuffle(full_data)
        split_idx = int(num_samples * train_rate)
        self.data = full_data[:split_idx] if train else full_data[split_idx:]
        print(f"{'Train' if train else 'Test'} Dataset Size = {len(self.data)}")

    def _random_number(self, rng, digits: int) -> int:
        if digits == 1:
            return rng.randint(0, 9)
        return rng.randint(10 ** (digits - 1), 10 ** digits - 1)

    def _maybe_add_leading_zero(self, rng, value: int) -> str:
        s = str(value)
        if rng.random() < self.leading_zero_prob:
            max_pad = max(1, self.max_digits - len(s))
            pad_len = rng.randint(1, max_pad)
            s = "0" * pad_len + s
        return s

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_addition_batch(batch, pad_id: int = -1):
    """加法任务的 batch 整理函数：padding 到统一长度。"""
    if pad_id == -1:
        pad_id = 13  # '_' 的索引（vocab 中最后一位）

    xs, ys = zip(*batch)
    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=pad_id)
    ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=pad_id)
    return xs, ys
