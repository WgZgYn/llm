"""PyTorch DataLoader 集成 —— DDP/FSDP 感知的 LM 数据管道。

核心设计:
    LMStream(IterableDataset): 无限随机流，内部做 DDP 分片
    create_dataloader():       工厂函数，返回配置好的 DataLoader

与 DistributedSampler 方案的区别:
    - DistributedSampler 需要 torch.randperm(len(dataset)) → 大数据集（9B+ tokens）
      会导致 CPU OOM（randperm(9B) ≈ 72 GB）
    - LMStream 直接在分片内 randint → 内存恒定，适合任意规模数据集
    - LM 训练不需要 epoch 概念 → 无限随机流更自然

用法:
    from llm.data.loader import create_dataloader

    loader = create_dataloader("data/openwebtext", "train",
                               batch_size=12, block_size=1024)
    for X, Y in loader:    # 无限循环，自动分片，pin_memory
        ...
"""

import os
import sys
import mmap as _mmap_module
import warnings
from typing import Iterator, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


# 防止 mmap 对象被 GC 回收导致底层内存被释放
_mmap_refs: list = []


def _open_mmap(filepath: str) -> np.ndarray:
    """打开 .bin 文件的 mmap，带 MADV_RANDOM 建议（读完即丢，不污染 page cache）。

    MADV_RANDOM 告诉内核：我们对这个文件的访问是随机的，
    不要预读相邻页，读完的页可以尽快回收。
    对几十 GB 的 LM 训练数据只扫一次或不规则访问，此举大幅降低 RAM 占用量。
    """
    fd = os.open(filepath, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        mm = _mmap_module.mmap(fd, size, access=_mmap_module.ACCESS_READ)
    finally:
        os.close(fd)  # mmap 创建后可安全关闭

    _mmap_refs.append(mm)
    # Linux: 告诉内核随机访问，不要预读，读完可回收
    if hasattr(mm, 'madvise') and hasattr(_mmap_module, 'MADV_RANDOM'):
        mm.madvise(_mmap_module.MADV_RANDOM)

    raw = np.frombuffer(mm, dtype=np.uint16)
    return raw


# ═══════════════════════════════════════════════════════════════════════════════
# LMStream —— 无限随机流 IterableDataset
# ═══════════════════════════════════════════════════════════════════════════════

class LMStream(IterableDataset):
    """从 .bin 文件无限产出随机 batch 的 IterableDataset。

    DDP 分片:
        rank=0 → 读取文件 [0, N/4)
        rank=1 → 读取文件 [N/4, N/2)
        ...

    DataLoader worker 分片:
        num_workers=2 → 每个 worker 再平分 rank 的分片

    这样做:
    - 不调用 randperm → 不 OOM
    - 每个 worker 进程独立 mmap → 无锁
    - 向量化索引 → 一次读取整个 batch
    """

    def __init__(self, filepath: str, batch_size: int, block_size: int):
        self.filepath = filepath
        self.batch_size = batch_size
        self.block_size = block_size

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # ── 获取 DDP 信息 ──
        rank = 0
        world_size = 1
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        # ── 获取 DataLoader worker 信息 ──
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        # ── 打开 mmap（每个 worker 独立）──
        raw = _open_mmap(self.filepath)  # MADV_RANDOM → 读完即丢
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*non-writable.*")
            raw_t = torch.from_numpy(raw)  # 零拷贝，只读
        total = len(raw) - self.block_size

        # ── 计算本 worker 的分片范围 ──
        # 先按 DDP rank 分，再按 worker_id 分
        rank_shard = total // world_size
        rank_start = rank * rank_shard
        rank_end = (rank + 1) * rank_shard if rank < world_size - 1 else total

        worker_shard = (rank_end - rank_start) // num_workers
        shard_start = rank_start + worker_id * worker_shard
        shard_end = (rank_start + (worker_id + 1) * worker_shard
                     if worker_id < num_workers - 1 else rank_end)

        max_start = shard_end - self.block_size
        if max_start <= shard_start:
            raise RuntimeError(
                f"Worker {worker_id}/{num_workers} (rank {rank}/{world_size}): "
                f"分片太小 ({shard_end - shard_start} tokens)，"
                f"block_size={self.block_size}。请减少 num_workers 或增大数据量。"
            )

        # ── 预计算 offset tensor（形状 [1, block_size]）──
        offsets = torch.arange(self.block_size, dtype=torch.long)

        # ── 无限循环产出 batch ──
        while True:
            starts = torch.randint(shard_start, max_start, (self.batch_size,))
            idx_x = starts.unsqueeze(1) + offsets.unsqueeze(0)       # [B, T]
            idx_y = idx_x + 1                                        # [B, T]

            # 从 mmap view 批量读取 + 转 int64
            x = raw_t[idx_x].long()
            y = raw_t[idx_y].long()

            yield x, y


# ═══════════════════════════════════════════════════════════════════════════════
# create_dataloader
# ═══════════════════════════════════════════════════════════════════════════════

def create_dataloader(
    data_dir: str,
    split: str,
    batch_size: int,
    block_size: int,
    num_workers: int | None = None,
    prefetch_factor: int = 2,
    method: str = "mmap",
) -> DataLoader:
    """创建无限循环的 DataLoader，自动处理 DDP/FSDP + worker 数据分片。

    参数:
        data_dir:         包含 train.bin / val.bin 的目录
        split:            'train' 或 'val'
        batch_size:       每 GPU 的 micro-batch 大小
        block_size:       GPT 的上下文窗口
        num_workers:      DataLoader 子进程数（None=自动）
        prefetch_factor:  每个 worker 预取的 batch 数
        method:           'mmap' (默认) 或 'pread' (零 page cache 污染)

    返回:
        DataLoader，迭代器永不耗尽。X, Y 已 pin_memory。
    """
    if num_workers is None:
        num_workers = 0 if sys.platform == 'win32' else 2

    filepath = os.path.join(data_dir, f'{split}.bin')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}")

    if method == "pread":
        from .pread_loader import PReadLMStream
        stream = PReadLMStream(filepath, batch_size, block_size)
    else:
        stream = LMStream(filepath, batch_size, block_size)

    loader = DataLoader(
        stream,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )
    return loader
