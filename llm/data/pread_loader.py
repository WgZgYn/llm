"""pread 数据加载器 —— 直接系统调用读取，零 page cache 污染。

与 mmap 方案的对比:
    mmap:  缺页 → 4KB page cache → 长期驻留 → RAM 膨胀
    pread: syscall → 用户 buffer → 随用随丢 → RAM 仅占当前 batch

适合: 几十 GB 数据只扫一次、随机采样、内存受限环境
不适合: 反复遍历同份数据（mmap 的 page cache 复用更有优势）

用法:
    from llm.data.pread_loader import create_pread_dataloader

    loader = create_pread_dataloader("data/openwebtext", "train",
                                     batch_size=12, block_size=1024)
    for X, Y in loader:
        ...  # X, Y [B, T] int64, 已 pin_memory
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

# ── 跨平台 pread ──
if hasattr(os, 'pread'):
    def _pread(fd, n_bytes: int, offset: int) -> bytes:
        return os.pread(fd, n_bytes, offset)
else:
    import threading
    _fd_lock = threading.Lock()
    def _pread(fd, n_bytes: int, offset: int) -> bytes:
        """Windows 回退: lseek + read（线程锁保护）。"""
        with _fd_lock:
            os.lseek(fd, offset, os.SEEK_SET)
            return os.read(fd, n_bytes)


class PReadLMStream(IterableDataset):
    """pread 版无限流 IterableDataset。

    与 LMStream 功能完全一致（DDP 分片、worker 分片、无限随机采样），
    区别仅在底层用 pread 替代 mmap。
    """

    def __init__(self, filepath: str, batch_size: int, block_size: int):
        self.filepath = filepath
        self.batch_size = batch_size
        self.block_size = block_size

    def __iter__(self):
        from torch.utils.data import get_worker_info

        # ── DDP + worker 信息 ──
        rank, world_size = 0, 1
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        worker_info = get_worker_info()
        w_id = worker_info.id if worker_info is not None else 0
        n_w = worker_info.num_workers if worker_info is not None else 1

        # ── 打开文件 ──
        fd = os.open(self.filepath, os.O_RDONLY)
        file_size = os.fstat(fd).st_size
        total_tokens = file_size // 2  # uint16 = 2 bytes

        # ── 计算分片 ──
        rank_shard = (total_tokens - self.block_size) // world_size
        r_start = rank * rank_shard
        r_end = (rank + 1) * rank_shard if rank < world_size - 1 else (total_tokens - self.block_size)

        w_shard = (r_end - r_start) // n_w
        shard_start = r_start + w_id * w_shard
        shard_end = r_start + (w_id + 1) * w_shard if w_id < n_w - 1 else r_end

        max_start = shard_end - self.block_size

        # ── 预分配 buffer ──
        bs = self.batch_size
        blk = self.block_size
        item_bytes = blk * 2  # uint16 → 2 bytes/token

        # ── 无限循环 ──
        while True:
            starts = np.random.randint(shard_start, max_start, bs)

            # 逐条 pread → 直接读到 bytes → 零拷贝包装
            xs = [np.frombuffer(_pread(fd, item_bytes, s * 2), dtype=np.uint16)
                  .astype(np.int64) for s in starts]
            ys = [np.frombuffer(_pread(fd, item_bytes, (s + 1) * 2), dtype=np.uint16)
                  .astype(np.int64) for s in starts]

            x = torch.from_numpy(np.stack(xs))
            y = torch.from_numpy(np.stack(ys))
            yield x, y


def create_pread_dataloader(
    data_dir: str, split: str, batch_size: int, block_size: int,
    num_workers: int | None = None, prefetch_factor: int = 2,
) -> DataLoader:
    """创建 pread 版 DataLoader，其他参数同 create_dataloader。"""
    if num_workers is None:
        num_workers = 0 if sys.platform == 'win32' else 2

    filepath = os.path.join(data_dir, f'{split}.bin')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}")

    stream = PReadLMStream(filepath, batch_size, block_size)
    return DataLoader(
        stream,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )
