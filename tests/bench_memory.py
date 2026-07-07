"""显存占用对比 —— 单卡 vs DDP vs FSDP。

测量: 同模型在不同后端下的峰值显存、最大 batch size。
纯随机数据，无 I/O，专注测量分布式本身的内存开销。

用法:
    # 单卡基线
    python tests/bench_memory.py --backend single

    # DDP
    torchrun --nproc-per-node=4 tests/bench_memory.py --backend ddp

    # FSDP ZeRO-3
    torchrun --nproc-per-node=4 tests/bench_memory.py --backend fsdp

输出: 模型参数量、峰值显存/GPU、可用显存占比、每 GPU 能塞下的最大 batch
"""

import os, sys, argparse, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def _make_model(n_layer, n_embd, n_head, vocab=1000, block_size=1024):
    return nn.Sequential(
        nn.Embedding(vocab, n_embd),
        *[nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=4*n_embd,
            batch_first=True, norm_first=True, dropout=0.0)
          for _ in range(n_layer)],
        nn.LayerNorm(n_embd),
        nn.Linear(n_embd, vocab),
    )


def _find_max_batch(model, device, seq_len, scaler, optimizer, ddp=False, max_search=128):
    """二分搜索不 OOM 的最大 batch size。"""
    lo, hi = 1, max_search
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            x = torch.randint(0, 1000, (mid, seq_len), device=device)
            y = torch.randint(0, 1000, (mid, seq_len), device=device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1))
            if ddp:
                with model.no_sync():
                    scaler.scale(loss).backward()
            else:
                scaler.scale(loss).backward()
            optimizer.zero_grad()
            best = mid
            lo = mid + 1
        except RuntimeError as e:
            if "out of memory" in str(e):
                hi = mid - 1
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["single","ddp","fsdp"], default="single")
    parser.add_argument("--model", choices=["small","medium","large","xl"], default="medium")
    parser.add_argument("--seq-len", type=int, default=1024)
    args = parser.parse_args()

    # ── 分布式 ──
    rank, world_size = 0, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_dist = int(os.environ.get("RANK", -1)) != -1

    if is_dist:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        torch.cuda.set_device(device)

    # ── 模型配置 ──
    cfgs = {
        "small":  dict(n_layer=8,  n_embd=512, n_head=8),
        "medium": dict(n_layer=12, n_embd=768, n_head=12),
        "large":  dict(n_layer=24, n_embd=1024, n_head=16),
        "xl":     dict(n_layer=36, n_embd=1280, n_head=20),
    }
    cfg = cfgs[args.model]

    # ── 模型 ──
    model = _make_model(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    raw_model = model

    if is_dist and args.backend == "ddp":
        model = DDP(model, device_ids=[device.index or 0])
    elif is_dist and args.backend == "fsdp":
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        mp = MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float16)
        torch.cuda.set_device(device)
        for layer in model:
            fully_shard(layer, mp_policy=mp, reshard_after_forward=True)
        fully_shard(model, mp_policy=mp, reshard_after_forward=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # ── 找最大 batch ──
    torch.cuda.reset_peak_memory_stats()
    use_ddp = is_dist and args.backend == "ddp"
    max_bs = _find_max_batch(model, device, args.seq_len, scaler, optimizer, ddp=use_ddp)
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    if rank == 0:
        print(f"\n{'='*55}")
        print(f"  Backend:      {args.backend} x {world_size} GPU")
        print(f"  Model:        {args.model} ({n_params:.1f}M params)")
        print(f"  Max batch/GPU: {max_bs}")
        print(f"  Global batch:  {max_bs * world_size}")
        print(f"  Peak VRAM:     {peak_mem:.1f} GiB / GPU")
        print(f"  VRAM per 1M param: {peak_mem/n_params*1000:.1f} MiB")
        print(f"{'='*55}")

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
