"""分布式后端基准测试 —— 单卡 vs DDP vs FSDP。

测量: 每 iter 时间、tok/s、峰值显存、最大 batch size。
完全独立，不依赖配置文件或 DataLoader。

用法:
    python tests/bench_backends.py                    # 单卡
    torchrun --nproc-per-node=4 tests/bench_backends.py  # DDP/FSDP (改 --backend)

参数:
    --backend     single | ddp | fsdp (默认 fsdp)
    --model       tiny | small | medium (默认 small)
    --batch       每 GPU micro-batch (默认 auto: 逐步增大至 OOM)
    --steps       测试步数 (默认 20)
"""

import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# ═══════════════════════════════════════════════════════════
# 简单 GPT-like 模型 (足够复现显存/吞吐特征)
# ═══════════════════════════════════════════════════════════

def _make_model(n_layer, n_embd, n_head, vocab=1000, block_size=1024):
    """构建一个类似 GPT 的模型。"""
    return nn.Sequential(
        nn.Embedding(vocab, n_embd),
        *[nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=4*n_embd,
            batch_first=True, norm_first=True, dropout=0.0)
          for _ in range(n_layer)],
        nn.LayerNorm(n_embd),
        nn.Linear(n_embd, vocab),
    )


MODEL_CFGS = {
    "tiny":   dict(n_layer=4,  n_embd=256, n_head=4),
    "small":  dict(n_layer=8,  n_embd=512, n_head=8),
    "medium": dict(n_layer=12, n_embd=768, n_head=12),
    "large":  dict(n_layer=24, n_embd=1024, n_head=16),
}

# ═══════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════

def _n_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def _gpu_mem():
    return torch.cuda.memory_allocated() / 1024**3, torch.cuda.max_memory_allocated() / 1024**3


# ═══════════════════════════════════════════════════════════
# 主测试
# ═══════════════════════════════════════════════════════════

def run_bench():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["single","ddp","fsdp"], default="fsdp")
    parser.add_argument("--model", choices=list(MODEL_CFGS), default="small")
    parser.add_argument("--batch", type=int, default=0, help="0=auto search")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--grad-accum", type=int, default=1)
    args = parser.parse_args()

    # ── 分布式初始化 ──
    rank, world_size = 0, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_dist = int(os.environ.get("RANK", -1)) != -1

    if is_dist:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        torch.cuda.set_device(device)

    # ── 模型 ──
    cfg = MODEL_CFGS[args.model]
    model = _make_model(**cfg).to(device)
    raw_model = model

    if is_dist and args.backend == "ddp":
        model = DDP(model, device_ids=[device.index if device.index is not None else 0])
    elif is_dist and args.backend == "fsdp":
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        mp = MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float16)
        torch.cuda.set_device(device)
        for m in model.modules():
            if len(list(m.children())) == 0:
                continue
        for layer in model:
            fully_shard(layer, mp_policy=mp, reshard_after_forward=True)
        fully_shard(model, mp_policy=mp, reshard_after_forward=True)

    # ── 优化器 ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # ── 找最大 batch size ──
    bs = args.batch
    grad_accum = args.grad_accum
    if bs == 0:
        bs = 2
        while True:
            try:
                x = torch.randint(0, 1000, (bs, args.seq_len), device=device)
                y = torch.randint(0, 1000, (bs, args.seq_len), device=device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1))
                scaler.scale(loss).backward()
                optimizer.zero_grad()
                bs *= 2
            except RuntimeError as e:
                if "out of memory" in str(e):
                    bs //= 2
                    torch.cuda.empty_cache()
                    break
                raise
        if rank == 0:
            print(f"max batch size: {bs}")

    # ── 预热 ──
    torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        for _ in range(grad_accum):
            x = torch.randint(0, 1000, (bs, args.seq_len), device=device)
            y = torch.randint(0, 1000, (bs, args.seq_len), device=device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)) / grad_accum
            if is_dist and args.backend == "ddp":
                is_last = True  # simplified
                if not is_last:
                    with model.no_sync():
                        scaler.scale(loss).backward()
                else:
                    scaler.scale(loss).backward()
            else:
                scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats()

    # ── 计时 ──
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(args.steps):
        for _ in range(grad_accum):
            x = torch.randint(0, 1000, (bs, args.seq_len), device=device)
            y = torch.randint(0, 1000, (bs, args.seq_len), device=device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)) / grad_accum
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    mem_peak = torch.cuda.max_memory_allocated() / 1024**3

    if rank == 0:
        params = _n_params(raw_model if hasattr(model, 'module') else model)
        tokens_per_sec = (bs * args.seq_len * grad_accum * args.steps * world_size) / dt
        print(f"\n{'='*55}")
        print(f"  Backend:    {args.backend} x {world_size} GPU")
        print(f"  Model:      {args.model} ({params:.1f}M params)")
        print(f"  Batch:      {bs} x {grad_accum} x {world_size} = {bs*grad_accum*world_size}")
        print(f"  Time:       {dt:.1f}s ({dt/args.steps:.1f}s/iter)")
        print(f"  Throughput: {tokens_per_sec:,.0f} tok/s")
        print(f"  Peak VRAM:  {mem_peak:.1f} GiB / GPU")
        print(f"{'='*55}")

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    run_bench()
