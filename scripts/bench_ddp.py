"""DDP 加速效果验证脚本 —— 最小化基准测试。

完全剥离框架代码，只用纯 PyTorch DDP + 随机数据，
测量单卡 vs 多卡的实际吞吐量。

用法:
    # 单卡
    python scripts/bench_ddp.py

    # 4 卡 DDP
    torchrun --nproc-per-node=4 scripts/bench_ddp.py

输出: 每个 GPU 的 tokens/sec 和总吞吐量
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# ── 与 GPT-2 124M 等价的简单模型 ──
class SimpleGPT(nn.Module):
    """和 GPT-2 124M 计算量相同的简化模型，用随机数据，无文件 I/O。"""

    def __init__(self, vocab_size=50304, n_embd=768, n_layer=12,
                 n_head=12, block_size=1024):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=n_embd, nhead=n_head, dim_feedforward=4 * n_embd,
                batch_first=True, norm_first=True,  # Pre-LN
                dropout=0.0,
            )
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, y):
        # x: [B, T] int, y: [B, T] int
        h = self.embed(x)
        h = self.blocks(h)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )


def bench():
    # ── DDP 初始化 ──
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device('cuda')

    # ── 模型 ──
    model = SimpleGPT().to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # 编译（可选）
    # model = torch.compile(model)

    # ── 优化器 ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=True)  # fp16

    # ── 参数 ──
    batch_size = 12
    block_size = 1024
    grad_accum = 40
    warmup_iters = 3
    bench_iters = 10

    # ── 预热 ──
    if rank == 0:
        print(f"World: {world_size} GPU(s), batch={batch_size}, "
              f"accum={grad_accum}, warmup={warmup_iters}...")

    for _ in range(warmup_iters):
        for micro_step in range(grad_accum):
            is_last = (micro_step == grad_accum - 1)
            # 随机数据（无 I/O，纯计算）
            x = torch.randint(0, 50304, (batch_size, block_size), device=device)
            y = torch.randint(0, 50304, (batch_size, block_size), device=device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                loss = model(x, y)
                loss = loss / grad_accum

            if ddp and not is_last:
                with model.no_sync():
                    scaler.scale(loss).backward()
            else:
                scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    if ddp:
        dist.barrier()

    # ── 基准测试 ──
    if rank == 0:
        print(f"Benchmarking {bench_iters} iters...")

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    for _ in range(bench_iters):
        for micro_step in range(grad_accum):
            is_last = (micro_step == grad_accum - 1)
            x = torch.randint(0, 50304, (batch_size, block_size), device=device)
            y = torch.randint(0, 50304, (batch_size, block_size), device=device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                loss = model(x, y)
                loss = loss / grad_accum

            if ddp and not is_last:
                with model.no_sync():
                    scaler.scale(loss).backward()
            else:
                scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0

    # ── 统计 ──
    tokens_per_iter = batch_size * block_size * grad_accum
    tokens_total = tokens_per_iter * bench_iters
    tok_per_sec_per_gpu = tokens_total / dt
    tok_per_sec_total = tok_per_sec_per_gpu * world_size

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  GPUs:          {world_size}")
        print(f"  Time:           {dt:.1f}s ({dt/bench_iters:.1f}s/iter)")
        print(f"  Tokens/GPU/s:   {tok_per_sec_per_gpu:,.0f}")
        print(f"  Tokens TOTAL/s: {tok_per_sec_total:,.0f}")
        if world_size > 1:
            # 与单卡对比（需要之前跑过单卡，记录在文件中）
            print(f"  预期(无通信):   {tok_per_sec_per_gpu * world_size:,.0f}")
        print(f"{'='*60}")

    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    bench()
