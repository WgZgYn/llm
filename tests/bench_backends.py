"""分布式后端统一基准 —— 吞吐 + 显存 + 最大 batch。

单文件 self-contained，使用因果 GPT-like 模型逼近真实 LLM 训练特征。

用法:
    # 吞吐基准 (默认)
    python tests/bench_backends.py --backend single --model medium --batch 12
    torchrun --nproc-per-node=4 tests/bench_backends.py --backend ddp --model medium --batch 12

    # 显存探索 (二分找最大 batch)
    python tests/bench_backends.py --backend single --model large --find-max

    # 一键对比 (DDP vs FSDP 同一模型)
    torchrun --nproc-per-node=4 tests/bench_backends.py --model xl --compare

    # 梯度累积模拟
    python tests/bench_backends.py --batch 4 --grad-accum 10

模型预设 (vocab=50304):
    tiny   :  4L  256d  →   ~4M
    small  :  8L  512d  →  ~25M
    medium : 12L  768d  →  ~86M
    large  : 24L 1024d  → ~350M
    xl     : 36L 1280d  → ~711M
    2b     : 48L 1600d  → ~1.5B

注意:
    本脚本使用 nn.TransformerEncoderLayer + is_causal=True 构建因果模型，
    与 llm.model.GPT 共享相同的自回归约束，但实现细节不同（无 weight tying、
    无 GQA/RoPE 等），因此测得的吞吐/显存作为后端对比参考而非精确预测。
"""

import os, time, argparse, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# ═══════════════════════════════════════════════════════════
# 模型 (vocab=50304, 因果自注意力)
# ═══════════════════════════════════════════════════════════

VOCAB = 50304

MODELS = {
    "tiny":   dict(n_layer=4,  n_embd=256,  n_head=4),
    "small":  dict(n_layer=8,  n_embd=512,  n_head=8),
    "medium": dict(n_layer=12, n_embd=768,  n_head=12),
    "large":  dict(n_layer=24, n_embd=1024, n_head=16),
    "xl":     dict(n_layer=36, n_embd=1280, n_head=20),
    "2b":     dict(n_layer=48, n_embd=1600, n_head=25),
}


class CausalTransformer(nn.Module):
    """因果 GPT-like 模型（用于后端 benchmark）。

    使用 nn.TransformerEncoderLayer + causal mask 确保自回归约束，
    避免双向 attention 导致的吞吐/显存测量偏差。
    """

    def __init__(self, vocab_size: int, n_layer: int, n_embd: int, n_head: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd, nhead=n_head, dim_feedforward=4 * n_embd,
                batch_first=True, norm_first=True, dropout=0.0)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        # 按输入序列长度生成因果 mask（上三角为 -inf）
        T = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask)
        x = self.ln_f(x)
        return self.lm_head(x)


def make_model(name="medium"):
    """构造因果 GPT-like 模型。"""
    c = MODELS[name]
    return CausalTransformer(
        vocab_size=VOCAB,
        n_layer=c["n_layer"],
        n_embd=c["n_embd"],
        n_head=c["n_head"],
    )


def n_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


# ═══════════════════════════════════════════════════════════
# 分布式包装
# ═══════════════════════════════════════════════════════════

def _check_fsdp_available():
    """检查当前 PyTorch 是否支持 FSDP2 API (>= 2.5)。"""
    try:
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy  # noqa: F401
        return True
    except ImportError:
        return False


def wrap_model(model, backend, is_dist, device):
    """DDP / FSDP / Single 包装。返回 (wrapped_model, is_ddp)。"""
    is_ddp = False
    if is_dist and backend == "ddp":
        device_id = device.index if device.index is not None else 0
        model = DDP(model, device_ids=[device_id])
        is_ddp = True
    elif is_dist and backend == "fsdp":
        if not _check_fsdp_available():
            raise ImportError(
                "FSDP2 (fully_shard / MixedPrecisionPolicy) 需要 PyTorch >= 2.5。"
                "请升级 PyTorch 或使用 --backend ddp。"
            )
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        mp = MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float16)
        torch.cuda.set_device(device)
        # 逐子模块分片：embed → 每层 → ln_f → lm_head
        fully_shard(model.embed, mp_policy=mp, reshard_after_forward=True)
        for layer in model.layers:
            fully_shard(layer, mp_policy=mp, reshard_after_forward=True)
        fully_shard(model.ln_f, mp_policy=mp, reshard_after_forward=True)
        fully_shard(model.lm_head, mp_policy=mp, reshard_after_forward=True)
        fully_shard(model, mp_policy=mp, reshard_after_forward=True)
    return model, is_ddp


# ═══════════════════════════════════════════════════════════
# 显存搜索
# ═══════════════════════════════════════════════════════════

def _is_oom_error(error: RuntimeError) -> bool:
    """跨语言检测 CUDA OOM 错误。"""
    msg = str(error).lower()
    # 覆盖英文 / 中文 / 西班牙语 等常见 locale 的 OOM 消息
    return any(kw in msg for kw in [
        "out of memory",
        "内存不足",
        "fuera de memoria",
        "mémoire insuffisante",
        "speicher erschöpft",
    ])


def find_max_batch(model, device, seq_len, scaler, optimizer, is_ddp=False):
    """二分搜索不 OOM 的最大 micro-batch size。

    注意: 在 FSDP2 下，所有 rank 必须执行相同的控制流（backward 触发集合通信）。
    若 rank 间显存压力不同导致二分路径分歧，可能导致挂起。
    FSDP 用户建议使用 --batch 显式指定。
    """
    lo, hi, best = 1, 128, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            x = torch.randint(0, VOCAB, (mid, seq_len), device=device)
            y = torch.randint(0, VOCAB, (mid, seq_len), device=device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            if is_ddp:
                with model.no_sync():
                    scaler.scale(loss).backward()
            else:
                scaler.scale(loss).backward()
            optimizer.zero_grad()
            best, lo = mid, mid + 1
        except RuntimeError as e:
            if _is_oom_error(e):
                hi = mid - 1
                torch.cuda.empty_cache(); gc.collect()
            else:
                raise
    return best


# ═══════════════════════════════════════════════════════════
# 吞吐基准
# ═══════════════════════════════════════════════════════════

def bench_throughput(model, device, is_ddp, batch_size, seq_len, grad_accum,
                     scaler, optimizer, steps=20, world_size=1):
    """测量 per-iter 时间、tok/s、峰值显存。

    DDP 模式下使用 no_sync() 避免中间 micro-batch 的冗余梯度同步。
    """
    # warmup
    for _ in range(5):
        for i in range(grad_accum):
            x = torch.randint(0, VOCAB, (batch_size, seq_len), device=device)
            y = torch.randint(0, VOCAB, (batch_size, seq_len), device=device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) / grad_accum
            _ddp_backward(loss, scaler, model, is_ddp, i == grad_accum - 1)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(steps):
        for i in range(grad_accum):
            x = torch.randint(0, VOCAB, (batch_size, seq_len), device=device)
            y = torch.randint(0, VOCAB, (batch_size, seq_len), device=device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) / grad_accum
            _ddp_backward(loss, scaler, model, is_ddp, i == grad_accum - 1)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3

    return dt, peak, (batch_size * seq_len * grad_accum * steps * world_size) / dt


def _ddp_backward(loss, scaler, model, is_ddp, is_last_micro):
    """DDP 模式下仅在最后 micro-batch 执行梯度 all-reduce。"""
    if is_ddp and not is_last_micro:
        with model.no_sync():
            scaler.scale(loss).backward()
    else:
        scaler.scale(loss).backward()


def print_result(backend, model_name, world_size, batch_size, grad_accum,
                 dt, steps, peak, tok_per_sec, params):
    """打印格式化的 benchmark 结果。"""
    tokens_per_step = batch_size * grad_accum * world_size
    print(f"\n{'='*60}")
    print(f"  Backend:       {backend} x {world_size} GPU")
    print(f"  Model:         {model_name} ({params:.1f}M params, vocab={VOCAB})")
    print(f"  Per-GPU batch: {batch_size} x {grad_accum} = {batch_size * grad_accum}")
    print(f"  Global batch:  {tokens_per_step} tokens/step")
    print(f"  Time:          {dt:.2f}s ({dt/steps:.3f}s/iter)")
    print(f"  Throughput:    {tok_per_sec:,.0f} tok/s")
    print(f"  Peak VRAM:     {peak:.2f} GiB / GPU")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════
# 对比模式
# ═══════════════════════════════════════════════════════════

def run_compare(args):
    """同一模型下 DDP vs FSDP 对比。"""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if dist.is_initialized() else "cuda")

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  Backend Comparison: {args.model} on {world_size} GPU(s)")
        print(f"  {'Backend':<10s} {'MaxBS':>6s} {'PeakVRAM':>10s} {'Time/iter':>10s} {'Tok/s':>10s}")
        print(f"  {'-'*55}")

    for backend in (["single"] if world_size == 1 else ["ddp", "fsdp"]):
        torch.cuda.empty_cache(); gc.collect()
        raw = make_model(args.model).to(device)
        model, is_ddp = wrap_model(raw, backend, world_size > 1, device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        sc = torch.amp.GradScaler("cuda", enabled=True)

        # 找最大 batch
        max_bs = args.batch if args.batch > 0 else find_max_batch(
            model, device, args.seq_len, sc, opt, is_ddp=is_ddp)
        bs = min(max_bs, 64) if world_size > 1 else max_bs  # 多卡时不炸

        # 测吞吐
        dt, peak, tok_s = bench_throughput(
            model, device, is_ddp, bs, args.seq_len, args.grad_accum,
            sc, opt, steps=args.steps, world_size=world_size)

        if rank == 0:
            print(f"  {backend:<10s} {max_bs:>6d} {peak:>9.2f}G {(dt/args.steps):>9.3f}s {tok_s:>10,.0f}")

        del model, raw, opt, sc


# ═══════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="分布式后端统一基准")
    p.add_argument("--backend", choices=["single","ddp","fsdp"], default="single")
    p.add_argument("--model", choices=list(MODELS), default="small")
    p.add_argument("--batch", type=int, default=0, help="0=自动二分搜索最大 batch")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--find-max", action="store_true", help="纯显存搜索")
    p.add_argument("--compare", action="store_true", help="一键 DDP/FSDP 对比")
    args = p.parse_args()

    # ── 分布式 ──
    is_dist = int(os.environ.get("RANK", -1)) != -1
    if is_dist:
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank() if is_dist else 0
    world_size = dist.get_world_size() if is_dist else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(
        f"cuda:{local_rank}" if is_dist
        else ("cuda" if torch.cuda.is_available() else "cpu"))

    if is_dist:
        torch.cuda.set_device(device)

    if args.compare:
        run_compare(args)
        if is_dist: dist.destroy_process_group()
        return

    # ── 模型 + 包装 ──
    raw = make_model(args.model).to(device)
    params = n_params(raw)
    model, is_ddp = wrap_model(raw, args.backend, is_dist, device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    sc = torch.amp.GradScaler("cuda", enabled=True)

    # ── 搜最大 batch ──
    bs = args.batch
    if bs == 0:
        bs = find_max_batch(model, device, args.seq_len, sc, opt, is_ddp=is_ddp)
    if bs == 0:
        if rank == 0:
            print("ERROR: 连 batch_size=1 都 OOM，无法运行 benchmark。")
        if is_dist: dist.destroy_process_group()
        return
    if rank == 0:
        print(f"max batch: {bs}")

    if args.find_max:
        peak = torch.cuda.max_memory_allocated() / 1024**3
        if rank == 0:
            print(f"Peak VRAM at max batch: {peak:.2f} GiB")
        if is_dist: dist.destroy_process_group()
        return

    # ── 吞吐测试 ──
    dt, peak, tok_s = bench_throughput(
        model, device, is_ddp, bs, args.seq_len, args.grad_accum,
        sc, opt, steps=args.steps, world_size=world_size)

    if rank == 0:
        print_result(args.backend, args.model, world_size, bs,
                     args.grad_accum, dt, args.steps, peak, tok_s, params)

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
