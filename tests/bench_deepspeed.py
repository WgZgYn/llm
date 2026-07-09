"""DeepSpeed ZeRO 基准测试 —— 不同 Stage 的显存与吞吐对比。

独立脚本，不依赖项目训练框架。

用法:
    # 单卡测试
    deepspeed tests/bench_deepspeed.py --stage 2

    # 多卡测试
    deepspeed --num_gpus=4 tests/bench_deepspeed.py --stage 3

    # 对比所有 stage
    deepspeed tests/bench_deepspeed.py --compare

测量: 峰值显存、每 iter 时间、最大 batch size、是否 OOM
"""

import os, sys, argparse, gc, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 简单模型 (同 bench_backends 风格)
# ═══════════════════════════════════════════════════════════════

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


MODELS = {
    "small":  dict(n_layer=8,  n_embd=512, n_head=8),
    "medium": dict(n_layer=12, n_embd=768, n_head=12),
    "large":  dict(n_layer=24, n_embd=1024, n_head=16),
}


def _n_params(model): return sum(p.numel() for p in model.parameters()) / 1e6


# ═══════════════════════════════════════════════════════════════
# 主测试
# ═══════════════════════════════════════════════════════════════

def run_bench(args):
    # ── 解析参数 ──
    model_name = args.model
    batch_size = args.batch
    seq_len = args.seq_len
    steps = args.steps
    stage = args.stage

    cfg = MODELS[model_name]

    # ── 构建模型 ──
    model = _make_model(**cfg)
    n_params = _n_params(model)

    # ── 检测 DeepSpeed ──
    try:
        import deepspeed
    except ImportError:
        print("[ERROR] DeepSpeed 未安装。 pip install deepspeed")
        return

    # ── DeepSpeed 配置 ──
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": True},
        "zero_optimization": {"stage": stage},
        "wall_clock_breakdown": False,
    }

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank, world_size = 0, 1

    # ── DeepSpeed 初始化 ──
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    # 找最大 batch（如果未指定）
    if batch_size == 0:
        batch_size = _find_max_batch(model, ds_config, cfg, seq_len, max_search=128)
        if rank == 0:
            print(f"Auto-detected max batch size: {batch_size}")
        ds_config["train_micro_batch_size_per_gpu"] = batch_size

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(),
        config_params=ds_config)

    # ── 预热 ──
    for _ in range(5):
        x = torch.randint(0, cfg.get("vocab", 1000), (batch_size, seq_len),
                          device=model_engine.device)
        y = torch.randint(0, cfg.get("vocab", 1000), (batch_size, seq_len),
                          device=model_engine.device)
        loss = model_engine(x, y)
        model_engine.backward(loss)
        model_engine.step()

    torch.cuda.reset_peak_memory_stats()

    # ── 计时 ──
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(steps):
        x = torch.randint(0, cfg.get("vocab", 1000), (batch_size, seq_len),
                          device=model_engine.device)
        y = torch.randint(0, cfg.get("vocab", 1000), (batch_size, seq_len),
                          device=model_engine.device)
        loss = model_engine(x, y)
        model_engine.backward(loss)
        model_engine.step()

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    if rank == 0:
        tokens_per_sec = (batch_size * seq_len * steps * world_size) / dt
        print(f"\n{'='*55}")
        print(f"  DeepSpeed ZeRO-{stage} x {world_size} GPU")
        print(f"  Model:    {model_name} ({n_params:.1f}M params)")
        print(f"  Batch:    {batch_size} x {world_size} = {batch_size*world_size}")
        print(f"  Time:     {dt:.1f}s ({dt/steps:.1f}s/iter)")
        print(f"  Tok/s:    {tokens_per_sec:,.0f}")
        print(f"  PeakVRAM: {peak_mem:.1f} GiB / GPU")
        print(f"{'='*55}")


def _find_max_batch(model, ds_config_base, cfg, seq_len, max_search=128):
    """二分搜索最大的不 OOM batch size。"""
    lo, hi = 1, max_search
    best = 1

    # 简单版：线性递增直到 OOM
    for bs in [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]:
        if bs > max_search:
            break
        try:
            ds_cfg = dict(ds_config_base)
            ds_cfg["train_micro_batch_size_per_gpu"] = bs
            import deepspeed
            m = _make_model(**cfg)
            engine, _, _, _ = deepspeed.initialize(
                model=m, model_parameters=m.parameters(), config_params=ds_cfg)
            x = torch.randint(0, cfg.get("vocab", 1000), (bs, seq_len),
                              device=engine.device)
            y = torch.randint(0, cfg.get("vocab", 1000), (bs, seq_len),
                              device=engine.device)
            loss = engine(x, y)
            engine.backward(loss)
            engine.step()
            best = bs
            # cleanup
            del engine, m, x, y, loss
            torch.cuda.empty_cache()
            gc.collect()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                del m
                torch.cuda.empty_cache()
                gc.collect()
                break
            raise
    return best


# ═══════════════════════════════════════════════════════════════
# 对比模式
# ═══════════════════════════════════════════════════════════════

def run_compare(args):
    """依次跑 ZeRO-1/2/3 对比。"""
    model_name = args.model
    cfg = MODELS[model_name]
    model = _make_model(**cfg)
    n_params = _n_params(model)

    try:
        import deepspeed
    except ImportError:
        print("[ERROR] DeepSpeed 未安装")
        return

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank, world_size = 0, 1

    if rank == 0:
        print(f"\nDeepSpeed ZeRO Comparison: {model_name} ({n_params:.1f}M params)")
        print(f"  {'Stage':<10s} {'MaxBS':>6s} {'PeakVRAM':>10s} {'Time/iter':>10s}")
        print(f"  {'-'*40}")

    for stage in [1, 2, 3]:
        # find max batch
        ds_cfg_base = {"fp16": {"enabled": True},
                       "gradient_accumulation_steps": 1,
                       "zero_optimization": {"stage": stage},
                       "wall_clock_breakdown": False}
        max_bs = _find_max_batch(_make_model(**cfg), ds_cfg_base, cfg, args.seq_len)

        # actual benchmark
        ds_cfg = dict(ds_cfg_base)
        ds_cfg["train_micro_batch_size_per_gpu"] = max_bs
        m = _make_model(**cfg)
        engine, _, _, _ = deepspeed.initialize(
            model=m, model_parameters=m.parameters(), config_params=ds_cfg)

        for _ in range(3):
            x = torch.randint(0, 1000, (max_bs, args.seq_len), device=engine.device)
            y = torch.randint(0, 1000, (max_bs, args.seq_len), device=engine.device)
            loss = engine(x, y)
            engine.backward(loss)
            engine.step()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            x = torch.randint(0, 1000, (max_bs, args.seq_len), device=engine.device)
            y = torch.randint(0, 1000, (max_bs, args.seq_len), device=engine.device)
            loss = engine(x, y)
            engine.backward(loss)
            engine.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**3

        if rank == 0:
            print(f"  ZeRO-{stage:<5s} {max_bs:>6d} {peak:>9.1f}G {(dt/args.steps):>9.2f}s")

        del engine, m
        torch.cuda.empty_cache()
        gc.collect()


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DeepSpeed ZeRO 基准测试")
    parser.add_argument("--model", choices=list(MODELS), default="small")
    parser.add_argument("--batch", type=int, default=0, help="0=auto detect max")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2, 3],
                        help="ZeRO stage")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--compare", action="store_true",
                        help="依次对比 ZeRO-1/2/3")
    args = parser.parse_args()

    if args.compare:
        run_compare(args)
    else:
        run_bench(args)


if __name__ == "__main__":
    main()
