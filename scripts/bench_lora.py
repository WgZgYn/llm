"""LoRA 微调基准测试 —— 全量微调 vs LoRA 多配置对比。

测量: 可训练参数、GPU显存、训练速度(tok/s)、每iter时间。

产出: 直接回应导师"训练时间从5天降到5小时"的对比数据表。

用法:
    python scripts/bench_lora.py                    # 默认 OWT 数据
    python scripts/bench_lora.py --steps 50         # 更快
    python scripts/bench_lora.py --no-full-ft       # 跳过全量微调（OOM风险）
"""

import os
import sys
import time
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

from llm.config import GPTConfig, TrainingConfig
from llm.data.loader import create_dataloader
from llm.lora import LoRAConfig, LoRAGPT, count_lora_params, print_lora_info


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def get_gpu_memory() -> dict:
    """获取 GPU 显存信息（GiB）。"""
    if not torch.cuda.is_available():
        return {}
    return {
        'allocated': torch.cuda.memory_allocated() / (1024 ** 3),
        'reserved': torch.cuda.memory_reserved() / (1024 ** 3),
        'max_allocated': torch.cuda.max_memory_allocated() / (1024 ** 3),
    }


def reset_memory_stats():
    """重置 CUDA 显存统计。"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def count_trainable(model: nn.Module) -> int:
    """统计可训练参数数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model: nn.Module) -> int:
    """统计总参数数量。"""
    return sum(p.numel() for p in model.parameters())


# ═══════════════════════════════════════════════════════════════
# 基准测试
# ═══════════════════════════════════════════════════════════════

def benchmark_config(
    name: str,
    build_model_fn,
    steps: int = 100,
    batch_size: int = 4,
    block_size: int = 512,
    lr: float = 2e-4,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = None,
):
    """对一种微调配置进行基准测试。

    参数:
        name: 配置名称（如 "LoRA r=8"）
        build_model_fn: 返回 (model, raw_model) 的函数
        steps: 测试步数
        batch_size: micro-batch size
        block_size: 序列长度
        lr: 学习率
        dtype: 计算精度
        device: 设备

    返回:
        dict: 基准测试结果
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 60}")
    print(f"  Benchmark: {name}")
    print(f"{'=' * 60}")

    # ── 构建模型 ──
    reset_memory_stats()
    model, raw_model = build_model_fn()
    model = model.to(device)
    model.train()

    mem_after_load = get_gpu_memory()
    trainable = count_trainable(raw_model)
    total = count_total(raw_model)

    print(f"  Trainable: {trainable/1e6:.2f}M / {total/1e6:.1f}M "
          f"({trainable/total*100:.2f}%)")
    print(f"  GPU memory after load: {mem_after_load.get('allocated', 0):.2f} GiB")

    # ── 数据加载器 ──
    data_dir = os.path.join(PROJECT_ROOT, "data", "openwebtext")
    if not os.path.exists(os.path.join(data_dir, "train.bin")):
        # 使用随机数据作为 fallback
        print("  [WARN] OWT data not found, using random data")
        use_random = True
    else:
        use_random = False

    if use_random:
        # 随机数据模拟
        def random_batch():
            x = torch.randint(0, 50257, (batch_size, block_size), device=device)
            y = torch.randint(0, 50257, (batch_size, block_size), device=device)
            return x, y
        get_batch = random_batch
    else:
        train_loader = create_dataloader(data_dir, "train", batch_size, block_size)
        train_iter = iter(train_loader)
        def get_batch():
            X, Y = next(train_iter)
            return X.to(device), Y.to(device)

    # ── 优化器 ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr, betas=(0.9, 0.95),
        weight_decay=1e-2, fused=torch.cuda.is_available(),
    )

    # ── Warmup ──
    print("  Warming up (10 steps)...")
    for _ in range(10):
        x, y = get_batch()
        with torch.autocast(device_type='cuda', dtype=dtype):
            logits, loss = raw_model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ── 正式测试 ──
    reset_memory_stats()
    torch.cuda.synchronize()

    times = []
    mem_peak = {'allocated': 0}

    print(f"  Benchmarking ({steps} steps)...")
    for i in range(steps):
        x, y = get_batch()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.autocast(device_type='cuda', dtype=dtype):
            logits, loss = raw_model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        mem_peak = get_gpu_memory()

        if (i + 1) % 25 == 0:
            avg_time = sum(times[-25:]) / 25
            tok_per_sec = (batch_size * block_size) / avg_time
            print(f"    step {i+1:3d}/{steps}: "
                  f"avg_dt={avg_time*1000:.0f}ms, "
                  f"tok/s={tok_per_sec:,.0f}, "
                  f"loss={loss.item():.4f}, "
                  f"gmem={mem_peak.get('allocated', 0):.2f}GiB")

    torch.cuda.synchronize()

    # ── 汇总 ──
    avg_time = sum(times) / len(times)
    tok_per_sec = (batch_size * block_size) / avg_time

    result = {
        'name': name,
        'trainable_params': trainable,
        'total_params': total,
        'trainable_ratio': trainable / total * 100,
        'gmem_allocated': mem_peak.get('allocated', 0),
        'gmem_reserved': mem_peak.get('reserved', 0),
        'gmem_peak': mem_peak.get('max_allocated', 0),
        'avg_iter_ms': avg_time * 1000,
        'tok_per_sec': tok_per_sec,
        'steps': steps,
    }

    print(f"\n  >>> {name}: "
          f"dt={result['avg_iter_ms']:.0f}ms, "
          f"tok/s={result['tok_per_sec']:,.0f}, "
          f"gmem={result['gmem_peak']:.2f}GiB")

    # 清理显存
    del model, raw_model, optimizer
    torch.cuda.empty_cache()
    reset_memory_stats()

    return result


# ═══════════════════════════════════════════════════════════════
# 打印对比表格
# ═══════════════════════════════════════════════════════════════

def print_comparison_table(results: list[dict]):
    """打印多配置对比表格。"""
    print(f"\n{'=' * 90}")
    print("  LoRA Fine-Tuning Benchmark Results")
    print(f"{'=' * 90}")
    print(f"  {'Configuration':<24s} {'Trainable':>10s} {'Ratio':>7s} "
          f"{'GPU Mem':>8s} {'Time/iter':>9s} {'tok/s':>10s}")
    print(f"  {'-' * 24} {'-' * 10} {'-' * 7} {'-' * 8} {'-' * 9} {'-' * 10}")

    baseline_tok = None
    for r in results:
        if r['name'] == 'Full Fine-Tuning':
            baseline_tok = r['tok_per_sec']

    for r in results:
        ratio = f"{r['trainable_ratio']:.1f}%"
        gmem = f"{r['gmem_peak']:.2f}G"
        dt = f"{r['avg_iter_ms']:.0f}ms"
        tok = f"{r['tok_per_sec']:,.0f}"

        # 相对于全量微调的加速比
        speedup = ""
        if baseline_tok and r['name'] != 'Full Fine-Tuning':
            ratio_speed = r['tok_per_sec'] / baseline_tok
            speedup = f" ({ratio_speed:.1f}x)"

        print(f"  {r['name']:<24s} {r['trainable_params']/1e6:>8.2f}M {ratio:>7s} "
              f"{gmem:>8s} {dt:>9s} {tok:>10s}{speedup}")

    print(f"  {'=' * 90}")

    # 显存对比
    print(f"\n  GPU Memory Reduction (vs Full Fine-Tuning):")
    for r in results:
        if r['name'] == 'Full Fine-Tuning':
            full_ft_mem = r['gmem_peak']
            print(f"    {r['name']:<24s}: {r['gmem_peak']:.2f} GiB (baseline)")
        else:
            reduction = (1 - r['gmem_peak'] / full_ft_mem) * 100 if full_ft_mem > 0 else 0
            print(f"    {r['name']:<24s}: {r['gmem_peak']:.2f} GiB (-{reduction:.0f}%)")

    # 关键结论
    print(f"\n  Key Takeaways:")
    for r in results:
        if r['name'] != 'Full Fine-Tuning':
            mem_ratio = r['gmem_peak'] / full_ft_mem * 100 if full_ft_mem > 0 else 0
            speed_ratio = r['tok_per_sec'] / baseline_tok if baseline_tok else 0
            print(f"    {r['name']}: uses {mem_ratio:.0f}% memory, "
                  f"{speed_ratio:.1f}x training speed vs Full FT")


# ═══════════════════════════════════════════════════════════════
# 构建不同配置的模型
# ═══════════════════════════════════════════════════════════════

def build_full_ft():
    """全量微调：GPT-2 124M，所有参数可训练。"""
    model_cfg = GPTConfig(
        vocab_size=50257, block_size=512,
        n_layer=12, n_head=12, n_embd=768,
        dropout=0.0, bias=True,
    )
    from llm.model.gpt import GPT
    from transformers import GPT2LMHeadModel

    # 创建模型并加载 HF 权重
    model = GPT(model_cfg)
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    sd_hf = model_hf.state_dict()

    # 映射权重（简化版，处理关键层）
    sd_local = model.state_dict()
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                  'mlp.c_fc.weight', 'mlp.c_proj.weight']
    mlp_map = {'mlp.c_fc': 'mlp.net.0', 'mlp.c_proj': 'mlp.net.2'}

    for hf_key, hf_tensor in sd_hf.items():
        if hf_key.endswith(('.attn.masked_bias', '.attn.bias')):
            continue
        local_key = hf_key
        for hf_n, local_n in mlp_map.items():
            if hf_n in local_key:
                local_key = local_key.replace(hf_n, local_n)
                break
        if local_key not in sd_local:
            continue
        if any(hf_key.endswith(w) for w in transposed):
            if hf_tensor.shape == sd_local[local_key].shape and hf_tensor.shape[0] != hf_tensor.shape[1]:
                sd_local[local_key].copy_(hf_tensor)
            else:
                sd_local[local_key].copy_(hf_tensor.t().contiguous())
        else:
            if hf_tensor.shape == sd_local[local_key].shape:
                sd_local[local_key].copy_(hf_tensor)

    del model_hf
    # 所有参数可训练
    for p in model.parameters():
        p.requires_grad = True
    return model, model


def build_lora(r: int, alpha: float, target_modules: str,
               train_wpe: bool = False, train_ln: bool = False):
    """LoRA 微调：只训练低秩矩阵。"""
    model_cfg = GPTConfig(
        vocab_size=50257, block_size=512,
        n_layer=12, n_head=12, n_embd=768,
        dropout=0.0, bias=True,
    )
    lora_cfg = LoRAConfig(
        r=r, alpha=alpha, dropout=0.0,
        target_modules=target_modules, target_layers=None,
        variant='lora', train_wpe=train_wpe, train_ln=train_ln,
    )
    model = LoRAGPT.from_pretrained("gpt2", lora_cfg)
    return model, model


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning Benchmark")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of benchmark steps per config (default: 100)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Micro-batch size (default: 4)")
    parser.add_argument("--block-size", type=int, default=512,
                        help="Sequence length (default: 512)")
    parser.add_argument("--no-full-ft", action="store_true",
                        help="Skip full fine-tuning (avoids OOM on 8GB GPUs)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    dtype = dtype_map[args.dtype]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type != 'cuda':
        print("[ERROR] CUDA not available. This benchmark requires GPU.")
        sys.exit(1)

    # bf16 兼容性检查
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("[WARN] bfloat16 not supported, falling back to float16")
        dtype = torch.float16

    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name} ({props.total_mem / (1024**3):.0f} GiB)")
    print(f"dtype: {args.dtype} | steps: {args.steps} | "
          f"batch_size: {args.batch_size} | block_size: {args.block_size}")

    results = []

    # ── Config 1: Full Fine-Tuning ──
    if not args.no_full_ft:
        try:
            r = benchmark_config(
                "Full Fine-Tuning",
                lambda: build_full_ft(),
                steps=args.steps,
                batch_size=args.batch_size,
                block_size=args.block_size,
                dtype=dtype,
                device=device,
            )
            results.append(r)
        except torch.cuda.OutOfMemoryError:
            print("  [SKIP] Full Fine-Tuning OOM (expected on 8GB GPUs)")
            torch.cuda.empty_cache()
    else:
        print("\n  [SKIP] Full Fine-Tuning (--no-full-ft)")

    # ── Config 2: LoRA r=8 (attn only) ──
    r = benchmark_config(
        "LoRA r=8 (attn only)",
        lambda: build_lora(r=8, alpha=16.0, target_modules='attn'),
        steps=args.steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        dtype=dtype,
        device=device,
    )
    results.append(r)

    # ── Config 3: LoRA r=16 (all) ──
    r = benchmark_config(
        "LoRA r=16 (attn+mlp)",
        lambda: build_lora(r=16, alpha=32.0, target_modules='all'),
        steps=args.steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        dtype=dtype,
        device=device,
    )
    results.append(r)

    # ── Config 4: LoRA r=32 (all, 高容量) ──
    r = benchmark_config(
        "LoRA r=32 (attn+mlp)",
        lambda: build_lora(r=32, alpha=64.0, target_modules='all'),
        steps=args.steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        dtype=dtype,
        device=device,
    )
    results.append(r)

    # ── 打印对比表 ──
    print_comparison_table(results)

    print("\nDone!")


if __name__ == "__main__":
    main()
