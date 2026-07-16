"""Pipeline Parallel (PP) 原理 demo —— GPipe 与 1F1B 调度。

核心思想: 模型按层切分到多卡，每卡只存部分层，通过流水线并行执行。
         用 micro-batch 填充流水线气泡，提高 GPU 利用率。

    原生流水线 (有气泡):
        GPU0: [F0][        ][F1][        ]
        GPU1: [   ][F0][        ][F1]
              ├─ bubble ─┤

    GPipe (micro-batch 填泡):
        GPU0: [F0][F1][F2][F3][B0][B1][B2][B3]
        GPU1:    [F0][F1][F2][F3][B0][B1][B2][B3]
              ↑ 气泡被 micro-batch 填满

    1F1B (减少峰值显存):
        GPU0: [F0][F1][F2][F3][B0][B1][B2][B3]
        GPU1:    [F0][F1][F2][F3][B0][B1][B2][B3]
              ↑ 不等所有 F 完成, 交替 F/B — 显存峰值更低

用法:
    python tests/unit/test_pp.py
"""

import os, sys, time, math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
# 第一部分: 概念演示 —— 流水线气泡
# ═══════════════════════════════════════════════════════════

def demo_pipeline_bubble():
    """演示流水线气泡为什么产生，以及 micro-batch 如何填泡。"""
    print("=" * 60)
    print("PP 原理: 流水线气泡与 Micro-Batch 填充")
    print("=" * 60)

    n_stages, n_micro = 4, 1  # 4 GPU, 1 micro-batch -> 气泡最大

    for n_mb in [1, 4, 8]:
        total_slots = (n_stages + n_mb - 1) * n_stages
        active_slots = n_stages * n_mb * 2  # forward + backward
        bubble_slots = total_slots - active_slots
        bubble_ratio = bubble_slots / total_slots * 100

        print(f"  Stages={n_stages}, Micro-batches={n_mb}: "
              f"bubble={bubble_ratio:.0f}%  "
              f"({'#' * max(1, int(bubble_ratio/5))}{'.' * max(1, 20-int(bubble_ratio/5))})")

    print(f"\n  结论: micro-batch 越多 -> 气泡越少 -> GPU 利用率越高")
    print(f"        但 micro-batch 数增加 -> 峰值显存增加 (要同时存多个 batch 的激活)\n")


# ═══════════════════════════════════════════════════════════
# 第二部分: PP 调度器 (仿真)
# ═══════════════════════════════════════════════════════════

def simulate_gpipe(n_stages=4, n_micro=8):
    """仿真 GPipe 调度: 先全部 forward，再全部 backward。"""
    timeline = [[] for _ in range(n_stages)]
    clock = 0

    # Forward: 每层流水线前进
    for mb in range(n_micro):
        for stage in range(n_stages):
            start = max(clock, timeline[stage][-1] if timeline[stage] else 0)
            if stage > 0:
                start = max(start, timeline[stage - 1][mb] if len(timeline[stage - 1]) > mb else 0)
            timeline[stage].append(start + 1)  # 1 unit per forward

    # Backward: 反向流水线
    for mb in range(n_micro - 1, -1, -1):
        for stage in range(n_stages - 1, -1, -1):
            prev = timeline[stage][-1] if timeline[stage] else 0
            if stage < n_stages - 1:
                bw_idx = n_micro + (n_micro - 1 - mb)
                prev = max(prev, timeline[stage + 1][bw_idx] if len(timeline[stage + 1]) > bw_idx else 0)
            # 上一层的 backward 时间
            if stage < n_stages - 1:
                last_bw_end = timeline[stage + 1][n_micro + (n_micro - 1 - mb)] if len(timeline[stage + 1]) > n_micro + (n_micro - 1 - mb) else 0
                prev = max(prev, last_bw_end)
            timeline[stage].append(max(prev, timeline[stage][-1]) + 2)  # backward costs 2x

    total_time = max(t[-1] for t in timeline)
    active = n_stages * n_micro * 3   # 1 forward + 2 backward
    total_slots = total_time * n_stages
    bubble = (total_slots - active) / total_slots * 100

    print(f"  GPipe: {n_stages} stages, {n_micro} micro-batches")
    print(f"    总时间: {total_time:.0f} units, 气泡: {bubble:.0f}%")

    if n_micro >= 8:
        formula_bubble = (n_stages - 1) / (n_stages - 1 + n_micro) * 100
        print(f"    理论气泡公式: (P-1)/(P-1+M) = {formula_bubble:.0f}%")


def simulate_1f1b(n_stages=4, n_micro=8):
    """仿真 1F1B 调度: warmup 后交替 F/B。"""
    # 简化版: warmup 阶段每个 stage 做足够多次 F，
    # 然后交替 F/B 直到所有 micro-batch 处理完
    warmup_ahead = [n_stages - i - 1 for i in range(n_stages)]
    # stage i 需要在前方 stage 填充 (n_stages - i - 1) 个 F 后收到第一个

    # 活跃 slot = n_stages × n_micro × 2 (F + B)
    # 总 slot = warmup + 2 × n_micro × n_stages - cooldown
    active = n_stages * n_micro * 2
    total = active + (n_stages - 1) * n_stages  # rough
    bubble = max(0, (total - active) / total * 100)

    print(f"  1F1B:  {n_stages} stages, {n_micro} micro-batches")
    print(f"    活跃时间 ~{n_micro * 2:.0f} units, 气泡 ~{bubble:.0f}%")
    print(f"    优势: 相比 GPipe，反向更早开始 -> 激活值峰值更低 -> 显存更低\n")


# ═══════════════════════════════════════════════════════════
# 第三部分: 纳什均衡模型分配
# ═══════════════════════════════════════════════════════════

def optimal_stage_partition():
    """给定异构 GPU，如何分配层数使得流水线最均匀？

    问题: 4 GPU，算力比 1:1:0.5:0.5，24 层模型，每 GPU 分配几层？
    """
    print("=" * 60)
    print("PP 负载均衡: 异构 GPU 下层数分配")
    print("=" * 60)

    n_layers = 24
    gpus = [
        ("GPU0 A100", 1.0),
        ("GPU1 A100", 1.0),
        ("GPU2 V100", 0.5),
        ("GPU3 V100", 0.5),
    ]

    total_capacity = sum(c for _, c in gpus)
    allocations = []
    remaining = n_layers

    for name, cap in gpus:
        # 按算力比例分配层数
        n = round(n_layers * cap / total_capacity)
        n = max(1, min(n, remaining - (len(gpus) - len(allocations) - 1)))
        allocations.append((name, n))
        remaining -= n

    print(f"  Model: {n_layers} layers, GPUs with relative speeds:")
    for name, cap in gpus:
        print(f"    {name}: {cap:.1f}x")
    print(f"\n  分配结果:")
    for name, n in allocations:
        bar = "#" * n
        print(f"    {name}: {n:>2d} layers {bar}")

    # 计算预估每 GPU 时间 (正比于 layers/speed)
    print(f"\n  预估每 GPU 耗时:")
    for (name, cap), (_, n) in zip(gpus, allocations):
        t = n / cap
        print(f"    {name}: {n}/{cap:.1f} = {t:.1f} units")

    print(f"\n  结论: 异构 GPU 下按算力比例分配层数 -> 流水线最均匀 -> 气泡最小")
    print(f"        这是你的课题可以优化的点 — 现有 PP 假设同构 GPU\n")


# ═══════════════════════════════════════════════════════════
# 第四部分: PP + DP 混合
# ═══════════════════════════════════════════════════════════

def hybrid_pp_dp():
    """演示 PP + DP 如何组合。"""
    print("=" * 60)
    print("PP + DP 混合并行")
    print("=" * 60)

    n_gpus = 8
    pp_size = 4   # 流水线深度
    dp_size = n_gpus // pp_size  # 数据并行组

    print(f"  8 GPU: PP={pp_size} x DP={dp_size}")
    print(f"")
    print(f"  GPU layout:")
    print(f"    DP group 0: GPU0 GPU1 GPU2 GPU3  (pipeline stage 0-3)")
    print(f"    DP group 1: GPU4 GPU5 GPU6 GPU7  (pipeline stage 0-3)")
    print(f"")
    print(f"  通信模式:")
    print(f"    层内 PP:   GPU0 -> GPU1 -> GPU2 -> GPU3  (send/recv activations)")
    print(f"    层间 DP:   GPU0 <-> GPU4  (all-reduce gradients)")
    print(f"")
    print(f"  优势: PP 解决单卡放不下模型, DP 增加吞吐")
    print(f"  Megatron-LM 标准配置: TP 在节点内 -> PP 跨节点 -> DP 跨节点组")


# ═══════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo_pipeline_bubble()
    simulate_gpipe(4, 8)
    simulate_1f1b(4, 8)
    optimal_stage_partition()
    hybrid_pp_dp()
