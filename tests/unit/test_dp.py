"""Data Parallel (DP / DDP) 原理 demo。

核心思想: 数据分片到多卡，每卡完整模型，各自 forward/backward，梯度同步。

    GPU0: [模型副本] 处理 batch[0:2]  → 局部梯度 g0
    GPU1: [模型副本] 处理 batch[2:4]  → 局部梯度 g1   } all-reduce → (g0+g1)/2
    GPU2: [模型副本] 处理 batch[4:6]  → 局部梯度 g2
    GPU3: [模型副本] 处理 batch[6:8]  → 局部梯度 g3

用法:
    python tests/unit/test_dp.py                     # 单进程演示
    torchrun --nproc-per-node=2 tests/unit/test_dp.py  # 真 2 卡 DDP
"""

import os, sys, time, math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
# 第一部分: DP 原理 —— 为什么数据可以并行
# ═══════════════════════════════════════════════════════════

def demo_dp_principle():
    """演示: batch 里每个样本的计算是独立的，可以并行。"""
    print("=" * 55)
    print("DP 原理: Y = WX + b，batch 内各列独立")
    print("=" * 55)

    N, L, B = 300, 400, 4  # out_dim, in_dim, batch
    X = torch.randn(L, B)
    W = torch.randn(N, L)
    b = torch.randn(N, 1)

    # 全量计算
    Y_full = W @ X + b

    # 逐列计算 (模拟 DP: 每张卡算一部分)
    Y_parts = []
    for i in range(B):
        y_i = W @ X[:, i:i+1] + b
        Y_parts.append(y_i)
    Y_split = torch.cat(Y_parts, dim=1)

    print(f"  W: [{N}, {L}], X: [{L}, {B}], b: [{N}, 1]")
    print(f"  Y_full = W@X+b  shape: {tuple(Y_full.shape)}")
    print(f"  Y_split(逐列)    shape: {tuple(Y_split.shape)}")
    print(f"  allclose: {torch.allclose(Y_full, Y_split)}")
    print(f"  结论: Y 的每一列只依赖 X 的对应列 → 可以分到不同 GPU 各自算 → DP\n")


# ═══════════════════════════════════════════════════════════
# 第二部分: 梯度同步 —— All-Reduce 的原理
# ═══════════════════════════════════════════════════════════

def demo_allreduce():
    """演示: 多卡各自算梯度 → all-reduce 求平均 → 每卡得到相同梯度。"""
    if not dist.is_initialized():
        print("=" * 55)
        print("DP 梯度同步: All-Reduce (仿真 4 卡)")
        print("=" * 55)

        # 仿真 4 张卡各自算出的梯度
        torch.manual_seed(42)
        grads = [torch.randn(5) for _ in range(4)]
        for i, g in enumerate(grads):
            print(f"  GPU{i} 局部梯度: {g.numpy().round(3)}")

        # All-reduce: 先求和再除以 world_size
        avg_grad = sum(grads) / 4
        print(f"  All-reduce 后每卡得到: {avg_grad.numpy().round(3)}")
        print(f"  结论: 梯度同步后每卡参数更新一致 → 模型权重一致\n")


def run_real_ddp(world_size: int):
    """真 DDP 测试: 2 卡跑同一个 Linear，看梯度是否一致。"""
    if world_size < 2:
        return

    rank = dist.get_rank()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)

    # 各卡创建相同模型
    torch.manual_seed(42)
    model = nn.Linear(64, 32).to(device)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index])

    # 各卡不同数据
    torch.manual_seed(rank * 100)
    x = torch.randn(4, 64, device=device)
    y = torch.randn(4, 32, device=device)

    loss = F.mse_loss(ddp_model(x), y)
    loss.backward()

    # 打印第一行权重的梯度（应该各卡相同）
    grad_sample = model.weight.grad[0, :5]
    gathered = [torch.zeros_like(grad_sample) for _ in range(world_size)]
    dist.all_gather(gathered, grad_sample)

    if rank == 0:
        print(f"  DDP 梯度 all-reduce 验证: {world_size} 卡")
        for i, g in enumerate(gathered):
            print(f"    rank {i}: {g.cpu().numpy().round(6)}")
        same = all(torch.allclose(g, gathered[0]) for g in gathered)
        print(f"    所有 rank 梯度一致: {same}")


# ═══════════════════════════════════════════════════════════
# 第三部分: 梯度累积 —— 用时间换大 batch
# ═══════════════════════════════════════════════════════════

def demo_grad_accum():
    """演示: 4 个 micro-batch 的梯度累积等价于 1 个 big batch。"""
    print(f"\n{'='*55}")
    print(f"梯度累积等价性: micro-batch × 4 = big batch × 1")
    print(f"{'='*55}")

    torch.manual_seed(42)
    model = nn.Linear(8, 4)
    X = torch.randn(16, 8)     # total batch = 16
    Y = torch.randn(16, 4)

    # 方案 A: 一次性算
    m1 = nn.Linear(8, 4)
    m1.load_state_dict(model.state_dict())
    loss_a = F.mse_loss(m1(X), Y)
    loss_a.backward()

    # 方案 B: 拆 4 个 micro-batch，梯度累积
    m2 = nn.Linear(8, 4)
    m2.load_state_dict(model.state_dict())
    for i in range(0, 16, 4):
        loss_b = F.mse_loss(m2(X[i:i+4]), Y[i:i+4]) / 4
        loss_b.backward()  # 梯度自动累加到 .grad

    # 比较
    g1 = m1.weight.grad.flatten()[:8]
    g2 = m2.weight.grad.flatten()[:8]
    print(f"  big batch grad:  {g1.numpy().round(6)}")
    print(f"  4x micro-batch:   {g2.numpy().round(6)}")
    print(f"  allclose: {torch.allclose(g1, g2)}")
    print(f"  结论: 梯度累积 = 等效大 batch，显存只需 1/4\n")


# ═══════════════════════════════════════════════════════════
# 第四部分: 性能对比 —— 单卡 vs DDP 加速比
# ═══════════════════════════════════════════════════════════

def bench_dp_overhead():
    """测 DDP 的 all-reduce 通信开销占比。"""
    print(f"{'='*55}")
    print(f"DDP 开销分析: 通信 vs 计算")
    print(f"{'='*55}")

    N, L, B = 4096, 4096, 256
    repeats = 100

    # 纯计算
    W = torch.randn(N, L, device="cuda")
    X = torch.randn(L, B, device="cuda")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        Y = W @ X
    torch.cuda.synchronize()
    t_compute = (time.perf_counter() - t0) / repeats * 1000

    # 纯通信 (all-reduce 一个等大的 tensor)
    g = torch.randn(N, L, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        torch.distributed.all_reduce(g, op=dist.ReduceOp.AVG) if dist.is_initialized() else None
    torch.cuda.synchronize()
    t_comm = (time.perf_counter() - t0) / repeats * 1000 if dist.is_initialized() else 0

    print(f"  MatMul {N}x{L}: {t_compute:.1f}ms")
    if dist.is_initialized():
        print(f"  All-Reduce {N}x{L}:   {t_comm:.1f}ms")
        ratio = t_comm / (t_compute + t_comm) * 100
        print(f"  通信占比: {ratio:.1f}%  (越小越好)")
        print(f"  结论: 模型越大 → 计算/通信比越高 → DDP 加速越接近 N×")
    else:
        print(f"  (单卡模式，跳过通信测试。用 torchrun 启动可测)")


# ═══════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    is_dist = int(os.environ.get("RANK", -1)) != -1
    world_size = 0

    if is_dist:
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0:
        demo_dp_principle()
        demo_allreduce()
        demo_grad_accum()

    bench_dp_overhead()

    if is_dist:
        run_real_ddp(world_size)
        dist.destroy_process_group()
