"""Tensor Parallel (TP) 原理 demo —— Megatron-LM 风格的列切 + 行切。

核心思想: 权重矩阵太大一张卡放不下 → 按列/行切到多卡 → 计算时通信。

    MLP = X @ A @ B

    列并行 (A 按列切):
        GPU0: X @ A[:, :half]  → 输出列切  → 给 GeLU（不通信！）
        GPU1: X @ A[:, half:]  → 输出列切
        → GeLU 是 element-wise，列切的数据经过 GeLU 仍是列切的

    行并行 (B 按行切):
        GPU0: GeLU_out @ B[:half, :]  → 部分和  → all-reduce → 完整输出
        GPU1: GeLU_out @ B[half:, :]  → 部分和  →

    整个 MLP: 只做一次 all-reduce！中间 GeLU 零通信。

用法:
    python tests/unit/test_tp.py                       # 单进程仿真
    torchrun --nproc-per-node=2 tests/unit/test_tp.py    # 真 2 卡 TP
"""

import os, sys, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ═══════════════════════════════════════════════════════════
# 第一部分: 列并行 Linear (ColumnParallel)
# ═══════════════════════════════════════════════════════════

class ColumnParallelLinear(nn.Module):
    """列并行 Linear: 权重按输出维度均分到多卡。

    W_full [out, in] → GPU0: W[:out/2, :], GPU1: W[out/2:, :]

    Forward:
        每卡用自己那份 W 做 matmul → 输出天然是列切分的
        如果 gather_output=True: all-gather 拼回完整输出
        如果 gather_output=False: 保持列切（给下游 GeLU 用，不通信）
    """

    def __init__(self, in_features, out_features, gather_output=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        world_size = dist.get_world_size() if dist.is_initialized() else 2
        assert out_features % world_size == 0

        self.out_per_gpu = out_features // world_size
        self.weight = nn.Parameter(torch.randn(self.out_per_gpu, in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_per_gpu))

    def forward(self, x):
        # x: [B, in]  (不切分，每卡拿完整输入)
        y_local = F.linear(x, self.weight, self.bias)  # [B, out_per_gpu]

        if self.gather_output and dist.is_initialized():
            # all-gather: 把各卡的分片拼成完整输出
            gathered = [torch.zeros_like(y_local) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, y_local)
            y = torch.cat(gathered, dim=-1)
        else:
            y = y_local  # 保持列切，给下游
        return y


# ═══════════════════════════════════════════════════════════
# 第二部分: 行并行 Linear (RowParallel)
# ═══════════════════════════════════════════════════════════

class RowParallelLinear(nn.Module):
    """行并行 Linear: 权重按输入维度均分到多卡。

    W_full [out, in] → GPU0: W[:, :in/2], GPU1: W[:, in/2:]

    Forward:
        输入已是列切分的（来自上游 ColumnParallel 或 GeLU）
        每卡用自己那份 W 做 matmul → 得到"部分和"
        all-reduce 把这些部分和加起来 → 完整输出
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        world_size = dist.get_world_size() if dist.is_initialized() else 2
        assert in_features % world_size == 0

        self.in_per_gpu = in_features // world_size
        self.weight = nn.Parameter(torch.randn(out_features, self.in_per_gpu))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # x: [B, in_per_gpu] (已是列切分)
        y_local = F.linear(x, self.weight, self.bias)  # [B, out]

        if dist.is_initialized():
            # all-reduce: 把各卡的部分和加起来 = 完整结果
            dist.all_reduce(y_local, op=dist.ReduceOp.SUM)
        # 注意: 单进程仿真时 all-reduce 不执行，需要手动模拟
        return y_local


# ═══════════════════════════════════════════════════════════
# 第三部分: TP MLP —— 列切 + GeLU + 行切
# ═══════════════════════════════════════════════════════════

class TP_MLP(nn.Module):
    """Megatron 风格 TP MLP: ColumnParallel → GeLU → RowParallel。

    整个过程中只在最后做一次 all-reduce！中间的 GeLU 是 element-wise 的，
    对列切分的数据天然兼容，不需要任何通信。
    """

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.c_fc = ColumnParallelLinear(in_dim, hidden_dim, gather_output=False)
        self.c_proj = RowParallelLinear(hidden_dim, out_dim)

    def forward(self, x):
        # x: [B, in_dim]   每卡完整输入
        h = F.gelu(self.c_fc(x))    # [B, hidden/gpu]  列切，不通信！
        y = self.c_proj(h)           # [B, out_dim]      all-reduce 后完整
        return y


# ═══════════════════════════════════════════════════════════
# 第四部分: 仿真验证 (单进程，模拟 2 卡)
# ═══════════════════════════════════════════════════════════

def simulate_tp():
    """在单进程内模拟 TP 的数据流动，验证数值正确性。"""
    print("=" * 55)
    print("TP 仿真: ColumnParallel → GeLU → RowParallel (2 GPU)")
    print("=" * 55)

    B, in_dim, hidden, out_dim = 4, 128, 512, 128
    x = torch.randn(B, in_dim)

    # ── 原始 MLP (单卡) ──
    torch.manual_seed(42)
    fc1_full = nn.Linear(in_dim, hidden)
    fc2_full = nn.Linear(hidden, out_dim)
    y_full = fc2_full(F.gelu(fc1_full(x)))

    # ── TP MLP (模拟 2 卡) ──
    # GPU0: 权重的左半
    W1_0 = fc1_full.weight[:hidden//2, :]      # [256, 128]
    b1_0 = fc1_full.bias[:hidden//2]            # [256]
    W2_0 = fc2_full.weight[:, :hidden//2]       # [128, 256]
    b2_0 = fc2_full.bias                        # [128]  注意: bias 不切！

    # GPU1: 权重的右半
    W1_1 = fc1_full.weight[hidden//2:, :]       # [256, 128]
    b1_1 = fc1_full.bias[hidden//2:]            # [256]
    W2_1 = fc2_full.weight[:, hidden//2:]       # [128, 256]
    b2_1 = torch.zeros_like(b2_0)               # RowParallel bias 只在一张卡上

    # 模拟前向
    # Step 1: ColumnParallel (输入不切)
    h0 = F.linear(x, W1_0, b1_0)                # [B, 256]
    h1 = F.linear(x, W1_1, b1_1)                # [B, 256]

    # Step 2: GeLU (element-wise，对切分透明)
    g0 = F.gelu(h0)                              # GPU0
    g1 = F.gelu(h1)                              # GPU1  零通信！

    # Step 3: RowParallel (all-reduce)
    y0 = F.linear(g0, W2_0, b2_0)               # [B, 128]  部分和
    y1 = F.linear(g1, W2_1, b2_1)               # [B, 128]  部分和
    y_tp = y0 + y1                               # all-reduce → 补上 GPU1 的 bias

    print(f"  Y_full  shape: {tuple(y_full.shape)}")
    print(f"  Y_tp    shape: {tuple(y_tp.shape)}")
    print(f"  allclose: {torch.allclose(y_full, y_tp, atol=1e-4)}")

    # 通信量分析
    hidden_bytes = hidden * 2  # fp16
    out_bytes = out_dim * 2
    comm_per_forward = out_bytes  # RowParallel 做一次 all-reduce 输出
    no_tp_comm = hidden_bytes + out_bytes  # 如果不用 TP，MLP 输入输出都要传

    print(f"\n  通信量分析 (此 MLP, fp16):")
    print(f"    TP:        {comm_per_forward}B / forward  (只在 RowParallel 端通信)")
    print(f"    不用 TP:   {no_tp_comm}B / forward  (输入输出都要传)")
    print(f"    节省:      {(1 - comm_per_forward/no_tp_comm)*100:.0f}%")
    print(f"    关键: GeLU 零通信 —— element-wise 操作对列切透明\n")


# ═══════════════════════════════════════════════════════════
# 第五部分: Attention 的 TP
# ═══════════════════════════════════════════════════════════

def simulate_attention_tp():
    """演示 Attention 如何做 TP: QKV 列切 + Output 行切。

    标准 Attention: O = softmax(QK^T/sqrt(d)) @ V
                   Q = X@Wq, K = X@Wk, V = X@Wv, O = attn@Wo

    TP 方案: Wq, Wk, Wv 按 head 数量切
             Wo 按行切
             通信量: 只在最后 all-reduce 一次
    """
    print("=" * 55)
    print("TP Attention: QKV 列切 + Output 行切")
    print("=" * 55)

    B, T, C = 2, 128, 256    # batch, seq, hidden
    n_heads, hd = 8, C // 8  # 8 heads, dim 32/head
    n_gpus = 2

    # 随机输入 + 权重
    torch.manual_seed(42)
    x = torch.randn(B, T, C)
    Wq = torch.randn(C, C)
    Wk = torch.randn(C, C)
    Wv = torch.randn(C, C)
    Wo = torch.randn(C, C)

    # ── 全量 Attention ──
    Q_full = x @ Wq
    K_full = x @ Wk
    V_full = x @ Wv

    Q_heads = Q_full.view(B, T, n_heads, hd).transpose(1, 2)
    K_heads = K_full.view(B, T, n_heads, hd).transpose(1, 2)
    V_heads = V_full.view(B, T, n_heads, hd).transpose(1, 2)

    scale = 1.0 / math.sqrt(hd)
    attn = F.softmax(Q_heads @ K_heads.transpose(-2, -1) * scale, dim=-1)
    attn_out = (attn @ V_heads).transpose(1, 2).contiguous().view(B, T, C)
    O_full = attn_out @ Wo

    # ── TP Attention (2 GPU, 每 GPU 4 heads) ──
    heads_per_gpu = n_heads // n_gpus
    h_start = 0 * heads_per_gpu * hd   # GPU0: head 0-3
    h_end = (0 + 1) * heads_per_gpu * hd

    # 每 GPU 只切 QKV 的 head 维度对应的 slice
    Wq_0 = torch.cat([Wq[:, h_start:h_end],
                       Wq[:, n_heads*hd + h_start:n_heads*hd + h_end],
                       Wq[:, 2*n_heads*hd + h_start:2*n_heads*hd + h_end]], dim=-1)

    # 简化演示: 直接按 head 维度切 Q, K, V
    # GPU0: head [0:4]，GPU1: head [4:8]
    Q0 = Q_heads[:, :4, :, :]   # GPU0, 4 heads
    K0 = K_heads[:, :4, :, :]
    V0 = V_heads[:, :4, :, :]

    Q1 = Q_heads[:, 4:, :, :]   # GPU1, 4 heads
    K1 = K_heads[:, 4:, :, :]
    V1 = V_heads[:, 4:, :, :]

    # 各自算 attention
    attn0 = F.softmax(Q0 @ K0.transpose(-2, -1) * scale, dim=-1)
    out0 = attn0 @ V0                        # [B, 4, T, hd]
    out0_flat = out0.transpose(1, 2).contiguous().view(B, T, heads_per_gpu * hd)

    attn1 = F.softmax(Q1 @ K1.transpose(-2, -1) * scale, dim=-1)
    out1 = attn1 @ V1                        # [B, 4, T, hd]
    out1_flat = out1.transpose(1, 2).contiguous().view(B, T, heads_per_gpu * hd)

    # RowParallel: 各自乘自己那部分 Wo，再 all-reduce
    Wo_0 = Wo[:, :heads_per_gpu * hd]        # [C, 128]
    Wo_1 = Wo[:, heads_per_gpu * hd:]        # [C, 128]

    # F.linear(x, W) = x @ W.T，x 需 reshape 到 [B*T, partial_C]
    O0_partial = F.linear(out0_flat.reshape(-1, heads_per_gpu * hd), Wo_0)
    O1_partial = F.linear(out1_flat.reshape(-1, heads_per_gpu * hd), Wo_1)
    O_tp = (O0_partial + O1_partial).view(B, T, C)  # all-reduce

    print(f"  O_full shape: {tuple(O_full.shape)}")
    print(f"  O_tp   shape: {tuple(O_tp.shape)}")
    print(f"  allclose: {torch.allclose(O_full, O_tp, atol=1e-3)}")
    print(f"  通信: 整个 Attention 只在最后 RowParallel 做一次 all-reduce!")


# ═══════════════════════════════════════════════════════════
# 第六部分: 真 TP (2 卡 torchrun)
# ═══════════════════════════════════════════════════════════

def run_real_tp():
    """2 卡跑真 TP MLP，验证数值正确性。"""
    if not dist.is_initialized():
        return

    rank = dist.get_rank()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)

    B, in_dim, hidden, out_dim = 4, 128, 512, 128
    x = torch.randn(B, in_dim, device=device)

    # 原始
    torch.manual_seed(42)
    fc1_full = nn.Linear(in_dim, hidden).to(device)
    fc2_full = nn.Linear(hidden, out_dim).to(device)
    y_full = fc2_full(F.gelu(fc1_full(x)))

    # TP
    torch.manual_seed(42)
    tp = TP_MLP(in_dim, hidden, out_dim).to(device)
    y_tp = tp(x)

    if rank == 0:
        print(f"\n{'='*55}")
        print(f"真 TP (2 GPU) 验证:")
        print(f"  Y_full: {y_full[0, :5].cpu().numpy().round(4)}")
        print(f"  Y_tp:   {y_tp[0, :5].cpu().numpy().round(4)}")
        print(f"  allclose: {torch.allclose(y_full, y_tp, atol=1e-3)}")


# ═══════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    is_dist = int(os.environ.get("RANK", -1)) != -1

    if is_dist:
        dist.init_process_group(backend="nccl")

    if not is_dist or dist.get_rank() == 0:
        simulate_tp()
        simulate_attention_tp()

    run_real_tp()

    if is_dist:
        dist.destroy_process_group()
