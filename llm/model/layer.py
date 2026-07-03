"""GPT 模型的基础层组件。

包含:
    - LayerNorm: nn.LayerNorm 别名（PyTorch ≥ 2.12 支持 bias 参数）
    - CausalSelfAttention: 融合 QKV 投影 + Flash Attention 的因果自注意力
    - MLP: GELU 激活 + 4x 扩展的前馈网络
    - Block: Pre-LN 风格的完整 Transformer block

所有组件都从 GPTConfig 读取超参数，保持一致的接口。
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..config.model_config import GPTConfig


# ═══════════════════════════════════════════════════════════════════════════════
# LayerNorm
# ═══════════════════════════════════════════════════════════════════════════════

# PyTorch ≥ 2.12 原生支持 bias 参数，无需自定义实现
LayerNorm = nn.LayerNorm


# ═══════════════════════════════════════════════════════════════════════════════
# CausalSelfAttention
# ═══════════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """因果自注意力机制（Pre-LN 风格）。

    设计要点（相比 MiniGPT 的改进）:
    1. 融合 QKV 投影: 一个 Linear(n_embd, 3*n_embd) 代替三个独立 Linear，
       减少 kernel launch 次数，约 30% 加速。
    2. Flash Attention: 优先使用 F.scaled_dot_product_attention
      （PyTorch 2.0+），自动调度到最优 CUDA kernel。
    3. 独立 dropout: attn_dropout 用于注意力权重，resid_dropout 用于输出残差路径。
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # ── 融合 QKV 投影：一次矩阵乘法 → split 为 Q, K, V ──
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # ── 输出投影 ──
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # ── Dropout ──
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # ── Flash Attention 检测 ──
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # 回退到手动实现时需要 causal mask buffer
            bias = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer("bias", bias.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数:
            x: [batch_size, seq_len, n_embd]

        返回:
            [batch_size, seq_len, n_embd]
        """
        B, T, C = x.size()

        # ── 融合 QKV 投影 + split ──
        # 一次 Big MM → split 为 Q, K, V（比三个独立 Linear 快）
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # ── 重塑为多头格式: (B, n_head, T, head_dim) ──
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # ── 计算注意力 ──
        if self.flash:
            # Flash Attention: 硬件加速，O(n) 显存
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # 手动实现（回退方案，PyTorch < 2.0 时使用）
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # ── 合并多头: (B, n_head, T, head_dim) → (B, T, n_embd) ──
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # ── 输出投影 + resid dropout ──
        y = self.resid_dropout(self.c_proj(y))
        return y


# ═══════════════════════════════════════════════════════════════════════════════
# MLP
# ═══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """Position-wise Feed-Forward Network。

    标准结构: Linear(n_embd → 4*n_embd) → GELU → Linear(4*n_embd → n_embd) → Dropout

    FFN 占了 Transformer 约 2/3 的参数量，是大模型参数存储的主体。
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Block
# ═══════════════════════════════════════════════════════════════════════════════

class Block(nn.Module):
    """Pre-LN Transformer Block。

    数据流（Pre-LN 风格，GPT-2/3、LLaMA 的标准做法）:
        x → LayerNorm → CausalSelfAttention → Dropout → + x（残差连接）
        x → LayerNorm → MLP → Dropout → + x（残差连接）

    相比 Post-LN（原始 Transformer）:
    - 梯度可通过残差路径无衰减地回传
    - 深层训练更稳定，无需 learning rate warmup 即可收敛
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数:
            x: [batch_size, seq_len, n_embd]

        返回:
            [batch_size, seq_len, n_embd]
        """
        # 子层 1: Self-Attention（Pre-LN + 残差）
        x = x + self.attn(self.ln_1(x))
        # 子层 2: MLP（Pre-LN + 残差）
        x = x + self.mlp(self.ln_2(x))
        return x
