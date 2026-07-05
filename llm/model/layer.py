"""Transformer 基础层组件 —— 框架无关，只依赖显式参数。

包含:
    - LayerNorm
    - CausalSelfAttention:  融合 QKV + Flash Attention
    - MLP:                  GELU 激活的标准 FFN
    - Block:                Pre-LN Transformer Block（可插拔 ffn）

所有组件接受显式参数（n_embd, n_head 等），不依赖 GPTConfig。
这样可以被 GPT、MoEGPT、Llama 等任意模型复用。
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# LayerNorm
# ═══════════════════════════════════════════════════════════════════════════════

LayerNorm = nn.LayerNorm


# ═══════════════════════════════════════════════════════════════════════════════
# CausalSelfAttention
# ═══════════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """因果自注意力（Pre-LN 风格）。

    参数:
        n_embd:      隐藏维度
        n_head:      注意力头数
        block_size:  最大序列长度（仅 Flash Attn 不可用时用于 causal mask）
        bias:        Linear 是否使用 bias
        dropout:     attention + resid dropout
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int,
                 bias: bool = True, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.dropout = dropout

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            mask = torch.tril(torch.ones(block_size, block_size))
            self.register_buffer("bias", mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


# ═══════════════════════════════════════════════════════════════════════════════
# MLP
# ═══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """标准 FFN: Linear → GELU → Linear → Dropout。

    参数:
        n_embd:   隐藏维度
        bias:     Linear bias
        dropout:  resid dropout
        expand:   扩展倍数（默认 4x）
        gelu:     激活函数（默认 nn.GELU()）
    """

    def __init__(self, n_embd: int, bias: bool = True, dropout: float = 0.0,
                 expand: int = 4, gelu: nn.Module | None = None):
        super().__init__()
        hidden = expand * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden, bias=bias),
            gelu if gelu is not None else nn.GELU(),
            nn.Linear(hidden, n_embd, bias=bias),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Block
# ═══════════════════════════════════════════════════════════════════════════════

class Block(nn.Module):
    """Pre-LN Transformer Block，可插拔 ffn。

    参数:
        n_embd, n_head, block_size, bias, dropout: 标准 transformer 参数
        ffn:  可选的自定义 FFN（如 MoE_FFN），None 则用默认 MLP
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int,
                 bias: bool = True, dropout: float = 0.0,
                 ffn: nn.Module | None = None):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, bias, dropout)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = ffn if ffn is not None else MLP(n_embd, bias, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
