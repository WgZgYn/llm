"""LoRAAttention —— 使用 LoRA 的因果自注意力。

继承 CausalSelfAttention，在 QKV 投影和/或输出投影上使用 LoRALinear。
"""

import torch
import torch.nn as nn

from ..model.layer import CausalSelfAttention
from .linear import LoRALinear
from .config import LoRAConfig


class LoRAAttention(CausalSelfAttention):
    """使用 LoRA 的因果自注意力。

    根据 lora_config.target_modules 决定哪些投影使用 LoRA:
        - 'c_attn' / 'attn' / 'all' → QKV 融合投影用 LoRALinear
        - 'c_proj' / 'attn' / 'all' → 输出投影用 LoRALinear

    forward 行为与父类完全一致，只是投影层被替换为 LoRALinear。
    KV-Cache、Flash Attention 等功能全部继承。
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        bias: bool = True,
        dropout: float = 0.0,
        lora_config: LoRAConfig | None = None,
    ):
        # ── 先构造父类（获取标准 attn 结构）──
        super().__init__(n_embd, n_head, block_size, bias, dropout)

        if lora_config is None:
            return  # 没有 LoRA 配置，退化为普通 CausalSelfAttention

        lc = lora_config
        targets = lc.target_modules
        if isinstance(targets, str):
            targets = {targets}  # 转为 set 便于查找
        else:
            targets = set(targets)

        # ── 替换 c_attn（融合 QKV 投影: n_embd → 3*n_embd）──
        if self._should_apply('c_attn', targets):
            old = self.c_attn
            self.c_attn = LoRALinear(
                n_embd, 3 * n_embd,
                r=lc.r, alpha=lc.alpha, dropout=lc.dropout,
                bias=bias,
            )
            self.c_attn.init_from_pretrained(old.weight.data)
            if old.bias is not None:
                with torch.no_grad():
                    self.c_attn.bias.data.copy_(old.bias.data)

        # ── 替换 c_proj（输出投影: n_embd → n_embd）──
        if self._should_apply('c_proj', targets):
            old = self.c_proj
            self.c_proj = LoRALinear(
                n_embd, n_embd,
                r=lc.r, alpha=lc.alpha, dropout=lc.dropout,
                bias=bias,
            )
            self.c_proj.init_from_pretrained(old.weight.data)
            if old.bias is not None:
                with torch.no_grad():
                    self.c_proj.bias.data.copy_(old.bias.data)

    @staticmethod
    def _should_apply(module_name: str, targets: set) -> bool:
        """判断该模块是否需要应用 LoRA。"""
        if 'all' in targets:
            return True
        return module_name in targets or 'attn' in targets
