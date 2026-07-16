"""LoRABlock —— 使用 LoRA 的 Transformer Block。

继承 Block，根据 lora_config 决定 attn 和/或 mlp 是否使用 LoRA 版本。
"""

import torch.nn as nn

from ..model.layer import Block
from .attention import LoRAAttention
from .mlp import LoRAMLP
from .config import LoRAConfig


class LoRABlock(Block):
    """使用 LoRA 的 Transformer Block。

    继承 Block 的全部功能（Pre-LN、KV-Cache、梯度检查点）。
    根据 lora_config 选择性地将 attention 和/或 MLP 替换为 LoRA 版本。

    参数:
        n_embd, n_head, block_size, bias, dropout: 标准 Block 参数
        lora_config: LoRA 配置（None = 退化为普通 Block）
        checkpoint: 是否启用梯度检查点
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        bias: bool = True,
        dropout: float = 0.0,
        lora_config: LoRAConfig | None = None,
        checkpoint: bool = False,
    ):
        # ── 先构造父类 Block（标准 attn + mlp）──
        # ffn=None → Block 使用默认 MLP
        super().__init__(
            n_embd, n_head, block_size, bias, dropout,
            ffn=None, checkpoint=checkpoint,
        )

        if lora_config is None:
            return

        lc = lora_config
        targets = lc.target_modules
        if isinstance(targets, str):
            targets = {targets}
        else:
            targets = set(targets)

        # ── 替换 attention（如 target 包含 attn 相关模块）──
        if self._should_apply_attn(targets):
            self.attn = LoRAAttention(
                n_embd, n_head, block_size,
                bias=bias, dropout=dropout,
                lora_config=lc,
            )

        # ── 替换 MLP（如 target 包含 mlp）──
        if self._should_apply_mlp(targets):
            self.mlp = LoRAMLP(
                n_embd,
                bias=bias, dropout=dropout,
                expand=lc.mlp_expand,
                lora_config=lc,
            )

    @staticmethod
    def _should_apply_attn(targets: set) -> bool:
        """判断 attention 是否需要 LoRA。"""
        return bool({'c_attn', 'c_proj', 'attn', 'all'} & targets)

    @staticmethod
    def _should_apply_mlp(targets: set) -> bool:
        """判断 MLP 是否需要 LoRA。"""
        return bool({'mlp', 'all'} & targets)
