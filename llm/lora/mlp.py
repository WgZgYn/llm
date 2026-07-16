"""LoRAMLP —— 使用 LoRA 的前馈网络。

继承 MLP，在 FFN 的两个 Linear 层上使用 LoRALinear。
"""

import torch
import torch.nn as nn

from ..model.layer import MLP
from .linear import LoRALinear
from .config import LoRAConfig


class LoRAMLP(MLP):
    """使用 LoRA 的前馈网络。

    根据 lora_config.target_modules 决定在哪些 Linear 上应用 LoRA:
        - 'mlp' / 'all' → c_fc (n_embd→4*n_embd) + c_proj (4*n_embd→n_embd)

    MLP 结构（父类）:
        net = Sequential(
            nn.Linear(n_embd, hidden),  # [0] — c_fc
            GELU(),                     # [1]
            nn.Linear(hidden, n_embd),  # [2] — c_proj
            Dropout/Identity,           # [3]
        )
    """

    def __init__(
        self,
        n_embd: int,
        bias: bool = True,
        dropout: float = 0.0,
        expand: int = 4,
        gelu: nn.Module | None = None,
        lora_config: LoRAConfig | None = None,
    ):
        # ── 构造父类 MLP ──
        super().__init__(n_embd, bias=bias, dropout=dropout, expand=expand, gelu=gelu)

        if lora_config is None:
            return

        lc = lora_config
        targets = lc.target_modules
        if isinstance(targets, str):
            targets = {targets}
        else:
            targets = set(targets)

        hidden = expand * n_embd

        # ── 替换 net[0]（c_fc: n_embd → hidden）──
        if self._should_apply(targets):
            old_fc = self.net[0]
            new_fc = LoRALinear(
                n_embd, hidden,
                r=lc.r, alpha=lc.alpha, dropout=lc.dropout,
                bias=bias,
            )
            new_fc.init_from_pretrained(old_fc.weight.data)
            if old_fc.bias is not None:
                with torch.no_grad():
                    new_fc.bias.data.copy_(old_fc.bias.data)
            self.net[0] = new_fc

        # ── 替换 net[2]（c_proj: hidden → n_embd）──
        if self._should_apply(targets):
            old_proj = self.net[2]
            new_proj = LoRALinear(
                hidden, n_embd,
                r=lc.r, alpha=lc.alpha, dropout=lc.dropout,
                bias=bias,
            )
            new_proj.init_from_pretrained(old_proj.weight.data)
            if old_proj.bias is not None:
                with torch.no_grad():
                    new_proj.bias.data.copy_(old_proj.bias.data)
            self.net[2] = new_proj

    @staticmethod
    def _should_apply(targets: set) -> bool:
        """判断 MLP 层是否需要应用 LoRA。"""
        return bool({'mlp', 'all'} & targets)
