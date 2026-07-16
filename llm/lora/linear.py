"""LoRALinear —— 低秩适配器 Linear 层。

核心公式:
    训练时: y = W₀x + (alpha/r) · B · A · x
    融合时: y = (W₀ + (alpha/r) · B · A) · x  （等价于普通 nn.Linear）

设计要点:
    1. base weight (W₀) 以 buffer 存储，冻结不参与梯度
    2. lora_A 用 kaiming uniform 初始化，lora_B 用 zeros 初始化
       → 训练开始时 LoRA 增量为 0，不影响原始输出
    3. merge() / unmerge() 实现训练↔推理零开销切换
    4. 完全兼容现有训练流程（AMP、梯度累积、DDP/FSDP）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """低秩适配器 Linear 层。

    包装一个冻结的 Linear 权重 + 两个可训练的低秩矩阵 A 和 B。

    参数:
        in_features:  输入维度
        out_features: 输出维度
        r:            低秩维度（默认 8）
        alpha:        缩放系数（默认 16.0），实际 scale = alpha / r
        dropout:      LoRA dropout 概率（默认 0.0）
        bias:         是否使用 bias（默认 True，与 GPT-2 一致）
        dtype:        compute dtype（默认 None = 跟随模型）
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = alpha / r
        self.merged = False  # 当前是否已融合

        # ── Base weight（冻结，以 buffer 存储，不参与 optimizer）──
        weight = torch.empty(out_features, in_features, dtype=dtype)
        self.register_buffer('weight', weight)

        # ── Base bias（冻结）──
        if bias:
            bias_tensor = torch.zeros(out_features, dtype=dtype)
            self.register_buffer('bias', bias_tensor)
        else:
            self.register_buffer('bias', None)

        # ── LoRA A: in_features → r（降维）──
        # kaiming uniform 初始化（与 nn.Linear 默认一致）
        self.lora_A = nn.Linear(in_features, r, bias=False, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        # ── LoRA B: r → out_features（升维）──
        # zeros 初始化 → 训练开始时不改变原始输出
        self.lora_B = nn.Linear(r, out_features, bias=False, dtype=dtype)
        nn.init.zeros_(self.lora_B.weight)

        # ── Dropout ──
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ── 保存原始 weight，用于 merge/unmerge ──
        self.register_buffer('_lora_stored_weight', None)

    # ═══════════════════════════════════════════════════════════════
    # 预训练权重加载
    # ═══════════════════════════════════════════════════════════════

    def init_from_pretrained(self, pretrained_weight: torch.Tensor):
        """从预训练权重初始化 base weight（原地复制）。

        用法:
            lora_linear = LoRALinear(768, 768, r=8)
            lora_linear.init_from_pretrained(original_linear.weight.data)
        """
        with torch.no_grad():
            self.weight.data.copy_(pretrained_weight.to(self.weight.dtype))

    # ═══════════════════════════════════════════════════════════════
    # 前向传播
    # ═══════════════════════════════════════════════════════════════

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        融合模式: y = W₀x + b        （等价于 nn.Linear）
        训练模式: y = W₀x + b + (alpha/r) · dropout(B · A(x))
        """
        # ── 融合模式: 纯 Linear forward ──
        if self.merged:
            return F.linear(x, self.weight, self.bias)

        # ── 训练/未融合推理模式 ──
        result = F.linear(x, self.weight, self.bias)

        # LoRA 增量: (alpha/r) * B(A(dropout(x)))
        lora_x = self.lora_dropout(x)
        lora_x = self.lora_A(lora_x)
        lora_x = self.lora_B(lora_x)
        result = result + lora_x * self.scaling

        return result

    # ═══════════════════════════════════════════════════════════════
    # Merge / Unmerge
    # ═══════════════════════════════════════════════════════════════

    def merge(self):
        """将 LoRA 增量融合进 base weight。

        W := W + (alpha/r) · B @ A

        融合后 forward 等价于普通 nn.Linear，推理零开销。
        保存原始 weight 到 _lora_stored_weight 以支持 unmerge。
        """
        if self.merged:
            return

        with torch.no_grad():
            # 保存原始 weight
            self._lora_stored_weight = self.weight.data.clone()

            # 计算 LoRA 增量 delta = (alpha/r) * B @ A
            delta = (self.lora_B.weight.data @ self.lora_A.weight.data) * self.scaling
            self.weight.data.add_(delta.to(self.weight.dtype))

        self.merged = True

    def unmerge(self):
        """从 base weight 中移除 LoRA 增量，恢复可训练状态。

        如果从未 merge 过（_lora_stored_weight 为 None），则不操作。
        """
        if not self.merged:
            return
        if self._lora_stored_weight is None:
            return

        with torch.no_grad():
            self.weight.data.copy_(self._lora_stored_weight)
            self._lora_stored_weight = None

        self.merged = False

    # ═══════════════════════════════════════════════════════════════
    # 属性代理 —— 让 LoRALinear 看起来像 nn.Linear
    # ═══════════════════════════════════════════════════════════════

    @property
    def requires_grad_(self):
        """代理 requires_grad_ 调用到 lora_A 和 lora_B。
        用于外部 freeze/unfreeze 操作。
        """
        # 返回一个可调用对象
        class _RequiresGrad:
            def __init__(self, lora_A, lora_B):
                self.lora_A = lora_A
                self.lora_B = lora_B

            def __call__(self, requires_grad: bool = True):
                self.lora_A.weight.requires_grad = requires_grad
                self.lora_B.weight.requires_grad = requires_grad

        return _RequiresGrad(self.lora_A, self.lora_B)

    # ═══════════════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════════════

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'r={self.r}, scale={self.scaling:.2f}, merged={self.merged}')
