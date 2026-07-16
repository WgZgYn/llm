"""LoRA 配置 —— 纯 dataclass，类型安全。"""

from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) 所有超参数。

    预设参考:
        r=8, alpha=16 → scale=2.0（标准设置，GPT-2 124M 适用）
        r=4, alpha=8  → scale=2.0（较小 rank，快速实验）
        r=16, alpha=32 → scale=2.0（较大 rank，效果更好但参数更多）

    用法:
        lora = LoRAConfig(r=8, alpha=16.0)
        lora = LoRAConfig(r=4, target_modules='all', variant='dora')
    """

    # ── 核心参数 ──
    r: int = 8                          # 低秩维度
    alpha: float = 16.0                 # 缩放系数（实际 scale = alpha / r）

    # ── 正则化 ──
    dropout: float = 0.0                # LoRA dropout

    # ── 目标模块选择 ──
    # 支持的值:
    #   'attn'   → QKV 投影 (c_attn) + 输出投影 (c_proj)
    #   'mlp'    → FFN 第一层 + 第二层
    #   'all'    → attn + mlp
    #   列表形式  → ['c_attn', 'c_proj'] 精确控制
    target_modules: str | list[str] = 'attn'

    # ── 目标层范围 ──
    # None = 全部层
    # (start, end) = 范围（如 (0, 6) → 第 0~5 层）
    # [0, 2, 4] = 指定具体层索引
    target_layers: tuple | list | None = None

    # ── 变种 ──
    variant: str = 'lora'               # 'lora' | 'dora'

    # ── LoRA+ ──
    # B 矩阵学习率 = base_lr × lora_plus_lr_ratio
    # A 矩阵学习率 = base_lr / lora_plus_lr_ratio
    # None = 不使用 LoRA+（A 和 B 相同 LR）
    lora_plus_lr_ratio: float | None = None

    # ── 冻结选项 ──
    # 对位置敏感的算术任务，需要让模型学习新的位置语义
    train_wpe: bool = False             # True = WPE 参与训练（位置嵌入）
    train_ln: bool = False              # True = LayerNorm 参与训练

    # ── 高级 ──
    init_style: str = 'default'         # 'default' (kaiming_A + zeros_B) | 'pissa'
    mlp_expand: int = 4                 # MLP 扩展倍数（与 MLP 默认保持一致）

    @property
    def scale(self) -> float:
        """LoRA 缩放系数 alpha / r。"""
        return self.alpha / self.r

    def to_dict(self) -> dict:
        """序列化为字典。"""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LoRAConfig":
        """从字典反序列化。"""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})
