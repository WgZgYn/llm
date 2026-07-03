"""GPT 模型架构配置 —— 纯 dataclass，类型安全，可 JSON 序列化。"""

from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    """GPT 模型架构的所有超参数。

    预设参考:
        gpt2:         n_layer=12, n_head=12, n_embd=768   (124M)
        gpt2-medium:  n_layer=24, n_head=16, n_embd=1024  (350M)
        gpt2-large:   n_layer=36, n_head=20, n_embd=1280  (774M)
        gpt2-xl:      n_layer=48, n_head=25, n_embd=1600  (1558M)
    """

    # ── 词汇与序列 ──
    vocab_size: int = 0           # 必须显式设置
    block_size: int = 0           # 必须显式设置

    # ── 架构维度 ──
    n_layer: int = 0              # 必须显式设置
    n_head: int = 0               # 必须显式设置
    n_embd: int = 0               # 必须显式设置

    # ── 正则化 ──
    dropout: float = 0.0         # 预训练建议 0.0，微调建议 0.1+
    bias: bool = True           # Linears 和 LayerNorms 中是否使用 bias

    def __post_init__(self):
        if self.n_head > 0 and self.n_embd > 0:
            assert self.n_embd % self.n_head == 0, \
                f"n_embd ({self.n_embd}) 必须能被 n_head ({self.n_head}) 整除"

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度 (d_k = n_embd / n_head)。"""
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        """序列化为字典（用于 checkpoint 保存）。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        """从字典反序列化（用于 checkpoint 加载）。"""
        valid_keys = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "GPTConfig":
        """从预设名称创建配置。

        支持: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
        """
        presets = {
            "gpt2":         dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium":  dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large":   dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl":      dict(n_layer=48, n_head=25, n_embd=1600),
        }
        if name not in presets:
            raise ValueError(f"未知预设 '{name}'。可选: {list(presets.keys())}")

        cfg = presets[name]
        cfg.update(overrides)
        return cls(**cfg)
