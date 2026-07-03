"""训练配置 —— 纯 dataclass，所有训练超参数集中管理。"""

from dataclasses import dataclass, asdict


@dataclass
class TrainingConfig:
    """所有训练相关的超参数。

    用法:
        config = TrainingConfig(max_iters=100, dataset="shakespeare_char")
        config = TrainingConfig.from_dict(saved_dict)
    """

    # ═════════════════════════════════════════════════════════════
    # 输入/输出
    # ═════════════════════════════════════════════════════════════
    out_dir: str = "out"
    init_from: str = "scratch"       # 'scratch' | 'resume' | 'gpt2' | 'gpt2-medium' | ...
    always_save_checkpoint: bool = True
    eval_only: bool = False

    # ═════════════════════════════════════════════════════════════
    # 数据
    # ═════════════════════════════════════════════════════════════
    dataset: str = ""                  # 必须显式设置（数据目录名）
    data_dir: str = ""                 # 数据文件路径（为空时自动从 dataset 推导）
    batch_size: int = 0                # 必须显式设置
    gradient_accumulation_steps: int = 0  # 必须显式设置

    # ═════════════════════════════════════════════════════════════
    # 优化器 (AdamW)
    # ═════════════════════════════════════════════════════════════
    learning_rate: float = 0.0         # 必须显式设置
    weight_decay: float = 1e-1        # L2 正则（仅用于 2D 参数）
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0            # 梯度裁剪阈值（0.0 = 禁用）

    # ═════════════════════════════════════════════════════════════
    # 学习率调度
    # ═════════════════════════════════════════════════════════════
    decay_lr: bool = True
    warmup_iters: int = 0              # 必须显式设置
    lr_decay_iters: int = 0            # 必须显式设置
    min_lr: float = 0.0                # 必须显式设置

    # ═════════════════════════════════════════════════════════════
    # 训练控制
    # ═════════════════════════════════════════════════════════════
    max_iters: int = 0                 # 必须显式设置
    eval_interval: int = 0             # 必须显式设置
    eval_iters: int = 0                # 必须显式设置
    log_interval: int = 1             # 每隔多少步打印日志

    # ═════════════════════════════════════════════════════════════
    # 系统
    # ═════════════════════════════════════════════════════════════
    backend: str = "single"           # 'single' | 'ddp' | 'deepspeed'
    device: str = "cuda"              # 'cpu' | 'cuda' | 'mps'
    dtype: str = "bfloat16"           # 'float32' | 'bfloat16' | 'float16'
    compile: bool = True              # 是否使用 torch.compile

    # DDP 专用
    ddp_backend: str = "nccl"         # 'nccl' | 'gloo'

    # FSDP 专用
    fsdp_reshard: bool = True         # True=FULL_SHARD(ZeRO-3) | False=SHARD_GRAD_OP(ZeRO-2)

    # ═════════════════════════════════════════════════════════════
    # 日志
    # ═════════════════════════════════════════════════════════════
    wandb_log: bool = False
    wandb_project: str = "llm"
    wandb_run_name: str = ""

    @property
    def tokens_per_iter(self, block_size: int) -> int:
        """每次优化器 step 处理的总 token 数。"""
        return (
            self.gradient_accumulation_steps
            * self.batch_size
            * block_size
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        valid_keys = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


