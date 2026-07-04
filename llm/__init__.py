"""llm — 模块化 GPT 训练框架。

快速开始:
    from llm import GPT, GPTConfig, Trainer, TrainingConfig, create_backend

    model_config = GPTConfig.from_preset("gpt2")
    model = GPT(model_config)
    train_config = TrainingConfig(max_iters=100)
    backend = create_backend(train_config)
    trainer = Trainer(model, train_config, backend)
    trainer.train()
"""

from .config import GPTConfig, TrainingConfig, validate_configs
from .model import GPT, LayerNorm, CausalSelfAttention, MLP, Block
from .training import (
    Trainer,
    TrainingBackend,
    SingleDeviceBackend,
    DDPBackend,
    FSDPBackend,
    DeepSpeedBackend,
    create_backend,
    WarmupCosineSchedule,
    ConstantSchedule,
    GracefulStopper,
    TrainingLogger,
)

__all__ = [
    # 模型
    "GPT", "GPTConfig",
    "LayerNorm", "CausalSelfAttention", "MLP", "Block",
    # 训练
    "TrainingConfig",
    "Trainer",
    "TrainingBackend",
    "SingleDeviceBackend",
    "DDPBackend",
    "FSDPBackend",
    "DeepSpeedBackend",
    "create_backend",
    "WarmupCosineSchedule",
    "ConstantSchedule",
    "GracefulStopper",
    "TrainingLogger",
]
