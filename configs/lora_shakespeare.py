"""Shakespeare 字符级 LoRA 快速验证配置。

用途: 验证 LoRA 训练流程完整性（train → eval → checkpoint → generate）
训练时间: ~30 秒 (CUDA)
模型规模: ~0.1M 参数，LoRA ~3K 参数

用法:
    python scripts/train_lora.py configs/lora_shakespeare.py
"""

from llm.config import GPTConfig, TrainingConfig
from llm.lora import LoRAConfig

# ═══════════════════════════════════════════════════════════════════════════
# 模型架构
# ═══════════════════════════════════════════════════════════════════════════

model = GPTConfig(
    vocab_size=65,              # shakespeare_char 字符集大小
    block_size=128,             # 上下文长度
    n_layer=2,                  # 2 层
    n_head=4,                   # 4 个注意力头
    n_embd=128,                 # 隐藏维度
    dropout=0.0,
    bias=True,
    gradient_checkpointing=False,
)

# ═══════════════════════════════════════════════════════════════════════════
# LoRA 配置
# ═══════════════════════════════════════════════════════════════════════════

lora = LoRAConfig(
    r=4,                        # 低秩维度
    alpha=8.0,                  # scale = 2.0
    dropout=0.0,
    target_modules='attn',      # 注意力投影用 LoRA
    target_layers=None,         # 全部层
    variant='lora',
)

# ═══════════════════════════════════════════════════════════════════════════
# 训练参数
# ═══════════════════════════════════════════════════════════════════════════

training = TrainingConfig(
    out_dir="out/lora_shakespeare",
    init_from="scratch",
    always_save_checkpoint=True,
    eval_only=False,

    dataset="shakespeare_char",
    data_dir="data/shakespeare_char",
    batch_size=32,
    gradient_accumulation_steps=4,

    learning_rate=3e-4,
    weight_decay=1e-1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,

    decay_lr=True,
    warmup_iters=20,
    lr_decay_iters=200,
    min_lr=3e-5,

    max_iters=200,
    eval_interval=50,
    eval_iters=20,
    log_interval=1,

    backend="single",
    device="cuda",
    dtype="bfloat16",
    compile=False,

    ddp_backend="nccl",
    wandb_log=False,
)
