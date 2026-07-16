"""LoRA 加法算术任务微调。

任务: 两位数加法 "a+b=c#"
评估: 准确率（答案完全匹配）→ 最直观的硬指标
预期: 从 ~0% 提升到 >80%，效果感受非常明显
时间: ~10 分钟 (RTX 4060, 5000 steps)

用法:
    python scripts/train_lora.py configs/lora_addition.py
"""

from llm.config import GPTConfig, TrainingConfig
from llm.lora import LoRAConfig

# ═══════════════════════════════════════════════════════════════════════════
# 模型架构（小模型，快速训练）
# ═══════════════════════════════════════════════════════════════════════════

model = GPTConfig(
    vocab_size=14,              # 0-9 + = # _ (14 个字符)
    block_size=32,              # max_digits=5 → 最长约 17 字符
    n_layer=4,                  # 4 层
    n_head=4,                   # 4 头
    n_embd=128,                 # 128 维
    dropout=0.0,                # 无 dropout
    bias=True,
    ignore_index=13,            # 忽略 PAD token '_'
    gradient_checkpointing=False,
)

# ═══════════════════════════════════════════════════════════════════════════
# LoRA 配置
# ═══════════════════════════════════════════════════════════════════════════

lora = LoRAConfig(
    r=8,                        # rank 8 → ~18K LoRA 参数
    alpha=16.0,                 # scale = 2.0
    dropout=0.0,
    target_modules='all',       # attn + mlp 全改
    target_layers=None,         # 全部 4 层
    variant='lora',
)

# ═══════════════════════════════════════════════════════════════════════════
# 训练参数
# ═══════════════════════════════════════════════════════════════════════════

training = TrainingConfig(
    out_dir="out/lora_addition",
    init_from="scratch",
    always_save_checkpoint=True,
    eval_only=False,

    # dataset="addition" 触发加法任务模式（使用 CharTokenizer + AdditionDataset）
    dataset="addition",
    data_dir="",                # 加法任务不需要 .bin 文件
    batch_size=128,             # 加法序列短，可以开大 batch
    gradient_accumulation_steps=1,

    learning_rate=1e-3,         # 从零训练 + LoRA，LR 可以高一些
    weight_decay=1e-1,
    beta1=0.9, beta2=0.95,
    grad_clip=1.0,

    decay_lr=True,
    warmup_iters=200,
    lr_decay_iters=5000,
    min_lr=1e-4,

    max_iters=5000,
    eval_interval=500,          # 每 500 步评估准确率
    eval_iters=100,             # 100 batch 平均 loss
    log_interval=50,            # 每 50 步打印

    backend="single",
    device="cuda",
    dtype="bfloat16",
    compile=False,

    ddp_backend="nccl",
    wandb_log=False,
)
