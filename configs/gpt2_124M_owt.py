"""GPT-2 124M 标准训练配置（OpenWebText）。

用途: 在 OpenWebText 上训练完整 GPT-2 124M，目标 loss ~2.85
需要: 8×A100 40GB，约 5 天
参考: nanoGPT config/train_gpt2.py

用法:
    # 单节点 8 GPU
    torchrun --standalone --nproc_per_node=8 scripts/train.py configs/gpt2_124M_owt.py
"""

from llm.config import GPTConfig, TrainingConfig

# ═══════════════════════════════════════════════════════════════════════════
# 模型架构
# ═══════════════════════════════════════════════════════════════════════════

model = GPTConfig(
    vocab_size=50304,          # GPT-2 vocab 向上取整到 64 的倍数
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,               # 预训练 dropout=0
    bias=True,                 # GPT-2 使用 bias
)

# ═══════════════════════════════════════════════════════════════════════════
# 训练参数
# ═══════════════════════════════════════════════════════════════════════════

training = TrainingConfig(
    # 输入/输出
    out_dir="out/gpt2_124M",
    init_from="scratch",
    always_save_checkpoint=True,
    eval_only=False,
    # 数据
    dataset="openwebtext",
    data_dir="data/openwebtext",
    batch_size=12,              # 每 GPU micro-batch（增大可减少循环次数 → 更快）
    gradient_accumulation_steps=40,  # 40×12×4GPU×1024 ≈ 0.5M tokens/step
    # 优化器 (AdamW)
    learning_rate=6e-4,
    weight_decay=1e-1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    # 学习率调度 (Cosine + Warmup)
    decay_lr=True,
    warmup_iters=2000,
    lr_decay_iters=600000,       # 与 max_iters 相同 → full cosine
    min_lr=6e-5,                  # learning_rate / 10
    # 训练控制
    max_iters=600000,
    eval_interval=2000,
    eval_iters=200,
    log_interval=1,
    # 系统
    backend="ddp",               # 多卡 DDP
    device="cuda",
    dtype="float16",
    compile=True,
    # DDP
    ddp_backend="nccl",
    # 日志
    wandb_log=False,
    wandb_project="llm",
    wandb_run_name="gpt2-124M-owt",
)
