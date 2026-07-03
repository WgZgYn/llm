"""Shakespeare 字符级快速验证配置。

用途: 验证训练流程完整性（train → eval → checkpoint → generate）
训练时间: ~30 秒 (CUDA), ~2 分钟 (CPU)
模型规模: ~1.5M 参数

用法:
    python scripts/train.py configs/shakespeare_char.py
"""

from llm.config import GPTConfig, TrainingConfig

# ═══════════════════════════════════════════════════════════════════════════
# 模型架构
# ═══════════════════════════════════════════════════════════════════════════

model = GPTConfig(
    # 词汇与序列
    vocab_size=65,            # shakespeare_char 字符集大小
    block_size=128,           # 上下文长度（数据总量才 1M chars，128 足够）
    # 架构
    n_layer=2,                # 轻量: 2 层
    n_head=4,                 # 4 个注意力头
    n_embd=128,               # 隐藏维度（d_model）
    # 正则化
    dropout=0.0,              # 预训练无需 dropout
    bias=True,                # 与 GPT-2 行为一致
)

# ═══════════════════════════════════════════════════════════════════════════
# 训练参数
# ═══════════════════════════════════════════════════════════════════════════

training = TrainingConfig(
    # 输入/输出
    out_dir="out/shakespeare_char",
    init_from="scratch",
    always_save_checkpoint=True,
    eval_only=False,
    # 数据
    dataset="shakespeare_char",
    data_dir="data/shakespeare_char",
    batch_size=32,
    gradient_accumulation_steps=4,   # 等效 batch = 32 * 4 = 128
    # 优化器
    learning_rate=3e-4,
    weight_decay=1e-1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    # 学习率调度
    decay_lr=True,
    warmup_iters=20,
    lr_decay_iters=200,             # 200 步 cosine 衰减
    min_lr=3e-5,                     # learning_rate / 10
    # 训练控制
    max_iters=200,                   # 总步数
    eval_interval=50,                # 每 50 步评估一次
    eval_iters=20,                   # 评估时采样 20 个 batch
    log_interval=1,                  # 每步打印日志
    # 系统
    backend="single",                # 单卡训练
    device="cuda",
    dtype="bfloat16",
    compile=False,                   # Windows 无 Triton，关闭
    # DDP
    ddp_backend="nccl",
    # 日志
    wandb_log=False,
    wandb_project="llm",
    wandb_run_name="shakespeare_char",
)
