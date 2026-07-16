"""GPT-2 Small (124M) LoRA 微调 —— OpenWebText 领域适配。

目标硬件: RTX 4060 laptop 8GB
显存占用: ~3.5 GB（LoRA r=8, batch=4, grad_accum=8）
训练时间: ~2 小时 (5000 steps @ ~1.4s/step)

用法:
    python scripts/train_lora.py configs/lora_gpt2_owt.py
"""

from llm.config import GPTConfig, TrainingConfig
from llm.lora import LoRAConfig

# ═══════════════════════════════════════════════════════════════════════════
# 模型架构（GPT-2 Small 124M）
# ═══════════════════════════════════════════════════════════════════════════

model = GPTConfig(
    vocab_size=50257,           # GPT-2 实际 vocab（与 HF/tiktoken 一致）
    block_size=512,             # 训练序列长度（RTX 4060 友好）
    n_layer=12,                 # GPT-2 标准 12 层
    n_head=12,                  # 12 头
    n_embd=768,                 # 768 维
    dropout=0.1,                # 微调时加 dropout 防过拟合
    bias=True,                  # GPT-2 使用 bias
    gradient_checkpointing=True,  # 省显存
)

# ═══════════════════════════════════════════════════════════════════════════
# LoRA 配置
# ═══════════════════════════════════════════════════════════════════════════

lora = LoRAConfig(
    r=8,                        # 标准 rank（0.29M LoRA 参数）
    alpha=16.0,                 # scale = 2.0
    dropout=0.1,                # LoRA dropout
    target_modules='attn',      # 仅注意力投影（QKV + output）
    target_layers=None,         # 全部 12 层
    variant='lora',
)

# ═══════════════════════════════════════════════════════════════════════════
# 训练参数（RTX 4060 8GB 优化）
# ═══════════════════════════════════════════════════════════════════════════

training = TrainingConfig(
    out_dir="out/lora_gpt2_owt",
    init_from="gpt2",           # 从 HuggingFace 加载 GPT-2 124M 预训练权重
    always_save_checkpoint=True,
    eval_only=False,

    dataset="openwebtext",
    data_dir="data/openwebtext",
    batch_size=4,               # 每步 4 个样本（RTX 4060 8GB 保守设置）
    gradient_accumulation_steps=8,  # 有效 batch = 4 × 8 = 32

    # 微调 LR（比预训练小，但可以比全量微调高一点）
    learning_rate=2e-4,
    weight_decay=1e-2,          # 微调时稍低的 weight decay
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,

    # 余弦衰减
    decay_lr=True,
    warmup_iters=100,            # 短 warmup
    lr_decay_iters=5000,        # 与 max_iters 对齐
    min_lr=2e-5,                # lr / 10

    max_iters=5000,
    eval_interval=500,          # 每 500 步评估
    eval_iters=50,              # 50 个 batch 平均
    log_interval=10,            # 每 10 步打印日志

    backend="single",           # 单卡
    device="cuda",
    dtype="bfloat16",           # RTX 4060 支持 bf16
    compile=False,              # 关闭 compile 避免首次编译延迟

    ddp_backend="nccl",
    wandb_log=False,
    wandb_project="llm-lora",
    wandb_run_name="gpt2-lora-owt",
)
