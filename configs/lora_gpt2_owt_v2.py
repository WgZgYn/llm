"""GPT-2 Small (124M) LoRA 微调 v2 —— 优化配置。

相比 v1 的改进:
1. LR: 2e-4 → 5e-5     （减少震荡）
2. rank: 8 → 16        （更多容量）
3. target: attn → all   （也改 MLP，知识存储关键层）
4. dropout: 0.1 → 0.0   （去掉噪声）
5. max_iters: 5000 → 2000（更短验证，也更匹配小任务）

用法:
    python scripts/train_lora.py configs/lora_gpt2_owt_v2.py
"""

from llm.config import GPTConfig, TrainingConfig
from llm.lora import LoRAConfig

model = GPTConfig(
    vocab_size=50257,
    block_size=512,
    n_layer=12, n_head=12, n_embd=768,
    dropout=0.0,                # ← 去掉 dropout，减少噪声
    bias=True,
    gradient_checkpointing=True,
)

lora = LoRAConfig(
    r=16,                       # ← rank 翻倍：8 → 16
    alpha=32.0,                 # scale = 2.0 保持不变
    dropout=0.0,                # ← 去掉 LoRA dropout
    target_modules='all',       # ← attn + mlp 全改
    target_layers=None,
    variant='lora',
)

training = TrainingConfig(
    out_dir="out/lora_gpt2_owt_v2",
    init_from="gpt2",
    always_save_checkpoint=True,

    dataset="openwebtext",
    data_dir="data/openwebtext",
    batch_size=4,
    gradient_accumulation_steps=8,   # effective batch = 32

    learning_rate=5e-5,             # ← LR 降 4 倍，稳定训练
    weight_decay=1e-2,
    beta1=0.9, beta2=0.95,
    grad_clip=1.0,

    decay_lr=True,
    warmup_iters=200,               # ← 更长 warmup
    lr_decay_iters=2000,            # 与 max_iters 对齐
    min_lr=5e-6,                    # lr / 10

    max_iters=2000,
    eval_interval=200,
    eval_iters=50,
    log_interval=10,

    backend="single", device="cuda", dtype="bfloat16",
    compile=False,
    ddp_backend="nccl",
    wandb_log=False,
)
