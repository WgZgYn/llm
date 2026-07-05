"""GPT-2 124M FSDP 训练配置。

FSDP 将参数/梯度/优化器状态分片到多 GPU:
- fsdp_reshard=True  → FULL_SHARD  (类 ZeRO-3)：每层前向后释放参数，最省显存
- fsdp_reshard=False → SHARD_GRAD_OP (类 ZeRO-2)：保留参数，仅分片梯度

相比 DDP，FSDP 允许训练更大模型或使用更大 batch size。

用法（4 GPU）:
    torchrun --standalone --nproc_per_node=4 scripts/train.py configs/gpt2_124M_fsdp.py
"""

from llm.config import GPTConfig, TrainingConfig

model = GPTConfig(
    vocab_size=50304,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    bias=True,
)

training = TrainingConfig(
    out_dir="out/gpt2_124M_fsdp",
    init_from="scratch",
    always_save_checkpoint=True,
    eval_only=False,
    dataset="openwebtext",
    data_dir="data/openwebtext",
    batch_size=12,
    gradient_accumulation_steps=40,
    learning_rate=6e-4,
    weight_decay=1e-1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    decay_lr=True,
    warmup_iters=2000,
    lr_decay_iters=600000,
    min_lr=6e-5,
    max_iters=600000,
    eval_interval=2000,
    eval_iters=200,
    log_interval=1,
    backend="fsdp",               # ← FSDP 后端
    device="cuda",
    dtype="float16",
    compile=True,
    ddp_backend="nccl",
    fsdp_reshard=True,            # FULL_SHARD (ZeRO-3)
    wandb_log=False,
    wandb_project="llm",
    wandb_run_name="gpt2-124M-fsdp",
)
