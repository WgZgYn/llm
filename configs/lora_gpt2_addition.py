"""GPT-2 124M + LoRA 微调 —— 加法算术任务（GPT-2 BPE tokenizer）v2。

改进:
  - max_digits=3（混合 1-3 位），覆盖 1000 个不同数字 token
  - 100K 训练样本 vs 1M+ 可能的组合 → 只覆盖 ~10% → 强制泛化
  - 预期：模型学会 token → 数值 → 加法 → token 的完整映射链

BPE 挑战:
  GPT-2 把 "123" 映射为单个 token id，和 "124" 完全无关。
  模型需要学习每个 token 对应的数值，无法靠子词结构泛化。
  这是 LLM 做算术的根本瓶颈。

用法:
    python scripts/train_lora.py configs/lora_gpt2_addition.py
"""

from llm.config import GPTConfig, TrainingConfig
from llm.lora import LoRAConfig

model = GPTConfig(
    vocab_size=50257, block_size=24,
    n_layer=12, n_head=12, n_embd=768,
    dropout=0.0, bias=True,
    gradient_checkpointing=True,
)

lora = LoRAConfig(
    r=16, alpha=32.0, dropout=0.0,
    target_modules='all', target_layers=None,
    variant='lora',
)

training = TrainingConfig(
    out_dir="out/lora_gpt2_addition_v2",
    init_from="gpt2",
    always_save_checkpoint=True,
    eval_only=False,
    dataset="addition_gpt2",
    data_dir="",
    batch_size=64, gradient_accumulation_steps=2,
    learning_rate=1e-4, weight_decay=1e-2,
    beta1=0.9, beta2=0.95, grad_clip=1.0,
    decay_lr=True, warmup_iters=200,
    lr_decay_iters=8000, min_lr=1e-5,
    max_iters=8000, eval_interval=1000,
    eval_iters=100, log_interval=50,
    backend="single", device="cuda",
    dtype="bfloat16", compile=False,
    ddp_backend="nccl", wandb_log=False,
)
