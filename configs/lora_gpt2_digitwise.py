"""GPT-2 124M + LoRA — Digit-wise 加法（真正的算术泛化）。

核心创新: 不用 GPT-2 BPE 对数字的自然切分（"46"→单token），
而是逐位编码：每个 digit 是独立的 GPT-2 token。

  "12+34=46" → [tok('1'),tok('2'),'+',tok('3'),tok('4'),'=',tok('4'),tok('6')]

优势:
  - 模型学 digit-wise 加法 + 进位规则（而非记忆 token 对）
  - 同位数泛化: (a*997+b)%5 确保 test pair 完全不可见
  - 可扩展到更高位数

预期: 训练 2 位 → 测试未见过的 2 位组合 → 高准确率 → 真正学会了加法！

用法:
    python scripts/train_lora.py configs/lora_gpt2_digitwise.py
"""

from llm.config import GPTConfig, TrainingConfig
from llm.lora import LoRAConfig

model = GPTConfig(
    vocab_size=50257,
    block_size=32,          # "123+456=579" → 12 tokens
    n_layer=12, n_head=12, n_embd=768,
    dropout=0.0, bias=True,
    gradient_checkpointing=True,
)

lora = LoRAConfig(
    r=16, alpha=32.0, dropout=0.0,
    target_modules='all', target_layers=None,
    variant='lora',
    train_wpe=True,           # ← 位置嵌入可训练（加法需要位置感知）
    train_ln=True,            # ← LayerNorm 可训练（数字分布 vs 语言分布）
)

training = TrainingConfig(
    out_dir="out/lora_gpt2_digitwise",
    init_from="gpt2",
    always_save_checkpoint=True,
    eval_only=False,

    # addition_gpt2_digitwise = digit-wise 编码 + modulo train/test split
    dataset="addition_gpt2_digitwise",
    data_dir="",
    batch_size=64, gradient_accumulation_steps=2,

    learning_rate=1e-4, weight_decay=1e-2,
    beta1=0.9, beta2=0.95, grad_clip=1.0,
    decay_lr=True, warmup_iters=200,
    lr_decay_iters=4000, min_lr=1e-5,
    max_iters=4000, eval_interval=500,
    eval_iters=30, log_interval=50,

    backend="single", device="cuda",
    dtype="bfloat16", compile=False,
    ddp_backend="nccl", wandb_log=False,
)
