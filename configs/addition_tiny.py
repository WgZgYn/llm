"""Addition 任务 — Tiny 模型 (0.8M params)。

用途: 验证最小模型能否学到加法规律（预期：不能完全学会）
"""

from llm.config import GPTConfig, TrainingConfig

model = GPTConfig(
    vocab_size=14,          # 0-9 + = # _
    block_size=32,          # max_digits=5 → "01234+05678=12345#" ≈ 17 chars
    n_layer=2, n_head=4, n_embd=64,
    dropout=0.0, bias=True,
    ignore_index=13,        # 忽略 PAD token '_' (vocab idx 13)，不参与 loss
)

training = TrainingConfig(
    out_dir="out/addition_tiny",
    init_from="scratch",
    always_save_checkpoint=True,
    dataset="addition",
    data_dir="",
    batch_size=128,
    gradient_accumulation_steps=1,
    learning_rate=1e-3, weight_decay=1e-1,
    decay_lr=True, warmup_iters=200, lr_decay_iters=5000, min_lr=1e-4,
    max_iters=5000, eval_interval=500, eval_iters=100, log_interval=10,
    backend="single", device="cuda", dtype="float16",
    compile=False,
)
