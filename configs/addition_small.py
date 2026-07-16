"""Addition 任务 — Small 模型 (3.1M params)。

GPT-2 124M 的 1/40 规模，预期能部分学会加法。
"""

from llm.config import GPTConfig, TrainingConfig

model = GPTConfig(
    vocab_size=14, block_size=32,
    n_layer=4, n_head=4, n_embd=128,
    dropout=0.0, bias=True, ignore_index=13,        # PAD '_' idx=13
)

training = TrainingConfig(
    out_dir="out/addition_small",
    init_from="scratch", always_save_checkpoint=True,
    dataset="addition", data_dir="",
    batch_size=256, gradient_accumulation_steps=1,
    learning_rate=1e-3, weight_decay=1e-1,
    decay_lr=True, warmup_iters=500, lr_decay_iters=8000, min_lr=1e-4,
    max_iters=8000, eval_interval=1000, eval_iters=200, log_interval=10,
    backend="single", device="cuda", dtype="float16", compile=False,
)
