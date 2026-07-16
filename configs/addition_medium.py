"""Addition 任务 — Medium 模型 (10.6M params)。

预期能较好学会加法，作为 scaling law 对比的中间点。
"""

from llm.config import GPTConfig, TrainingConfig

model = GPTConfig(
    vocab_size=14, block_size=32,
    n_layer=6, n_head=6, n_embd=384,
    dropout=0.0, bias=True, ignore_index=13,        # PAD '_' idx=13
)

training = TrainingConfig(
    out_dir="out/addition_medium",
    init_from="scratch", always_save_checkpoint=True,
    dataset="addition", data_dir="",
    batch_size=256, gradient_accumulation_steps=1,
    learning_rate=1e-3, weight_decay=1e-1,
    decay_lr=True, warmup_iters=500, lr_decay_iters=10000, min_lr=1e-4,
    max_iters=10000, eval_interval=1000, eval_iters=200, log_interval=10,
    backend="single", device="cuda", dtype="float16", compile=False,
)
