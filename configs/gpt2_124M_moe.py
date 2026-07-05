"""GPT-2 124M + MoE 变种配置。

在 OpenWebText 上训练，中间层替换为 MoE_FFN（8 experts, top-2 激活）。

用法:
    torchrun --nproc-per-node=4 scripts/train.py configs/gpt2_124M_moe.py
"""

from llm.config import GPTConfig, TrainingConfig
from llm.model.moe_gpt import MoEGPT

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
    out_dir="out/gpt2_124M_moe",
    init_from="scratch",
    always_save_checkpoint=True,
    dataset="openwebtext",
    data_dir="data/openwebtext",
    batch_size=12,
    gradient_accumulation_steps=40,
    learning_rate=6e-4,
    weight_decay=1e-1,
    beta1=0.9, beta2=0.95, grad_clip=1.0,
    decay_lr=True, warmup_iters=2000, lr_decay_iters=600000, min_lr=6e-5,
    max_iters=600000, eval_interval=2000, eval_iters=200,
    backend="ddp", device="cuda", dtype="float16",
    compile=False,
)

# ── 变种模型：config 直接构造 ──
model_obj = MoEGPT(
    model,
    num_experts=8,        # 8 个 expert
    top_k=2,              # 每 token 激活 2 个
    moe_layers=(3, 9),    # 第 3~8 层用 MoE，首尾保持标准 MLP
)
