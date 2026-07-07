"""GPT-2 30M + MoE 变种配置。

在 OpenWebText 上训练，中间层替换为 MoE_FFN（8 experts, top-2 激活）。

用法:
    torchrun --nproc-per-node=4 scripts/train.py configs/gpt2_124M_moe.py
"""

from llm.config import GPTConfig, TrainingConfig
from llm.model.moe_gpt import MoEGPT

model = GPTConfig(
    vocab_size=50304,          # 与你 OWT 预处理时的 vocab 保持一致
    block_size=512,            # 略微提升序列长度，更贴近真实训练
    n_layer=6,                 # 6 层，约 4M 参数
    n_head=6,                  # 头数保持整除
    n_embd=384,                # 384 维
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
    gradient_accumulation_steps=20,
    learning_rate=3e-4,
    weight_decay=1e-1,
    beta1=0.9, beta2=0.95, grad_clip=1.0,
    decay_lr=True, warmup_iters=200, lr_decay_iters=6000, min_lr=3e-5,
    max_iters=6000, eval_interval=600, eval_iters=200,
    backend="single", device="cuda", dtype="bfloat16",
    compile=False,
)

# ── 变种模型：config 直接构造 ──
model_obj = MoEGPT(
    model,
    num_experts=8,        # 8 个 expert
    top_k=2,              # 每 token 激活 2 个
    moe_layers=(6, 7),    # 最后一次层用 MoE，首尾保持标准 MLP
)
