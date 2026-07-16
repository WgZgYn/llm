"""
GPT-2 小模型（~4M 参数）– 使用 OpenWebText，RTX 4060 快速验证（~25 分钟）。

用途: 验证你已处理好的 OpenWebText 数据是否能被正确加载和训练。
      不追求收敛，只验证 loss 下降和梯度更新正常。
硬件: RTX 4060 8GB，单卡。
数据: 直接使用你下载并处理好的 data/openwebtext。
用法:
    python scripts/train.py configs/gpt2_lite.py
"""

from llm.config import GPTConfig, TrainingConfig

# ═══════════════════════════════════════════════════════════════════════════
# 模型架构（比纯玩具稍大，显存占用 ~3.5GB，能体现 OWT 的多样性）
# ═══════════════════════════════════════════════════════════════════════════

model = GPTConfig(
    vocab_size=50304,          # 与你 OWT 预处理时的 vocab 保持一致
    block_size=512,            # 略微提升序列长度，更贴近真实训练
    n_layer=6,                 # 6 层，约 4M 参数
    n_head=6,                  # 头数保持整除
    n_embd=384,                # 384 维
    dropout=0.0,
    bias=True,
    gradient_checkpointing=False,
)

# ═══════════════════════════════════════════════════════════════════════════
# 训练参数（针对 OWT 调整，总 step ~6000，耗时约 25 分钟）
# ═══════════════════════════════════════════════════════════════════════════

training = TrainingConfig(
    # 输入/输出
    out_dir="out/gpt2_lite",
    init_from="scratch",
    always_save_checkpoint=False,   # 节省磁盘 I/O，只保存最后一次
    eval_only=False,

    # ═══ 关键：使用你已处理好的 OpenWebText ═══
    dataset="openwebtext",          # 对应 data/openwebtext 下的 train.bin / val.bin
    data_dir="data/openwebtext",    # 确认路径无误
    batch_size=16,                   # 每卡 batch size
    gradient_accumulation_steps=4,  # 有效 batch = 16*4 = 64 样本/步
                                    # 每步 token 数 = 64 * 512 = 32,768 tokens

    # 优化器
    learning_rate=3e-4,
    weight_decay=1e-1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,

    # 学习率调度（余弦衰减，适配 6000 步）
    decay_lr=True,
    warmup_iters=200,               # 前 200 步 warmup
    lr_decay_iters=6000,            # 与 max_iters 对齐
    min_lr=3e-5,                    # lr/10

    # 训练控制（设定 6000 步，RTX 4060 约 0.25 秒/步 → 约 25 分钟）
    max_iters=6000,
    eval_interval=500,              # 每 500 步验证一次（可观察到 loss 稳步下降）
    eval_iters=50,                  # 验证时用 50 个 batch
    log_interval=20,                # 每 20 步打印一次 loss

    # 系统（单卡，强烈建议关闭 compile，否则首次编译会浪费 3~5 分钟）
    backend="single",
    device="cuda",
    dtype="bfloat16",               # 节省显存且精度足够
    compile=False,                  # 关闭编译，保证前几秒就有日志输出

    # DDP（单卡不启用）
    ddp_backend="nccl",

    # 日志（可选，如需观察可开启 wandb，但会占用几秒初始化时间）
    wandb_log=False,
    wandb_project="llm-owt-debug",
    wandb_run_name="gpt2-owt-4060",
)