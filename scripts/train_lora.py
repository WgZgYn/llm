"""LoRA 微调训练入口。

支持两种任务模式:
1. 标准 LM 微调（OpenWebText、Shakespeare 等）
   python scripts/train_lora.py configs/lora_gpt2_owt.py

2. 加法算术任务（accuracy 评估）
   python scripts/train_lora.py configs/lora_addition.py

用法:
    python scripts/train_lora.py configs/lora_shakespeare.py      # smoke test
    python scripts/train_lora.py configs/lora_gpt2_owt.py         # 领域微调
    python scripts/train_lora.py configs/lora_addition.py         # 算术任务
"""

import os
import sys
import argparse
import importlib.util

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from llm import GPT, GPTConfig, TrainingConfig, validate_configs
from llm.data.loader import create_dataloader
from llm.data.dataset import CharTokenizer, AdditionDataset, collate_addition_batch
from llm.training.trainer import Trainer
from llm.lora import (
    LoRAConfig, LoRAGPT, LoRALinear,
    save_lora_adapters, load_lora_adapters, print_lora_info,
    merge_all_lora, unmerge_all_lora,
)


# ═══════════════════════════════════════════════════════════════
# 配置加载
# ═══════════════════════════════════════════════════════════════

def load_config(config_path: str):
    """动态加载 Python 配置文件，返回 (model_cfg, train_cfg, lora_cfg, module)。"""
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    spec = importlib.util.spec_from_file_location("task_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    missing = []
    if not hasattr(module, "model"): missing.append("model (GPTConfig)")
    if not hasattr(module, "training"): missing.append("training (TrainingConfig)")
    if not hasattr(module, "lora"): missing.append("lora (LoRAConfig)")
    if missing:
        raise ValueError(f"配置文件缺少: {missing}")

    if not isinstance(module.model, GPTConfig):
        raise TypeError(f"model 必须是 GPTConfig")
    if not isinstance(module.training, TrainingConfig):
        raise TypeError(f"training 必须是 TrainingConfig")
    if not isinstance(module.lora, LoRAConfig):
        raise TypeError(f"lora 必须是 LoRAConfig")
    return module.model, module.training, module.lora, module


# ═══════════════════════════════════════════════════════════════
# DDP 环境
# ═══════════════════════════════════════════════════════════════

def _init_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    return rank, local_rank, world_size, device


# ═══════════════════════════════════════════════════════════════
# 模型构造
# ═══════════════════════════════════════════════════════════════

def build_model(model_cfg: GPTConfig, train_cfg: TrainingConfig,
                lora_config: LoRAConfig, resume: bool,
                device: torch.device, is_dist: bool, local_rank: int):
    """创建 LoRAGPT 模型 + 分布式包装。"""
    if train_cfg.init_from in ("scratch",):
        print(f"Initializing LoRAGPT from scratch "
              f"(r={lora_config.r}, alpha={lora_config.alpha})")
        raw_model = LoRAGPT(model_cfg, lora_config)
    elif train_cfg.init_from in ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"):
        print(f"Loading LoRAGPT from HuggingFace: {train_cfg.init_from}")
        raw_model = LoRAGPT.from_pretrained(
            train_cfg.init_from, lora_config,
            override_args={'dropout': model_cfg.dropout} if model_cfg.dropout != 0 else None,
        )
        if model_cfg.block_size < raw_model.config.block_size:
            raw_model.crop_block_size(model_cfg.block_size)
    elif train_cfg.init_from == "resume":
        print(f"Resuming from {train_cfg.out_dir}/ckpt.pt")
        raw_model = LoRAGPT(model_cfg, lora_config)
    else:
        raise ValueError(f"Unknown init_from: {train_cfg.init_from}")

    print_lora_info(raw_model)

    backend = train_cfg.backend.lower()
    ddp = False

    if backend == "ddp" and is_dist:
        raw_model = raw_model.to(device)
        model = DDP(raw_model, device_ids=[local_rank])
        ddp = True
    else:
        raw_model = raw_model.to(device)
        model = raw_model

    if train_cfg.compile and hasattr(torch, 'compile'):
        try:
            import triton  # noqa: F401
        except ImportError:
            print("[WARNING] Triton not available, skipping torch.compile")
        else:
            print("compiling the model...")
            model = torch.compile(model)

    return model, raw_model, ddp


# ═══════════════════════════════════════════════════════════════
# 优化器（支持 LoRA+）
# ═══════════════════════════════════════════════════════════════

def build_optimizer(model, config: TrainingConfig, lora_config: LoRAConfig):
    """构建优化器，可选 LoRA+（A/B 不同学习率）。"""
    raw = model.module if hasattr(model, 'module') else model
    base_lr = config.learning_rate
    ratio = lora_config.lora_plus_lr_ratio

    if ratio is not None:
        lora_A_params, lora_B_params, other_params = [], [], []
        for n, p in raw.named_parameters():
            if not p.requires_grad:
                continue
            if 'lora_A' in n:
                lora_A_params.append(p)
            elif 'lora_B' in n:
                lora_B_params.append(p)
            else:
                other_params.append(p)

        groups = []
        if lora_A_params:
            groups.append({'params': lora_A_params, 'lr': base_lr / ratio,
                          'weight_decay': 0.0})
        if lora_B_params:
            groups.append({'params': lora_B_params, 'lr': base_lr * ratio,
                          'weight_decay': 0.0})
        if other_params:
            groups.append({'params': other_params, 'lr': base_lr,
                          'weight_decay': config.weight_decay})
        print(f"  LoRA+: A_lr={base_lr/ratio:.2e}, B_lr={base_lr*ratio:.2e} "
              f"(ratio={ratio})")
    else:
        trainable = [p for p in raw.parameters() if p.requires_grad]
        groups = [{'params': trainable, 'lr': base_lr,
                   'weight_decay': config.weight_decay}]

    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=base_lr,
                             betas=(config.beta1, config.beta2), fused=fused)


# ═══════════════════════════════════════════════════════════════
# 加法任务专用
# ═══════════════════════════════════════════════════════════════
#
# 支持两种模式:
#   dataset="addition"       → CharTokenizer, 字符级 (vocab=14)
#   dataset="addition_gpt2"  → GPT-2 BPE tokenizer (vocab=50257)
# ═══════════════════════════════════════════════════════════════

def _is_addition_task(train_cfg: TrainingConfig) -> bool:
    return train_cfg.dataset.startswith("addition")

@torch.no_grad()
def evaluate_digitwise_accuracy(model, loader, device, max_batches=50) -> float:
    """Digit-wise 加法准确率（teacher forcing 版，快速）。

    不逐 token 生成（太慢），而是用教师强制：
    在完整序列上做一次 forward，比较 "=" 之后的预测 token 与 target。
    """
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    eq_token = enc.encode("=")[0]
    digit_tokens = set(enc.encode(str(d))[0] for d in range(10))

    model.eval()
    correct, total = 0, 0

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits, _ = model(x, y)  # full forward, teacher forcing
        preds = logits.argmax(dim=-1)

        for b in range(x.size(0)):
            eq_mask = (x[b] == eq_token).nonzero(as_tuple=True)[0]
            if len(eq_mask) == 0:
                continue
            eq_pos = eq_mask[0].item()

            # 在 "=" 之后，比较预测和 target 的所有 digit tokens
            match = True
            ans_len = 0
            for t in range(eq_pos, min(preds.size(1), y.size(1))):
                targ_tok = y[b, t].item()
                if targ_tok == 0:  # padding
                    break
                if targ_tok not in digit_tokens:
                    break  # 非 digit token（不应该出现在答案中）
                pred_tok = preds[b, t].item()
                ans_len += 1
                if pred_tok != targ_tok:
                    match = False

            if ans_len > 0:
                total += 1
                if match:
                    correct += 1

    model.train()
    return correct / total if total > 0 else 0.0


def _is_gpt2_addition(train_cfg: TrainingConfig) -> bool:
    return train_cfg.dataset in ("addition_gpt2", "addition_gpt2_digitwise")

def _is_digitwise_addition(train_cfg: TrainingConfig) -> bool:
    return train_cfg.dataset == "addition_gpt2_digitwise"


# ── CharTokenizer 模式（字符级 vocab=14）──

def create_addition_loaders(train_cfg: TrainingConfig, tokenizer: CharTokenizer):
    bs = train_cfg.batch_size
    pad_id = tokenizer.pad_id
    train_ds = AdditionDataset(tokenizer, train=True, num_samples=100000, max_digits=5)
    val_ds = AdditionDataset(tokenizer, train=False, num_samples=100000, max_digits=5)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              collate_fn=lambda b: collate_addition_batch(b, pad_id),
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            collate_fn=lambda b: collate_addition_batch(b, pad_id),
                            pin_memory=True, drop_last=True)
    return train_loader, val_loader


@torch.no_grad()
def evaluate_addition_accuracy(model, loader, tokenizer, device, max_batches=50) -> float:
    model.eval()
    correct, total = 0, 0
    eq_id = tokenizer.stoi['=']
    for i, (x, y) in enumerate(loader):
        if i >= max_batches: break
        x, y = x.to(device), y.to(device)
        logits, _ = model(x, y)
        preds = logits.argmax(dim=-1)
        for b in range(x.size(0)):
            eq_mask = (x[b] == eq_id).nonzero(as_tuple=True)[0]
            if len(eq_mask) == 0: continue
            start = eq_mask[0].item() + 1
            end_mask = (y[b] == tokenizer.eos_id).nonzero(as_tuple=True)[0]
            end = end_mask[0].item() if len(end_mask) > 0 else y.size(1)
            valid_len = end - start
            if valid_len <= 0 or start + valid_len > preds.size(1): continue
            target = y[b, start:end]
            pred = preds[b, start:start + valid_len]
            total += 1
            if pred.shape == target.shape and (pred == target).all():
                correct += 1
    model.train()
    return correct / total if total > 0 else 0.0


# ── GPT-2 BPE tokenizer 模式 ──

class GPT2AdditionDataset(torch.utils.data.Dataset):
    """用 GPT-2 BPE tokenizer 编码的加法数据集。

    每个样本格式: "a+b=c"（用 GPT-2 tokenizer 编码）
    GPT-2 tokenizer 中多位数如 "12" "46" 是单个 token。

    训练目标是标准 LM: 给定 "a+b=", 预测 "c" token。
    """

    def __init__(self, train: bool = True, num_samples: int = 100000,
                 max_digits: int = 2, seed: int = 42):
        import random
        import tiktoken

        self.enc = tiktoken.get_encoding('gpt2')
        self.eq_token = self.enc.encode("=")[0]  # token id for "="

        rng = random.Random(seed)

        # 位数分布
        digit_probs = [0.5, 0.3, 0.2][:max_digits]
        s = sum(digit_probs)
        digit_probs = [p / s for p in digit_probs]

        self.data = []
        for _ in range(num_samples):
            digits_a = rng.choices(range(1, max_digits + 1), weights=digit_probs, k=1)[0]
            digits_b = rng.choices(range(1, max_digits + 1), weights=digit_probs, k=1)[0]
            a = rng.randint(10 ** (digits_a - 1), 10 ** digits_a - 1) if digits_a > 1 else rng.randint(0, 9)
            b = rng.randint(10 ** (digits_b - 1), 10 ** digits_b - 1) if digits_b > 1 else rng.randint(0, 9)

            text = f"{a}+{b}={a + b}"
            tokens = self.enc.encode(text)
            # input = all tokens except last, target = all tokens except first
            self.data.append((
                torch.tensor(tokens[:-1], dtype=torch.long),
                torch.tensor(tokens[1:], dtype=torch.long),
            ))

        rng.shuffle(self.data)
        split_idx = int(num_samples * 0.8)
        self.data = self.data[:split_idx] if train else self.data[split_idx:]
        print(f"GPT-2 Addition {'Train' if train else 'Test'}: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Digit-wise 编码模式 ──

class DigitWiseAdditionDataset(torch.utils.data.Dataset):
    """用 GPT-2 tokenizer 逐位编码的加法数据集。

    核心改进: 不依赖 GPT-2 BPE 对数字的任意切分，
    而是将每个 digit 映射为独立 token:
      "123+456=579"
      -> [tok('1'), tok('2'), tok('3'), tok('+'),
          tok('4'), tok('5'), tok('6'), tok('='),
          tok('5'), tok('7'), tok('9')]

    这样模型可以学 digit-wise 算术 + 进位规则，
    而非记忆 BPE token 对。

    Train/test 分割: 用 (a * 997 + b) % 5 的方式确定划分，
    确保同一个 (a,b) 对不会同时出现在 train 和 test 中。
    """

    def __init__(self, train: bool = True, max_digits: int = 2,
                 num_samples: int = 100000, seed: int = 42):
        import random
        import tiktoken

        enc = tiktoken.get_encoding('gpt2')

        # GPT-2 digit tokens: '0'-'9' → 15-24
        self.digit_map = {str(d): enc.encode(str(d))[0] for d in range(10)}
        self.plus_tok = enc.encode('+')[0]   # 10
        self.eq_tok = enc.encode('=')[0]     # 28

        rng = random.Random(seed)

        # 位数分布：偏向短数字
        digit_probs = [0.5, 0.3, 0.2][:max_digits]
        s = sum(digit_probs)
        digit_probs = [p / s for p in digit_probs]

        self.data = []
        for _ in range(num_samples):
            da = rng.choices(range(1, max_digits + 1), weights=digit_probs, k=1)[0]
            db = rng.choices(range(1, max_digits + 1), weights=digit_probs, k=1)[0]
            lo_a = 10 ** (da - 1) if da > 1 else 0
            hi_a = 10 ** da - 1
            lo_b = 10 ** (db - 1) if db > 1 else 0
            hi_b = 10 ** db - 1
            a = rng.randint(lo_a, hi_a)
            b = rng.randint(lo_b, hi_b)

            # Modulo-based train/test split: (a*997 + b) % 5 < 4 → train
            bucket = (a * 997 + b) % 5
            is_train_sample = (bucket < 4)
            if is_train_sample != train:
                continue

            tokens = self._encode(a, b)
            self.data.append((
                torch.tensor(tokens[:-1], dtype=torch.long),
                torch.tensor(tokens[1:], dtype=torch.long),
            ))

        print(f"DigitWise Addition {'Train' if train else 'Test'}: "
              f"{len(self.data)} samples (max_digits={max_digits})")

    def _encode(self, a: int, b: int) -> list[int]:
        """逐位编码 'a+b=c'。"""
        tokens = []
        for ch in str(a):
            tokens.append(self.digit_map[ch])
        tokens.append(self.plus_tok)
        for ch in str(b):
            tokens.append(self.digit_map[ch])
        tokens.append(self.eq_tok)
        for ch in str(a + b):
            tokens.append(self.digit_map[ch])
        return tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_gpt2_addition(batch, pad_value: int = 0):
    """Padding 到统一长度（GPT-2 没有特殊 PAD token，用 0）。"""
    xs, ys = zip(*batch)
    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=pad_value)
    ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=pad_value)
    return xs, ys


def create_gpt2_addition_loaders(train_cfg: TrainingConfig):
    """用 GPT-2 tokenizer 创建加法 DataLoader。

    支持两种模式:
      addition_gpt2           → BPE 自然编码（多位数=单token）
      addition_gpt2_digitwise → 逐位编码（每个digit=独立token，支持泛化）
    """
    bs = train_cfg.batch_size
    max_digits = getattr(train_cfg, 'addition_max_digits', None) or 2
    num_samples = getattr(train_cfg, 'addition_num_samples', None) or 100000

    if _is_digitwise_addition(train_cfg):
        train_ds = DigitWiseAdditionDataset(train=True, max_digits=max_digits,
                                            num_samples=num_samples)
        val_ds = DigitWiseAdditionDataset(train=False, max_digits=max_digits,
                                          num_samples=num_samples)
    else:
        train_ds = GPT2AdditionDataset(train=True, num_samples=num_samples,
                                       max_digits=max_digits)
        val_ds = GPT2AdditionDataset(train=False, num_samples=num_samples,
                                     max_digits=max_digits)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              collate_fn=collate_gpt2_addition,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            collate_fn=collate_gpt2_addition,
                            pin_memory=True, drop_last=True)
    return train_loader, val_loader


@torch.no_grad()
def evaluate_gpt2_addition_accuracy(model, loader, device, max_batches=50) -> float:
    """GPT-2 tokenizer 版加法准确率。

    GPT-2 BPE 把答案数字映射为 1-2 个 token。
    评估逻辑: 找到 "=" 位置 → 逐 token 生成 → 拼接解码 → 与期望答案比较。
    """
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    eq_token = enc.encode("=")[0]

    model.eval()
    correct, total = 0, 0

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)

        for b in range(x.size(0)):
            # 找到 "=" 位置
            eq_mask = (x[b] == eq_token).nonzero(as_tuple=True)[0]
            if len(eq_mask) == 0:
                continue
            eq_pos = eq_mask[0].item()

            # 从 "=" 位置开始逐 token greedy 生成答案
            # 输入只取到 "="（含）之前
            input_ids = x[b:b+1, :eq_pos + 1]
            answer_tokens = []
            for _ in range(3):  # 最多 3 个 token
                logits, _ = model(input_ids)
                next_token = logits[0, -1, :].argmax().item()
                decoded = enc.decode([next_token])
                if not decoded.strip().isdigit():
                    break
                answer_tokens.append(next_token)
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[next_token]], device=device)
                ], dim=1)

            pred_text = enc.decode(answer_tokens).strip()

            # 期望答案：y 中 "=" 之后的所有 token
            targ_tokens = []
            for t in range(eq_pos, y.size(1)):
                tok = y[b, t].item()
                if tok == 0:
                    break  # padding
                targ_tokens.append(tok)
            targ_text = enc.decode(targ_tokens).strip()

            if not pred_text:
                continue

            total += 1
            if pred_text == targ_text:
                correct += 1

    model.train()
    return correct / total if total > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    # ── 1. 加载 & 校验配置 ──
    args = parse_args()
    print(f"Loading config: {args.config}")
    model_cfg, train_cfg, lora_config, config_module = load_config(args.config)
    validate_configs(model_cfg, train_cfg, strict=not args.no_strict)
    print("  Config validation passed")

    # ── 2. 分布式环境 ──
    is_dist = int(os.environ.get('RANK', -1)) != -1
    rank, local_rank, world_size = 0, 0, 1
    device = torch.device(train_cfg.device if torch.cuda.is_available() else 'cpu')

    if is_dist:
        rank, local_rank, world_size, device = _init_distributed()

    addition_mode = _is_addition_task(train_cfg)

    # ── 3. 打印信息 ──
    if rank == 0:
        n_params = (12 * model_cfg.n_layer * model_cfg.n_embd ** 2) / 1e6
        print(f"\n{'=' * 60}")
        print(f"  Model:     {model_cfg.n_layer}L {model_cfg.n_head}H "
              f"{model_cfg.n_embd}D  (~{n_params:.1f}M params)")
        print(f"  LoRA:      r={lora_config.r}, alpha={lora_config.alpha}, "
              f"target={lora_config.target_modules}")
        if addition_mode:
            print(f"  Task:      ADDITION (accuracy evaluation)")
        if lora_config.lora_plus_lr_ratio:
            print(f"             LoRA+ ratio={lora_config.lora_plus_lr_ratio}")
        total_batch = train_cfg.batch_size * train_cfg.gradient_accumulation_steps * world_size
        print(f"  Training:  {train_cfg.max_iters} steps, "
              f"global_batch={total_batch}")
        print(f"             lr={train_cfg.learning_rate}, "
              f"dtype={train_cfg.dtype}, backend={train_cfg.backend}")
        print(f"{'=' * 60}\n")

    # ── 4. 模型 ──
    model, raw_model, ddp = build_model(
        model_cfg, train_cfg, lora_config,
        resume=(train_cfg.init_from == "resume"),
        device=device, is_dist=is_dist, local_rank=local_rank,
    )

    # ── 5. 优化器 + scaler ──
    optimizer = build_optimizer(model, train_cfg, lora_config)
    use_fp16 = train_cfg.dtype == 'float16' and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16) if use_fp16 else None

    # ── 6. bf16 兼容性 ──
    if device.type == 'cuda' and train_cfg.dtype == 'bfloat16':
        if not torch.cuda.is_bf16_supported():
            print(f"[WARNING] {torch.cuda.get_device_name(device)} 不支持 bfloat16, 自动切换 float16")
            train_cfg.dtype = 'float16'
            scaler = torch.amp.GradScaler('cuda', enabled=True)

    # ── 7. 数据 ──
    tokenizer = None
    gpt2_addition_mode = _is_gpt2_addition(train_cfg)
    digitwise_mode = _is_digitwise_addition(train_cfg)

    if gpt2_addition_mode:
        import tiktoken
        enc = tiktoken.get_encoding('gpt2')
        if digitwise_mode:
            print(f"GPT-2 DigitWise Addition: each digit is a separate token")
            print(f"  Example: '12+34=46' -> [tok('1'),tok('2'),tok('+'),"
                  f"tok('3'),tok('4'),tok('='),tok('4'),tok('6')]")
            print(f"  Train/test split: (a*997+b) % 5 < 4 → train (NO overlap!)")
        else:
            print(f"GPT-2 BPE Addition: vocab_size={enc.n_vocab}")
            print(f"  Example: '12+34=46' -> {enc.encode('12+34=46')}")
        train_loader, val_loader = create_gpt2_addition_loaders(train_cfg)
        print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    elif addition_mode:
        # ── CharTokenizer 加法任务 ──
        tokenizer = CharTokenizer()
        print(f"Addition task: vocab_size={tokenizer.vocab_size} "
              f"(tokens: {''.join(tokenizer.vocab)})")
        train_loader, val_loader = create_addition_loaders(train_cfg, tokenizer)
        print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    else:
        # ── 标准 LM 数据 ──
        data_dir = train_cfg.data_dir or os.path.join(PROJECT_ROOT, "data", train_cfg.dataset)
        if not os.path.exists(os.path.join(data_dir, "train.bin")):
            print(f"[ERROR] 未找到训练数据: {data_dir}/train.bin")
            sys.exit(1)
        train_loader = create_dataloader(
            data_dir, "train", train_cfg.batch_size, model_cfg.block_size)
        val_loader = create_dataloader(
            data_dir, "val", train_cfg.batch_size, model_cfg.block_size)

    # ── 8. Trainer ──
    trainer = Trainer(
        model=model, optimizer=optimizer, config=train_cfg,
        train_loader=train_loader, val_loader=val_loader,
        raw_model=raw_model, scaler=scaler, ddp=ddp,
    )

    # ── 9. 加法任务：注入 accuracy 评估回调 ──
    if gpt2_addition_mode and rank == 0:
        orig_eval = trainer._eval_and_save

        if digitwise_mode:
            def eval_with_accuracy():
                orig_eval()
                acc = evaluate_digitwise_accuracy(raw_model, val_loader, device)
                print(f"  [eval] digitwise addition accuracy: {acc*100:.1f}%")
        else:
            def eval_with_accuracy():
                orig_eval()
                acc = evaluate_gpt2_addition_accuracy(raw_model, val_loader, device)
                print(f"  [eval] addition accuracy (GPT-2 BPE): {acc*100:.1f}%")

        trainer._eval_and_save = eval_with_accuracy

    elif addition_mode and tokenizer is not None and rank == 0:
        orig_eval = trainer._eval_and_save

        def eval_with_accuracy():
            orig_eval()
            acc = evaluate_addition_accuracy(
                raw_model, val_loader, tokenizer, device)
            print(f"  [eval] accuracy: {acc*100:.1f}%")

        trainer._eval_and_save = eval_with_accuracy

    # ── 10. 注入 lora_args 到 checkpoint（Trainer 不知道 LoRA）──
    _orig_save = trainer._save_checkpoint

    def _save_with_lora():
        _orig_save()
        # 把 lora_config 写入已保存的 checkpoint
        ckpt_path = os.path.join(train_cfg.out_dir, 'ckpt.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            ckpt['lora_args'] = lora_config.to_dict()
            torch.save(ckpt, ckpt_path)

    trainer._save_checkpoint = _save_with_lora

    # ── 11. 恢复训练 ──
    if train_cfg.init_from == "resume":
        trainer.load_checkpoint()
        raw_model.freeze_base_weights()

    # ── 12. 训练 ──
    trainer.train()

    # ── 13. 最终评估（加法任务）──
    if gpt2_addition_mode and rank == 0:
        if digitwise_mode:
            final_acc = evaluate_digitwise_accuracy(
                raw_model, val_loader, device, max_batches=200)
            print(f"\n{'=' * 50}")
            print(f"  Final digitwise addition accuracy: {final_acc*100:.1f}%")
            print(f"{'=' * 50}")
        else:
            final_acc = evaluate_gpt2_addition_accuracy(
                raw_model, val_loader, device, max_batches=200)
            print(f"\n{'=' * 50}")
            print(f"  Final GPT-2 BPE addition accuracy: {final_acc*100:.1f}%")
            print(f"{'=' * 50}")

    elif addition_mode and tokenizer is not None and rank == 0:
        final_acc = evaluate_addition_accuracy(
            raw_model, val_loader, tokenizer, device, max_batches=200)
        print(f"\n{'=' * 50}")
        print(f"  Final addition accuracy: {final_acc*100:.1f}%")
        print(f"{'=' * 50}")

    # ── 13. 保存 LoRA adapter ──
    if rank == 0:
        adapter_path = os.path.join(train_cfg.out_dir, 'lora_adapters.pt')
        save_lora_adapters(raw_model, adapter_path, lora_config)

    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA 微调训练")
    parser.add_argument("config", type=str, help="配置文件路径")
    parser.add_argument("--no-strict", action="store_true",
                        help="跳过配置校验警告")
    parser.add_argument("--resume", action="store_true",
                        help="从最新 checkpoint 恢复训练")
    return parser.parse_args()


if __name__ == "__main__":
    main()
