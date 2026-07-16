"""Addition 任务训练 + 评估。

用法:
    python scripts/train_addition.py configs/addition_tiny.py
    python scripts/train_addition.py configs/addition_small.py
    python scripts/train_addition.py configs/addition_medium.py

输出: out/<name>/ 下 checkpoint + 终端打印 eval accuracy
"""

import os, sys, argparse, importlib.util

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from llm import GPT, GPTConfig, TrainingConfig, validate_configs
from llm.data.dataset import CharTokenizer, AdditionDataset, collate_addition_batch
from llm.training.trainer import Trainer


def load_config(path):
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("cfg", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.model, m.training, m


def make_loaders(cfg: TrainingConfig, tokenizer):
    """创建 train/val DataLoader。"""
    bs = cfg.batch_size
    pad_id = tokenizer.pad_id

    train_ds = AdditionDataset(tokenizer, train=True, num_samples=100000, max_digits=5)
    val_ds   = AdditionDataset(tokenizer, train=False, num_samples=100000, max_digits=5)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              collate_fn=lambda b: collate_addition_batch(b, pad_id),
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=bs, shuffle=False,
                              collate_fn=lambda b: collate_addition_batch(b, pad_id),
                              pin_memory=True, drop_last=True)
    return train_loader, val_loader, train_ds, val_ds


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, max_batches=50):
    """计算加法准确率：答案部分是否完全预测正确。"""
    model.eval()
    correct = 0
    total = 0
    eq_id = tokenizer.stoi['=']
    pad_id = tokenizer.stoi['_']

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        B, T = x.shape
        # 传 targets=y 触发返回全体 logits（不是仅最后一位置）
        logits, _ = model(x, y)
        preds = logits.argmax(dim=-1)  # [B, T]

        for b in range(B):
            # 找 '=' 位置 → 答案起点
            eq_mask = (x[b] == eq_id).nonzero(as_tuple=True)[0]
            if len(eq_mask) == 0:
                continue
            start = eq_mask[0].item() + 1

            # 找答案终点（'#' 或 pad）
            end_mask = ((y[b] == tokenizer.stoi['#']) | (y[b] == pad_id)).nonzero(as_tuple=True)[0]
            end = end_mask[0].item() if len(end_mask) > 0 else T

            valid_len = end - start
            if valid_len <= 0 or start + valid_len > preds.size(1):
                continue

            target = y[b, start:end]
            pred = preds[b, start:start + valid_len]

            total += 1
            if pred.shape == target.shape and (pred == target).all():
                correct += 1

    model.train()
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    model_cfg, train_cfg, cfg_module = load_config(args.config)
    validate_configs(model_cfg, train_cfg, strict=False)

    if args.compile:
        train_cfg.compile = True

    # ── Tokenizer ──
    tokenizer = CharTokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    # ── 数据 ──
    train_loader, val_loader, _, val_ds = make_loaders(train_cfg, tokenizer)

    # ── 模型 ──
    raw_model = GPT(model_cfg)
    device = torch.device(train_cfg.device if torch.cuda.is_available() else 'cpu')
    raw_model = raw_model.to(device)
    model = raw_model
    if train_cfg.compile and hasattr(torch, 'compile'):
        print("compiling model...")
        model = torch.compile(model)

    # ── 优化器 ──
    params = {pn: p for pn, p in raw_model.named_parameters() if p.requires_grad}
    decay = [p for p in params.values() if p.dim() >= 2]
    nodecay = [p for p in params.values() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {'params': decay, 'weight_decay': train_cfg.weight_decay},
        {'params': nodecay, 'weight_decay': 0.0},
    ], lr=train_cfg.learning_rate, betas=(train_cfg.beta1, train_cfg.beta2), fused=True)
    scaler = torch.amp.GradScaler('cuda', enabled=(train_cfg.dtype == 'float16'))

    n_params = sum(p.numel() for p in raw_model.parameters()) / 1e6
    print(f"Model params: {n_params:.2f}M")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── Trainer ──
    trainer = Trainer(
        model=model, optimizer=optimizer, config=train_cfg,
        train_loader=train_loader, val_loader=val_loader,
        raw_model=raw_model, scaler=scaler, ddp=False,
    )

    # ── 训练（注入评估回调）──
    orig_eval = trainer._eval_and_save

    def eval_with_acc():
        orig_eval()
        acc = evaluate(raw_model, val_loader, tokenizer, device)
        print(f"  [eval] accuracy: {acc*100:.1f}%")

    trainer._eval_and_save = eval_with_acc

    print(f"\n{'='*50}")
    print(f"Training {n_params:.1f}M params on Addition task")
    print(f"{'='*50}\n")

    trainer.train()

    # ── 最终评估 ──
    final_acc = evaluate(raw_model, val_loader, tokenizer, device, max_batches=200)
    print(f"\nFinal accuracy: {final_acc*100:.1f}%")


if __name__ == "__main__":
    main()
