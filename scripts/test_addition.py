"""手动测试 Addition 模型 —— 输入算式，看模型能否正确计算。

用法:
    # 随机测试
    python scripts/test_addition.py out/addition_small/ckpt.pt --random 20

    # 指定算式
    python scripts/test_addition.py out/addition_small/ckpt.pt --expr "123+456"
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from llm.data.dataset import CharTokenizer, AdditionDataset, collate_addition_batch
from llm.model.gpt import GPT
from llm.config.model_config import GPTConfig


def load_model(ckpt_path: str, device: str = "cpu"):
    """从 checkpoint 恢复模型。"""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("model_args", {})
    cfg = GPTConfig.from_dict(cfg_dict)
    model = GPT(cfg)
    sd = ckpt["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    iter_num = ckpt.get("iter_num", "?")
    return model, cfg, iter_num


@torch.no_grad()
def solve(model, expr: str, tokenizer, device="cpu", max_len=32) -> str:
    """给定算式字符串，返回模型的计算结果。

    流程:
      1. tokenize: "123+456=" → token ids
      2. 自回归生成直到 '#' 或达到 max_len
      3. 解码返回
    """
    # 确保以 '=' 结尾
    if not expr.endswith("="):
        expr = expr + "="
    ids = torch.tensor([tokenizer.encode(expr)], dtype=torch.long, device=device)

    for _ in range(max_len - ids.size(1)):
        # 只取最后 block_size 个 token
        ctx = ids if ids.size(1) <= model.config.block_size else ids[:, -model.config.block_size:]
        logits, _ = model(ctx)
        next_token = logits[:, -1, :].argmax(dim=-1)  # [1]
        ids = torch.cat([ids, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == tokenizer.stoi["#"]:
            break

    result = tokenizer.decode(ids[0].tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description="手动测试 Addition 模型")
    parser.add_argument("checkpoint", type=str, help="checkpoint 路径")
    parser.add_argument("--random", type=int, default=0, help="随机测试 N 道题")
    parser.add_argument("--expr", nargs="*", help="直接测试算式，如 123+456")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # ── 加载模型 ──
    tokenizer = CharTokenizer()
    model, cfg, iter_num = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.2f}M params, trained {iter_num} iters")
    print(f"Config: {cfg.n_layer}L {cfg.n_head}H {cfg.n_embd}D, block_size={cfg.block_size}")
    print()

    # ── 指定算式 ──
    if args.expr:
        for expr in args.expr:
            result = solve(model, expr, tokenizer, device)
            # 解析结果
            try:
                a_str, rest = expr.split("+")
                b_str = rest.rstrip("=")
                expected = int(a_str) + int(b_str)
                pred_str = result.split("=")[-1].rstrip("#")
                pred_val = int(pred_str) if pred_str.isdigit() else "?"
                ok = "OK" if pred_val == expected else f"FAIL (expected {expected})"
            except Exception:
                ok = "?"
            print(f"  {expr} → {result}  {ok}")
        return

    # ── 随机测试 ──
    if args.random > 0:
        import random
        rng = random.Random(42)
        correct = 0
        total = 0

        for _ in range(args.random):
            # 生成随机算式（1-5 位数）
            digits_a = rng.randint(1, 5)
            digits_b = rng.randint(1, 5)
            a = rng.randint(0, 10**digits_a - 1)
            b = rng.randint(0, 10**digits_b - 1)
            expr = f"{a}+{b}="
            result = solve(model, expr, tokenizer, device)
            pred_str = result.split("=")[-1].rstrip("#")
            try:
                pred_val = int(pred_str)
            except ValueError:
                pred_val = None
            ok = pred_val == a + b
            if ok:
                correct += 1
            total += 1

            # 只打印错误或每隔 50 题
            if not ok or total % max(1, args.random // 10) == 0:
                status = "OK" if ok else f"FAIL expected {a + b}"
                print(f"  {expr}{result}  {status}")

        print(f"\nAccuracy: {correct}/{total} = {100*correct/total:.1f}%")
        return

    # ── 交互模式 ──
    print("输入算式（如 123+456），输入 q 退出")
    print("模型会自动补 '='，生成到 '#' 为止")
    print()
    while True:
        try:
            expr = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if expr.lower() in ("q", "quit", "exit"):
            break
        if not expr:
            continue
        result = solve(model, expr, tokenizer, device)
        print(f"  {result}")
        print()


if __name__ == "__main__":
    main()
