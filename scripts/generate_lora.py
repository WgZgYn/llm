"""LoRA 模型生成 & 测试脚本。

支持的加载方式:
  1. 完整 checkpoint:  --checkpoint out/lora_addition/ckpt.pt
  2. LoRA adapter:     --lora out/lora_addition/lora_adapters.pt --base gpt2
  3. 两者同时:         --checkpoint ... --lora ...

功能:
  --task lm           标准文本生成（GPT-2 BPE tokenizer）
  --task addition_gpt2  GPT-2 BPE 加法测试
  --task addition_char  字符级加法测试

泛化测试:
  --digits 3           用 N 位数测试（训练用 2 位，测试用 3 位 = 泛化能力）

用法:
  # GPT-2 BPE 加法（同分布）
  python scripts/generate_lora.py --checkpoint out/lora_gpt2_addition/ckpt.pt

  # 泛化测试：2位训练 → 3位测试
  python scripts/generate_lora.py --checkpoint out/lora_gpt2_addition/ckpt.pt --digits 3

  # 交互模式
  python scripts/generate_lora.py --checkpoint out/lora_gpt2_addition/ckpt.pt --interactive

  # 文本生成对比
  python scripts/generate_lora.py --lora out/lora_owt/lora_adapters.pt --compare
"""

import os
import sys
import argparse
import pickle
import random

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import tiktoken
from llm.config import GPTConfig
from llm.lora import (
    LoRAConfig, LoRAGPT,
    merge_all_lora, unmerge_all_lora, print_lora_info,
)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="LoRA Model Generation & Testing")

    # 模型加载
    p.add_argument("--checkpoint", type=str, default=None,
                   help="完整 checkpoint 路径 (ckpt.pt)")
    p.add_argument("--lora", type=str, default=None,
                   help="LoRA adapter 路径 (lora_adapters.pt)")
    p.add_argument("--base", type=str, default="gpt2",
                   help="Base model: gpt2, gpt2-medium, gpt2-large, gpt2-xl")

    # 任务模式
    p.add_argument("--task", type=str, default="auto",
                   choices=["auto", "lm", "addition_gpt2", "addition_digitwise",
                            "addition_char"],
                   help="任务模式 (auto=自动检测)")

    # 加法测试参数
    p.add_argument("--digits", type=int, default=None,
                   help="测试位数（默认=训练位数，设更大值=泛化测试）")
    p.add_argument("--num_tests", type=int, default=20,
                   help="批量测试样本数")

    # 文本生成参数
    p.add_argument("--prompt", type=str, default=None,
                   help="输入文本（LM 模式）")
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)

    # 模式
    p.add_argument("--interactive", action="store_true", help="交互模式")
    p.add_argument("--compare", action="store_true",
                   help="对比 base model vs LoRA model")
    p.add_argument("--merge", action="store_true",
                   help="推理前 merge LoRA 到 base weights")

    # 系统
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda | cpu")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════════

def _infer_lora_config_from_sd(sd: dict) -> LoRAConfig:
    """从 state_dict 推断 LoRA 配置。"""
    ranks = set()
    has_mlp = False
    for k, v in sd.items():
        if 'lora_A.weight' in k:
            ranks.add(v.shape[0])
        if 'mlp' in k and 'lora_' in k:
            has_mlp = True
    r = max(ranks) if ranks else 8
    target = 'all' if has_mlp else 'attn'
    return LoRAConfig(r=r, alpha=r * 2.0, target_modules=target)


def _should_use_hf_base(task_type: str, model_args: dict) -> bool:
    """判断是否需要从 HuggingFace 加载预训练权重。"""
    if task_type == "addition_char":
        return False
    if model_args.get('n_embd', 768) < 768:
        return False
    if model_args.get('vocab_size', 50257) < 100:
        return False
    return True


def load_lora_model(args):
    """加载 LoRA 模型。返回 (model, task_type, train_digits)。"""
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    task_type = args.task
    train_digits = None  # 训练时使用的位数
    lora_config = None

    # ── 方式 1: 完整 checkpoint ──
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model_args = ckpt.get('model_args', {})
        lora_args = ckpt.get('lora_args', {})

        # 推断任务
        if task_type == "auto":
            vocab = model_args.get('vocab_size', 50257)
            train_cfg = ckpt.get('config', {})
            dataset_name = train_cfg.get('dataset', '')
            if vocab <= 20:
                task_type = "addition_char"
            elif 'digitwise' in dataset_name:
                task_type = "addition_digitwise"
            elif lora_args or 'addition' in dataset_name:
                task_type = "addition_gpt2"
            else:
                task_type = "lm"

        # 推断训练位数（从 block_size 推断；block_size 越大→位数越多）
        train_digits = model_args.get('block_size', 32) // 6  # 粗略估计

        # 构建 LoRA config
        config = GPTConfig.from_dict(model_args)
        if lora_args:
            lora_config = LoRAConfig.from_dict(lora_args)
        else:
            lora_config = _infer_lora_config_from_sd(ckpt['model'])
            print(f"  Inferred LoRA: r={lora_config.r}, target={lora_config.target_modules}")

        # 创建模型
        if _should_use_hf_base(task_type, model_args):
            print(f"  Loading base from HuggingFace: {args.base}")
            model = LoRAGPT.from_pretrained(args.base, lora_config,
                                            override_args={'dropout': 0.0})
            if config.block_size < model.config.block_size:
                model.crop_block_size(config.block_size)
        else:
            print(f"  Creating from scratch (n_embd={config.n_embd})")
            model = LoRAGPT(config, lora_config)

        # 加载权重
        sd = ckpt['model']
        for k in list(sd.keys()):
            if k.startswith('_orig_mod.'):
                sd[k[len('_orig_mod.'):]] = sd.pop(k)

        # strict=False 并只报告真正的错误
        model_sd = model.state_dict()
        real_missing = []
        for k in model_sd:
            if k not in sd and 'lora_' not in k:
                real_missing.append(k)
        if real_missing:
            print(f"  [WARN] {len(real_missing)} non-LoRA keys missing from checkpoint")

        model.load_state_dict(sd, strict=False)
        model = model.to(device)

    # ── 方式 2: LoRA adapter + HF base ──
    elif args.lora:
        print(f"Loading LoRA adapter: {args.lora}")
        lora_data = torch.load(args.lora, map_location=device, weights_only=False)
        lora_config_dict = lora_data.get('lora_config', {})
        lora_config = LoRAConfig.from_dict(lora_config_dict) if lora_config_dict else LoRAConfig()
        lora_state = lora_data.get('lora_state', lora_data)

        print(f"  Loading base from HuggingFace: {args.base}")
        model = LoRAGPT.from_pretrained(args.base, lora_config,
                                        override_args={'dropout': 0.0})
        model.load_state_dict(lora_state, strict=False)
        model = model.to(device)

        if task_type == "auto":
            task_type = "lm"

    # ── 方式 3: 仅 HF base ──
    else:
        print(f"Loading base model: {args.base}")
        lora_config = LoRAConfig()
        model = LoRAGPT.from_pretrained(args.base, lora_config,
                                        override_args={'dropout': 0.0})
        model = model.to(device)
        task_type = "lm"

    model.eval()

    if args.merge:
        print("Merging LoRA into base weights...")
        merge_all_lora(model)

    print_lora_info(model)
    print(f"  Task: {task_type}, block_size: {model.config.block_size}")

    return model, task_type, train_digits


# ═══════════════════════════════════════════════════════════════
# 加法测试
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def test_addition_gpt2(model, a: int, b: int):
    """GPT-2 BPE tokenizer 加法测试。

    关键: GPT-2 BPE 把多位数映射为 1-2 个 token。
    先编码正确答案得出期望 token 数，再生成恰好那么多 token 后比较。
    这避免了模型"不知道何时停止"的问题。
    """
    enc = tiktoken.get_encoding("gpt2")
    device = next(model.parameters()).device
    block_size = model.config.block_size

    prompt = f"{a}+{b}="
    correct = str(a + b)

    # 编码正确答案，确定期望 token 数量
    correct_tokens = enc.encode(correct)
    num_answer_tokens = len(correct_tokens)

    ids = enc.encode(prompt)
    if len(ids) >= block_size:
        return prompt, "TOO_LONG", correct, False

    x = torch.tensor([ids], dtype=torch.long, device=device)

    # 生成恰好 num_answer_tokens 个 token
    predicted_tokens = []
    for _ in range(num_answer_tokens):
        logits, _ = model(x)
        next_token = logits[0, -1, :].argmax().item()
        predicted_tokens.append(next_token)
        x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)
        if x.size(1) >= block_size:
            break

    predicted = enc.decode(predicted_tokens).strip() if predicted_tokens else "?"
    return prompt, predicted, correct, predicted == correct


@torch.no_grad()
def test_addition_char(model, a: int, b: int):
    """CharTokenizer 加法测试（逐 token greedy 直到 EOS/PAD）。"""
    from llm.data.dataset import CharTokenizer
    enc = CharTokenizer()
    device = next(model.parameters()).device
    block_size = model.config.block_size

    prompt = f"{a}+{b}="
    correct = str(a + b)

    ids = enc.encode(prompt)
    if len(ids) >= block_size:
        return prompt, "TOO_LONG", correct, False

    x = torch.tensor([ids], dtype=torch.long, device=device)

    # 逐 token 生成
    generated = []
    for _ in range(20):  # 足够大
        logits, _ = model(x)  # [B, T, V]
        next_token = logits[0, -1, :].argmax().item()

        if next_token in (enc.eos_id, enc.pad_id):
            break
        generated.append(next_token)
        x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)

        if x.size(1) >= block_size:  # 防止超出 block_size
            break

    predicted = enc.decode(generated)
    return prompt, predicted, correct, predicted == correct


def _random_number(digits: int, rng: random.Random) -> int:
    """生成指定位数的随机数。"""
    if digits <= 1:
        return rng.randint(0, 9)
    lo = 10 ** (digits - 1)
    hi = 10 ** digits - 1
    return rng.randint(lo, hi)


@torch.no_grad()
def test_addition_digitwise(model, a: int, b: int):
    """Digit-wise 加法测试（逐 token greedy 生成 digit tokens）。"""
    enc = tiktoken.get_encoding("gpt2")
    device = next(model.parameters()).device
    block_size = model.config.block_size

    prompt = f"{a}+{b}="
    correct = str(a + b)

    # Digit-wise 编码 prompt
    digit_map = {str(d): enc.encode(str(d))[0] for d in range(10)}
    plus_tok = enc.encode('+')[0]
    eq_tok = enc.encode('=')[0]
    digit_tokens = set(digit_map.values())

    tokens = []
    for ch in str(a):
        tokens.append(digit_map[ch])
    tokens.append(plus_tok)
    for ch in str(b):
        tokens.append(digit_map[ch])
    tokens.append(eq_tok)

    if len(tokens) >= block_size:
        return prompt, "TOO_LONG", correct, False

    x = torch.tensor([tokens], dtype=torch.long, device=device)

    # 确定正确答案需要几个 token，生成恰好那么多
    correct_tokens = enc.encode(correct)
    num_answer_tokens = len(correct_tokens)

    gen = []
    for _ in range(num_answer_tokens):
        logits, _ = model(x)
        next_tok = logits[0, -1, :].argmax().item()
        gen.append(next_tok)
        x = torch.cat([x, torch.tensor([[next_tok]], device=device)], dim=1)
        if x.size(1) >= block_size:
            break

    predicted = enc.decode(gen).strip() if gen else "?"
    return prompt, predicted, correct, predicted == correct


def run_addition_tests(model, task_type: str, test_digits: int,
                       num_tests: int = 20, interactive: bool = False):
    """运行加法测试，返回 (correct, total)。"""
    if task_type == "addition_digitwise":
        test_fn = test_addition_digitwise
    elif task_type == "addition_gpt2":
        test_fn = test_addition_gpt2
    else:
        test_fn = test_addition_char
    rng = random.Random(42)

    if interactive:
        print(f"Interactive Addition Test (digits={test_digits})")
        print("Enter 'a+b' or 'quit'.\n")
        correct = 0
        total = 0
        while True:
            inp = input("calc> ").strip()
            if inp.lower() in ('quit', 'exit', 'q'):
                break
            if not inp:
                continue
            try:
                if '+' in inp:
                    a_str, b_str = inp.split('+', 1)
                    a, b = int(a_str), int(b_str)
                else:
                    print("  Format: a+b (e.g., 12+34)")
                    continue
            except ValueError:
                print("  Invalid format. Use: a+b")
                continue

            prompt, predicted, correct_ans, is_correct = test_fn(model, a, b)
            total += 1
            if is_correct:
                correct += 1
            status = "OK" if is_correct else f"WRONG (expected {correct_ans})"
            print(f"  {prompt}{predicted}  {status}")

        if total > 0:
            print(f"\n  Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
        return correct, total

    else:
        print(f"Addition Test (digits={test_digits}, {num_tests} samples)\n")
        pairs = [(_random_number(test_digits, rng), _random_number(test_digits, rng))
                 for _ in range(num_tests)]

        correct = 0
        for a, b in pairs:
            prompt, predicted, correct_ans, is_correct = test_fn(model, a, b)
            if is_correct:
                correct += 1
            status = "OK" if is_correct else f"WRONG (expected {correct_ans})"
            print(f"  {prompt}{predicted}  {status}")

        acc = correct / len(pairs) * 100
        print(f"\n  Accuracy: {correct}/{len(pairs)} = {acc:.1f}%")
        return correct, len(pairs)


# ═══════════════════════════════════════════════════════════════
# 文本生成
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_text(model, prompt: str, args, enc=None):
    """标准文本生成。"""
    device = next(model.parameters()).device
    if enc is None:
        enc = tiktoken.get_encoding("gpt2")

    ids = enc.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    output_ids = model.generate(
        x,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    return enc.decode(output_ids[0].tolist())


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    model, task_type, train_digits = load_lora_model(args)

    # ── 确定测试位数 ──
    if args.digits is not None:
        test_digits = args.digits
    else:
        test_digits = train_digits or 2

    is_generalization = (train_digits is not None and test_digits > train_digits)

    print(f"\n{'=' * 60}")

    # ── 加法模式 ──
    if task_type in ("addition_gpt2", "addition_char", "addition_digitwise"):
        if is_generalization:
            print(f"GENERALIZATION TEST: trained on {train_digits}-digit, testing on {test_digits}-digit")
            print("(This tests whether the model learned ADDITION, not just memorization)\n")

        run_addition_tests(
            model, task_type, test_digits,
            num_tests=args.num_tests,
            interactive=args.interactive,
        )

        # 如果是泛化测试，也跑一下同分布作为参考
        if is_generalization and not args.interactive:
            print(f"\n{'─' * 40}")
            print(f"In-distribution reference ({train_digits}-digit):\n")
            run_addition_tests(
                model, task_type, train_digits,
                num_tests=args.num_tests,
                interactive=False,
            )

    # ── 文本生成模式 ──
    else:
        enc = tiktoken.get_encoding("gpt2")
        prompt = args.prompt or "Hello, I'm a language model,"

        if args.interactive:
            print("Text Generation - Interactive Mode")
            print("Type prompts or 'quit' to exit.\n")
            while True:
                p = input("prompt> ").strip()
                if p.lower() in ('quit', 'exit', 'q'):
                    break
                if not p:
                    continue
                try:
                    print(generate_text(model, p, args, enc))
                    print("─" * 40)
                except Exception as e:
                    print(f"  Error: {e}")

        elif args.compare:
            print(f"Prompt: {prompt}\n")
            print("─" * 40)
            print("[Base model (without LoRA)]")
            unmerge_all_lora(model)
            try:
                print(generate_text(model, prompt, args, enc))
            except Exception as e:
                print(f"Error: {e}")
            print("─" * 40)
            print("[With LoRA adapter]")
            merge_all_lora(model)
            try:
                print(generate_text(model, prompt, args, enc))
            except Exception as e:
                print(f"Error: {e}")
            print("─" * 40)
        else:
            print(f"Prompt: {prompt}\n")
            print(generate_text(model, prompt, args, enc))


if __name__ == "__main__":
    main()
