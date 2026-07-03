"""文本生成入口脚本。

用法:
    # 从 GPT-2 预训练权重生成
    python scripts/generate.py --model=gpt2 --prompt="Hello, my name is"

    # 从本地 checkpoint 生成
    python scripts/generate.py --checkpoint=out/ckpt.pt --prompt="The quick brown"

    # 交互模式
    python scripts/generate.py --model=gpt2 --interactive
"""

import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import tiktoken
from llm import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser(description="GPT Text Generation")

    p.add_argument("--model", type=str, default="gpt2",
                   help="预训练模型: gpt2, gpt2-medium, gpt2-large, gpt2-xl")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="本地 checkpoint 路径（优先级高于 --model）")
    p.add_argument("--prompt", type=str, default="Hello, I'm a language model,")
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--interactive", action="store_true",
                   help="交互模式")
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda | cpu")

    return p.parse_args()


def load_model(args) -> tuple[GPT, tiktoken.Encoding]:
    """加载模型和 tokenizer。"""
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if args.checkpoint:
        # 从本地 checkpoint 加载
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        config = GPTConfig.from_dict(ckpt['model_args'])
        model = GPT(config)
        # 移除 _orig_mod. 前缀
        sd = ckpt['model']
        for k in list(sd.keys()):
            if k.startswith('_orig_mod.'):
                sd[k[len('_orig_mod.'):]] = sd.pop(k)
        model.load_state_dict(sd)
        model = model.to(device)
    else:
        # 从 HuggingFace GPT-2 加载
        print(f"Loading pretrained: {args.model}")
        model = GPT.from_pretrained(args.model)
        model = model.to(device)

    model.eval()

    # Tokenizer: 使用 GPT-2 BPE
    enc = tiktoken.get_encoding("gpt2")

    return model, enc


def generate(model, enc, prompt, args):
    """生成文本。"""
    device = next(model.parameters()).device
    ids = enc.encode(prompt)
    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)

    output_ids = model.generate(
        input_tensor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    generated = enc.decode(output_ids[0].tolist())
    return generated


def main():
    args = parse_args()
    model, enc = load_model(args)

    print(f"\nModel loaded. Vocab size: {model.config.vocab_size}")
    print(f"Parameters: {model.get_num_params()/1e6:.2f}M\n")
    print("=" * 60)

    if args.interactive:
        print("Interactive mode. Type 'quit' to exit.\n")
        while True:
            prompt = input("Prompt> ").strip()
            if prompt.lower() in ('quit', 'exit', 'q'):
                break
            if not prompt:
                continue
            print("=" * 40)
            output = generate(model, enc, prompt, args)
            print(output)
            print("=" * 40)
    else:
        output = generate(model, enc, args.prompt, args)
        print(output)


if __name__ == "__main__":
    main()
