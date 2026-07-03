"""字符级模型文本生成。

用法:
    python scripts/generate_char.py out/shakespeare_char/ckpt.pt --prompt="FIRST" --max_tokens=100
"""

import sys
import os
import pickle
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from llm import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser(description="Character-level text generation")
    p.add_argument("checkpoint", type=str, help="Checkpoint 路径")
    p.add_argument("--prompt", type=str, default="FIRST CITIZEN:")
    p.add_argument("--max_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    # ── 1. 加载 checkpoint ──
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_cfg = GPTConfig.from_dict(ckpt["model_args"])
    model = GPT(model_cfg)
    sd = ckpt["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    print(f"Model: {model_cfg.n_layer}L/{model_cfg.n_head}H/{model_cfg.n_embd}D, "
          f"{model.get_num_params()/1e6:.2f}M params")

    # ── 2. 加载字符级 tokenizer（从 meta.pkl）──
    train_cfg = ckpt.get("config", {})
    data_dir = train_cfg.get("data_dir", "data/shakespeare_char")
    meta_path = os.path.join(data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        # 兼容 out_dir 相对路径
        meta_path = "data/shakespeare_char/meta.pkl"

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    itos = meta["itos"]  # id -> char
    stoi = meta["stoi"]  # char -> id

    def encode(text):
        return [stoi[c] for c in text]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    # ── 3. 生成 ──
    print(f"\nPrompt ({len(args.prompt)} chars): {repr(args.prompt)}")
    print(f"\n{'='*60}")

    with torch.no_grad():
        ids = encode(args.prompt)
        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    generated = decode(output_ids[0].tolist())
    print(generated)
    print(f"\n{'='*60}")
    print(f"Generated {len(output_ids[0]) - len(ids)} new characters")


if __name__ == "__main__":
    main()
