"""
Prepare the Shakespeare dataset for character-level language modeling.
Encodes characters as integers and saves train.bin, val.bin, meta.pkl.

Output: data/shakespeare_char/
    input.txt    — raw text (downloaded)
    train.bin    — training tokens (90%)
    val.bin      — validation tokens (10%)
    meta.pkl     — vocab info (vocab_size, stoi, itos)

Usage:
    python llm/data/prepare/shakespeare_char.py
"""

import os
import pickle
import requests
import numpy as np

# ── 输出目录：项目根目录/data/shakespeare_char/ ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "shakespeare_char")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. 下载原始文本 ──
input_file_path = os.path.join(OUT_DIR, "input.txt")
if not os.path.exists(input_file_path):
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open(input_file_path, "w", encoding="utf-8") as f:
        f.write(requests.get(data_url).text)
    print(f"Downloaded Shakespeare text to {input_file_path}")

with open(input_file_path, "r", encoding="utf-8") as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# ── 2. 构建字符级词汇表 ──
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", "".join(chars))
print(f"vocab size: {vocab_size:,}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join([itos[i] for i in ids])


# ── 3. 训练/验证集划分 ──
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# ── 4. 导出 .bin 文件 ──
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_bin = os.path.join(OUT_DIR, "train.bin")
val_bin = os.path.join(OUT_DIR, "val.bin")
train_ids.tofile(train_bin)
val_ids.tofile(val_bin)
print(f"Saved: {train_bin}")
print(f"Saved: {val_bin}")

# ── 5. 保存元信息 ──
meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}
meta_path = os.path.join(OUT_DIR, "meta.pkl")
with open(meta_path, "wb") as f:
    pickle.dump(meta, f)
print(f"Saved: {meta_path}")
