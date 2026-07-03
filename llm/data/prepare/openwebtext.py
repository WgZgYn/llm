"""
Prepare the OpenWebText dataset with GPT-2 BPE tokenization.
Downloads the dataset via HuggingFace datasets, encodes with tiktoken,
and saves as continuous uint16 token id arrays.

Output: data/openwebtext/
    train.bin    — training tokens (~9B, ~17GB)
    val.bin      — validation tokens (~4M, ~8.5MB)

Requirements:
    pip install datasets tiktoken tqdm

Usage:
    python llm/data/prepare/openwebtext.py
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import multiprocessing as mp

# ── 输出目录：项目根目录/data/openwebtext/ ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "openwebtext")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 配置 ──
NUM_PROC = 8          # CPU 核心数，用于并行编码
BATCH_SIZE = 2000     # 每批文档数（内存 ~4-6MB 临时数组）
enc = tiktoken.get_encoding("gpt2")


# ── 多进程 worker 初始化 ──
def init_worker():
    """每个子进程初始化时加载编码器（避免重复加载）"""
    global worker_enc
    worker_enc = tiktoken.get_encoding("gpt2")


def encode_single(text):
    """单个文本的编码（在子进程中运行）"""
    ids = worker_enc.encode_ordinary(text)
    ids.append(worker_enc.eot_token)  # 每个文档末尾添加 EOT token
    return ids


# ── 核心写入函数 ──
def write_split(split_name, dataset_split, batch_size=2000, num_proc=8):
    """将数据集的一个 split 编码并写入 .bin 文件。"""
    filename = os.path.join(OUT_DIR, f"{split_name}.bin")
    total_docs = len(dataset_split)
    total_batches = (total_docs + batch_size - 1) // batch_size

    total_tokens = 0
    processed_docs = 0

    with mp.Pool(processes=num_proc, initializer=init_worker) as pool:
        with open(filename, "wb") as f:
            for batch in tqdm(
                dataset_split.iter(batch_size=batch_size),
                desc=f"Writing {split_name}",
                unit="batch",
                total=total_batches,
            ):
                texts = batch["text"]

                # 多进程并行编码当前批次
                list_of_ids = pool.map(encode_single, texts)

                # 展平并写入
                flat_ids = np.concatenate(
                    [np.array(ids, dtype=np.uint16) for ids in list_of_ids]
                )
                flat_ids.tofile(f)

                total_tokens += len(flat_ids)
                processed_docs += len(texts)

                del list_of_ids, flat_ids

    print(f"{split_name}: {processed_docs} docs, {total_tokens:,} tokens, "
          f"saved to {filename}")


# ── 保存元信息 ──
def save_meta(vocab_size):
    """保存词汇表大小等元信息。"""
    import pickle
    meta = {"vocab_size": vocab_size}
    meta_path = os.path.join(OUT_DIR, "meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved: {meta_path}")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Windows 下多进程必须放在 if __name__ == "__main__" 内

    # 1. 加载数据集
    print("Loading OpenWebText dataset...")
    dataset = load_dataset("Skylion007/openwebtext", num_proc=NUM_PROC)
    split_dataset = dataset["train"].train_test_split(
        test_size=0.0005, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")

    # 2. 编码并写入
    print(f"\nOutput directory: {OUT_DIR}")
    write_split("train", split_dataset["train"], batch_size=BATCH_SIZE, num_proc=NUM_PROC)
    write_split("val", split_dataset["val"], batch_size=BATCH_SIZE, num_proc=NUM_PROC)

    # 3. 保存元信息
    save_meta(enc.max_token_value + 1)

    print("\nDone! Files in", OUT_DIR)
    for f in os.listdir(OUT_DIR):
        size_mb = os.path.getsize(os.path.join(OUT_DIR, f)) / (1024 * 1024)
        print(f"  {f}  ({size_mb:.1f} MB)")
