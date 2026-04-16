#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
from collections import defaultdict
from tqdm import tqdm

def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def build_npy_index(npy_root):
    print(f"[*] 正在全盘扫描 NPY 根目录，构建全局文件索引...")
    npy_index = {}
    for root, _, files in os.walk(npy_root):
        cls_dir = os.path.relpath(root, npy_root)  # e.g. "Zhihu" or "Baidu"
        for f in files:
            if f.endswith('.npy'):
                base_name = f.replace('.npy', '').replace('.pcap', '')
                # 用 "类名/文件名" 做 key，避免跨类同名文件覆盖
                key = f"{cls_dir}/{base_name}"
                npy_index[key] = os.path.join(root, f)
    print(f"  扫描完毕！共找到 {len(npy_index)} 个 .npy 文件。")
    return npy_index

def split_npy(npy_root, train_jsonl, test_jsonl, output_root):
    train_data = read_jsonl(train_jsonl)
    test_data  = read_jsonl(test_jsonl)

    print(f"train samples: {len(train_data)}")
    print(f"test  samples: {len(test_data)}")

    # 1. 核心大招：构建全局字典
    npy_index = build_npy_index(npy_root)

    if len(npy_index) == 0:
        print("\n❌ 致命错误：在你的 npy_root 目录下没有找到任何 .npy 文件！请检查传入的 --npy_root 路径是否正确！")
        return

    splits = {"train": train_data, "test":  test_data}
    stats = defaultdict(lambda: {"success": 0, "missing": 0})

    for split_name, data in splits.items():
        print(f"\nProcessing {split_name}...")
        for sample in tqdm(data, desc=split_name):
            rel = sample.get("sample_relpath", "")
            cls = sample.get("class", "unknown")
            if not rel:
                continue

            # 2. 从 JSONL 提取唯一标识符：类名/文件名
            jsonl_filename = os.path.basename(rel)
            base_name = jsonl_filename.replace('.npy', '').replace('.pcap', '')
            key = f"{cls}/{base_name}"

            # 3. 用 "类名/文件名" 精确匹配，避免跨类同名碰撞
            if key in npy_index:
                src_path = npy_index[key]
                # 目标路径: output_root / train或test / 语义化的类名(如 UC Browser) / filename.npy
                dst_dir  = os.path.join(output_root, split_name, cls)
                dst_path = os.path.join(dst_dir, os.path.basename(src_path))

                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                stats[split_name]["success"] += 1
            else:
                stats[split_name]["missing"] += 1
                if stats[split_name]["missing"] <= 3:
                    print(f"\n  [WARN] npy not found: {key}")

    print("\n" + "=" * 60)
    print("SPLIT REPORT")
    print("=" * 60)
    for split_name in ("train", "test"):
        s = stats[split_name]
        print(f"{split_name:<8}: success={s['success']:>6}  missing={s['missing']:>6}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_root",    required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--test_jsonl",  required=True)
    parser.add_argument("--output_root", required=True)
    # 不再需要 label_map，因为我们直接用文件名硬匹配了！
    parser.add_argument("--label_map",   default=None)
    args = parser.parse_args()

    split_npy(
        npy_root    = args.npy_root,
        train_jsonl = args.train_jsonl,
        test_jsonl  = args.test_jsonl,
        output_root = args.output_root
    )

if __name__ == "__main__":
    main()