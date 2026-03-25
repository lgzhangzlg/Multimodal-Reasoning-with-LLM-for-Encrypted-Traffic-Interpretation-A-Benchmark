#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按照 train.jsonl / test.jsonl 的 split 字段，
将 npy 文件复制到 train / test 目录，供 NetMamba 训练使用。

Usage:
    python split_npy_for_netmamba.py \
        --npy_root     /root/autodl-tmp/Datasets/USTC-TFC-2016/USTC-TFC-2016_npy_v3_balacned \
        --train_jsonl  /root/autodl-tmp/Datasets/USTC-TFC-2016/nlp_output/train.jsonl \
        --test_jsonl   /root/autodl-tmp/Datasets/USTC-TFC-2016/nlp_output/test.jsonl \
        --output_root  /root/autodl-tmp/Datasets/USTC-TFC-2016/USTC-TFC-2016_npy_v3_netmamba
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pcap_rel_to_npy_rel(sample_relpath: str) -> str:
    base, ext = os.path.splitext(sample_relpath)
    return sample_relpath if ext.lower() == ".npy" else (base + ".npy")


def split_npy(npy_root, train_jsonl, test_jsonl, output_root):
    train_data = read_jsonl(train_jsonl)
    test_data  = read_jsonl(test_jsonl)

    print(f"train samples: {len(train_data)}")
    print(f"test  samples: {len(test_data)}")

    splits = {
        "train": train_data,
        "test":  test_data,
    }

    stats = defaultdict(lambda: {"success": 0, "missing": 0})

    for split_name, data in splits.items():
        print(f"\nProcessing {split_name}...")
        for sample in tqdm(data, desc=split_name):
            rel = sample.get("sample_relpath", "")
            cls = sample.get("class", "")
            if not rel:
                stats[split_name]["missing"] += 1
                continue
            if not cls:
                cls = "unknown"  # 不再从 target 里读，新格式 target 没有 class 字段

            npy_rel  = pcap_rel_to_npy_rel(rel)
            src_path = os.path.join(npy_root, npy_rel)
            print(f"Checking: {src_path}")
            if not os.path.exists(src_path):
                stats[split_name]["missing"] += 1
                continue

            # 目标路径: output_root / train|test / ClassName / filename.npy
            dst_dir  = os.path.join(output_root, split_name, cls)
            dst_path = os.path.join(dst_dir, os.path.basename(npy_rel))

            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            stats[split_name]["success"] += 1

    # 统计报告
    print("\n" + "=" * 60)
    print("SPLIT REPORT")
    print("=" * 60)
    for split_name in ("train", "test"):
        s = stats[split_name]
        print(f"{split_name:<8}: success={s['success']:>6}  missing={s['missing']:>6}")

    # 按类别统计
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    for split_name in ("train", "test"):
        split_dir = os.path.join(output_root, split_name)
        if not os.path.exists(split_dir):
            continue
        print(f"\n[{split_name}]")
        total = 0
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                n = len([f for f in os.listdir(cls_dir) if f.endswith(".npy")])
                print(f"  {cls:<22}: {n:>6}")
                total += n
        print(f"  {'TOTAL':<22}: {total:>6}")
    print("=" * 60)
    print(f"\nOutput directory: {output_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_root",    required=True,
                        help="原始 npy 根目录")
    parser.add_argument("--train_jsonl", required=True,
                        help="train.jsonl 路径")
    parser.add_argument("--test_jsonl",  required=True,
                        help="test.jsonl 路径")
    parser.add_argument("--output_root", required=True,
                        help="输出根目录，会在其下创建 train/ 和 test/ 子目录")
    args = parser.parse_args()

    split_npy(
        npy_root    = args.npy_root,
        train_jsonl = args.train_jsonl,
        test_jsonl  = args.test_jsonl,
        output_root = args.output_root,
    )


if __name__ == "__main__":
    main()