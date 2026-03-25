#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import argparse
import re
from collections import defaultdict
from typing import Dict, List


def extract_label(label_sentence: str) -> str:
    """
    从 'This is BitTorrent traffic.' 提取 'BitTorrent'
    """
    if not label_sentence:
        return ""
    m = re.search(r"This\s+is\s+(.+?)\s+traffic\.?$", label_sentence, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return label_sentence.strip()


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(path: str, data: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def stratified_split(
    data: List[Dict],
    test_ratio: float,
    seed: int = 42,
):
    random.seed(seed)

    # 按类别分组
    buckets = defaultdict(list)
    for sample in data:
        label = extract_label(sample.get("label_sentence", ""))
        buckets[label].append(sample)

    train_data, test_data = [], []

    for label, samples in buckets.items():
        n = len(samples)
        if n < 2:
            # 极小类：全部进训练集
            train_data.extend(samples)
            continue

        random.shuffle(samples)
        n_test = max(1, int(n * test_ratio))
        test_samples = samples[:n_test]
        train_samples = samples[n_test:]

        train_data.extend(train_samples)
        test_data.extend(test_samples)

    random.shuffle(train_data)
    random.shuffle(test_data)

    return train_data, test_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True, help="原始 jsonl")
    parser.add_argument("--train_jsonl", type=str, required=True, help="输出 train.jsonl")
    parser.add_argument("--test_jsonl", type=str, required=True, help="输出 test.jsonl")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    data = load_jsonl(args.input_jsonl)
    train_data, test_data = stratified_split(
        data,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    save_jsonl(args.train_jsonl, train_data)
    save_jsonl(args.test_jsonl, test_data)

    print("===== Split Done =====")
    print(f"Total samples : {len(data)}")
    print(f"Train samples : {len(train_data)}")
    print(f"Test samples  : {len(test_data)}")
    print(f"Test ratio    : {len(test_data) / max(1, len(data)):.3f}")


if __name__ == "__main__":
    main()
#
# python split_train_test_jsonl.py \
#   --input_jsonl /root/autodl-tmp/Datasets/USTC-TFC-2016/nlp_description_output/USTC-TFC-2016_descriptions_clean.jsonl \
#   --train_jsonl /root/autodl-tmp/Datasets/USTC-TFC-2016/train.jsonl \
#   --test_jsonl  /root/autodl-tmp/Datasets/USTC-TFC-2016/test.jsonl \
#   --test_ratio 0.2 \
#   --seed 42
