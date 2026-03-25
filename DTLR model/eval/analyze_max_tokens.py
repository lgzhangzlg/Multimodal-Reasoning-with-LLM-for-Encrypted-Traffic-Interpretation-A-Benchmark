#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 jsonl 文件中 target 字段的 token 长度分布，推荐合适的 --max_new_tokens 值。

用法：
    python3 analyze_max_tokens.py <jsonl_path> --tokenizer <tokenizer_path>

示例：
    python3 analyze_max_tokens.py \
        /root/autodl-tmp/Datasets/CSTNet-TLS1.3/nlp_output_no_LLMclass_0_6000/test.jsonl \
        --tokenizer /root/autodl-tmp/train_out/CSTNet-TLS1.3/.../checkpoint-288

    # 若不指定 tokenizer，使用字符数估算（tokens ≈ chars / 3.5）
    python3 analyze_max_tokens.py test.jsonl
"""

import argparse
import json
import sys
import os
import numpy as np


def load_lengths_with_tokenizer(jsonl_path, tokenizer_path, field):
    from transformers import AutoTokenizer
    print(f"[INFO] 加载 tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    lengths = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.get(field, "")
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            lengths.append(len(ids))
    return lengths, "token（tokenizer精确计算）"


def load_lengths_estimated(jsonl_path, field):
    """用字符数 / 3.5 估算 token 数（英文约4字符/token，中英混合约3.5）"""
    lengths = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.get(field, "")
            if not text:
                continue
            lengths.append(int(len(text) / 3.5))
    return lengths, "token（字符数/3.5 估算）"


def print_stats(lengths, unit_label, margin=1.2):
    lengths = sorted(lengths)
    n = len(lengths)
    arr = np.array(lengths, dtype=np.float64)

    p50  = int(np.percentile(arr, 50))
    p90  = int(np.percentile(arr, 90))
    p95  = int(np.percentile(arr, 95))
    p99  = int(np.percentile(arr, 99))
    p100 = int(arr.max())
    mean = int(arr.mean())

    recommended = int(p99 * margin / 10 + 0.9) * 10  # 向上取整到10的倍数

    print(f"\n{'='*55}")
    print(f"  样本数:        {n}")
    print(f"  单位:          {unit_label}")
    print(f"  最小值:        {int(arr.min())}")
    print(f"  平均值:        {mean}")
    print(f"  p50:           {p50}")
    print(f"  p90:           {p90}")
    print(f"  p95:           {p95}")
    print(f"  p99:           {p99}")
    print(f"  最大值:        {p100}")
    print(f"{'='*55}")
    print(f"  推荐 --max_new_tokens = {recommended}")
    print(f"  （p99={p99} × {margin:.1f} 余量，向上取整到10的倍数）")
    print(f"  覆盖率：")
    for threshold in [recommended, p99, p95, p90]:
        coverage = sum(1 for l in lengths if l <= threshold) / n * 100
        print(f"    max_new_tokens={threshold:<6} → 覆盖 {coverage:.2f}% 样本")
    print(f"{'='*55}\n")

    return recommended


def main():
    parser = argparse.ArgumentParser(description="分析 jsonl target 长度，推荐 max_new_tokens")
    parser.add_argument("jsonl_path", help="jsonl 文件路径")
    parser.add_argument("--tokenizer", default=None,
                        help="tokenizer 路径（checkpoint 目录），不填则用字符数估算")
    parser.add_argument("--field", default="target",
                        help="要分析的字段名（默认 target）")
    parser.add_argument("--margin", type=float, default=1.2,
                        help="p99 的余量倍数（默认 1.2，即 20%% 余量）")
    args = parser.parse_args()

    if not os.path.exists(args.jsonl_path):
        print(f"[ERROR] 文件不存在: {args.jsonl_path}")
        sys.exit(1)

    print(f"[INFO] 分析文件: {args.jsonl_path}")
    print(f"[INFO] 目标字段: {args.field}")

    if args.tokenizer:
        lengths, unit = load_lengths_with_tokenizer(
            args.jsonl_path, args.tokenizer, args.field)
    else:
        print("[INFO] 未指定 tokenizer，使用字符数/3.5 估算")
        lengths, unit = load_lengths_estimated(args.jsonl_path, args.field)

    if not lengths:
        print(f"[ERROR] 未找到任何 '{args.field}' 字段内容")
        sys.exit(1)

    print_stats(lengths, unit, margin=args.margin)


if __name__ == "__main__":
    main()
