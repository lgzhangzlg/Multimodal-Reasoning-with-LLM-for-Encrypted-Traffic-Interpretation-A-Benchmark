#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库合并脚本
用法：python3 merge_kb.py <kb1.json> <kb2.json> [kb3.json ...] -o <output.json>
"""

import argparse
import json
import os
import sys


def load_kb(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_kbs(paths, conflict="latest"):
    """
    合并多个知识库，后面的文件会覆盖前面的同名条目（conflict="latest"）
    或保留最早的条目（conflict="first"）
    """
    merged = {}
    stats = {}

    for path in paths:
        kb = load_kb(path)
        new_keys = 0
        overwritten_keys = 0

        for key, value in kb.items():
            if key in merged:
                if conflict == "latest":
                    merged[key] = value
                    overwritten_keys += 1
                else:  # first
                    pass  # 保留已有的，不覆盖
            else:
                merged[key] = value
                new_keys += 1

        stats[path] = {"total": len(kb), "new": new_keys, "overwritten": overwritten_keys}

    return merged, stats


def main():
    parser = argparse.ArgumentParser(description="合并多个 knowledge_base.json 文件")
    parser.add_argument("inputs", nargs="+", help="输入的 knowledge_base.json 文件路径（可多个）")
    parser.add_argument("-o", "--output", required=True, help="输出文件路径")
    parser.add_argument(
        "--conflict",
        choices=["latest", "first"],
        default="latest",
        help="同名条目冲突处理策略：latest=后面的覆盖前面的（默认），first=保留最早的",
    )
    args = parser.parse_args()

    # 检查输入文件是否存在
    for path in args.inputs:
        if not os.path.exists(path):
            print(f"[ERROR] 文件不存在: {path}")
            sys.exit(1)

    print(f"合并策略: {args.conflict}")
    print(f"输入文件数: {len(args.inputs)}")
    print()

    merged, stats = merge_kbs(args.inputs, conflict=args.conflict)

    # 打印统计
    for path, s in stats.items():
        print(f"  {os.path.basename(path):<60} 条目: {s['total']:>4}  新增: {s['new']:>4}  覆盖: {s['overwritten']:>4}")

    print()
    print(f"合并结果: {len(merged)} 条条目")

    # 写出
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"已保存至: {args.output}")


if __name__ == "__main__":
    main()