#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import argparse
from typing import Any, Dict

# 1) 匹配 Path: xxx.pcap.（非贪婪，支持多次）
PATH_RE = re.compile(r"(?i)\bPath:\s*.*?\.pcap\.\s*")

# 2) 标点与空格清洗规则
MULTI_SPACE_RE = re.compile(r"\s+")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?])")
MULTI_PUNCT_RE = re.compile(r"([.!?]){2,}")

TARGET_FIELDS = {"sample_description", "nl_description", "label_sentence"}


def clean_grammar(text: Any) -> Any:
    if not isinstance(text, str):
        return text

    s = text

    # (1) 删除 Path 泄露
    s = PATH_RE.sub(" ", s)

    # (2) 删除多余空格
    s = MULTI_SPACE_RE.sub(" ", s)

    # (3) 标点前不留空格
    s = SPACE_BEFORE_PUNCT_RE.sub(r"\1", s)

    # (4) 连续标点压缩为 1 个
    s = MULTI_PUNCT_RE.sub(r"\1", s)

    # (5) 修复句号后无空格的问题
    s = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", s)

    # (6) 去掉首尾杂质
    s = s.strip(" .;:,")

    return s.strip()


def process_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    for k in TARGET_FIELDS:
        if k in obj:
            obj[k] = clean_grammar(obj[k])
    return obj


def is_jsonl(path: str) -> bool:
    return path.lower().endswith(".jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="input .jsonl or .json")
    ap.add_argument("--out_path", required=True, help="output .jsonl or .json")
    args = ap.parse_args()

    if is_jsonl(args.in_path):
        count = 0
        with open(args.in_path, "r", encoding="utf-8") as fin, \
             open(args.out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                obj = process_obj(obj)
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1
        print(f"[OK] processed {count} lines -> {args.out_path}")
    else:
        with open(args.in_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, dict):
            obj = process_obj(obj)
        elif isinstance(obj, list):
            obj = [process_obj(x) if isinstance(x, dict) else x for x in obj]
        else:
            raise ValueError("Unsupported JSON root type")

        with open(args.out_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

        print(f"[OK] saved -> {args.out_path}")


if __name__ == "__main__":
    main()
