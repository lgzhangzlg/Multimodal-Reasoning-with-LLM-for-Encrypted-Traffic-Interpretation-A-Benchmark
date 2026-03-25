#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert existing train.jsonl / test.jsonl to LLM-only format.

For each sample:
  1. Load the npy file from --npy-root
  2. Convert bytes to hex string
  3. Replace user_text with text-only version (no <image>, no <cls_placeholder>)
  4. Write new train.jsonl / test.jsonl to --output-dir

Usage:
    python convert_to_llmonly_jsonl.py \
        --train-jsonl /root/autodl-tmp/Datasets/ISCXVPN2016/nlp_output_noLLMclass_200_6000/train.jsonl \
        --test-jsonl  /root/autodl-tmp/Datasets/ISCXVPN2016/nlp_output_noLLMclass_200_6000/test.jsonl \
        --npy-root    /root/autodl-tmp/Datasets/ISCXVPN2016/ISCXVPN2016_npy_split_npy_v3_balacned_200_6000 \
        --output-dir  /root/autodl-tmp/Datasets/ISCXVPN2016/nlp_output_llmonly_200_6000 \
        --max-bytes   1600
"""

import argparse
import json
import os
import numpy as np
from collections import OrderedDict
from tqdm import tqdm


# ============================================================
# 十六进制转换
# ============================================================

def npy_to_hex_string(npy_path: str, max_bytes: int = 1600) -> str:
    """
    Load npy file (uint8, shape (1, L)) and convert to hex string.
    Example: "0b 45 48 7b 29 e2 ..."
    """
    arr = np.load(npy_path)
    arr = np.asarray(arr, dtype=np.uint8).reshape(-1)
    arr = arr[:max_bytes]
    return " ".join(f"{b:02x}" for b in arr)


# ============================================================
# user_text 构建（纯文本，无 image token，无 cls_placeholder）
# ============================================================

def build_llmonly_user_text(class_list: list, hex_str: str) -> str:
    classes_str = ", ".join(sorted(class_list))
    return (
        "The following is a raw network traffic capture represented as a "
        "hexadecimal byte sequence (5 packets × 160 bytes each, "
        "IP addresses zeroed, ports bucketed):\n\n"
        f"{hex_str}\n\n"
        f"The possible traffic categories are: {classes_str}.\n\n"
        "Based on the byte sequence above, return a single JSON object "
        "with the following keys:\n\n"
        "- class: the predicted traffic category (must be one of the listed categories).\n\n"
        "- traits: an object that objectively describes the byte-level characteristics "
        "of this flow, with exactly these keys:\n"
        "    - has_tls_record: boolean. True if TLS record header pattern is detected "
        "(content-type byte 0x14~0x17 followed by version byte 0x03xx).\n"
        "    - has_http_method: boolean. True if HTTP tokens such as GET, POST, HTTP/1.x, "
        "Host:, or User-Agent: are present in the payload.\n"
        "    - ascii_ratio_bucket: one of 'low', 'mid', 'high'. Indicates the proportion "
        "of printable ASCII bytes (0x20~0x7E) in the non-zero payload.\n"
        "    - entropy_bucket: one of 'low', 'mid', 'high'. Indicates Shannon entropy of "
        "the non-zero payload. High entropy suggests encrypted or compressed data.\n"
        "    - zero_pad_ratio_bucket: one of 'low', 'mid', 'high'. Indicates the proportion "
        "of zero bytes. High ratio means the flow is short relative to the capture window.\n\n"
        "- evidence: a list of 2~4 strings. Each string should describe a concrete "
        "byte-level observation or protocol pattern that supports the classification result. "
        "Focus on what is actually visible in the byte features, such as header patterns, "
        "payload characteristics, or entropy signatures.\n\n"
        "- description: a single paragraph of 2~3 sentences that explains what this traffic "
        "is, what protocol or application it belongs to, and what the byte features reveal "
        "about its behavior. Be specific to the classified category.\n\n"
        "- notes: a single sentence with a security-relevant observation or recommendation "
        "about this traffic category, such as whether it could be misused, what to monitor, "
        "or any anomaly indicators."
    )


# ============================================================
# npy 路径解析
# ============================================================

def resolve_npy_path(sample_relpath: str, npy_root: str) -> str:
    """
    Convert sample_relpath (pcap relative path) to npy path under npy_root.
    Handles both .pcap and .npy extensions.
    """
    base, ext = os.path.splitext(sample_relpath)
    npy_rel = base + ".npy"
    return os.path.join(npy_root, npy_rel)


# ============================================================
# 主转换逻辑
# ============================================================

def convert_jsonl(
    input_path: str,
    output_path: str,
    npy_root: str,
    max_bytes: int,
    split_label: str,
):
    print(f"\nConverting {split_label}: {input_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 第一遍：读取所有样本，收集 class_list
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    class_list = sorted(set(r["class"] for r in rows))
    print(f"  Classes ({len(class_list)}): {class_list}")

    ok = skip = 0
    out_rows = []

    for row in tqdm(rows, desc=f"  {split_label}", ncols=80):
        relpath = row.get("sample_relpath", "")
        npy_path = resolve_npy_path(relpath, npy_root)

        if not os.path.exists(npy_path):
            # 尝试直接用 sample_path 的目录结构
            sample_path = row.get("sample_path", "")
            alt_rel = os.path.splitext(os.path.basename(sample_path))[0] + ".npy"
            cls = row.get("class", "")
            npy_path_alt = os.path.join(npy_root, cls, alt_rel)
            if os.path.exists(npy_path_alt):
                npy_path = npy_path_alt
            else:
                skip += 1
                continue

        try:
            hex_str = npy_to_hex_string(npy_path, max_bytes=max_bytes)
        except Exception as e:
            print(f"  [WARN] Failed to load {npy_path}: {e}")
            skip += 1
            continue

        new_user_text = build_llmonly_user_text(class_list, hex_str)

        new_row = OrderedDict([
            ("sample_path",    row.get("sample_path", "")),
            ("sample_relpath", relpath),
            ("sample_id",      row.get("sample_id", "")),
            ("class",          row.get("class", "")),
            ("split",          split_label),
            ("user_text",      new_user_text),
            ("target",         row.get("target", "")),
        ])
        out_rows.append(new_row)
        ok += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  Done: ok={ok}, skip(npy missing)={skip} → {output_path}")
    return out_rows


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Convert jsonl to LLM-only hex format")
    parser.add_argument("--train-jsonl", required=True, help="Path to existing train.jsonl")
    parser.add_argument("--test-jsonl",  required=True, help="Path to existing test.jsonl")
    parser.add_argument("--npy-root",    required=True, help="Root directory of npy files")
    parser.add_argument("--output-dir",  required=True, help="Output directory for new jsonl files")
    parser.add_argument("--max-bytes",   type=int, default=800,
                        help="Max bytes to include in hex string (default: 800, i.e. 5 packets x 160 bytes)")
    args = parser.parse_args()

    convert_jsonl(
        input_path=args.train_jsonl,
        output_path=os.path.join(args.output_dir, "train.jsonl"),
        npy_root=args.npy_root,
        max_bytes=args.max_bytes,
        split_label="train",
    )
    convert_jsonl(
        input_path=args.test_jsonl,
        output_path=os.path.join(args.output_dir, "test.jsonl"),
        npy_root=args.npy_root,
        max_bytes=args.max_bytes,
        split_label="test",
    )

    print("\nDone. Run with:")
    print(f"  --data-path {os.path.join(args.output_dir, 'train.jsonl')}")


if __name__ == "__main__":
    main()