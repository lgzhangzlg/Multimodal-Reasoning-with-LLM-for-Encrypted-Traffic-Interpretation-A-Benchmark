#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 ISCX-Tor-2016 Tor_split_pcap 目录下的细粒度子文件夹，
按应用类型合并到 8 个大类文件夹中。

合并规则：
- 前缀匹配（AUDIO_* → AUDIO，CHAT_* → CHAT，等）
- 不规范命名手动映射（tor_spotify2-* → AUDIO，tor_p2p_* → P2P）

合并方式：将子文件夹内的所有 pcap 文件复制/移动到目标大类文件夹，
重名时自动加上来源子文件夹名作为前缀，避免覆盖。

用法：
    python3 merge_tor_classes.py <input_dir> <output_dir> [--move]

示例：
    python3 merge_tor_classes.py \
        /root/autodl-tmp/Datasets/ISCX-Tor-2016/ISCX-Tor-2016_pcap/Tor_split_pcap \
        /root/autodl-tmp/Datasets/ISCX-Tor-2016/ISCX-Tor-2016_pcap/Tor_merged \
        
    # 使用 --move 移动而非复制（节省磁盘空间）
    python3 merge_tor_classes.py <input_dir> <output_dir> --move
"""

import os
import sys
import shutil
import argparse
from collections import defaultdict


# ── 手动映射：不规范命名 → 大类 ──
MANUAL_MAP = {
    "tor_spotify2-1":          "AUDIO",
    "tor_spotify2-2":          "AUDIO",
    "tor_p2p_multipleSpeed2-1": "P2P",
    "tor_p2p_vuze-2-1":        "P2P",
}

# ── 前缀映射（按前缀自动匹配） ──
PREFIX_MAP = {
    "AUDIO":         "AUDIO",
    "BROWSING":      "BROWSING",
    "CHAT":          "CHAT",
    "FILE-TRANSFER": "FILE-TRANSFER",
    "MAIL":          "MAIL",
    "P2P":           "P2P",
    "VIDEO":         "VIDEO",
    "VOIP":          "VOIP",
}


def get_category(folder_name: str) -> str:
    """根据文件夹名推断所属大类，返回大类名或 None。"""
    if folder_name in MANUAL_MAP:
        return MANUAL_MAP[folder_name]
    for prefix, category in PREFIX_MAP.items():
        if folder_name.upper().startswith(prefix.upper()):
            return category
    return None


def merge(input_dir: str, output_dir: str, move: bool = False):
    if not os.path.isdir(input_dir):
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        sys.exit(1)

    subfolders = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])

    # 统计映射情况
    category_map = {}
    unrecognized = []
    for folder in subfolders:
        cat = get_category(folder)
        if cat:
            category_map[folder] = cat
        else:
            unrecognized.append(folder)

    # 打印映射预览
    print(f"{'='*65}")
    print(f"{'子文件夹':<40} {'→':<3} {'大类'}")
    print(f"{'-'*65}")
    by_cat = defaultdict(list)
    for folder, cat in category_map.items():
        by_cat[cat].append(folder)
    for cat in sorted(by_cat):
        for folder in by_cat[cat]:
            print(f"  {folder:<38} → {cat}")
    print(f"{'='*65}")

    if unrecognized:
        print(f"\n[WARNING] 以下文件夹无法识别类别，将跳过：")
        for f in unrecognized:
            print(f"  {f}")
        print()

    # 执行合并
    op = "移动" if move else "复制"
    print(f"\n开始{op}文件...\n")

    stats = defaultdict(lambda: {"files": 0, "skipped": 0})

    for folder, cat in category_map.items():
        src_dir = os.path.join(input_dir, folder)
        dst_dir = os.path.join(output_dir, cat)
        os.makedirs(dst_dir, exist_ok=True)

        pcap_files = [f for f in os.listdir(src_dir) if f.endswith(".pcap")]
        for fname in pcap_files:
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)

            # 重名时加来源前缀
            if os.path.exists(dst_path):
                new_fname = f"{folder}__{fname}"
                dst_path  = os.path.join(dst_dir, new_fname)
                stats[cat]["skipped"] += 1

            if move:
                shutil.move(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
            stats[cat]["files"] += 1

        print(f"  [{cat}] ← {folder}: {len(pcap_files)} 个文件")

    # 汇总
    print(f"\n{'='*65}")
    print(f"{'大类':<20} {'文件数':>10} {'重名处理':>10}")
    print(f"{'-'*65}")
    total_files = 0
    for cat in sorted(stats):
        s = stats[cat]
        print(f"  {cat:<18} {s['files']:>10} {s['skipped']:>10}")
        total_files += s['files']
    print(f"{'-'*65}")
    print(f"  {'合计':<18} {total_files:>10}")
    print(f"{'='*65}")
    print(f"\n输出目录: {output_dir}")
    print(f"操作方式: {op}")
    print("完成！")


def main():
    parser = argparse.ArgumentParser(description="合并 Tor 数据集细粒度子文件夹到大类")
    parser.add_argument("input_dir",  help="原始 Tor_split_pcap 目录")
    parser.add_argument("output_dir", help="合并后的输出目录")
    parser.add_argument("--move", action="store_true",
                        help="移动文件而非复制（节省磁盘，原目录文件会消失）")
    args = parser.parse_args()
    merge(args.input_dir, args.output_dir, move=args.move)


if __name__ == "__main__":
    main()
