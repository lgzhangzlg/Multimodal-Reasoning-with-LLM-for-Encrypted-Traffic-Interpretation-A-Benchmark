#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用脚本：递归遍历指定根目录下所有 pcap 文件，按五元组拆分为子 pcap。
拆分结果保存到独立的输出目录，保留与输入目录一致的相对路径结构。

规则：
- 优先按 TCP/UDP 五元组拆分
- IPv6 TCP/UDP 同样支持
- 若某个 pcap 无任何 TCP/UDP 流量（全是 ICMP/ARP/QUIC 等），
  则将整个 pcap 作为一条流保存，保证类别不丢失
- 双向流分开保存（A->B 和 B->A 各自独立为一个子 pcap）
- 命名格式：{app_name}_{srcIP}_{dstIP}_{srcPort}_{dstPort}_{proto}.pcap

用法：
    python3 split_pcap_by_5tuple.py <输入根目录> <输出根目录> [--workers N]

示例：
    python3 split_pcap_by_5tuple.py \
        /root/autodl-tmp/Datasets/CrossPlatform/CrossPlatform_pcaps/android \
        /root/autodl-tmp/Datasets/CrossPlatform/CrossPlatform_android_split/android \
        --workers 8
"""

import os
import sys
import shutil
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def get_flow_key(pkt):
    """
    从数据包提取五元组 key，支持 IPv4 和 IPv6。
    注意：scapy import 放在函数内，避免多进程 fork 问题。
    """
    from scapy.all import IP, TCP, UDP
    try:
        from scapy.layers.inet6 import IPv6
        has_ipv6 = True
    except ImportError:
        has_ipv6 = False

    src_ip = dst_ip = None

    if IP in pkt:
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
    elif has_ipv6 and IPv6 in pkt:
        src_ip = pkt[IPv6].src
        dst_ip = pkt[IPv6].dst
    else:
        return None

    if TCP in pkt:
        return (src_ip, dst_ip, pkt[TCP].sport, pkt[TCP].dport, "TCP")
    elif UDP in pkt:
        return (src_ip, dst_ip, pkt[UDP].sport, pkt[UDP].dport, "UDP")
    else:
        return None


def split_pcap_worker(args_tuple):
    """
    子进程 worker：处理单个 pcap 文件。
    所有 scapy import 放在函数内部，避免多进程 fork 问题。
    """
    from scapy.all import PcapReader, PcapWriter

    pcap_path, input_root, output_root = args_tuple

    app_name   = os.path.splitext(os.path.basename(pcap_path))[0]
    rel_dir    = os.path.relpath(os.path.dirname(pcap_path), input_root)
    output_dir = os.path.join(output_root, rel_dir, app_name)

    flows      = defaultdict(list)
    other_pkts = []
    total_pkts = 0

    try:
        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                total_pkts += 1
                key = get_flow_key(pkt)
                if key is None:
                    other_pkts.append(pkt)
                else:
                    flows[key].append(pkt)
    except Exception as e:
        return {"app": app_name, "status": "error", "msg": str(e), "flows": 0}

    os.makedirs(output_dir, exist_ok=True)

    # ── 情况1：有 TCP/UDP 流，正常拆分 ──
    if flows:
        written = 0
        for (src_ip, dst_ip, sport, dport, proto), pkts in flows.items():
            src_ip_s = src_ip.replace(".", "_").replace(":", "_")
            dst_ip_s = dst_ip.replace(".", "_").replace(":", "_")
            filename = f"{app_name}_{src_ip_s}_{dst_ip_s}_{sport}_{dport}_{proto}.pcap"
            out_path = os.path.join(output_dir, filename)
            try:
                with PcapWriter(out_path, sync=True) as writer:
                    for pkt in pkts:
                        writer.write(pkt)
                written += 1
            except Exception:
                pass

        if other_pkts:
            fallback_path = os.path.join(output_dir, f"{app_name}_other.pcap")
            try:
                with PcapWriter(fallback_path, sync=True) as writer:
                    for pkt in other_pkts:
                        writer.write(pkt)
            except Exception:
                pass

        return {"app": app_name, "status": "ok", "flows": written,
                "total_pkts": total_pkts, "other_pkts": len(other_pkts)}

    # ── 情况2：无 TCP/UDP，保存全部包为单文件 ──
    if other_pkts:
        fallback_path = os.path.join(output_dir, f"{app_name}_all.pcap")
        try:
            with PcapWriter(fallback_path, sync=True) as writer:
                for pkt in other_pkts:
                    writer.write(pkt)
        except Exception:
            shutil.copy2(pcap_path, fallback_path)
        return {"app": app_name, "status": "fallback", "flows": 1,
                "total_pkts": total_pkts, "other_pkts": len(other_pkts)}

    # ── 情况3：pcap 完全为空 ──
    fallback_path = os.path.join(output_dir, f"{app_name}_all.pcap")
    shutil.copy2(pcap_path, fallback_path)
    return {"app": app_name, "status": "empty", "flows": 1,
            "total_pkts": 0, "other_pkts": 0}


def find_pcap_files(root_dir):
    pcap_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".pcap"):
                pcap_files.append(os.path.join(dirpath, fname))
    return sorted(pcap_files)


def main():
    parser = argparse.ArgumentParser(description="按五元组拆分 pcap 文件（多进程版）")
    parser.add_argument("input_root",  help="输入根目录")
    parser.add_argument("output_root", help="输出根目录")
    parser.add_argument("--workers", type=int, default=4,
                        help="并行进程数（默认4，建议不超过CPU核心数）")
    args = parser.parse_args()

    if not os.path.isdir(args.input_root):
        print(f"[ERROR] 输入目录不存在: {args.input_root}")
        sys.exit(1)

    pcap_files = [
        p for p in find_pcap_files(args.input_root)
        if "mitmdump" not in os.path.basename(p)
    ]

    if not pcap_files:
        print("[WARN] 未找到任何 pcap 文件，退出。")
        sys.exit(0)

    print(f"[INFO] 输入根目录: {args.input_root}")
    print(f"[INFO] 输出根目录: {args.output_root}")
    print(f"[INFO] 并行进程数: {args.workers}")
    print(f"[INFO] 共 {len(pcap_files)} 个 pcap 文件，开始处理...\n")

    tasks = [(p, args.input_root, args.output_root) for p in pcap_files]

    total_flows    = 0
    fallback_count = 0
    error_count    = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(split_pcap_worker, t): t[0] for t in tasks}
        iterator = tqdm(as_completed(futures), total=len(futures), desc="splitting") \
                   if HAS_TQDM else as_completed(futures)

        for fut in iterator:
            pcap_path = futures[fut]
            try:
                result = fut.result()
                total_flows += result["flows"]
                if result["status"] in ("fallback", "empty"):
                    fallback_count += 1
                    if not HAS_TQDM:
                        print(f"  [FALLBACK] {result['app']}: {result['status']}")
                elif result["status"] == "error":
                    error_count += 1
                    print(f"  [ERROR] {result['app']}: {result['msg']}")
            except Exception as e:
                error_count += 1
                print(f"  [ERROR] {pcap_path}: {e}")

    # 验证输出类别数
    output_classes = len([
        d for d in os.listdir(args.output_root)
        if os.path.isdir(os.path.join(args.output_root, d))
    ]) if os.path.isdir(args.output_root) else 0

    print(f"\n{'='*60}")
    print(f"处理文件数:   {len(pcap_files)}")
    print(f"生成子流总数: {total_flows}")
    print(f"fallback类别: {fallback_count}")
    print(f"错误数:       {error_count}")
    print(f"输出类别数:   {output_classes}  (期望={len(pcap_files)})")
    if output_classes != len(pcap_files):
        print(f"[WARN] 类别数不一致，差异={len(pcap_files) - output_classes}")
    else:
        print(f"[OK] 类别数完全一致！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()