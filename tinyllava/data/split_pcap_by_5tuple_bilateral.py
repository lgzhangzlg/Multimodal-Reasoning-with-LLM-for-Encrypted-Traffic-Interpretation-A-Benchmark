"""
通用脚本：递归遍历指定根目录下所有 pcap 文件，按五元组拆分为子 pcap。
拆分结果保存到独立的输出目录，保留与输入目录一致的相对路径结构。

规则：
- 只处理 TCP / UDP 流量，其他协议丢弃
- 双向流合并（A->B 与 B->A 归为同一条流）
- 命名格式：{app_name}_{srcIP}_{dstIP}_{srcPort}_{dstPort}_{proto}.pcap
  （IP 中的 . 替换为 _，避免文件名歧义）

目录结构示例：
  输入：/Datasets/CrossPlatform/android/com.abc.pcap
  输出：/Datasets/CrossPlatform_split/android/com.abc/com.abc_ip1_ip2_80_443_TCP.pcap

用法：
    python3 split_pcap_by_5tuple.py <输入根目录> <输出根目录>

示例：
    python3 split_pcap_by_5tuple.py /root/autodl-tmp/Datasets/CrossPlatform \
                                    /root/autodl-tmp/Datasets/CrossPlatform_split

依赖：
    pip install scapy
"""

import os
import sys
from collections import defaultdict
from scapy.all import PcapReader, PcapWriter, IP, TCP, UDP


def get_flow_key(pkt):
    """
    从数据包中提取规范化五元组 key（双向统一）。
    返回 (src_ip, dst_ip, src_port, dst_port, proto) 或 None（非TCP/UDP则丢弃）。
    """
    if not pkt.haslayer(IP):
        return None

    ip = pkt[IP]
    proto = None
    sport, dport = 0, 0

    if pkt.haslayer(TCP):
        proto = "TCP"
        sport = pkt[TCP].sport
        dport = pkt[TCP].dport
    elif pkt.haslayer(UDP):
        proto = "UDP"
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport
    else:
        return None  # 丢弃非 TCP/UDP

    src_ip, dst_ip = ip.src, ip.dst

    # 规范化：保证双向流 key 一致
    # 比较 (ip, port) 元组，较小的放前面
    if (src_ip, sport) > (dst_ip, dport):
        src_ip, dst_ip = dst_ip, src_ip
        sport, dport = dport, sport

    return (src_ip, dst_ip, sport, dport, proto)


def split_pcap(pcap_path, input_root, output_root):
    """
    将单个 pcap 文件按五元组拆分，保存到输出目录中对应的子目录下。
    保留与输入目录一致的相对路径结构。
    """
    app_name = os.path.splitext(os.path.basename(pcap_path))[0]

    # 计算相对路径，映射到输出目录
    rel_dir = os.path.relpath(os.path.dirname(pcap_path), input_root)
    output_dir = os.path.join(output_root, rel_dir, app_name)

    # 第一步：读取所有包，按五元组分组
    flows = defaultdict(list)
    total_pkts = 0
    dropped_pkts = 0

    try:
        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                total_pkts += 1
                key = get_flow_key(pkt)
                if key is None:
                    dropped_pkts += 1
                    continue
                flows[key].append(pkt)
    except Exception as e:
        print(f"  [ERROR] 读取失败: {pcap_path} -> {e}")
        return 0

    if not flows:
        print(f"  [SKIP] 无有效 TCP/UDP 流: {pcap_path}")
        return 0

    # 第二步：创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 第三步：写出每条流
    written = 0
    for (src_ip, dst_ip, sport, dport, proto), pkts in flows.items():
        src_ip_s = src_ip.replace(".", "_")
        dst_ip_s = dst_ip.replace(".", "_")
        filename = f"{app_name}_{src_ip_s}_{dst_ip_s}_{sport}_{dport}_{proto}.pcap"
        out_path = os.path.join(output_dir, filename)

        try:
            with PcapWriter(out_path, sync=True) as writer:
                for pkt in pkts:
                    writer.write(pkt)
            written += 1
        except Exception as e:
            print(f"  [ERROR] 写出失败: {out_path} -> {e}")

    print(f"  [OK] {app_name}: "
          f"总包数={total_pkts}, 丢弃={dropped_pkts}, "
          f"流数={written}, 输出目录={output_dir}")
    return written


def find_pcap_files(root_dir):
    """
    递归遍历 root_dir，找出所有 .pcap 文件。
    """
    pcap_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".pcap"):
                pcap_files.append(os.path.join(dirpath, fname))
    return sorted(pcap_files)


def main():
    if len(sys.argv) < 3:
        print("用法: python3 split_pcap_by_5tuple.py <输入根目录> <输出根目录>")
        print("示例: python3 split_pcap_by_5tuple.py /root/autodl-tmp/Datasets/CrossPlatform \\")
        print("                                       /root/autodl-tmp/Datasets/CrossPlatform_split")
        sys.exit(1)

    input_root = sys.argv[1]
    output_root = sys.argv[2]

    if not os.path.isdir(input_root):
        print(f"[ERROR] 输入目录不存在: {input_root}")
        sys.exit(1)

    print(f"[INFO] 输入根目录: {input_root}")
    print(f"[INFO] 输出根目录: {output_root}")
    print(f"[INFO] 正在递归搜索 pcap 文件...")

    pcap_files = find_pcap_files(input_root)

    if not pcap_files:
        print("[WARN] 未找到任何 pcap 文件，退出。")
        sys.exit(0)

    print(f"[INFO] 共找到 {len(pcap_files)} 个 pcap 文件，开始处理...\n")

    total_flows = 0
    for i, pcap_path in enumerate(pcap_files, 1):
        print(f"[{i}/{len(pcap_files)}] {pcap_path}")
        flows = split_pcap(pcap_path, input_root, output_root)
        total_flows += flows

    print(f"\n{'='*60}")
    print(f"全部完成！处理文件数: {len(pcap_files)}，生成子流总数: {total_flows}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()