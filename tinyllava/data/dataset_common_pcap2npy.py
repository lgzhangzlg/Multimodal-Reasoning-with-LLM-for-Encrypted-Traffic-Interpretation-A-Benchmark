# import os
# import numpy as np
# from scapy.all import rdpcap
# from scapy.layers.inet import IP, TCP, UDP
# from scapy.layers.inet6 import IPv6
# from scapy.layers.l2 import ARP
# from scapy.packet import Raw
#
# # ==========================
# # 全局可调参数
# # ==========================
# NUM_PACKETS = 10          # K
# BYTES_PER_PACKET = 160    # L
# HEADER_BYTES = 64         # header: 1B proto_id + 63B header_body
# PAYLOAD_BYTES = 96        # payload bytes
#
# REMOVE_IP = True          # 是否脱敏 IP/IPv6（src/dst -> 0）
# KEEP_PAYLOAD = True       # 是否保留 payload
#
# # 方案1：在 pcap->npy 阶段彻底禁用“端口捷径”
# MASK_PORT = True          # 是否处理端口
# PORT_MODE = "bucket"      # "zero" 或 "bucket"（推荐 bucket：只保留端口等级信息）
#
# # 安全：禁止 fallback 把 L2/MAC 等环境指纹塞进 header
# DISABLE_L2_FALLBACK = True
# # ==========================
#
# assert HEADER_BYTES + PAYLOAD_BYTES == BYTES_PER_PACKET, "HEADER_BYTES + PAYLOAD_BYTES must equal BYTES_PER_PACKET"
# assert HEADER_BYTES >= 1, "HEADER_BYTES must be >= 1 (needs 1 byte for proto_id)"
#
#
# def find_files(data_path, extension=".pcap"):
#     pcap_files = []
#     for root, _, files in os.walk(data_path):
#         for file in files:
#             if file.endswith(extension):
#                 pcap_files.append(os.path.join(root, file))
#     return pcap_files
#
#
# def port_bucket(p: int) -> int:
#     """1: well-known, 2: registered, 3: dynamic"""
#     if p <= 1023:
#         return 1
#     if p <= 49151:
#         return 2
#     return 3
#
#
# def get_proto_id(pkt) -> int:
#     """
#     更细粒度：区分 L3 + L4（避免只靠端口，但允许模型知道 TCP/UDP 这种“行为级”信息）
#       IPv4: 10, IPv4+TCP:11, IPv4+UDP:12
#       IPv6: 20, IPv6+TCP:21, IPv6+UDP:22
#       ARP:  3
#       other:255
#     """
#     if IP in pkt:
#         if TCP in pkt:
#             return 11
#         if UDP in pkt:
#             return 12
#         return 10
#     if IPv6 in pkt:
#         if TCP in pkt:
#             return 21
#         if UDP in pkt:
#             return 22
#         return 20
#     if ARP in pkt:
#         return 3
#     return 255
#
#
# def get_l4_payload_bytes(pkt) -> bytes:
#     """Prefer TCP/UDP payload. Raw is fallback."""
#     if not KEEP_PAYLOAD:
#         return b""
#     if TCP in pkt:
#         return bytes(pkt[TCP].payload)
#     if UDP in pkt:
#         return bytes(pkt[UDP].payload)
#     if Raw in pkt:
#         return bytes(pkt[Raw])
#     return b""
#
#
# def _mask_ports_l4(l4):
#     """Mask ports in TCP/UDP header to avoid port-based shortcut."""
#     if not MASK_PORT:
#         return l4
#     if PORT_MODE == "zero":
#         l4.sport = 0
#         l4.dport = 0
#     elif PORT_MODE == "bucket":
#         l4.sport = port_bucket(int(l4.sport))
#         l4.dport = port_bucket(int(l4.dport))
#     else:
#         raise ValueError(f"Unsupported PORT_MODE: {PORT_MODE}")
#     return l4
#
#
# def get_l3l4_header_bytes(pkt) -> bytes:
#     """
#     Build header bytes = (IP/IPv6 header only) + (TCP/UDP header only), no L4 payload.
#     IP masking is applied here (field-level), so bytes() reflects masked addresses.
#     Also masks ports (scheme1) to prevent port shortcut.
#     """
#
#     # IPv4
#     if IP in pkt:
#         ip = pkt[IP].copy()
#         if REMOVE_IP:
#             ip.src = "0.0.0.0"
#             ip.dst = "0.0.0.0"
#         ip.remove_payload()
#         l3 = bytes(ip)
#
#         l4 = b""
#         if TCP in pkt:
#             tcp = pkt[TCP].copy()
#             tcp = _mask_ports_l4(tcp)
#             tcp.remove_payload()
#             l4 = bytes(tcp)
#         elif UDP in pkt:
#             udp = pkt[UDP].copy()
#             udp = _mask_ports_l4(udp)
#             udp.remove_payload()
#             l4 = bytes(udp)
#         return l3 + l4
#
#     # IPv6
#     if IPv6 in pkt:
#         ip6 = pkt[IPv6].copy()
#         if REMOVE_IP:
#             ip6.src = "::"
#             ip6.dst = "::"
#         ip6.remove_payload()
#         l3 = bytes(ip6)
#
#         l4 = b""
#         if TCP in pkt:
#             tcp = pkt[TCP].copy()
#             tcp = _mask_ports_l4(tcp)
#             tcp.remove_payload()
#             l4 = bytes(tcp)
#         elif UDP in pkt:
#             udp = pkt[UDP].copy()
#             udp = _mask_ports_l4(udp)
#             udp.remove_payload()
#             l4 = bytes(udp)
#         return l3 + l4
#
#     # ARP
#     if ARP in pkt:
#         arp = pkt[ARP].copy()
#         # ARP 也可能泄露地址信息，这里一并脱敏
#         if REMOVE_IP:
#             try:
#                 arp.psrc = "0.0.0.0"
#                 arp.pdst = "0.0.0.0"
#             except Exception:
#                 pass
#         arp.remove_payload()
#         return bytes(arp)
#
#     # Fallback: avoid leaking L2/MAC/device fingerprints
#     if DISABLE_L2_FALLBACK:
#         return b""
#
#     # (Not recommended) raw packet head (may include link layer)
#     return bytes(pkt)
#
#
# def pack_one_packet(pkt) -> np.ndarray:
#     """
#     Pack one packet into fixed 160 bytes:
#       [1B proto_id | 63B header_body] + [96B payload]
#     """
#     proto_id = get_proto_id(pkt)
#     hdr_bytes = get_l3l4_header_bytes(pkt)
#     pay_bytes = get_l4_payload_bytes(pkt)
#
#     header_buf = bytearray(HEADER_BYTES)
#     header_buf[0] = proto_id
#     body = hdr_bytes[: HEADER_BYTES - 1]
#     header_buf[1:1 + len(body)] = body  # rest remains 0
#
#     payload_buf = (pay_bytes + b"\x00" * PAYLOAD_BYTES)[:PAYLOAD_BYTES]
#
#     out = np.frombuffer(bytes(header_buf) + payload_buf, dtype=np.uint8)
#     # out.shape == (160,)
#     return out
#
#
# def select_packets(packets, K=NUM_PACKETS):
#     """
#     Selection strategy:
#       - First 2 packets
#       - Last 2 packets
#       - Middle: pick by payload length desc to fill remaining
#       - If still short: uniform sampling fallback
#     If n < K: repeat packets to fill (avoid all-zero padding blocks).
#     """
#     n = len(packets)
#     if n == 0:
#         return []
#
#     if n >= K:
#         head = packets[:2]
#         tail = packets[-2:] if n > 2 else []
#         middle = packets[2:-2] if n > 4 else packets[2:]
#
#         need_mid = K - len(head) - len(tail)
#         middle_sorted = sorted(middle, key=lambda p: len(get_l4_payload_bytes(p)), reverse=True)
#         picked = head + middle_sorted[:need_mid] + tail
#
#         if len(picked) < K:
#             need = K - len(picked)
#             idxs = np.linspace(0, n - 1, num=need, dtype=int).tolist()
#             picked += [packets[i] for i in idxs]
#
#         return picked[:K]
#
#     # n < K: repeat to fill (avoid all-zero padding blocks)
#     picked = list(packets)
#     i = 0
#     while len(picked) < K:
#         picked.append(packets[i % n])
#         i += 1
#     return picked[:K]
#
#
# def pcap_to_npy_array(pcap_filename) -> np.ndarray:
#     packets = list(rdpcap(pcap_filename))
#     if not packets:
#         return None
#
#     selected = select_packets(packets, K=NUM_PACKETS)
#     mat = np.stack([pack_one_packet(p) for p in selected], axis=0)  # (K,160), uint8
#
#     # Flatten: (1, K*L)
#     flow = mat.reshape(1, NUM_PACKETS * BYTES_PER_PACKET).astype(np.uint8, copy=False)  # (1,1600)
#     return flow
#
#
# def process_dataset(root_dir, output_dir, extension=".pcap"):
#     pcap_files = find_files(root_dir, extension)
#     os.makedirs(output_dir, exist_ok=True)
#
#     for pcap_file in pcap_files:
#         try:
#             flow = pcap_to_npy_array(pcap_file)
#             if flow is None:
#                 print(f"Warning: {pcap_file} contains no valid packets.")
#                 continue
#
#             relative_path = os.path.relpath(pcap_file, root_dir)
#             relative_dir = os.path.dirname(relative_path)
#             out_dir = os.path.join(output_dir, relative_dir)
#             os.makedirs(out_dir, exist_ok=True)
#
#             filename = os.path.basename(pcap_file).replace(".pcap", ".npy")
#             out_path = os.path.join(out_dir, filename)
#             np.save(out_path, flow)
#
#         except Exception as e:
#             print(f"Error processing {pcap_file}: {e}")
#
#
# if __name__ == "__main__":
#     root_dir = "/root/autodl-tmp/Datasets/ISCXVPN2016/ISCXVPN2016_pcap"
#     output_dir = "/root/autodl-tmp/Datasets/ISCXVPN2016/ISCXVPN2016_npy_1600_v3_noport"
#     process_dataset(root_dir, output_dir)
import os
import time
import random
import numpy as np
from collections import defaultdict
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import ARP
from scapy.packet import Raw

# ==========================
# 全局可调参数
# ==========================
NUM_PACKETS = 10          # K
BYTES_PER_PACKET = 160    # L
HEADER_BYTES = 64         # header: 1B proto_id + 63B header_body
PAYLOAD_BYTES = 96        # payload bytes

REMOVE_IP = True
KEEP_PAYLOAD = True

MASK_PORT = True
PORT_MODE = "bucket"

DISABLE_L2_FALLBACK = True

# ==========================
# 类别均衡参数
# ==========================
MAX_SAMPLES_PER_CLASS = 5000  # N: 每个类别最多保留的文件数，多截断少保持原样
RANDOM_SEED = 42

# ==========================

assert HEADER_BYTES + PAYLOAD_BYTES == BYTES_PER_PACKET
assert HEADER_BYTES >= 1


def format_time(seconds):
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def find_files(data_path, extension=".pcap"):
    pcap_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(extension):
                pcap_files.append(os.path.join(root, file))
    return pcap_files


def port_bucket(p: int) -> int:
    if p <= 1023:   return 1
    if p <= 49151:  return 2
    return 3


def get_proto_id(pkt) -> int:
    if IP in pkt:
        if TCP in pkt: return 11
        if UDP in pkt: return 12
        return 10
    if IPv6 in pkt:
        if TCP in pkt: return 21
        if UDP in pkt: return 22
        return 20
    if ARP in pkt: return 3
    return 255


def get_l4_payload_bytes(pkt) -> bytes:
    if not KEEP_PAYLOAD: return b""
    if TCP in pkt: return bytes(pkt[TCP].payload)
    if UDP in pkt: return bytes(pkt[UDP].payload)
    if Raw in pkt: return bytes(pkt[Raw])
    return b""


def _mask_ports_l4(l4):
    if not MASK_PORT: return l4
    if PORT_MODE == "zero":
        l4.sport = 0; l4.dport = 0
    elif PORT_MODE == "bucket":
        l4.sport = port_bucket(int(l4.sport))
        l4.dport = port_bucket(int(l4.dport))
    else:
        raise ValueError(f"Unsupported PORT_MODE: {PORT_MODE}")
    return l4


def get_l3l4_header_bytes(pkt) -> bytes:
    if IP in pkt:
        ip = pkt[IP].copy()
        if REMOVE_IP: ip.src = "0.0.0.0"; ip.dst = "0.0.0.0"
        ip.remove_payload(); l3 = bytes(ip); l4 = b""
        if TCP in pkt:
            tcp = pkt[TCP].copy(); tcp = _mask_ports_l4(tcp); tcp.remove_payload(); l4 = bytes(tcp)
        elif UDP in pkt:
            udp = pkt[UDP].copy(); udp = _mask_ports_l4(udp); udp.remove_payload(); l4 = bytes(udp)
        return l3 + l4
    if IPv6 in pkt:
        ip6 = pkt[IPv6].copy()
        if REMOVE_IP: ip6.src = "::"; ip6.dst = "::"
        ip6.remove_payload(); l3 = bytes(ip6); l4 = b""
        if TCP in pkt:
            tcp = pkt[TCP].copy(); tcp = _mask_ports_l4(tcp); tcp.remove_payload(); l4 = bytes(tcp)
        elif UDP in pkt:
            udp = pkt[UDP].copy(); udp = _mask_ports_l4(udp); udp.remove_payload(); l4 = bytes(udp)
        return l3 + l4
    if ARP in pkt:
        arp = pkt[ARP].copy()
        if REMOVE_IP:
            try: arp.psrc = "0.0.0.0"; arp.pdst = "0.0.0.0"
            except: pass
        arp.remove_payload(); return bytes(arp)
    if DISABLE_L2_FALLBACK: return b""
    return bytes(pkt)


def pack_one_packet(pkt) -> np.ndarray:
    proto_id = get_proto_id(pkt)
    hdr_bytes = get_l3l4_header_bytes(pkt)
    pay_bytes = get_l4_payload_bytes(pkt)
    header_buf = bytearray(HEADER_BYTES)
    header_buf[0] = proto_id
    body = hdr_bytes[: HEADER_BYTES - 1]
    header_buf[1:1 + len(body)] = body
    payload_buf = (pay_bytes + b"\x00" * PAYLOAD_BYTES)[:PAYLOAD_BYTES]
    return np.frombuffer(bytes(header_buf) + payload_buf, dtype=np.uint8)


def select_packets(packets, K=NUM_PACKETS):
    n = len(packets)
    if n == 0: return []
    if n >= K:
        head = packets[:2]
        tail = packets[-2:] if n > 2 else []
        middle = packets[2:-2] if n > 4 else packets[2:]
        need_mid = K - len(head) - len(tail)
        middle_sorted = sorted(middle, key=lambda p: len(get_l4_payload_bytes(p)), reverse=True)
        picked = head + middle_sorted[:need_mid] + tail
        if len(picked) < K:
            idxs = np.linspace(0, n - 1, num=K - len(picked), dtype=int).tolist()
            picked += [packets[i] for i in idxs]
        return picked[:K]
    picked = list(packets)
    i = 0
    while len(picked) < K:
        picked.append(packets[i % n]); i += 1
    return picked[:K]


def pcap_to_npy_array(pcap_filename) -> np.ndarray:
    packets = list(rdpcap(pcap_filename))
    if not packets: return None
    selected = select_packets(packets, K=NUM_PACKETS)
    mat = np.stack([pack_one_packet(p) for p in selected], axis=0)
    return mat.reshape(1, NUM_PACKETS * BYTES_PER_PACKET).astype(np.uint8, copy=False)


def get_class_from_path(pcap_file, root_dir):
    """第一级子文件夹名作为类别"""
    parts = os.path.relpath(pcap_file, root_dir).split(os.sep)
    return parts[0] if len(parts) > 1 else "unknown"


def process_dataset(root_dir, output_dir, N=MAX_SAMPLES_PER_CLASS, extension=".pcap"):
    # ---- Phase 1: 扫描 & 按类别分组 ----
    print("=" * 60)
    print("Phase 1: Scanning and grouping by class...")
    print("=" * 60)

    all_files = find_files(root_dir, extension)
    class_to_files = defaultdict(list)
    for f in all_files:
        class_to_files[get_class_from_path(f, root_dir)].append(f)

    # ---- Phase 2: 每类截断到 N（不足则保持原样） ----
    rng = random.Random(RANDOM_SEED)
    task_list = []

    print(f"\n  {'Class':<30s} {'Original':>10s} {'Keep':>10s} {'Action':>15s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*15}")

    for cls in sorted(class_to_files.keys()):
        files = class_to_files[cls]
        orig = len(files)
        if orig > N:
            selected = rng.sample(files, N)
            action = f"truncate -{orig - N}"
        else:
            selected = files
            action = "keep all"
        task_list.extend(selected)
        print(f"  {cls:<30s} {orig:>10d} {len(selected):>10d} {action:>15s}")

    print(f"\n  Total files to process: {len(task_list)}")

    # ---- Phase 3: 转换 ----
    print(f"\n{'='*60}")
    print(f"Phase 3: Converting pcap -> npy ...")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    total = len(task_list)
    success = error = skip = 0
    t0 = time.time()

    for i, pcap_file in enumerate(task_list, 1):
        try:
            flow = pcap_to_npy_array(pcap_file)
            if flow is None:
                skip += 1; status = "SKIP"
            else:
                rel_path = os.path.relpath(pcap_file, root_dir)
                out_path = os.path.join(output_dir, os.path.dirname(rel_path))
                os.makedirs(out_path, exist_ok=True)
                npy_name = os.path.basename(pcap_file).replace(extension, ".npy")
                np.save(os.path.join(out_path, npy_name), flow)
                success += 1; status = "OK"
        except Exception as e:
            error += 1; status = f"ERR: {e}"

        elapsed = time.time() - t0
        eta = elapsed / i * (total - i)
        msg = (
            f"\r[{i}/{total}] {i/total*100:5.1f}% | "
            f"Elapsed: {format_time(elapsed)} | "
            f"ETA: {format_time(eta)} | "
            f"{i/elapsed:.1f} files/s | {status}"
        )
        print(f"{msg:<120}", end="", flush=True)

        if i % 500 == 0:
            print(f"\n  [Checkpoint] OK:{success} Skip:{skip} Err:{error}")

    total_time = time.time() - t0
    print(f"\n\n{'='*60}")
    print(f"Done!  OK:{success}  Skip:{skip}  Err:{error}  Time:{format_time(total_time)}")
    print("=" * 60)


if __name__ == "__main__":
    root_dir = "/root/autodl-tmp/Datasets/USTC-TFC-2016/USTC-TFC-2016_pcap"
    output_dir = "/root/autodl-tmp/Datasets/USTC-TFC-2016/USTC-TFC-2016_npy_1600_v3_noport_balanced"

    MAX_SAMPLES_PER_CLASS = 5000

    process_dataset(root_dir, output_dir, N=MAX_SAMPLES_PER_CLASS)