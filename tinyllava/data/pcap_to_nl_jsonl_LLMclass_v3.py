#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified PCAP Processing Pipeline
=================================
完整数据处理链路：
  pcap → npy（落盘）→ jsonl（含字节特征 + 知识库描述）→ train/test 划分

v2 改动：
  - build_user_text 使用 <cls_placeholder> 占位符，训练/推理时在线替换为类别 special token
  - build_target 去掉 class 字段，由推理时代码拼接
  - build_evidence 强制 2~4 条，新增 stats 参数，更丰富的字节层证据
  - build_description 强制 2~3 句，覆盖字节特征/协议说明/流量行为
  - build_notes 聚焦安全相关，不再随机拼接 behavior
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tinyllava.utils.constants import DEFAULT_IMAGE_TOKEN
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from scapy.all import rdpcap
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6
    from scapy.layers.l2 import ARP
    from scapy.packet import Raw
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False

try:
    import pyshark
    HAS_PYSHARK = True
except ImportError:
    HAS_PYSHARK = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# ============================================================
# 全局参数
# ============================================================
NUM_PACKETS    = 10
BYTES_PER_PKT  = 160
HEADER_BYTES   = 64
PAYLOAD_BYTES  = 96
REMOVE_IP      = True
KEEP_PAYLOAD   = True
MASK_PORT      = True
PORT_MODE      = "bucket"
DISABLE_L2_FB  = True
MIN_CLASS_SAMPLES = 100

assert HEADER_BYTES + PAYLOAD_BYTES == BYTES_PER_PKT


# ============================================================
# 类别标签映射与过滤（CSTNet / CrossPlatform 清理用）
# ============================================================

LABEL_MAP: Dict[str, str] = {}   # old_dir_name -> new_label
REMOVE_SET: set = set()          # 要跳过的类别目录名


def load_label_map(path: str) -> Tuple[Dict[str, str], set]:
    """
    从 final_category_mapping.json 加载:
      - rename mapping:  old_dir_name -> new_label
      - removed list:    需要删除的类别集合
    """
    if not path or not os.path.exists(path):
        return {}, set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rename: Dict[str, str] = {}
    removed: set = set()
    for dataset_info in data.values():
        mapping = dataset_info.get("mapping", {})
        for old_key, val in mapping.items():
            # val 可能是 {"label": "xxx", "method": "known"} 或直接 "xxx"
            if isinstance(val, dict):
                rename[old_key] = val.get("label", old_key)
            else:
                rename[old_key] = val
        removed.update(dataset_info.get("removed", []))
    return rename, removed


def map_class(raw_cls: str) -> str:
    """将 PCAP 目录名映射为新标签名，找不到则原样返回"""
    return LABEL_MAP.get(raw_cls, raw_cls)


# ============================================================
# 知识库：Claude API 自动生成
# ============================================================

KB_PROMPT_TEMPLATE = """\
你是一位资深网络安全专家，请为以下网络流量类别生成结构化知识库。

类别列表：{class_list}

请为每个类别输出以下字段：
- protocol_hint: 一句话说明这是什么协议/应用（英文，20词以内）
- behaviors: 3~5条关键行为特征，列表形式（英文）
- packet_profile: 典型包大小、吞吐量、持续时间的描述（英文，30词以内）
- security_context: 从安全分析角度的备注或建议（英文，30词以内）
- distinguishing_from_similar: 与易混淆类别的区分要点（英文，30词以内）
- descriptions: 3条不同角度的自然语言描述，各50词左右（英文）

输出格式为标准 JSON 对象，key 为类别名，value 为上述字段的对象。
不要输出任何说明文字，只输出 JSON。
"""

KB_FALLBACK_DYNAMIC_FIELDS = {
    "protocol_hint": "Unknown traffic category",
    "behaviors": ["unclassified network traffic"],
    "packet_profile": "Variable packet sizes and throughput",
    "security_context": "Further analysis required to determine security implications.",
    "distinguishing_from_similar": "Insufficient information to distinguish from similar categories.",
    "descriptions": [
        "This traffic category has not been characterized in the knowledge base.",
        "Network flows in this category exhibit unclassified behavior patterns.",
        "Further domain expertise is required to describe this traffic type.",
    ],
}


def load_kb_cache(kb_path: str) -> Dict[str, Any]:
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_kb_cache(kb_path: str, kb: Dict[str, Any]):
    os.makedirs(os.path.dirname(os.path.abspath(kb_path)), exist_ok=True)
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)


def generate_kb_for_classes(
    classes: List[str],
    api_key: str,
    kb_cache: Dict[str, Any],
    kb_path: str,
) -> Dict[str, Any]:
    missing = [c for c in classes if c not in kb_cache]
    if not missing:
        print(f"  [KB] All {len(classes)} classes found in cache.")
        return kb_cache

    print(f"  [KB] Generating knowledge for {len(missing)} new classes: {missing}")

    if not HAS_ANTHROPIC:
        print("  [KB] anthropic not installed, using fallback descriptions.")
        for cls in missing:
            kb_cache[cls] = dict(KB_FALLBACK_DYNAMIC_FIELDS)
            kb_cache[cls]["descriptions"] = [
                f"{cls} traffic: unclassified category, no knowledge base entry available.",
                f"Network flows labeled as {cls} require further analysis.",
                f"Traffic patterns for {cls} have not been characterized.",
            ]
        save_kb_cache(kb_path, kb_cache)
        return kb_cache

    client = anthropic.Anthropic(api_key=api_key)
    prompt = KB_PROMPT_TEMPLATE.format(class_list=", ".join(missing))

    try:
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        generated = json.loads(raw)
        for cls, entry in generated.items():
            kb_cache[cls] = entry
            print(f"    [KB] Generated: {cls}")
        for cls in missing:
            if cls not in kb_cache:
                print(f"    [KB] Fallback: {cls}")
                kb_cache[cls] = _build_dynamic_fallback(cls)
    except Exception as e:
        print(f"  [KB] API error: {e}. Using fallback for all missing classes.")
        for cls in missing:
            kb_cache[cls] = _build_dynamic_fallback(cls)

    save_kb_cache(kb_path, kb_cache)
    return kb_cache


def _build_dynamic_fallback(cls: str) -> Dict[str, Any]:
    entry = dict(KB_FALLBACK_DYNAMIC_FIELDS)
    entry["descriptions"] = [
        f"{cls} traffic: unclassified category.",
        f"Network flows labeled as {cls} require further domain analysis.",
        f"Traffic patterns for {cls} have not been characterized in the knowledge base.",
    ]
    return entry


# ============================================================
# pcap → npy
# ============================================================

def port_bucket(p: int) -> int:
    if p <= 1023:  return 1
    if p <= 49151: return 2
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


def _mask_ports(l4):
    if not MASK_PORT: return l4
    if PORT_MODE == "zero":
        l4.sport = 0; l4.dport = 0
    elif PORT_MODE == "bucket":
        l4.sport = port_bucket(int(l4.sport))
        l4.dport = port_bucket(int(l4.dport))
    return l4


def get_l3l4_header_bytes(pkt) -> bytes:
    if IP in pkt:
        ip = pkt[IP].copy()
        if REMOVE_IP: ip.src = "0.0.0.0"; ip.dst = "0.0.0.0"
        ip.remove_payload(); l3 = bytes(ip); l4 = b""
        if TCP in pkt:
            tcp = pkt[TCP].copy(); tcp = _mask_ports(tcp); tcp.remove_payload(); l4 = bytes(tcp)
        elif UDP in pkt:
            udp = pkt[UDP].copy(); udp = _mask_ports(udp); udp.remove_payload(); l4 = bytes(udp)
        return l3 + l4
    if IPv6 in pkt:
        ip6 = pkt[IPv6].copy()
        if REMOVE_IP: ip6.src = "::"; ip6.dst = "::"
        ip6.remove_payload(); l3 = bytes(ip6); l4 = b""
        if TCP in pkt:
            tcp = pkt[TCP].copy(); tcp = _mask_ports(tcp); tcp.remove_payload(); l4 = bytes(tcp)
        elif UDP in pkt:
            udp = pkt[UDP].copy(); udp = _mask_ports(udp); udp.remove_payload(); l4 = bytes(udp)
        return l3 + l4
    if ARP in pkt:
        arp = pkt[ARP].copy()
        if REMOVE_IP:
            try: arp.psrc = "0.0.0.0"; arp.pdst = "0.0.0.0"
            except: pass
        arp.remove_payload(); return bytes(arp)
    if DISABLE_L2_FB: return b""
    return bytes(pkt)


def pack_one_packet(pkt) -> np.ndarray:
    proto_id  = get_proto_id(pkt)
    hdr_bytes = get_l3l4_header_bytes(pkt)
    pay_bytes = get_l4_payload_bytes(pkt)
    hbuf = bytearray(HEADER_BYTES)
    hbuf[0] = proto_id
    body = hdr_bytes[:HEADER_BYTES - 1]
    hbuf[1:1 + len(body)] = body
    pbuf = (pay_bytes + b"\x00" * PAYLOAD_BYTES)[:PAYLOAD_BYTES]
    return np.frombuffer(bytes(hbuf) + pbuf, dtype=np.uint8)


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


def pcap_to_npy_array(pcap_path: str) -> Optional[np.ndarray]:
    if not HAS_SCAPY:
        raise RuntimeError("scapy not installed.")
    packets = list(rdpcap(pcap_path))
    if not packets: return None
    selected = select_packets(packets, K=NUM_PACKETS)
    mat = np.stack([pack_one_packet(p) for p in selected], axis=0)
    return mat.reshape(1, NUM_PACKETS * BYTES_PER_PKT).astype(np.uint8, copy=False)


# ============================================================
# pcap 统计
# ============================================================

def analyze_pcap_stats(pcap_path: str, topk: int = 5) -> Dict[str, Any]:
    if not HAS_SCAPY:
        return _empty_stats()
    try:
        packets = rdpcap(pcap_path)
    except Exception:
        return _empty_stats()
    if not packets: 
        return _empty_stats()

    protocol_counts: Dict[str, int] = {}
    app_bytes: Dict[str, int] = {}
    total_bytes = 0
    first_ts = last_ts = None

    for pkt in packets:
        try:
            frame_len = len(pkt)
            total_bytes += frame_len
            if pkt.haslayer("TCP"):
                if pkt.haslayer("TLS") or (pkt.haslayer("Raw") and _looks_like_tls(bytes(pkt["TCP"].payload))):
                    proto = "TLS"
                elif pkt.haslayer("HTTP"):
                    proto = "HTTP"
                else:
                    proto = "TCP"
            elif pkt.haslayer("UDP"):
                if pkt.haslayer("DNS"):
                    proto = "DNS"
                elif pkt.haslayer("RTP"):
                    proto = "RTP"
                else:
                    proto = "UDP"
            elif pkt.haslayer("ICMP"):
                proto = "ICMP"
            else:
                proto = "OTHER"
            protocol_counts[proto] = protocol_counts.get(proto, 0) + 1
            app_bytes[proto] = app_bytes.get(proto, 0) + frame_len
            try:
                ts = float(pkt.time)
                if first_ts is None: first_ts = ts
                last_ts = ts
            except: pass
        except Exception:
            continue

    packet_count = len(packets)
    duration   = max(0.0, float(last_ts or 0) - float(first_ts or 0))
    avg_size   = total_bytes / packet_count if packet_count else 0.0
    throughput = total_bytes / duration if duration > 0 else 0.0
    top_items  = sorted(app_bytes.items(), key=lambda kv: (-kv[1], kv[0]))[:topk]
    app_bytes_topk = [
        {"app": n, "bytes": b,
         "share_pct": round(b / total_bytes * 100, 3) if total_bytes else 0.0}
        for n, b in top_items
    ]
    return {
        "packet_count":     packet_count,
        "total_bytes":      total_bytes,
        "duration_seconds": round(duration, 6),
        "avg_packet_size":  round(avg_size, 3),
        "throughput_Bps":   round(throughput, 3),
        "protocol_counts":  protocol_counts,
        "app_bytes_topk":   app_bytes_topk,
    }


def _looks_like_tls(payload: bytes) -> bool:
    if len(payload) < 3:
        return False
    return payload[0] in (0x14, 0x15, 0x16, 0x17) and payload[1] == 0x03


def _empty_stats() -> Dict[str, Any]:
    return {
        "packet_count": 0, "total_bytes": 0, "duration_seconds": 0.0,
        "avg_packet_size": 0.0, "throughput_Bps": 0.0,
        "protocol_counts": {}, "app_bytes_topk": [],
    }


# ============================================================
# 字节特征提取
# ============================================================

def load_npy_uint8(path: str) -> np.ndarray:
    x = np.load(path)
    x = np.asarray(x)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x.reshape(-1)


def shannon_entropy(arr: np.ndarray) -> float:
    if arr.size == 0: return 0.0
    hist = np.bincount(arr, minlength=256).astype(np.float64)
    s = hist.sum()
    if s <= 0: return 0.0
    p = hist / s; p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def ascii_ratio(arr: np.ndarray) -> float:
    if arr.size == 0: return 0.0
    return float(((arr >= 0x20) & (arr <= 0x7E)).sum()) / arr.size


def zero_pad_ratio(arr: np.ndarray) -> float:
    if arr.size == 0: return 1.0
    return float((arr == 0).sum()) / arr.size


def has_tls_record(arr: np.ndarray) -> bool:
    b = arr.tobytes()
    for ct in (0x14, 0x15, 0x16, 0x17):
        for ver in (0x00, 0x01, 0x02, 0x03, 0x04):
            if bytes([ct, 0x03, ver]) in b: return True
    return False


def has_http_method(arr: np.ndarray) -> bool:
    b = arr.tobytes()
    return any(t in b for t in [b"GET ", b"POST ", b"HEAD ", b"PUT ", b"HTTP/1.", b"Host:", b"User-Agent:"])


def bucketize(val: float, p33: float, p66: float) -> str:
    if val <= p33: return "low"
    if val <= p66: return "mid"
    return "high"


def compute_thresholds(npy_paths: List[str], max_samples: int = 20000, seed: int = 42) -> Dict[str, float]:
    rng = random.Random(seed)
    paths = rng.sample(npy_paths, min(max_samples, len(npy_paths)))
    ascii_v, ent_v, zero_v = [], [], []
    for p in paths:
        try:
            flat = load_npy_uint8(p)
            nz = flat[flat != 0]
            ascii_v.append(ascii_ratio(nz))
            ent_v.append(shannon_entropy(nz))
            zero_v.append(zero_pad_ratio(flat))
        except: pass

    def pct(v, q):
        return float(np.percentile(np.array(v, dtype=np.float64), q)) if v else 0.0

    return {
        "ascii_p33": pct(ascii_v, 33), "ascii_p66": pct(ascii_v, 66),
        "ent_p33":   pct(ent_v,   33), "ent_p66":   pct(ent_v,   66),
        "zero_p33":  pct(zero_v,  33), "zero_p66":  pct(zero_v,  66),
    }


def extract_byte_traits(flat: np.ndarray, thr: Dict[str, float]) -> Dict[str, Any]:
    nz = flat[flat != 0]
    return OrderedDict([
        ("has_tls_record",        bool(has_tls_record(flat))),
        ("has_http_method",       bool(has_http_method(flat))),
        ("ascii_ratio_bucket",    bucketize(ascii_ratio(nz),    thr["ascii_p33"], thr["ascii_p66"])),
        ("entropy_bucket",        bucketize(shannon_entropy(nz), thr["ent_p33"],   thr["ent_p66"])),
        ("zero_pad_ratio_bucket", bucketize(zero_pad_ratio(flat), thr["zero_p33"], thr["zero_p66"])),
    ])


# ============================================================
# stats 数值语义化
# ============================================================

def semanticize_stats(stats: Dict[str, Any], class_name: str = "") -> str:
    parts = []
    avg_size   = stats.get("avg_packet_size", 0.0)
    throughput = stats.get("throughput_Bps", 0.0)
    topk       = stats.get("app_bytes_topk", [])
    pkt_count  = stats.get("packet_count", 0)

    SMALL_PKT_CLASSES = {"Zeus", "Neris", "Virut", "Shifu", "Geodo", "Htbot", "Miuref", "Nsis-ay"}
    LARGE_PKT_CLASSES = {"BitTorrent", "Gmail", "Outlook", "WorldOfWarcraft", "Skype", "Facetime"}

    if pkt_count >= 5:
        if avg_size > 800 and class_name not in SMALL_PKT_CLASSES:
            parts.append(f"Large average packet size ({avg_size:.0f} bytes) suggests bulk data transfer.")
        elif avg_size < 100 and class_name not in LARGE_PKT_CLASSES:
            parts.append(f"Very small average packet size ({avg_size:.0f} bytes) consistent with C&C beacon or keepalive.")
        elif 100 <= avg_size <= 800:
            parts.append(f"Moderate average packet size ({avg_size:.0f} bytes) over {pkt_count} packets.")

    if throughput > 1_000_000:
        parts.append(f"High throughput ({throughput/1e6:.1f} MB/s) indicates active data exchange.")
    elif throughput > 0 and pkt_count >= 3:
        parts.append(f"Low throughput ({throughput:.0f} B/s) suggests sparse or control-plane traffic.")

    if topk:
        parts.append(f"Dominant protocol: {topk[0]['app']} ({topk[0]['share_pct']:.1f}% of bytes).")

    return " ".join(parts)


# ============================================================
# target 构建（v2：更丰富的 evidence/description/notes，去掉 class）
# ============================================================

def build_evidence(
    traits: Dict[str, Any],
    kb_entry: Dict[str, Any],
    stats: Dict[str, Any],
) -> List[str]:
    ev = []

    # ── 客观字节证据（基于 traits）──
    if traits["has_tls_record"]:
        ev.append(
            "TLS record header pattern detected: content-type byte (0x14~0x17) "
            "followed by version byte 0x03xx, indicating an encrypted TLS session."
        )
    if traits["has_http_method"]:
        ev.append(
            "HTTP tokens present in payload window (e.g., GET/POST/HTTP/1.x/Host:/User-Agent:), "
            "indicating plaintext HTTP application-layer communication."
        )

    # 熵 + ASCII 组合判断
    if traits["ascii_ratio_bucket"] == "low" and traits["entropy_bucket"] == "high":
        ev.append(
            "Low ASCII ratio combined with high Shannon entropy strongly indicates "
            "encrypted or compressed payload content."
        )
    elif traits["ascii_ratio_bucket"] == "high" and traits["entropy_bucket"] == "low":
        ev.append(
            "High ASCII ratio with low entropy suggests repetitive plaintext content, "
            "consistent with structured text-based protocols."
        )
    elif traits["ascii_ratio_bucket"] == "high":
        ev.append(
            "High proportion of printable ASCII bytes suggests plaintext "
            "application-layer content is present in the payload."
        )
    elif traits["entropy_bucket"] == "high":
        ev.append(
            "High Shannon entropy in the non-zero payload region indicates "
            "the data is likely encrypted, compressed, or binary."
        )

    # 零填充
    if traits["zero_pad_ratio_bucket"] == "high":
        ev.append(
            "High zero-padding ratio indicates this is a short flow: "
            "the actual payload occupies only a small portion of the fixed capture window."
        )
    elif traits["zero_pad_ratio_bucket"] == "low":
        ev.append(
            "Low zero-padding ratio indicates sustained payload activity "
            "across the full capture window, consistent with bulk data transfer."
        )

    # ── 协议统计证据（基于 stats）──
    topk = stats.get("app_bytes_topk", [])
    if topk:
        top = topk[0]
        ev.append(
            f"Dominant protocol is {top['app']} ({top['share_pct']:.1f}% of bytes), "
            f"consistent with the expected protocol profile of this traffic category."
        )

    # ── 知识库判别证据 ──
    distinguish = kb_entry.get("distinguishing_from_similar", "")
    if distinguish:
        ev.append(f"Distinguishing characteristic from similar categories: {distinguish}")

    # 强制至少 2 条，不足时从知识库 behaviors 补充
    if len(ev) < 2:
        behaviors = kb_entry.get("behaviors", [])
        for b in behaviors:
            if len(ev) >= 2:
                break
            ev.append(f"Expected behavioral pattern: {b}")

    return ev[:4]  # 最多 4 条


def build_description(
    traits: Dict[str, Any],
    kb_entry: Dict[str, Any],
    stats: Dict[str, Any],
    class_name: str = "",
) -> str:
    sentences = []

    # ── 句1：字节特征描述 ──
    if traits["has_tls_record"]:
        if traits["entropy_bucket"] == "high":
            sentences.append(
                "The byte window exhibits TLS record header patterns alongside high-entropy payload, "
                "confirming active encrypted communication."
            )
        else:
            sentences.append(
                "TLS record header patterns are present in the byte window, "
                "indicating an encrypted transport layer."
            )
    elif traits["has_http_method"]:
        sentences.append(
            "The byte window contains HTTP method tokens and readable ASCII content, "
            "indicating unencrypted HTTP application-layer traffic."
        )
    elif traits["ascii_ratio_bucket"] == "low" and traits["entropy_bucket"] == "high":
        sentences.append(
            "The byte window is dominated by high-entropy, low-ASCII content, "
            "consistent with an encrypted or compressed data stream."
        )
    elif traits["ascii_ratio_bucket"] == "high":
        sentences.append(
            "The byte window contains substantial readable ASCII content, "
            "suggesting plaintext or lightly encoded application data."
        )
    else:
        sentences.append(
            "The byte window shows mixed entropy and ASCII characteristics "
            "across the sampled packet slices."
        )

    # ── 句2：协议/应用说明（来自知识库）──
    protocol_hint = kb_entry.get("protocol_hint", "")
    if protocol_hint:
        sentences.append(protocol_hint + ".")
    else:
        profile = kb_entry.get("packet_profile", "")
        if profile:
            sentences.append(f"Typical traffic profile: {profile}.")

    # ── 句3：流量行为特征（stats + 知识库兜底）──
    stats_desc = semanticize_stats(stats, class_name=class_name)
    avg_size   = stats.get("avg_packet_size", 0.0)
    pkt_count  = stats.get("packet_count", 0)

    if stats_desc:
        sentences.append(stats_desc)
    elif traits["zero_pad_ratio_bucket"] == "high":
        sentences.append(
            "The high zero-padding ratio suggests this flow is short-lived "
            "or carries minimal payload per session."
        )
    elif avg_size > 800 and pkt_count >= 5:
        sentences.append(
            "Large average packet size indicates sustained bulk data transfer "
            "typical of this traffic category."
        )

    # 强制至少 2 句，不足时从知识库 descriptions 补一句
    if len(sentences) < 2:
        descs = kb_entry.get("descriptions", [])
        if descs:
            sentences.append(random.choice(descs))

    return " ".join(sentences[:3])  # 最多 3 句


def build_notes(kb_entry: Dict[str, Any], rng: random.Random) -> str:
    ctx         = kb_entry.get("security_context", "")
    behaviors   = kb_entry.get("behaviors", [])
    distinguish = kb_entry.get("distinguishing_from_similar", "")

    # 优先用 security_context，最聚焦安全
    if ctx:
        if len(ctx) < 60 and behaviors:
            behavior = rng.choice(behaviors)
            return f"{ctx} Notably, {behavior.lower()}."
        return ctx

    # security_context 为空时用 distinguishing 构造安全备注
    if distinguish:
        return (
            f"When analyzing this traffic, note that: {distinguish.lower()} "
            f"Misclassification risk should be considered during security monitoring."
        )

    # 最后兜底：用 behavior 构造
    if behaviors:
        behavior = rng.choice(behaviors)
        return f"Security note: this traffic is characterized by {behavior.lower()}."

    return "No specific security notes available for this traffic category."


def build_target(
    class_name: str,
    traits: Dict[str, Any],
    kb_entry: Dict[str, Any],
    stats: Dict[str, Any],
    rng: random.Random,
) -> str:

    target = OrderedDict([
        ("class", class_name),
        ("traits", traits),
        ("evidence", build_evidence(traits, kb_entry, stats)),
        ("description", build_description(traits, kb_entry, stats, class_name=class_name)),
        ("notes", build_notes(kb_entry, rng)),
    ])
    return json.dumps(target, ensure_ascii=False, separators=(",", ": "))



def build_user_text(class_list: List[str]) -> str:

    classes_str = ", ".join(sorted(class_list))
    return (
        f"{DEFAULT_IMAGE_TOKEN}\n"
        "Above are the raw traffic byte features extracted from a network flow.\n\n"
        "return a single JSON object with the following keys:\n\n"
        f"- class: the traffic category name. Must be exactly one of: {classes_str}.\n\n"
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
# 数据集扫描 & 均衡采样
# ============================================================

def scan_dataset(pcap_root: str) -> Dict[str, List[str]]:
    class_to_files: Dict[str, List[str]] = defaultdict(list)
    total = 0
    for root, _, files in os.walk(pcap_root):
        for f in files:
            if f.endswith(".pcap"):
                full = os.path.join(root, f)
                rel  = os.path.relpath(full, pcap_root)
                cls  = rel.split(os.sep)[0]
                class_to_files[cls].append(full)
                total += 1
                if total % 50000 == 0:
                    print(f"  [scan] {total} pcap files found so far...", flush=True)
    print(f"  [scan] Done: {total} pcap files total.")
    return dict(class_to_files)


def filter_and_balance(
    class_to_files: Dict[str, List[str]],
    max_per_class: int,
    min_samples: int,
    seed: int,
) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    result: Dict[str, List[str]] = {}

    print("\n" + "=" * 65)
    print("Phase 1: Scanning and filtering classes")
    print("=" * 65)
    print(f"  {'Class':<30} {'Original':>10} {'Keep':>10} {'Action':>12}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*12}")

    for cls in sorted(class_to_files.keys()):
        files = class_to_files[cls]
        orig  = len(files)
        if orig < min_samples:
            action = f"SKIP (<{min_samples})"
            print(f"  {cls:<30} {orig:>10} {'0':>10} {action:>12}")
            continue
        if orig > max_per_class:
            selected = rng.sample(files, max_per_class)
            action = f"truncate -{orig - max_per_class}"
        else:
            selected = files
            action = "keep all"
        result[cls] = selected
        print(f"  {cls:<30} {orig:>10} {len(selected):>10} {action:>12}")

    print(f"\n  Remaining classes: {len(result)}, Total files: {sum(len(v) for v in result.values())}")
    return result


def split_train_test(
    class_to_files: Dict[str, List[str]],
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    train_files, test_files = [], []
    for cls, files in class_to_files.items():
        shuffled = list(files)
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_ratio))
        test_files.extend(shuffled[:n_test])
        train_files.extend(shuffled[n_test:])
    rng.shuffle(train_files)
    rng.shuffle(test_files)
    return train_files, test_files


# ============================================================
# 并行 pcap → npy
# ============================================================

def _npy_worker(args_tuple):
    pcap_path, pcap_root, npy_root = args_tuple
    rel      = os.path.relpath(pcap_path, pcap_root)
    # 将目录名映射为新标签，NPY输出到新标签目录下
    parts    = rel.split(os.sep)
    parts[0] = map_class(parts[0])
    npy_rel  = os.path.splitext(os.sep.join(parts))[0] + ".npy"
    npy_path = os.path.join(npy_root, npy_rel)
    if os.path.exists(npy_path):
        return {"pcap": pcap_path, "npy": npy_path, "status": "exists"}
    try:
        arr = pcap_to_npy_array(pcap_path)
        if arr is None:
            return {"pcap": pcap_path, "npy": None, "status": "empty"}
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, arr)
        return {"pcap": pcap_path, "npy": npy_path, "status": "ok"}
    except Exception as e:
        return {"pcap": pcap_path, "npy": None, "status": f"error: {e}"}


def convert_pcaps_to_npy(
    pcap_files: List[str],
    pcap_root: str,
    npy_root: str,
    workers: int = 1,
) -> Dict[str, str]:
    print("\n" + "=" * 65)
    print(f"Phase 2: Converting {len(pcap_files)} pcap files to npy")
    print("=" * 65)

    pcap_to_npy: Dict[str, str] = {}
    tasks = [(p, pcap_root, npy_root) for p in pcap_files]
    ok = skip = err = 0
    t0 = time.time()

    def _process_result(res):
        nonlocal ok, skip, err
        if res["status"] == "ok":
            ok += 1
            pcap_to_npy[res["pcap"]] = res["npy"]
        elif res["status"] == "exists":
            skip += 1
            pcap_to_npy[res["pcap"]] = res["npy"]
        else:
            err += 1

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_npy_worker, t) for t in tasks]
            iterator = (
                tqdm(as_completed(futures), total=len(futures), desc="pcap→npy")
                if tqdm else as_completed(futures)
            )
            for fut in iterator:
                try:
                    _process_result(fut.result(timeout=30))  # 每个 pcap 最多 30 秒
                except TimeoutError:
                    err += 1
                    print(f"  [WARN] worker timeout, skipping")
                except Exception as e:
                    err += 1
                    print(f"  [WARN] worker error: {e}")
    else:
        iterator = tqdm(tasks, desc="pcap→npy") if tqdm else tasks
        for t in iterator:
            _process_result(_npy_worker(t))

    elapsed = time.time() - t0
    print(f"  Done: ok={ok} skip(exists)={skip} error={err} in {elapsed:.1f}s")
    return pcap_to_npy


# ============================================================
# jsonl 生成
# ============================================================

def build_jsonl_rows(
    pcap_files: List[str],
    pcap_to_npy: Dict[str, str],
    pcap_root: str,
    kb: Dict[str, Any],
    thr: Dict[str, float],
    class_list: List[str],
    seed: int,
    split_label: str,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows = []

    # 所有样本共用同一个 user_text（含 <cls_placeholder>，在线替换）
    user_text = build_user_text(class_list)

    for pcap_path in pcap_files:
        npy_path = pcap_to_npy.get(pcap_path)
        if not npy_path or not os.path.exists(npy_path):
            continue

        rel       = os.path.relpath(pcap_path, pcap_root)
        cls       = map_class(rel.split(os.sep)[0])
        sample_id = os.path.splitext(os.path.basename(pcap_path))[0]

        try:
            flat   = load_npy_uint8(npy_path)
            traits = extract_byte_traits(flat, thr)
        except Exception as e:
            print(f"  [WARN] byte traits failed for {rel}: {e}")
            continue

        try:
            stats = analyze_pcap_stats(pcap_path)
        except Exception:
            stats = _empty_stats()

        kb_entry = kb.get(cls, _build_dynamic_fallback(cls))
        if not isinstance(kb_entry, dict):
            kb_entry = _build_dynamic_fallback(cls)
        descs    = kb_entry.get("descriptions", [])
        class_description = rng.choice(descs) if descs else ""

        target_str = build_target(cls, traits, kb_entry, stats, rng)

        nl_description = (
            f"This is {cls} traffic. "
            f"{class_description} "
            f"{semanticize_stats(stats)}"
        ).strip()

        row = OrderedDict([
            ("sample_path",       pcap_path),
            ("sample_relpath",    rel),
            ("sample_id",         sample_id),
            ("class",             cls),
            ("split",             split_label),
            ("user_text",         user_text),
            ("class_description", class_description),
            ("nl_description",    nl_description),
            ("stats",             stats),
            ("byte_traits",       traits),
            ("target",            target_str),
        ])
        rows.append(row)

    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows)} rows → {path}")


# ============================================================
# Phase 6 多进程支持
# ============================================================

def _jsonl_single_worker(args_tuple):
    """
    处理单个 pcap 文件并返回一条 jsonl row（或 None）。
    作为顶层函数以满足 multiprocessing 的 pickle 要求。
    per_file_seed 由主进程用原始顺序 RNG 预先派生，保持与串行版本一致的随机逻辑。
    """
    pcap_path, pcap_to_npy, pcap_root, kb, thr, class_list, per_file_seed, split_label, user_text = args_tuple

    npy_path = pcap_to_npy.get(pcap_path)
    if not npy_path or not os.path.exists(npy_path):
        return None

    rel       = os.path.relpath(pcap_path, pcap_root)
    cls       = map_class(rel.split(os.sep)[0])
    sample_id = os.path.splitext(os.path.basename(pcap_path))[0]

    # 使用主进程顺序派生的 per_file_seed，保持原始随机逻辑
    rng = random.Random(per_file_seed)

    try:
        flat   = load_npy_uint8(npy_path)
        traits = extract_byte_traits(flat, thr)
    except Exception as e:
        print(f"  [WARN] byte traits failed for {rel}: {e}")
        return None

    try:
        stats = analyze_pcap_stats(pcap_path)
    except Exception:
        stats = _empty_stats()

    kb_entry = kb.get(cls, _build_dynamic_fallback(cls))
    if not isinstance(kb_entry, dict):
        kb_entry = _build_dynamic_fallback(cls)
    descs    = kb_entry.get("descriptions", [])
    class_description = rng.choice(descs) if descs else ""

    target_str = build_target(cls, traits, kb_entry, stats, rng)

    nl_description = (
        f"This is {cls} traffic. "
        f"{class_description} "
        f"{semanticize_stats(stats)}"
    ).strip()

    return OrderedDict([
        ("sample_path",       pcap_path),
        ("sample_relpath",    rel),
        ("sample_id",         sample_id),
        ("class",             cls),
        ("split",             split_label),
        ("user_text",         user_text),
        ("class_description", class_description),
        ("nl_description",    nl_description),
        ("stats",             stats),
        ("byte_traits",       traits),
        ("target",            target_str),
    ])


def build_jsonl_rows_parallel(
    pcap_files: List[str],
    pcap_to_npy: Dict[str, str],
    pcap_root: str,
    kb: Dict[str, Any],
    thr: Dict[str, float],
    class_list: List[str],
    seed: int,
    split_label: str,
    workers: int,
) -> List[Dict[str, Any]]:
    """
    build_jsonl_rows 的多进程版本。
    在主进程中用与原串行版本相同的顺序 RNG 为每个文件预先派生 per_file_seed，
    再由 worker 以该种子初始化各自的 RNG，保持原有随机逻辑不变。
    结果按原始文件列表顺序返回（futures 按提交顺序收集）。
    """
    # 与 build_jsonl_rows 保持一致：用相同顺序 RNG 为每个文件派生子种子
    rng = random.Random(seed)
    user_text = build_user_text(class_list)
    per_file_seeds = [rng.randint(0, 2**31 - 1) for _ in pcap_files]

    tasks = [
        (p, pcap_to_npy, pcap_root, kb, thr, class_list, s, split_label, user_text)
        for p, s in zip(pcap_files, per_file_seeds)
    ]
    rows = []
    err  = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        # 用 futures list 而非 as_completed，保证结果顺序与文件列表一致
        futures = [ex.submit(_jsonl_single_worker, t) for t in tasks]
        iterator = (
            tqdm(futures, total=len(futures), desc=f"jsonl({split_label})")
            if tqdm else futures
        )
        for fut in iterator:
            try:
                result = fut.result()
                if result is not None:
                    rows.append(result)
            except Exception as e:
                err += 1
                print(f"  [WARN] jsonl worker error: {e}")

    if err:
        print(f"  [WARN] {err} files failed during parallel jsonl generation.")
    return rows


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Unified PCAP Processing Pipeline v2")
    parser.add_argument("--pcap-root",      required=True,  help="pcap 数据集根目录")
    parser.add_argument("--npy-root",       required=True,  help="npy 输出目录")
    parser.add_argument("--output-dir",     required=True,  help="jsonl 输出目录")
    parser.add_argument("--kb-cache",       default="./knowledge_base.json", help="知识库缓存文件路径")
    parser.add_argument("--api-key",        default=None,   help="Anthropic API Key")
    parser.add_argument("--max-per-class",  type=int, default=5000, help="每类最多保留样本数")
    parser.add_argument("--min-per-class",  type=int, default=MIN_CLASS_SAMPLES, help="低于此数量的类别自动过滤")
    parser.add_argument("--test-ratio",     type=float, default=0.2, help="测试集比例")
    parser.add_argument("--workers",        type=int, default=1, help="并行进程数")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--label-map",     default=None,
                        help="final_category_mapping.json 路径，用于类别重命名/过滤（CSTNet/CrossPlatform）")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 加载类别映射表（不传则不做任何映射，兼容 ISCXVPN/Tor/USTC）
    global LABEL_MAP, REMOVE_SET
    if args.label_map:
        LABEL_MAP, REMOVE_SET = load_label_map(args.label_map)
        print(f"  [LabelMap] Loaded: {len(LABEL_MAP)} renames, {len(REMOVE_SET)} removals")

    class_to_files = scan_dataset(args.pcap_root)

    # 过滤被删除的类别 + 将目录名映射为新标签（iOS合并对自动合并）
    if LABEL_MAP or REMOVE_SET:
        mapped: Dict[str, List[str]] = {}
        for cls, files in class_to_files.items():
            if cls in REMOVE_SET:
                print(f"  [LabelMap] Removed class: {cls} ({len(files)} files)")
                continue
            new_cls = map_class(cls)
            if new_cls in mapped:
                # iOS 合并对（如 google-drive 和 google-drive-secure-...）会合并到同一个 new_cls
                mapped[new_cls].extend(files)
                print(f"  [LabelMap] Merged: {cls} -> {new_cls} (+{len(files)} files)")
            else:
                mapped[new_cls] = files
                if new_cls != cls:
                    print(f"  [LabelMap] Renamed: {cls} -> {new_cls}")
        class_to_files = mapped

    class_to_files = filter_and_balance(
        class_to_files,
        max_per_class=args.max_per_class,
        min_samples=args.min_per_class,
        seed=args.seed,
    )
    if not class_to_files:
        print("ERROR: No valid classes found after filtering.")
        return

    class_list = sorted(class_to_files.keys())

    print("\n" + "=" * 65)
    print("Phase 2: Knowledge base generation")
    print("=" * 65)
    kb_cache = load_kb_cache(args.kb_cache)
    kb_cache = generate_kb_for_classes(class_list, api_key, kb_cache, args.kb_cache)

    print("\n" + "=" * 65)
    print(f"Phase 3: Train/test split (test_ratio={args.test_ratio})")
    print("=" * 65)
    train_files, test_files = split_train_test(class_to_files, args.test_ratio, args.seed)
    print(f"  Train: {len(train_files)} files, Test: {len(test_files)} files")

    all_files = train_files + test_files

    pcap_to_npy = convert_pcaps_to_npy(
        all_files, args.pcap_root, args.npy_root, workers=args.workers
    )

    print("\n" + "=" * 65)
    print("Phase 5: Computing byte feature thresholds")
    print("=" * 65)
    npy_paths = [v for v in pcap_to_npy.values() if v and os.path.exists(v)]
    thr = compute_thresholds(npy_paths, max_samples=20000, seed=args.seed)
    print(f"  Thresholds computed from {len(npy_paths)} npy files.")
    for k, v in thr.items():
        print(f"    {k}: {v:.4f}")

    print("\n" + "=" * 65)
    print("Phase 6: Generating jsonl")
    print("=" * 65)

    if args.workers > 1:
        print(f"  Using {args.workers} parallel workers for jsonl generation.")
        train_rows = build_jsonl_rows_parallel(
            train_files, pcap_to_npy, args.pcap_root,
            kb_cache, thr, class_list, args.seed, split_label="train",
            workers=args.workers,
        )
        test_rows = build_jsonl_rows_parallel(
            test_files, pcap_to_npy, args.pcap_root,
            kb_cache, thr, class_list, args.seed, split_label="test",
            workers=args.workers,
        )
    else:
        train_rows = build_jsonl_rows(
            train_files, pcap_to_npy, args.pcap_root,
            kb_cache, thr, class_list, args.seed, split_label="train"
        )
        test_rows = build_jsonl_rows(
            test_files, pcap_to_npy, args.pcap_root,
            kb_cache, thr, class_list, args.seed, split_label="test"
        )

    train_out = os.path.join(args.output_dir, "train.jsonl")
    test_out  = os.path.join(args.output_dir, "test.jsonl")
    write_jsonl(train_out, train_rows)
    write_jsonl(test_out,  test_rows)

    print("\n" + "=" * 65)
    print("Pipeline complete!")
    print("=" * 65)
    print(f"  Classes:        {len(class_list)}")
    print(f"  Train samples:  {len(train_rows)}")
    print(f"  Test  samples:  {len(test_rows)}")
    print(f"  npy root:       {args.npy_root}")
    print(f"  Train jsonl:    {train_out}")
    print(f"  Test  jsonl:    {test_out}")
    print(f"  KB cache:       {args.kb_cache}")
    print("=" * 65)


if __name__ == "__main__":
    main()