#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import random
from collections import OrderedDict
from typing import Dict, Any, List

import numpy as np
from tqdm import tqdm

# ----------------------------
# Config (match your v3_noport)
# ----------------------------
K = 10
L = 160
HEADER_BYTES = 64
PAYLOAD_BYTES = 96

# 输入输出
IN_JSONL = "/root/autodl-tmp/Datasets/USTC-TFC-2016/nlp_description_output/test.jsonl"
NPY_ROOT = "/root/autodl-tmp/Datasets/ISCXVPN2016/ISCXVPN2016_npy_1600_v3_noport_balanced"
OUT_JSONL = "/root/autodl-tmp/Datasets/USTC-TFC-2016/nlp_description_output/test_byte_grounded.jsonl"
OUT_THRESH_JSON = "/root/autodl-tmp/Datasets/USTC-TFC-2016/nlp_description_output/test_byte_thresholds.json"

# 采样统计阈值（建议先用 20000；你数据大也可 50000）
MAX_STAT_SAMPLES = 20000
SEED = 42

assert HEADER_BYTES + PAYLOAD_BYTES == L

# ----------------------------
# Helpers
# ----------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def pcap_rel_to_npy_path(sample_relpath: str) -> str:
    base, ext = os.path.splitext(sample_relpath)
    npy_rel = base + ".npy" if ext.lower() != ".npy" else sample_relpath
    return os.path.join(NPY_ROOT, npy_rel)

def load_npy_uint8(path: str) -> np.ndarray:
    x = np.load(path)
    x = np.asarray(x)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    if x.ndim == 2 and x.shape[0] == 1:
        x = x.reshape(-1)
    elif x.ndim != 1:
        x = x.reshape(-1)
    return x

def reshape_packets(x: np.ndarray) -> np.ndarray:
    if x.size != K * L:
        if x.size > K * L:
            x = x[: K * L]
        else:
            pad = np.zeros(K * L - x.size, dtype=np.uint8)
            x = np.concatenate([x, pad], axis=0)
    return x.reshape(K, L)

def shannon_entropy(byte_arr: np.ndarray) -> float:
    if byte_arr.size == 0:
        return 0.0
    hist = np.bincount(byte_arr, minlength=256).astype(np.float64)
    s = hist.sum()
    if s <= 0:
        return 0.0
    p = hist / s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def ascii_ratio(byte_arr: np.ndarray) -> float:
    if byte_arr.size == 0:
        return 0.0
    printable = ((byte_arr >= 0x20) & (byte_arr <= 0x7E)).sum()
    return float(printable) / float(byte_arr.size)

def zero_pad_ratio(byte_arr: np.ndarray) -> float:
    if byte_arr.size == 0:
        return 1.0
    return float((byte_arr == 0).sum()) / float(byte_arr.size)

def repeat_slice_ratio(packets: np.ndarray) -> float:
    rows = [bytes(packets[i].tolist()) for i in range(packets.shape[0])]
    uniq = len(set(rows))
    return 1.0 - (float(uniq) / float(packets.shape[0]))

def has_tls_record(byte_arr: np.ndarray) -> bool:
    b = byte_arr.tobytes()
    for ct in (0x14, 0x15, 0x16, 0x17):
        for ver in (0x00, 0x01, 0x02, 0x03, 0x04):
            pat = bytes([ct, 0x03, ver])
            if pat in b:
                return True
    return False

def has_http_method(byte_arr: np.ndarray) -> bool:
    b = byte_arr.tobytes()
    tokens = [b"GET ", b"POST ", b"HEAD ", b"PUT ", b"HTTP/1.", b"Host:", b"User-Agent:"]
    return any(t in b for t in tokens)

def bucketize(val: float, p33: float, p66: float) -> str:
    if val <= p33:
        return "low"
    if val <= p66:
        return "mid"
    return "high"

def sanitize_notes(class_desc: str) -> str:
    if not class_desc:
        return ""
    for sep in [". ", ".\n", "\n", "\r\n"]:
        if sep in class_desc:
            class_desc = class_desc.split(sep)[0].strip()
            break
    bad_prefix = ["This capture contains", "Observed protocols", "Average packet size", "Top application protocols"]
    for bp in bad_prefix:
        if bp in class_desc:
            return ""
    return class_desc.strip()

# ----------------------------
# Evidence and Description templates
# ----------------------------
EVIDENCE_POOL = OrderedDict([
    ("tls", "TLS record header pattern detected (e.g., xx 03 xx)."),
    ("http", "HTTP tokens found in the payload window (e.g., GET/POST/HTTP/1.)."),
    ("enc", "Payload shows low ASCII ratio and high entropy (likely encrypted/compressed)."),
    ("plain", "Payload contains a notable amount of readable ASCII (likely plaintext application data)."),
    ("pad", "Large zero-padding ratio is present in the fixed-length window."),
    ("repeat", "Repeated packet slices observed (likely introduced by sampling repetition)."),
])

def build_evidence(traits: Dict[str, Any]) -> List[str]:
    ev = []
    if traits["has_tls_record"]:
        ev.append(EVIDENCE_POOL["tls"])
    if traits["has_http_method"]:
        ev.append(EVIDENCE_POOL["http"])

    if traits["ascii_ratio_bucket"] == "low" and traits["entropy_bucket"] == "high":
        ev.append(EVIDENCE_POOL["enc"])
    elif traits["ascii_ratio_bucket"] == "high":
        ev.append(EVIDENCE_POOL["plain"])

    if traits["zero_pad_ratio_bucket"] == "high":
        ev.append(EVIDENCE_POOL["pad"])

    if traits["repeat_slice_ratio_bucket"] == "high":
        ev.append(EVIDENCE_POOL["repeat"])

    return ev[:3]

def build_description(traits: Dict[str, Any], evidence: List[str]) -> str:
    sents = []
    if traits["ascii_ratio_bucket"] == "low" and traits["entropy_bucket"] == "high":
        sents.append(
            "The captured byte window is mostly high-entropy with limited readable ASCII, which is consistent with encrypted or compressed traffic."
        )
    elif traits["ascii_ratio_bucket"] == "high":
        sents.append(
            "The captured byte window contains a considerable amount of readable ASCII, suggesting plaintext application-layer content in the sampled packets."
        )
    else:
        sents.append("The captured byte window shows mixed characteristics across the fixed-length packet slices.")

    if traits["has_http_method"]:
        sents.append("HTTP-related tokens are present in the payload window.")
    if traits["has_tls_record"]:
        sents.append("TLS record header patterns are present in the byte window.")

    if traits["zero_pad_ratio_bucket"] in ("mid", "high"):
        sents.append("A noticeable portion of the fixed-length window is zero-padded, indicating limited captured content within the configured slice size.")
    if traits["repeat_slice_ratio_bucket"] == "high":
        sents.append("Repeated packet slices are observed, likely introduced by the sampling strategy when the original flow has fewer packets.")

    return " ".join(sents[:4])

# ----------------------------
# Main pipeline
# ----------------------------
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"[1/4] Loading jsonl: {IN_JSONL}")
    rows = read_jsonl(IN_JSONL)
    n = len(rows)
    print(f"  loaded rows: {n}")

    idxs = list(range(n))
    random.shuffle(idxs)
    idxs = idxs[: min(MAX_STAT_SAMPLES, n)]
    print(f"[2/4] Collecting feature stats for thresholds (samples={len(idxs)})")

    ascii_vals, ent_vals, zero_vals, rep_vals = [], [], [], []
    miss_npy_stat = 0

    for i in tqdm(idxs, desc="Collect stats", ncols=100):
        r = rows[i]
        rel = r.get("sample_relpath")
        if not rel:
            continue
        npy_path = pcap_rel_to_npy_path(rel)
        if not os.path.exists(npy_path):
            miss_npy_stat += 1
            continue

        x = load_npy_uint8(npy_path)
        pk = reshape_packets(x)

        flat = pk.reshape(-1)
        nz = flat[flat != 0]

        ascii_vals.append(ascii_ratio(nz))
        ent_vals.append(shannon_entropy(nz))
        zero_vals.append(zero_pad_ratio(flat))
        rep_vals.append(repeat_slice_ratio(pk))

    def pct(v, p):
        if len(v) == 0:
            return 0.0
        return float(np.percentile(np.asarray(v, dtype=np.float64), p))

    thr = {
        "ascii_ratio_p33": pct(ascii_vals, 33),
        "ascii_ratio_p66": pct(ascii_vals, 66),
        "entropy_p33": pct(ent_vals, 33),
        "entropy_p66": pct(ent_vals, 66),
        "zero_pad_p33": pct(zero_vals, 33),
        "zero_pad_p66": pct(zero_vals, 66),
        "repeat_p50": pct(rep_vals, 50),
        "n_stat_samples_used": len(ascii_vals),
        "n_stat_samples_missing_npy": miss_npy_stat,
    }

    os.makedirs(os.path.dirname(OUT_THRESH_JSON), exist_ok=True)
    with open(OUT_THRESH_JSON, "w", encoding="utf-8") as f:
        json.dump(thr, f, ensure_ascii=False, indent=2)

    print(f"[3/4] Building per-sample targets for ALL rows -> {OUT_JSONL}")
    out_rows = []
    miss_npy = 0

    for r in tqdm(rows, desc="Build targets", ncols=100):
        rel = r.get("sample_relpath")
        if not rel:
            out_rows.append(r)
            continue

        npy_path = pcap_rel_to_npy_path(rel)
        if not os.path.exists(npy_path):
            miss_npy += 1
            out_rows.append(r)
            continue

        x = load_npy_uint8(npy_path)
        pk = reshape_packets(x)
        flat = pk.reshape(-1)
        nz = flat[flat != 0]

        has_tls = has_tls_record(flat)
        has_http = has_http_method(flat)

        a = ascii_ratio(nz)
        e = shannon_entropy(nz)
        z = zero_pad_ratio(flat)
        rep = repeat_slice_ratio(pk)

        traits = OrderedDict([
            ("has_tls_record", bool(has_tls)),
            ("has_http_method", bool(has_http)),
            ("ascii_ratio_bucket", bucketize(a, thr["ascii_ratio_p33"], thr["ascii_ratio_p66"])),
            ("entropy_bucket", bucketize(e, thr["entropy_p33"], thr["entropy_p66"])),
            ("zero_pad_ratio_bucket", bucketize(z, thr["zero_pad_p33"], thr["zero_pad_p66"])),
            ("repeat_slice_ratio_bucket", "high" if rep >= thr["repeat_p50"] else "low"),
        ])

        evidence = build_evidence(traits)
        description = build_description(traits, evidence)

        notes = sanitize_notes(r.get("class_description", "") or "")
        if notes:
            notes = notes[:200]

        target = OrderedDict([
            ("class", r.get("class", "")),
            ("traits", traits),
            ("evidence", evidence),
            ("description", description),
            ("notes", notes),
        ])
        target_str = json.dumps(target, ensure_ascii=False, separators=(",", ": "))

        r2 = dict(r)
        r2["byte_traits"] = traits
        r2["byte_evidence"] = evidence
        r2["byte_description"] = description
        r2["target"] = target_str
        out_rows.append(r2)

    print(f"[4/4] Writing jsonl ({len(out_rows)} rows) ...")
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    write_jsonl(OUT_JSONL, out_rows)

    print("Done.")
    print("Saved:", OUT_JSONL)
    print("Thresholds:", OUT_THRESH_JSON)
    print("Missing npy (target build):", miss_npy)
    print("Missing npy (threshold stats):", miss_npy_stat)

if __name__ == "__main__":
    main()
