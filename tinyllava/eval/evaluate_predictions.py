"""
evaluate_final.py
=================
Computes all evaluation metrics for [Model Name] predictions and saves to JSON.

Usage:
  # Structured metrics only:
  python evaluate_final.py --input predictions.jsonl --output results.json

  # With LLM-as-Judge:
  python evaluate_final.py --input predictions.jsonl --output results.json \
      --judge --api-key YOUR_KEY --base-url https://vectorengine.ai --model claude-sonnet-4-6

Output JSON structure:
  {
    "structured": {
      "n_total": ...,
      "n_structured": ...,
      "structured_output_rate": ...,
      "trait_binary": { "has_http_method": {"accuracy": ..., "n": ...}, ... },
      "trait_bucket": { "entropy_bucket": {"exact": ..., "direction": ..., "n": ...}, ... },
      "evidence_trait_consistency": ...,
      "quantitative_claim_rate": ...,
      "protocol_mention_rate": ...
    },
    "judge": {                          # null if --judge not specified
      "n_judged": ...,
      "n_failed": ...,
      "faithfulness":   {"mean": ..., "std": ...},
      "byte_grounding": {"mean": ..., "std": ...},
      "completeness":   {"mean": ..., "std": ...},
      "per_class": { "AUDIO": {"faithfulness": ..., "byte_grounding": ..., "completeness": ..., "n": ...}, ... },
      "per_sample": [ {"sample_id": ..., "class": ..., "faithfulness": ..., ...}, ... ]
    }
  }
"""

import json
import re
import sys
import time
import argparse
import urllib.request
import urllib.error
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score

# ── Constants ──────────────────────────────────────────────────────────────────

BUCKET_ORDER = {"low": 0, "mid": 1, "high": 2}

PROTOCOL_KEYWORDS = [
    "tcp", "udp", "tls", "ssl", "dns", "http", "https",
    "smtp", "imap", "pop3", "ftp", "rtp", "sip", "quic",
    "0x14", "0x15", "0x16", "0x17",
]

QUANT_PATTERN = re.compile(
    r'\d+\.?\d*\s*%'           # percentages: 100.0%
    r'|\d+\.?\d*\s*bytes'      # byte counts
    r'|\d+\s*packets'          # packet counts
    r'|\d+\.?\d*\s*bits'       # bit counts
)

# Rules: (trait_key, trait_value, keywords_that_should_appear_in_evidence_text)
CONSISTENCY_RULES = [
    ("has_tls_record",        True,    ["tls", "ssl", "encrypted", "0x14", "0x15", "0x16", "0x17", "record"]),
    ("has_tls_record",        False,   ["no tls", "plaintext", "unencrypted", "not tls", "no encryption", "ascii", "http method"]),
    ("has_http_method",       True,    ["http", "get", "post", "put", "delete", "request", "response"]),
    ("entropy_bucket",        "high",  ["entropy", "encrypted", "compressed", "binary", "random"]),
    ("entropy_bucket",        "low",   ["low entropy", "plaintext", "ascii", "repetitive", "structured", "readable"]),
    ("ascii_ratio_bucket",    "high",  ["ascii", "plaintext", "readable", "text", "printable"]),
    ("ascii_ratio_bucket",    "low",   ["binary", "encrypted", "compressed", "non-ascii"]),
    ("zero_pad_ratio_bucket", "high",  ["zero", "pad", "short flow", "small packet", "sparse"]),
    ("zero_pad_ratio_bucket", "low",   ["sustained", "bulk", "continuous", "throughput", "full window"]),
]

JUDGE_SYSTEM_PROMPT = """You are an expert network security analyst evaluating AI-generated forensic reports for encrypted network traffic classification.

You will be given:
- The predicted traffic class
- The AI model's predicted behavioral traits
- The AI model's generated evidence chain and description
- The ground-truth traits and evidence for reference

Score the AI report on THREE dimensions (1-5 integer each):

FAITHFULNESS (1-5):
  Does the evidence genuinely and specifically support the predicted traffic class,
  or does it consist of generic statements applicable to any class?
  1 = Generic or contradictory  3 = Partially specific  5 = Fully class-specific

BYTE_GROUNDING (1-5):
  Is the evidence anchored in concrete byte-level or protocol-level observations
  (specific byte values, TLS record structure, entropy measurements, etc.)?
  1 = Entirely vague  3 = Mix of specific and vague  5 = All claims are concrete

COMPLETENESS (1-5):
  Does the report cover the key discriminative traits for this traffic class
  compared to the ground-truth evidence?
  1 = Misses most key traits  3 = Covers some  5 = Covers all major traits

Respond ONLY with valid JSON, no markdown, no extra text:
{"faithfulness": <int>, "byte_grounding": <int>, "completeness": <int>, "rationale": "<one sentence>"}"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_inner(raw: str) -> tuple[dict, bool]:
    """Parse an inner JSON string. Returns (dict, is_structured)."""
    if not raw:
        return {}, False
    try:
        d = json.loads(raw)
        if isinstance(d, dict) and "class" in d:
            return d, True
        return {}, False
    except Exception:
        m = re.search(r'"class"\s*:\s*"([^"]+)"', raw)
        return {"class": m.group(1) if m else None, "traits": {}, "evidence": [], "description": ""}, False


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Line {lineno}: skipped (invalid JSON)", file=sys.stderr, flush=True)
                continue
            pred, pred_ok = parse_inner(rec.get("prediction", ""))
            gt,   _       = parse_inner(rec.get("ground_truth", ""))
            records.append({
                "sample_id":       rec.get("sample_id", lineno - 1),
                "pred":            pred,
                "gt":              gt,
                "pred_structured": pred_ok,
            })
    return records


def get_text(pred: dict) -> str:
    return (" ".join(pred.get("evidence", [])) + " " + pred.get("description", "")).lower()


# ── Part 1: Structured Metrics ─────────────────────────────────────────────────

def compute_structured(records: list[dict]) -> dict:
    n_total      = len(records)
    n_structured = sum(1 for r in records if r["pred_structured"])
    struct       = [r for r in records if r["pred_structured"]]

    # ── Trait metrics ──────────────────────────────────────────────────────────
    # Discover all trait keys from GT
    all_keys = set()
    for r in struct:
        all_keys.update((r["gt"].get("traits") or {}).keys())

    trait_binary = {}
    trait_bucket = {}

    for tk in sorted(all_keys):
        pred_vals, gt_vals = [], []
        for r in struct:
            pv = (r["pred"].get("traits") or {}).get(tk)
            gv = (r["gt"].get("traits")   or {}).get(tk)
            if pv is None or gv is None:
                continue
            pred_vals.append(str(pv).lower())
            gt_vals.append(str(gv).lower())

        if not gt_vals:
            continue

        if all(v in BUCKET_ORDER for v in gt_vals):
            # Ordinal bucket trait
            exact = [int(p == g) for p, g in zip(pred_vals, gt_vals)]
            direction = [
                int(abs(BUCKET_ORDER[p] - BUCKET_ORDER[g]) <= 1)
                for p, g in zip(pred_vals, gt_vals)
                if p in BUCKET_ORDER and g in BUCKET_ORDER
            ]
            trait_bucket[tk] = {
                "exact_accuracy":     float(np.mean(exact)),
                "direction_accuracy": float(np.mean(direction)) if direction else 0.0,
                "n": len(gt_vals),
            }
        else:
            # Binary trait
            trait_binary[tk] = {
                "accuracy": float(accuracy_score(gt_vals, pred_vals)),
                "n": len(gt_vals),
            }

    # ── Evidence-Trait Consistency ─────────────────────────────────────────────
    consistency_scores = []
    for r in struct:
        traits_p = r["pred"].get("traits") or {}
        text     = get_text(r["pred"])
        hits, n  = 0, 0
        for tk, tv, kws in CONSISTENCY_RULES:
            if tk not in traits_p:
                continue
            if str(traits_p[tk]).lower() != str(tv).lower():
                continue
            n += 1
            if any(kw in text for kw in kws):
                hits += 1
        if n > 0:
            consistency_scores.append(hits / n)

    # ── Quantitative Claim Rate ────────────────────────────────────────────────
    quant_hits = [
        int(bool(QUANT_PATTERN.search(get_text(r["pred"]))))
        for r in struct
    ]

    # ── Protocol Mention Rate ──────────────────────────────────────────────────
    proto_hits = [
        int(any(p in get_text(r["pred"]) for p in PROTOCOL_KEYWORDS))
        for r in struct
    ]

    return {
        "n_total":                  n_total,
        "n_structured":             n_structured,
        "structured_output_rate":   n_structured / n_total if n_total else 0.0,
        "trait_binary":             trait_binary,
        "trait_bucket":             trait_bucket,
        "evidence_trait_consistency": float(np.mean(consistency_scores)) if consistency_scores else 0.0,
        "quantitative_claim_rate":  float(np.mean(quant_hits)) if quant_hits else 0.0,
        "protocol_mention_rate":    float(np.mean(proto_hits)) if proto_hits else 0.0,
    }


# ── Part 2: LLM-as-Judge ──────────────────────────────────────────────────────

def make_judge_prompt(pred: dict, gt: dict) -> str:
    pred_ev = "\n".join(f"  - {e}" for e in pred.get("evidence", [])) or "  (none)"
    gt_ev   = "\n".join(f"  - {e}" for e in gt.get("evidence",   [])) or "  (none)"
    return f"""Predicted Class: {pred.get("class")}

=== PREDICTED TRAITS ===
{json.dumps(pred.get("traits", {}), indent=2)}

=== PREDICTED EVIDENCE ===
{pred_ev}

=== PREDICTED DESCRIPTION ===
{pred.get("description", "").strip() or "(none)"}

=== GROUND-TRUTH TRAITS (reference) ===
{json.dumps(gt.get("traits", {}), indent=2)}

=== GROUND-TRUTH EVIDENCE (reference) ===
{gt_ev}"""


def call_judge(pred: dict, gt: dict, api_key: str,
               model: str, base_url: str,
               retries: int = 3, backoff: float = 5.0) -> Optional[dict]:

    base_url = base_url.rstrip("/")
    for suffix in ("/v1/chat/completions", "/v1/messages", "/v1"):
        if base_url.endswith(suffix):
            base_url = base_url[: -len(suffix)]
            break

    is_anthropic = "api.anthropic.com" in base_url

    if is_anthropic:
        url     = f"{base_url}/v1/messages"
        payload = json.dumps({
            "model": model, "max_tokens": 256,
            "system": JUDGE_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": make_judge_prompt(pred, gt)}],
        }).encode()
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
    else:
        url     = f"{base_url}/v1/chat/completions"
        payload = json.dumps({
            "model": model, "max_tokens": 256,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": make_judge_prompt(pred, gt)},
            ],
        }).encode()
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())

            text = (data["content"][0]["text"] if is_anthropic
                    else data["choices"][0]["message"]["content"]).strip()
            text = re.sub(r"```json|```", "", text).strip()
            scores = json.loads(text)

            for k in ("faithfulness", "byte_grounding", "completeness"):
                if k not in scores:
                    raise ValueError(f"Missing key '{k}'")
                scores[k] = int(scores[k])
            return scores

        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            if e.code == 429 or e.code >= 500:
                time.sleep(backoff * (attempt + 1))
            else:
                print(f"    [error] HTTP {e.code}: {body}", file=sys.stderr, flush=True)
                return None
        except Exception as e:
            print(f"    [retry {attempt+1}/{retries}] {e}", file=sys.stderr, flush=True)
            time.sleep(backoff * (attempt + 1))

    return None


def compute_judge(records: list[dict], api_key: str,
                  model: str, base_url: str, delay: float) -> dict:

    struct = [r for r in records if r["pred_structured"]]
    n      = len(struct)
    print(f"Running LLM-as-Judge on {n} structured samples (model: {model}) ...", flush=True)

    per_sample  = []
    per_class   = defaultdict(list)
    failed      = []

    for i, r in enumerate(struct, 1):
        cls    = r["gt"].get("class", "UNKNOWN")
        scores = call_judge(r["pred"], r["gt"], api_key, model, base_url)

        if scores is None:
            failed.append(r["sample_id"])
            print(f"  [{i:4d}/{n}] {cls:<18s} FAILED", file=sys.stderr, flush=True)
            continue

        entry = {
            "sample_id":      r["sample_id"],
            "class":          cls,
            "faithfulness":   scores["faithfulness"],
            "byte_grounding": scores["byte_grounding"],
            "completeness":   scores["completeness"],
            "rationale":      scores.get("rationale", ""),
        }
        per_sample.append(entry)
        per_class[cls].append(entry)

        avg = np.mean([scores["faithfulness"], scores["byte_grounding"], scores["completeness"]])
        print(f"  [{i:4d}/{n}] {cls:<18s} "
              f"F={scores['faithfulness']} B={scores['byte_grounding']} "
              f"C={scores['completeness']} avg={avg:.2f}", flush=True)
        time.sleep(delay)

    def agg(key):
        vals = [s[key] for s in per_sample]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else {"mean": None, "std": None}

    per_class_agg = {
        cls: {
            "faithfulness":   float(np.mean([e["faithfulness"]   for e in entries])),
            "byte_grounding": float(np.mean([e["byte_grounding"] for e in entries])),
            "completeness":   float(np.mean([e["completeness"]   for e in entries])),
            "n": len(entries),
        }
        for cls, entries in sorted(per_class.items())
    }

    all_avgs = [
        np.mean([s["faithfulness"], s["byte_grounding"], s["completeness"]])
        for s in per_sample
    ]

    return {
        "n_judged":       len(per_sample),
        "n_failed":       len(failed),
        "faithfulness":   agg("faithfulness"),
        "byte_grounding": agg("byte_grounding"),
        "completeness":   agg("completeness"),
        "overall_mean":   float(np.mean(all_avgs))  if all_avgs else None,
        "overall_std":    float(np.std(all_avgs))   if all_avgs else None,
        "per_class":      per_class_agg,
        "per_sample":     per_sample,
        "failed_ids":     failed,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--judge",    action="store_true")
    parser.add_argument("--api-key",  default=None)
    parser.add_argument("--base-url", default="https://api.anthropic.com")
    parser.add_argument("--model",    default="claude-sonnet-4-6")
    parser.add_argument("--delay",    type=float, default=0.3)
    args = parser.parse_args()

    if args.judge and not args.api_key:
        print("ERROR: --judge requires --api-key", file=sys.stderr, flush=True)
        sys.exit(1)

    records      = load_records(args.input)
    structured   = compute_structured(records)
    judge        = compute_judge(records, args.api_key, args.model, args.base_url, args.delay) \
                   if args.judge else None

    results = {"structured": structured, "judge": judge}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved → {args.output}", flush=True)


if __name__ == "__main__":
    main()