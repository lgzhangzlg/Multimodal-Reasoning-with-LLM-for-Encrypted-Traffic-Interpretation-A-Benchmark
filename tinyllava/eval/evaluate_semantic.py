"""
evaluate_semantic.py
====================
Evaluates semantic similarity between predicted and ground-truth
`evidence` and `description` fields using DeepSeek API.

Outputs a JSON file per dataset with per-sample scores and aggregate stats.

Features:
  - Incremental saving (safe to resume if interrupted)
  - Concurrent requests (configurable)
  - Progress display

Usage:
  python evaluate_semantic.py \
      --input  predictions.jsonl \
      --output semantic_results.json \
      --api-key YOUR_DEEPSEEK_KEY \
      --dataset ISCX-Tor-2016
"""

import json
import argparse
import time
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ── Dataset paths (fill in before running) ────────────────────────────────────

DATASET_PATHS = {
    "ISCX-Tor-2016":         "FILL_IN",
    "ISCXVPN2016":           "FILL_IN",
    "CSTNet-TLS1.3":         "FILL_IN",
    "CrossPlatform-iOS":     "FILL_IN",
    "CrossPlatform-Android": "FILL_IN",
    "USTC-TFC-2016":         "FILL_IN",
}

# ── API config ─────────────────────────────────────────────────────────────────

BASE_URL = "https://api.deepseek.com"
MODEL    = "deepseek-chat"

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert evaluator for network traffic forensic reports. "
    "You will be given a predicted report field and a ground-truth reference field. "
    "Your task is to assess the semantic similarity between them. "
    "Focus on whether the predicted content conveys the same meaning, "
    "key observations, and technical insights as the ground truth. "
    "Respond with ONLY a single JSON object. No explanation, no markdown."
)

def build_prompt(pred_evidence: str, gt_evidence: str,
                 pred_description: str, gt_description: str) -> str:
    return f"""Compare the predicted forensic report fields against the ground-truth references.

--- EVIDENCE ---
Predicted : {pred_evidence}
Ground Truth: {gt_evidence}

--- DESCRIPTION ---
Predicted : {pred_description}
Ground Truth: {gt_description}

Return ONLY this JSON object (no other text):
{{
  "evidence_similarity": <float 0.0-1.0>,
  "description_similarity": <float 0.0-1.0>,
  "overall_similarity": <float 0.0-1.0>
}}

Scoring guide (use any continuous value in [0.0, 1.0]):
  0.95-1.0  — same technical meaning; differences in wording or phrasing only
  0.80-0.94 — all key technical points present; minor differences in detail level
  0.65-0.79 — mostly correct; one key point missing or slightly imprecise
  0.45-0.64 — captures the general idea but misses several specific observations
  0.20-0.44 — weak overlap; major technical points missing or incorrect
  0.00-0.19 — mostly unrelated, contradictory, or empty

Important: Do NOT penalize for different phrasing, sentence structure, or verbosity.
Focus purely on whether the core technical meaning and key observations are preserved.
"""

# ── API call ──────────────────────────────────────────────────────────────────

def call_api(prompt: str, api_key: str,
             retries: int = 5, backoff: float = 5.0):
    url     = f"{BASE_URL}/chat/completions"
    payload = json.dumps({
        "model":    MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream":      False,
        "temperature": 0.0,
    }).encode("utf-8")
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=payload, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"].strip()
            # Strip markdown fences if present
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            if e.code in (429, 500, 502, 503):
                wait = backoff * (2 ** attempt)
                print(f"  [retry {attempt+1}/{retries}] HTTP {e.code}, wait {wait:.0f}s", flush=True)
                time.sleep(wait)
            else:
                print(f"  [API error] HTTP {e.code}: {body}", flush=True)
                return None
        except Exception as e:
            wait = backoff * (2 ** attempt)
            print(f"  [retry {attempt+1}/{retries}] {e}, wait {wait:.0f}s", flush=True)
            time.sleep(wait)
    return None

# ── Extract fields from prediction ───────────────────────────────────────────

def extract_fields(obj: dict):
    """Extract evidence and description from a prediction object."""
    # Prediction
    pred_raw = obj.get("prediction", {})
    if isinstance(pred_raw, str):
        try:
            pred_raw = json.loads(pred_raw)
        except Exception:
            pred_raw = {}

    pred_evidence    = " ".join(pred_raw.get("evidence", [])) if isinstance(
        pred_raw.get("evidence"), list) else str(pred_raw.get("evidence", ""))
    pred_description = pred_raw.get("description", "")

    # Ground truth
    gt_raw = obj.get("ground_truth", obj.get("target", {}))
    if isinstance(gt_raw, str):
        try:
            gt_raw = json.loads(gt_raw)
        except Exception:
            gt_raw = {}

    gt_evidence    = " ".join(gt_raw.get("evidence", [])) if isinstance(
        gt_raw.get("evidence"), list) else str(gt_raw.get("evidence", ""))
    gt_description = gt_raw.get("description", "")

    return pred_evidence, gt_evidence, pred_description, gt_description

# ── Evaluate one sample ───────────────────────────────────────────────────────

def evaluate_sample(idx: int, obj: dict, api_key: str):
    pred_ev, gt_ev, pred_desc, gt_desc = extract_fields(obj)

    # Skip if both predicted fields are empty
    if not pred_ev.strip() and not pred_desc.strip():
        return idx, {"evidence_similarity": 0.0,
                     "description_similarity": 0.0,
                     "overall_similarity": 0.0,
                     "skipped": True}

    prompt = build_prompt(pred_ev, gt_ev, pred_desc, gt_desc)
    result = call_api(prompt, api_key)

    if result is None:
        return idx, {"evidence_similarity": None,
                     "description_similarity": None,
                     "overall_similarity": None,
                     "failed": True}

    result["skipped"] = False
    result["failed"]  = False
    return idx, result

# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(input_path: str, output_path: str,
             api_key: str, dataset_name: str,
             workers: int = 8):

    # Load predictions
    samples = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"[{dataset_name}] Loaded {len(samples)} samples", flush=True)

    # Load existing results (resume support)
    out_path = Path(output_path)
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        scores = existing.get("per_sample", {})
        print(f"[{dataset_name}] Resuming — {len(scores)} already done", flush=True)
    else:
        scores = {}

    save_lock = Lock()

    def save():
        done   = [v for v in scores.values() if not v.get("failed")]
        ev_scores   = [v["evidence_similarity"]    for v in done if v.get("evidence_similarity")    is not None]
        desc_scores = [v["description_similarity"] for v in done if v.get("description_similarity") is not None]
        ov_scores   = [v["overall_similarity"]     for v in done if v.get("overall_similarity")     is not None]

        agg = {
            "dataset":               dataset_name,
            "n_total":               len(samples),
            "n_evaluated":           len(done),
            "n_failed":              sum(1 for v in scores.values() if v.get("failed")),
            "evidence_similarity":   {"mean": _mean(ev_scores),   "std": _std(ev_scores)},
            "description_similarity":{"mean": _mean(desc_scores), "std": _std(desc_scores)},
            "overall_similarity":    {"mean": _mean(ov_scores),   "std": _std(ov_scores)},
            "per_sample":            scores,
        }
        out_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False))

    def _mean(lst):
        return round(sum(lst) / len(lst), 6) if lst else None

    def _std(lst):
        if len(lst) < 2:
            return None
        m = sum(lst) / len(lst)
        return round((sum((x - m) ** 2 for x in lst) / len(lst)) ** 0.5, 6)

    # Build task list (skip already done)
    todo = [(i, s) for i, s in enumerate(samples) if str(i) not in scores]
    print(f"[{dataset_name}] Remaining: {len(todo)}", flush=True)

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(evaluate_sample, i, s, api_key): i
                   for i, s in todo}
        for fut in as_completed(futures):
            idx, result = fut.result()
            with save_lock:
                scores[str(idx)] = result
                completed += 1
                if completed % 50 == 0 or completed == len(todo):
                    save()
                    print(f"  [{dataset_name}] {completed}/{len(todo)} done", flush=True)

    save()
    print(f"[{dataset_name}] Finished. Output → {output_path}", flush=True)

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True, help="predictions.jsonl path")
    parser.add_argument("--output",  required=True, help="output JSON path")
    parser.add_argument("--api-key", required=True, help="DeepSeek API key")
    parser.add_argument("--dataset", default="unknown", help="Dataset name for logging")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent API requests (default 8)")
    args = parser.parse_args()

    evaluate(
        input_path=args.input,
        output_path=args.output,
        api_key=args.api_key,
        dataset_name=args.dataset,
        workers=args.workers,
    )

if __name__ == "__main__":
    main()