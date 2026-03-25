#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import random
from typing import Dict, Any, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from tinyllava.model import TinyLlavaForConditionalGeneration
from tinyllava.utils.constants import DEFAULT_IMAGE_TOKEN


# -------------------------
# jsonl
# -------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


# -------------------------
# remove leakage
# -------------------------
def strip_path_leak(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"Path:\s*.*?\.pcap\.\s*", "", str(text), flags=re.IGNORECASE)


def extract_class_from_label_sentence(label_sentence: str) -> str:
    s = strip_path_leak(str(label_sentence)).strip()

    m = re.search(r"(?i)\bthis\s+is\s+(.+?)\s+traffic\b", s)
    if m:
        return m.group(1).strip()

    m = re.search(r"(?i)\bthis\s+is\s+(.+?)\s*\.?\s*$", s)
    if m:
        return m.group(1).strip()

    return s.strip()


# -------------------------
# text cleanup
# -------------------------
def clean_keep_newlines(t: str) -> str:
    if t is None:
        return ""
    s = str(t)

    s = re.sub(r"^\s*(?:[>\-]+\s*)+", "", s)
    s = re.sub(r"^\s*(?:\.(?:png|jpg|jpeg)\b)\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*```.*?\n", "", s)
    s = re.sub(r"\n```$", "", s)

    s = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in s.splitlines())
    return s.strip()


def strip_prompt_echo(gen_text: str) -> str:
    """
    清掉模型复读的 <image> / Below is... / Rules... 这类前缀行，避免污染分类解析。
    """
    t = clean_keep_newlines(gen_text)
    if not t:
        return ""

    lines = [ln.strip() for ln in t.splitlines()]
    out = []
    skipping = True

    for ln in lines:
        low = ln.lower()
        if skipping:
            if low in {"<image>", "image:", "input:", "output:"}:
                continue
            if low.startswith("below is a network traffic sample"):
                continue
            if low.startswith("task:") or low.startswith("requirements:") or low.startswith("rules:"):
                continue
            if low.startswith("you must") or low.startswith("- do not") or low.startswith("1)") or low.startswith("2)"):
                continue
            if ln in {"", ".", "·"}:
                continue
            skipping = False
        out.append(ln)

    # 还可能残留 DEFAULT_IMAGE_TOKEN
    s = "\n".join(out).strip()
    s = s.replace(DEFAULT_IMAGE_TOKEN, "").strip()
    return s


def remove_leading_class_sentence(text: str) -> str:
    t = clean_keep_newlines(text)
    if not t:
        return ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return ""

    first = re.sub(r"^\s*[>\-\*]+\s*", "", lines[0]).strip()
    pats = [
        r"(?i)^\s*this\s+is\s+.+?\s+traffic(?:\s+sample)?\s*\.?\s*$",
        r"(?i)^\s*this\s+is\s+.+?\s*\.?\s*$",
        r"(?i)^\s*it\s+is\s+.+?\s+traffic\s*\.?\s*$",
    ]
    if any(re.match(p, first) for p in pats):
        return "\n".join(lines[1:]).strip()
    return "\n".join(lines).strip()


# -------------------------
# prompt build
# -------------------------
def build_user_text(sample: Dict[str, Any], eval_mode: str, use_hint: bool) -> str:
    """
    重要：不要出现 <CLASS> / Line 1 / Output format / Example
    这些会被模型复读，导致你现在那种 “Line 2+: ...” 垃圾输出。
    """
    parts = [
        "Below is a network traffic sample.",
        "Task: identify the traffic category, then describe its main characteristics.",
        "",
        'Requirement: the FIRST sentence must be exactly: "This is <category> traffic."',
        "Then write a short natural-language description.",
        "Do NOT write code/functions/JSON.",
        "Do NOT repeat the instructions.",
    ]

    if use_hint:
        parts += ["", f"Category hint: {strip_path_leak(sample.get('label_sentence', ''))}"]

    if eval_mode == "with_stats":
        parts += ["", f"Sample statistics: {strip_path_leak(sample.get('sample_description', ''))}"]
    elif eval_mode == "leak_free":
        pass
    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")

    return "\n".join(parts).strip()


def build_prompt(sample: Dict[str, Any], eval_mode: str, use_hint: bool) -> str:
    return DEFAULT_IMAGE_TOKEN + "\n" + build_user_text(sample, eval_mode, use_hint)


# -------------------------
# parse / match label
# -------------------------
DEFAULT_BANNED_LABELS = {"malware", "benign", "unknown", "other", "attack", "normal"}


def soft_match_anywhere(gen_text: str, label_set: Set[str], banned: Set[str]) -> str:
    t = clean_keep_newlines(gen_text)
    if not t:
        return ""
    tlow = t.lower()

    labels_sorted = sorted(label_set, key=lambda x: len(x), reverse=True)

    def ok_label(lab: str) -> bool:
        return lab.strip() and lab.lower() not in banned

    for lab in labels_sorted:
        if not ok_label(lab):
            continue
        lab_low = lab.lower()
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(lab_low)}(?![A-Za-z0-9])", tlow):
            return lab

    for lab in labels_sorted:
        if not ok_label(lab):
            continue
        if lab.lower() in tlow:
            return lab

    return ""


# -------------------------
# npy -> (C, L)
# -------------------------
def load_npy_as_CL(sample_relpath: str, image_folder: str) -> torch.Tensor:
    base, _ = os.path.splitext(sample_relpath)
    npy_path = os.path.join(image_folder, base + ".npy")
    arr = np.load(npy_path)
    x = torch.from_numpy(arr).float()

    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim == 2:
        if x.shape[0] > x.shape[1] and x.shape[1] <= 64:
            x = x.t().contiguous()
    elif x.ndim == 3:
        if x.shape[0] <= 64:
            x = x.flatten(1)
        else:
            x = x.permute(2, 0, 1).contiguous().flatten(1)
    else:
        x = x.reshape(x.shape[0], -1)

    return x


def is_bad_tensor(x: torch.Tensor) -> Tuple[bool, str]:
    if x is None:
        return True, "none"
    if torch.isnan(x).any():
        return True, "nan"
    if torch.isinf(x).any():
        return True, "inf"
    if torch.var(x).item() < 1e-12:
        return True, "near_const"
    return False, ""


# -------------------------
# metrics
# -------------------------
def try_metrics():
    bleu_fn = None
    rouge_fn = None

    try:
        import sacrebleu

        def corpus_bleu(hyps, refs):
            return float(sacrebleu.corpus_bleu(hyps, [refs]).score)

        bleu_fn = corpus_bleu
    except Exception:
        pass

    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        def rougeL_f1(hyps, refs):
            vals = []
            for h, r in zip(hyps, refs):
                vals.append(float(scorer.score(r, h)["rougeL"].fmeasure))
            return 100.0 * sum(vals) / max(1, len(vals))

        rouge_fn = rougeL_f1
    except Exception:
        pass

    return bleu_fn, rouge_fn


def hyp_ref_description_only(hyp: str, ref: str) -> Tuple[str, str]:
    hyp = clean_keep_newlines(hyp)
    ref = clean_keep_newlines(ref)

    hyp_desc = remove_leading_class_sentence(hyp).strip()
    ref_desc = remove_leading_class_sentence(ref).strip()

    if not hyp_desc:
        hyp_desc = hyp
    if not ref_desc:
        ref_desc = ref

    return hyp_desc, ref_desc


# -------------------------
# Dataset
# -------------------------
class EvalDataset(Dataset):
    def __init__(
        self,
        samples_all: List[Dict[str, Any]],
        image_folder: str,
        eval_mode: str,
        use_hint: bool,
        max_eval_samples: Optional[int],
        shuffle_subset: bool,
        seed: int = 42,
    ):
        self.samples_all = samples_all
        self.image_folder = image_folder
        self.eval_mode = eval_mode
        self.use_hint = use_hint

        self.label_set: Set[str] = set()
        for s in self.samples_all:
            self.label_set.add(extract_class_from_label_sentence(s.get("label_sentence", "")))

        self.samples = list(self.samples_all)
        if shuffle_subset:
            random.seed(seed)
            random.shuffle(self.samples)
        if max_eval_samples is not None:
            self.samples = self.samples[:max_eval_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        prompt = build_prompt(s, eval_mode=self.eval_mode, use_hint=self.use_hint)

        img = load_npy_as_CL(s["sample_relpath"], self.image_folder)
        bad, reason = is_bad_tensor(img)

        ref_text = s.get("nl_description", "") or ""
        if not ref_text:
            ref_text = strip_path_leak(s.get("sample_description", "") or "")
        ref_text = strip_path_leak(ref_text)

        true_label = extract_class_from_label_sentence(s.get("label_sentence", ""))

        return {
            "prompt": prompt,
            "image": img,
            "bad": bad,
            "bad_reason": reason,
            "ref": str(ref_text),
            "true_label": str(true_label),
            "sample_relpath": s.get("sample_relpath", ""),
        }


def collate_fn(batch, tokenizer):
    prompts = [b["prompt"] for b in batch]
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    images = [b["image"] for b in batch]
    if not all(isinstance(x, torch.Tensor) and x.shape == images[0].shape for x in images):
        shapes = [tuple(x.shape) for x in images]
        raise RuntimeError(f"发现 image shape 不一致：{shapes}")
    images = torch.stack(images, dim=0)

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "images": images,
        "refs": [b["ref"] for b in batch],
        "true_labels": [b["true_label"] for b in batch],
        "bad": [b["bad"] for b in batch],
        "bad_reason": [b["bad_reason"] for b in batch],
        "sample_relpath": [b["sample_relpath"] for b in batch],
    }


# -------------------------
# generation
# -------------------------
def _bad_words_ids(tokenizer):
    bad = ["/jpeg", "/png", "/jpg", "jpeg", "png", "jpg", ".png", ".jpg", ".jpeg"]
    ids = []
    for w in bad:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if toks:
            ids.append(toks)
    return ids


@torch.no_grad()
def tinyllava_generate(model, input_ids, attention_mask, images, args, tokenizer):
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    gen_kwargs = dict(
        inputs=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        pad_token_id=tokenizer.pad_token_id,
        bad_words_ids=_bad_words_ids(tokenizer),
    )
    if args.do_sample:
        gen_kwargs.update(dict(temperature=args.temperature, top_p=args.top_p))

    try:
        return model.generate(images=images, **gen_kwargs)
    except TypeError:
        return model.generate(image=images, **gen_kwargs)


def decode_new_tokens_only(gen_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer) -> List[str]:
    """
    关键修复：只 decode “新生成部分”，不要把 prompt 一起 decode。
    对每个样本，prompt 长度 = attention_mask.sum()
    """
    outs = []
    for i in range(gen_ids.size(0)):
        in_len = int(attention_mask[i].sum().item())
        new_ids = gen_ids[i, in_len:]
        txt = tokenizer.decode(new_ids, skip_special_tokens=True)
        outs.append(txt.strip())
    return outs


# -------------------------
# eval
# -------------------------
@torch.no_grad()
def run_eval(model, tokenizer, loader, device, args, label_set: Set[str], banned: Set[str]):
    model.eval()

    hyps_desc, refs_desc = [], []
    trues, pred_soft = [], []

    bad_count = 0
    bad_reasons: Dict[str, int] = {}
    rows = []

    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["images"].to(device)

        gen_ids = tinyllava_generate(model, input_ids, attention_mask, images, args, tokenizer)

        # ★ 只取新生成部分
        texts = decode_new_tokens_only(gen_ids, attention_mask, tokenizer)

        for yp, rhef, tl, is_bad, reason, relp in zip(
            texts, batch["refs"], batch["true_labels"], batch["bad"], batch["bad_reason"], batch["sample_relpath"]
        ):
            yp = strip_prompt_echo(str(yp))
            rhef = str(rhef).strip()
            tl = str(tl).strip()

            soft = soft_match_anywhere(yp, label_set, banned=banned)

            hyp_d, ref_d = hyp_ref_description_only(yp, rhef)
            hyps_desc.append(hyp_d)
            refs_desc.append(ref_d)

            trues.append(tl)
            pred_soft.append(soft)

            if is_bad:
                bad_count += 1
                bad_reasons[reason] = bad_reasons.get(reason, 0) + 1

            rows.append({
                "true_label": tl,
                "pred_soft": soft,
                "ref": rhef,
                "hyp": yp,
                "hyp_desc": hyp_d,
                "ref_desc": ref_d,
                "sample_relpath": relp,
            })

        if args.log_every > 0 and step % args.log_every == 0:
            done = min(step * args.batch_size, args.eval_n)
            print(f"[eval] step={step}  done={done}/{args.eval_n}")

    def acc(preds: List[str], trues_: List[str]) -> float:
        correct, total = 0, 0
        for t, p in zip(trues_, preds):
            if not t:
                continue
            total += 1
            if norm(t) == norm(p):
                correct += 1
        return 100.0 * correct / max(1, total)

    soft_acc = acc(pred_soft, trues)

    bleu_fn, rouge_fn = try_metrics()
    bleu = bleu_fn(hyps_desc, refs_desc) if bleu_fn else None
    rougeL = rouge_fn(hyps_desc, refs_desc) if rouge_fn else None

    return soft_acc, bleu, rougeL, rows, bad_count, bad_reasons


def apply_mode_defaults(args):
    if args.mode == "debug":
        if args.max_eval_samples is None:
            args.max_eval_samples = 100
        if args.max_new_tokens is None:
            args.max_new_tokens = 200
        if args.batch_size is None:
            args.batch_size = 4
        if args.num_beams is None:
            args.num_beams = 3
        if args.log_every is None:
            args.log_every = 10
        if args.shuffle_subset is None:
            args.shuffle_subset = True
    elif args.mode == "full":
        if args.max_new_tokens is None:
            args.max_new_tokens = 200
        if args.batch_size is None:
            args.batch_size = 4
        if args.num_beams is None:
            args.num_beams = 3
        if args.log_every is None:
            args.log_every = 200
        if args.shuffle_subset is None:
            args.shuffle_subset = False

    if args.do_sample is None:
        args.do_sample = False
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, choices=["debug", "full"], default="debug")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--eval_jsonl", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)

    ap.add_argument("--eval_mode", type=str, choices=["leak_free", "with_stats"], default="leak_free")
    ap.add_argument("--use_hint", action="store_true", help="sanity check only (leak gt label)")

    ap.add_argument("--max_eval_samples", type=int, default=None)
    ap.add_argument("--shuffle_subset", action="store_true", default=None)

    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--num_beams", type=int, default=None)

    ap.add_argument("--do_sample", action="store_true", default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dump_path", type=str, default=None)
    ap.add_argument("--log_every", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--banned_labels", type=str, default="Malware,Benign")

    args = ap.parse_args()
    args = apply_mode_defaults(args)

    ckpt_path = os.path.join(args.output_dir, args.checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    model = TinyLlavaForConditionalGeneration.from_pretrained(ckpt_path, trust_remote_code=True)

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    model.to(device)

    samples_all = load_jsonl(args.eval_jsonl)

    ds = EvalDataset(
        samples_all=samples_all,
        image_folder=args.image_folder,
        eval_mode=args.eval_mode,
        use_hint=args.use_hint,
        max_eval_samples=args.max_eval_samples,
        shuffle_subset=args.shuffle_subset,
        seed=args.seed,
    )

    label_set = ds.label_set
    args.eval_n = len(ds)

    banned = set([x.strip().lower() for x in (args.banned_labels.split(",") if args.banned_labels else [])])
    banned = banned.union(DEFAULT_BANNED_LABELS)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    soft_acc, bleu, rougeL, rows, bad_count, bad_reasons = run_eval(
        model, tokenizer, dl, device, args, label_set=label_set, banned=banned
    )

    print("\n===== EVAL =====")
    print(f"mode = {args.mode}")
    print(f"eval_mode = {args.eval_mode}")
    print(f"N = {len(rows)}")
    print(f"#Classes = {len(label_set)}")
    print(f"Soft Acc = {soft_acc:.2f}%")
    print(f"BLEU(desc-only) = {bleu:.2f}" if bleu is not None else "BLEU = (skip, pip install sacrebleu)")
    print(f"ROUGE-L F1(desc-only) = {rougeL:.2f}" if rougeL is not None else "ROUGE-L F1 = (skip, pip install rouge-score)")

    print("\n===== Bad Samples =====")
    print(f"bad_count = {bad_count}/{len(rows)}")
    if bad_reasons:
        print("bad_reasons =", bad_reasons)

    print("\n===== Examples (first 5) =====")
    for i, row in enumerate(rows[:5]):
        print("-" * 80)
        print(f"[{i}] TRUE_LABEL: {row['true_label']}")
        print(f"[{i}] PRED_SOFT:  {row['pred_soft']}")
        print(f"[{i}] REF(desc): {row['ref_desc'][:220]}")
        print(f"[{i}] HYP(desc): {row['hyp_desc'][:220]}")
        print(f"[{i}] HYP(raw):  {row['hyp'][:220]}")

    if args.dump_path:
        os.makedirs(os.path.dirname(args.dump_path) or ".", exist_ok=True)
        with open(args.dump_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nSaved to: {args.dump_path}")


if __name__ == "__main__":
    main()
