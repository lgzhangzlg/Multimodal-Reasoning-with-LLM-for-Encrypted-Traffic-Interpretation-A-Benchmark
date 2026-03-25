#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import transformers
from tqdm import tqdm
from torch.cuda.amp import autocast  # Import autocast for automatic mixed precision

# ---- TinyLLaVA imports (follow train.py style) ----
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.model import TinyLlavaForConditionalGeneration
from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from tinyllava.utils import logger_setting
from tinyllava.utils.arguments import ModelArguments, DataArguments, TrainingArguments
from tinyllava.utils.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


# ---------------------------
# Eval args (prefix with eval_ to avoid argparse conflicts)
# ---------------------------
@dataclass
class EvalArguments:
    eval_jsonl: str = field(metadata={"help": "Path to eval jsonl (same schema as train/test jsonl)."})
    eval_out_jsonl: str = field(metadata={"help": "Where to dump prediction jsonl."})
    eval_max_new_tokens: int = field(default=256)
    eval_batch_size: int = field(default=1)

    # per-class sampling
    eval_per_class_sample: int = field(default=0, metadata={"help": "0 means use all; otherwise sample N per class."})
    eval_sample_seed: int = field(default=42)

    # generation controls (stable defaults: greedy)
    eval_do_sample: bool = field(default=False)
    eval_temperature: float = field(default=0.0)
    eval_top_k: int = field(default=0)

    # image/byte settings
    eval_byte_length: int = field(default=1600)


# ---------------------------
# helpers: jsonl io
# ---------------------------
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------
# helpers: per-class sampling
# ---------------------------
def per_class_sample(rows: List[Dict[str, Any]], k: int, seed: int) -> List[Dict[str, Any]]:
    if k <= 0:
        return rows
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        c = r.get("class")
        if c is None:
            # fallback: parse from target
            try:
                c = json.loads(r.get("target", "{}")).get("class", "UNKNOWN")
            except Exception:
                c = "UNKNOWN"
        buckets.setdefault(c, []).append(r)

    out = []
    for c, lst in buckets.items():
        if len(lst) <= k:
            out.extend(lst)
        else:
            out.extend(rng.sample(lst, k))

    # keep deterministic order (optional): sort by class then relpath
    out.sort(key=lambda x: (x.get("class", ""), x.get("sample_relpath", "")))
    print(f"[INFO] per-class sampling: {k}/class | classes={len(buckets)} | sampled_total={len(out)}")
    return out


# ---------------------------
# helpers: npy loading (match your dataset.py)
# ---------------------------
def pcap_rel_to_npy_rel(sample_relpath: str) -> str:
    base, ext = os.path.splitext(sample_relpath)
    return sample_relpath if ext.lower() == ".npy" else (base + ".npy")


def load_npy_image(image_folder: str, sample_relpath: str, byte_length: int) -> torch.Tensor:
    npy_rel = pcap_rel_to_npy_rel(sample_relpath)
    npy_path = os.path.join(image_folder, npy_rel)
    arr = np.load(npy_path)
    arr = np.asarray(arr)

    # normalize shape -> (1, L)
    if arr.ndim == 2 and arr.shape[0] == 1:
        pass
    elif arr.ndim == 1:
        arr = arr[None, :]
    else:
        arr = arr.reshape(1, -1)

    L = int(byte_length)
    if arr.shape[1] < L:
        pad = np.zeros((1, L - arr.shape[1]), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=1)
    elif arr.shape[1] > L:
        arr = arr[:, :L]

    x = torch.from_numpy(arr.astype(np.float32)) / 255.0  # (1, L)
    return x


# ---------------------------
# prompt/tokenization (stable, minimal)
# ---------------------------
def _tokenizer_image_token(prompt: str, tokenizer, return_tensors: str = "pt") -> torch.Tensor:
    parts = prompt.split(DEFAULT_IMAGE_TOKEN)
    ids: List[int] = []
    for i, p in enumerate(parts):
        if p:
            ids.extend(tokenizer(p, add_special_tokens=False).input_ids)
        if i != len(parts) - 1:
            ids.append(IMAGE_TOKEN_INDEX)
    if return_tensors == "pt":
        return torch.tensor([ids], dtype=torch.long)
    return ids


def build_prompt_qwen_style(user_text: str, tokenizer) -> Tuple[torch.Tensor, str]:
    messages = [{"role": "user", "content": user_text}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = (
            "<|im_start|>user\n" + user_text + "\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    input_ids = _tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    return input_ids, prompt


# ---------------------------
# output postprocess: extract first JSON object
# ---------------------------
def extract_first_json_obj(text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if text is None:
        return None, None
    s = text.strip()

    start = s.find("{")
    if start < 0:
        return None, None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = s[start:i+1]
                    try:
                        obj = json.loads(cand)
                        return cand, obj
                    except Exception:
                        return None, None
    return None, None


# ---------------------------
# model loading (mirror train.py)
# ---------------------------
def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

    model_args = {}
    model_args["llm"] = {
        "model_name_or_path": model_arguments.model_name_or_path,
        "cache_dir": model_arguments.cache_dir,
        "attn_implementation": model_arguments.attn_implementation,
    }
    model_args["vision_tower"] = {
        "model_name_or_path": model_arguments.vision_tower.split(":")[-1],
    }
    if getattr(model_arguments, "vision_tower2", "") != "":
        model_args["vision_tower"]["model_name_or_path2"] = model_arguments.vision_tower2.split(":")[-1]
    model_args["connector"] = {
        "connector_type": model_arguments.connector_type,
    }
    return model_args


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, EvalArguments))
    model_args, data_args, train_args, eval_args = parser.parse_args_into_dataclasses()

    logger_setting(getattr(train_args, "output_dir", None))

    if getattr(model_args, "attn_implementation", None) in (None, "", "flash_attention_2"):
        model_args.attn_implementation = "sdpa"

    training_recipe = TrainingRecipeFactory(train_args.training_recipe)(train_args)

    margs = load_settings(model_args, data_args, train_args)
    margs = training_recipe.add_args(margs)

    cfg = TinyLlavaConfig()
    cfg.load_from_config(model_args)
    model = TinyLlavaForConditionalGeneration(cfg)

    if train_args.pretrained_model_path is None:
        raise ValueError("--pretrained_model_path must be set to your stage2 output root (NOT checkpoint subdir).")
    model = training_recipe.load(model, margs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    tokenizer = model.tokenizer

    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
            eos_ids.append(im_end_id)
    except Exception:
        pass
    eos_token_id = eos_ids[0] if len(eos_ids) == 1 else eos_ids

    rows = read_jsonl(eval_args.eval_jsonl)
    rows = per_class_sample(rows, eval_args.eval_per_class_sample, eval_args.eval_sample_seed)

    gen_kwargs = dict(
        max_new_tokens=int(eval_args.eval_max_new_tokens),
        do_sample=bool(eval_args.eval_do_sample),
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    if eval_args.eval_do_sample:
        if eval_args.eval_temperature and eval_args.eval_temperature > 0:
            gen_kwargs["temperature"] = float(eval_args.eval_temperature)
        if eval_args.eval_top_k and eval_args.eval_top_k > 0:
            gen_kwargs["top_k"] = int(eval_args.eval_top_k)

    out_rows = []
    correct = 0
    valid = 0
    per_class = {}

    bs = max(1, int(eval_args.eval_batch_size))
    for start in tqdm(range(0, len(rows), bs), desc="Eval", ncols=100):
        batch_rows = rows[start:start+bs]

        input_id_list = []
        image_list = []
        prompts = []

        gt_classes = []
        relpaths = []
        targets = []

        for r in batch_rows:
            rel = r.get("sample_relpath")
            if not rel:
                raise KeyError("eval jsonl row missing required field: sample_relpath")
            relpaths.append(rel)

            gt = r.get("class")
            if gt is None:
                try:
                    gt = json.loads(r.get("target", "{}")).get("class", "UNKNOWN")
                except Exception:
                    gt = "UNKNOWN"
            gt_classes.append(gt)

            targets.append(r.get("target", ""))

            user_text = (
                f"{DEFAULT_IMAGE_TOKEN}\n"
                "Return a single JSON object ONLY (no extra text). "
                "Keys MUST appear in this order: class, traits, evidence, description, notes."
            )
            input_ids, prompt = build_prompt_qwen_style(user_text, tokenizer)
            input_id_list.append(input_ids)
            prompts.append(prompt)

            img = load_npy_image(data_args.image_folder, rel, eval_args.eval_byte_length)  # (1,L)
            image_list.append(img)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [x.squeeze(0) for x in input_id_list],
            batch_first=True,
            padding_value=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        ).to(device)

        attn_mask = input_ids.ne(tokenizer.pad_token_id).to(device) if tokenizer.pad_token_id is not None else None

        images = torch.stack(image_list, dim=0).to(device)

        with torch.no_grad():
            # Using autocast for automatic mixed precision
            with torch.amp.autocast(device_type='cuda'):
                gen_ids = model.generate(
                    inputs=input_ids,
                    images=images,
                    attention_mask=attn_mask,
                    **gen_kwargs
                )

        for b, r in enumerate(batch_rows):
            gt = gt_classes[b]
            rel = relpaths[b]

            gen_text = tokenizer.decode(gen_ids[b], skip_special_tokens=True)

            pred_json, pred_obj = extract_first_json_obj(gen_text)
            pred_class = None
            if pred_obj is not None and isinstance(pred_obj, dict):
                pred_class = pred_obj.get("class")
                valid += 1

            ok = (pred_class == gt)

            if gt not in per_class:
                per_class[gt] = {"n": 0, "correct": 0, "valid": 0}
            per_class[gt]["n"] += 1
            per_class[gt]["valid"] += 1 if pred_obj is not None else 0
            per_class[gt]["correct"] += 1 if ok else 0

            correct += 1 if ok else 0

            out_rows.append({
                "sample_relpath": rel,
                "gt_class": gt,
                "target": r.get("target", ""),
                "prompt": prompts[b],
                "pred_text": gen_text,
                "pred_json": pred_json,
                "pred_obj": pred_obj,
                "pred_class": pred_class,
                "correct": ok,
            })

    total = len(rows)
    acc = correct / total if total else 0.0
    valid_rate = valid / total if total else 0.0

    print(f"[RESULT] total={total} | acc={acc:.4f} | valid_json_rate={valid_rate:.4f}")
    print("[RESULT] per-class accuracy:")
    for cls in sorted(per_class.keys()):
        n = per_class[cls]["n"]
        c = per_class[cls]["correct"]
        v = per_class[cls]["valid"]
        print(f"  - {cls:20s}  n={n:4d}  acc={c/n:.4f}  valid={v/n:.4f}")

    write_jsonl(eval_args.eval_out_jsonl, out_rows)

    summary_path = os.path.splitext(eval_args.eval_out_jsonl)[0] + ".summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"total={total}\n")
        f.write(f"acc={acc:.6f}\n")
        f.write(f"valid_json_rate={valid_rate:.6f}\n")
        f.write("per_class:\n")
        for cls in sorted(per_class.keys()):
            n = per_class[cls]["n"]
            c = per_class[cls]["correct"]
            v = per_class[cls]["valid"]
            f.write(f"{cls}\t{n}\t{c/n:.6f}\t{v/n:.6f}\n")

    print(f"[INFO] wrote: {eval_args.eval_out_jsonl}")
    print(f"[INFO] wrote: {summary_path}")


if __name__ == "__main__":
    main()
