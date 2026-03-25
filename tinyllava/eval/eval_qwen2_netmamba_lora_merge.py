#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import argparse
import random
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from peft import PeftModel

# TinyLLaVA imports (your repo)
from tinyllava.model import TinyLlavaForConditionalGeneration
from tinyllava.model.configuration_tinyllava import TinyLlavaConfig

# constants
from tinyllava.utils.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)

from tinyllava.utils.prompt_utils import (
    build_user_text,
    extract_class_from_label_sentence,
    strip_path_leak,
    remove_leading_class_sentence,
)

# ---------------------------
# jsonl helpers
# ---------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------
# traffic npy loader
# ---------------------------
def load_npy_traffic(sample_relpath: str, image_folder: str) -> torch.Tensor:
    base, _ = os.path.splitext(sample_relpath)
    npy_rel = base + ".npy"
    npy_path = os.path.join(image_folder, npy_rel)
    if not os.path.exists(npy_path):
        raise FileNotFoundError(npy_path)

    import numpy as np
    arr = np.load(npy_path)
    x = torch.from_numpy(arr).float()

    # Normalize to (C, L)
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

    # NetMamba forward asserts C==1
    if x.shape[0] > 1:
        x = x[:1, :]

    return x


# ---------------------------
# metrics
# ---------------------------
def _tokenize_simple(text: str) -> List[str]:
    text = (text or "").strip().lower()
    return re.findall(r"[a-z0-9]+", text)


def rouge_l_f1(pred: str, ref: str) -> float:
    p = _tokenize_simple(pred)
    r = _tokenize_simple(ref)
    if not p or not r:
        return 0.0

    n, m = len(p), len(r)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if p[i - 1] == r[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp

    lcs = dp[m]
    prec = lcs / n
    rec = lcs / m
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def bleu4(pred: str, ref: str, smooth: float = 1.0) -> float:
    p = _tokenize_simple(pred)
    r = _tokenize_simple(ref)
    if not p or not r:
        return 0.0

    def ngrams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(0, len(tokens) - n + 1)]

    bp = 1.0
    if len(p) < len(r):
        bp = math.exp(1 - len(r) / max(1, len(p)))

    weights = [0.25, 0.25, 0.25, 0.25]
    score_log = 0.0

    from collections import Counter
    for n, w in zip([1, 2, 3, 4], weights):
        pn = ngrams(p, n)
        rn = ngrams(r, n)
        if not pn:
            return 0.0
        pc = Counter(pn)
        rc = Counter(rn)

        match = 0
        total = 0
        for ng, cnt in pc.items():
            total += cnt
            match += min(cnt, rc.get(ng, 0))

        prec_n = (match + smooth) / (total + smooth)
        score_log += w * math.log(prec_n)

    return float(bp * math.exp(score_log))


# ---------------------------
# "mention gold => correct"
# ---------------------------
def mentioned_gold(gen_text: str, gold: str) -> bool:
    gen_text = gen_text or ""
    gold = (gold or "").strip()
    if not gold:
        return False
    pattern = r"(?i)(?<![A-Za-z0-9])" + re.escape(gold) + r"(?![A-Za-z0-9])"
    return re.search(pattern, gen_text) is not None


# ---------------------------
# stratified sampling
# ---------------------------
def stratified_sample(data: List[Dict[str, Any]], max_samples: int, seed: int = 0) -> List[Dict[str, Any]]:
    if max_samples is None or max_samples <= 0 or max_samples >= len(data):
        return data

    random.seed(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for s in data:
        cls = s.get("class") or extract_class_from_label_sentence(s.get("label_sentence", ""))
        cls = (cls or "").strip()
        buckets.setdefault(cls, []).append(s)

    classes = [c for c in buckets.keys() if c]
    if not classes:
        random.shuffle(data)
        return data[:max_samples]

    per = math.ceil(max_samples / len(classes))
    picked = []
    for c in classes:
        arr = buckets[c]
        random.shuffle(arr)
        picked.extend(arr[:per])

    if len(picked) > max_samples:
        random.shuffle(picked)
        picked = picked[:max_samples]
    elif len(picked) < max_samples:
        remain = [x for x in data if x not in picked]
        random.shuffle(remain)
        picked.extend(remain[: max_samples - len(picked)])

    return picked


# ---------------------------
# prompt + tokenizer helpers
# ---------------------------
def build_inputs(tokenizer, sample: Dict[str, Any]) -> torch.Tensor:
    user_text = build_user_text(sample, use_hint=False)
    content = DEFAULT_IMAGE_TOKEN + "\n" + user_text
    messages = [{"role": "user", "content": content}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return input_ids


def extract_ref_desc(sample: Dict[str, Any]) -> str:
    ref = sample.get("nl_description", "") or ""
    if not ref:
        ref = sample.get("sample_description", "") or ""
    ref = strip_path_leak(ref).strip()
    ref = remove_leading_class_sentence(ref).strip()
    return ref


def extract_gen_desc(gen_text: str) -> str:
    t = strip_path_leak(gen_text).strip()
    t = remove_leading_class_sentence(t).strip()
    return t


def pad_batch(input_ids_list: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    max_len = max(x.shape[1] for x in input_ids_list)
    out = []
    for x in input_ids_list:
        if x.shape[1] < max_len:
            pad = torch.full((1, max_len - x.shape[1]), pad_id, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        out.append(x)
    return torch.cat(out, dim=0)


def ensure_image_token(tokenizer, model):
    """
    Ensure tokenizer has DEFAULT_IMAGE_TOKEN, and we can map it to a real token id.
    If missing, add it and resize embeddings.
    """
    tok_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    if tok_id is None or tok_id == tokenizer.unk_token_id:
        added = tokenizer.add_special_tokens({"additional_special_tokens": [DEFAULT_IMAGE_TOKEN]})
        print(f"[Fix] Added {DEFAULT_IMAGE_TOKEN} to tokenizer. added={added}")
        if added > 0:
            model.resize_token_embeddings(len(tokenizer))
    tok_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    return tok_id


def replace_image_token_id_to_index(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    out = input_ids.clone()
    out[out == image_token_id] = IMAGE_TOKEN_INDEX
    return out


# ---------------------------
# debug helpers
# ---------------------------
def debug_print_features(model, images: torch.Tensor, prefix="[Debug]"):
    with torch.no_grad():
        feats = model.encode_images(images)
        print(f"{prefix} encode_images shape: {tuple(feats.shape)} dtype: {feats.dtype}")
        print(
            f"{prefix} feats stats: "
            f"min={float(feats.min()):.6f} max={float(feats.max()):.6f} "
            f"mean={float(feats.mean()):.6f} std={float(feats.std()):.6f}"
        )
        print(f"{prefix} feats finite: {bool(torch.isfinite(feats).all())}")
    return feats


def decode_batch_generated_only(tokenizer, out_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[str]:
    """
    Decode ONLY the generated part (new tokens after prompt),
    and also tolerate the case where model immediately ends (empty).
    """
    texts = []
    for i in range(out_ids.shape[0]):
        prompt_len = int(attention_mask[i].sum().item())
        gen_seq = out_ids[i][prompt_len:]
        gen_text = tokenizer.decode(gen_seq, skip_special_tokens=True).strip()
        texts.append(gen_text)
    return texts


def gen_new_tokens_len(out_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[int]:
    lens = []
    for i in range(out_ids.shape[0]):
        prompt_len = int(attention_mask[i].sum().item())
        lens.append(int(out_ids[i].shape[0] - prompt_len))
    return lens


# ---------------------------
# main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--eval_jsonl", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)

    parser.add_argument("--vision_tower", type=str, default="netmamba")
    parser.add_argument("--connector_type", type=str, default="mlp2x_gelu")
    parser.add_argument("--mm_vision_select_layer", type=int, default=-2)
    parser.add_argument("--image_aspect_ratio", type=str, default="square")
    parser.add_argument("--pretrained_vision_tower_path", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--merge_lora", action="store_true")
    parser.add_argument("--dump_jsonl", type=str, required=True)

    # debug
    parser.add_argument("--debug_check", action="store_true", help="print token/feature stats for first batch")
    parser.add_argument("--debug_contrast", action="store_true", help="run images vs zero-images contrast on first batch")

    args = parser.parse_args()

    ckpt_dir = os.path.join(args.model_dir, args.checkpoint)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(ckpt_dir)

    # ------------------------
    # load eval data
    # ------------------------
    data = read_jsonl(args.eval_jsonl)
    if args.max_samples and args.max_samples > 0:
        data = stratified_sample(data, args.max_samples, seed=args.seed)

    label_set = sorted({
        (s.get("class") or extract_class_from_label_sentence(s.get("label_sentence", "")) or "").strip()
        for s in data
    })

    print(f"[Eval] loaded {len(data)} samples, label_count={len(label_set)}")
    if args.max_samples and args.max_samples > 0:
        print(f"[Eval] stratified max_samples={args.max_samples}")

    # ------------------------
    # build model
    # ------------------------
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = TinyLlavaConfig.from_pretrained(args.model_dir)
    config.llm_model_name_or_path = args.llm_dir
    config.tokenizer_name_or_path = args.llm_dir
    config.vision_model_name_or_path = args.vision_tower
    config.connector_type = args.connector_type
    config.vision_feature_layer = args.mm_vision_select_layer
    config.image_aspect_ratio = args.image_aspect_ratio

    model = TinyLlavaForConditionalGeneration(config)

    model.load_llm(model_name_or_path=args.llm_dir)

    vt_kwargs = {}
    if args.pretrained_vision_tower_path:
        vt_kwargs["pretrained_vision_tower_path"] = args.pretrained_vision_tower_path
    model.load_vision_tower(model_name_or_path=args.vision_tower, **vt_kwargs)

    connector_ckpt = os.path.join(args.model_dir, "connector", "pytorch_model.bin")
    if os.path.exists(connector_ckpt):
        sd = torch.load(connector_ckpt, map_location="cpu")
        missing, unexpected = model.connector.load_state_dict(sd, strict=False)
        print(f"[Load] connector: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[Warn] connector ckpt not found: {connector_ckpt}")

    print(f"[Load] LoRA from: {ckpt_dir}")
    model = PeftModel.from_pretrained(model, ckpt_dir, is_trainable=False)

    if args.merge_lora:
        print("[Load] merge LoRA into base weights ...")
        model = model.merge_and_unload()

    model.eval().to(device)

    tokenizer = model.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_token_id = ensure_image_token(tokenizer, model)
    print(f"[Debug] tokenizer '{DEFAULT_IMAGE_TOKEN}' id = {image_token_id}, will be replaced to IMAGE_TOKEN_INDEX={IMAGE_TOKEN_INDEX}")

    # ------------------------
    # inference loop
    # ------------------------
    results = []
    correct = 0
    sum_rouge = 0.0
    sum_bleu = 0.0
    n_metric = 0

    bs = max(1, args.batch_size)
    pbar = tqdm(range(0, len(data), bs), desc="Eval", ncols=100)

    for start in pbar:
        batch = data[start:start + bs]

        input_ids_list = []
        images_list = []
        gold_list = []
        sample_id_list = []
        sample_relpath_list = []
        ref_desc_list = []

        for s in batch:
            gold = s.get("class") or extract_class_from_label_sentence(s.get("label_sentence", ""))
            gold = (gold or "").strip()

            inp = build_inputs(tokenizer, s)
            inp = replace_image_token_id_to_index(inp, image_token_id)
            input_ids_list.append(inp)

            rel = s.get("sample_relpath") or ""
            if not rel:
                sp = s.get("sample_path", "")
                rel = os.path.basename(sp)
            x = load_npy_traffic(rel, args.image_folder)
            images_list.append(x)

            gold_list.append(gold)
            sample_id_list.append(s.get("sample_id", ""))
            sample_relpath_list.append(rel)
            ref_desc_list.append(extract_ref_desc(s))

        input_ids = pad_batch(input_ids_list, tokenizer.pad_token_id).to(device)
        images = torch.stack(images_list, dim=0).to(device)
        if model.dtype in (torch.float16, torch.bfloat16):
            images = images.to(dtype=model.dtype)

        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device)

        if args.debug_check and start == 0:
            counts = [(input_ids[i] == IMAGE_TOKEN_INDEX).sum().item() for i in range(input_ids.shape[0])]
            print(f"[Debug] IMAGE_TOKEN_INDEX({IMAGE_TOKEN_INDEX}) count per sample: {counts}")
            debug_print_features(model, images[:1], prefix="[Debug]")

        # gen kwargs：do_sample=False 时不要传 temperature/top_p（不然全是 warning）
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
        )
        if args.do_sample:
            gen_kwargs.update(dict(temperature=args.temperature, top_p=args.top_p))

        with torch.no_grad():
            out_ids_A = model.generate(
                inputs=input_ids,
                images=images,
                attention_mask=attention_mask,
                **gen_kwargs
            )

            if args.debug_contrast and start == 0:
                images_zero = torch.zeros_like(images)
                out_ids_B = model.generate(
                    inputs=input_ids,
                    images=images_zero,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )

                txtA = decode_batch_generated_only(tokenizer, out_ids_A, attention_mask)
                txtB = decode_batch_generated_only(tokenizer, out_ids_B, attention_mask)
                lenA = gen_new_tokens_len(out_ids_A, attention_mask)
                lenB = gen_new_tokens_len(out_ids_B, attention_mask)

                print("\n========== [Debug] images vs zero-images (generated-only) ==========")
                for i in range(min(4, len(txtA))):
                    print(f"[{i}] gold={gold_list[i]}")
                    print(f"new_tokens A={lenA[i]} | B={lenB[i]}")
                    print("A(normal):", (txtA[i][:300] if txtA[i] else "<EMPTY>").replace("\n", "\\n"))
                    print("B(zero)  :", (txtB[i][:300] if txtB[i] else "<EMPTY>").replace("\n", "\\n"))
                    print("same?    :", txtA[i] == txtB[i])
                    print("-" * 60)
                print("===============================================================\n")

        # decode + metrics
        gen_texts = decode_batch_generated_only(tokenizer, out_ids_A, attention_mask)

        for i in range(len(batch)):
            gen_text = gen_texts[i]

            gold = gold_list[i]
            is_ok = mentioned_gold(gen_text, gold)
            if is_ok:
                correct += 1

            gen_desc = extract_gen_desc(gen_text)
            ref_desc = ref_desc_list[i]

            rL = rouge_l_f1(gen_desc, ref_desc)
            b4 = bleu4(gen_desc, ref_desc)

            sum_rouge += rL
            sum_bleu += b4
            n_metric += 1

            results.append(dict(
                sample_id=sample_id_list[i],
                sample_relpath=sample_relpath_list[i],
                gold=gold,
                pred=(gold if is_ok else ""),
                pred_raw="",
                is_correct=is_ok,
                gen_first_line=(gen_text.splitlines()[0] if gen_text else ""),
                gen_desc=gen_desc,
                ref_desc=ref_desc,
                rougeL_f1=rL,
                bleu4=b4,
                gen_text=gen_text,
            ))

        acc = correct / max(1, len(results))
        pbar.set_postfix({"acc": f"{acc:.4f}", "rougeL": f"{(sum_rouge/max(1,n_metric)):.4f}"})

    # summary
    acc = correct / max(1, len(results))
    avg_rouge = sum_rouge / max(1, n_metric)
    avg_bleu = sum_bleu / max(1, n_metric)

    print("\n========== Summary ==========")
    print(f"Samples: {len(results)}")
    print(f"Acc(mention-gold): {acc:.6f}")
    print(f"ROUGE-L(F1):       {avg_rouge:.6f}")
    print(f"BLEU-4:            {avg_bleu:.6f}")
    print("=============================\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.dump_jsonl)) or ".", exist_ok=True)
    write_jsonl(args.dump_jsonl, results)
    print(f"[Saved] {args.dump_jsonl}")


if __name__ == "__main__":
    main()
