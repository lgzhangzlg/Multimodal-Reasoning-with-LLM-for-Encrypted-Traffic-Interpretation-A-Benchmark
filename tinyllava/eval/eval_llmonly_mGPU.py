#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_llmonly_mGPU.py — Evaluation script for LLM-only ablation experiment.

LLM-only 模式说明：
  - user_text 已包含十六进制字节序列，无 <image> token，无 <cls_placeholder>
  - 不加载 npy 文件，不传 images 给模型，走纯文本推理路径
  - class 字段由 LLM 自行预测并输出到 JSON 中

Usage:
    # 单卡
    python eval_llmonly_mGPU.py \
        --checkpoint_path  /root/autodl-tmp/train_out/xxx/checkpoint-xxx \
        --vision_tower_path /root/autodl-tmp/.../checkpoint-best.pth \
        --eval_data_path   /root/autodl-tmp/Datasets/ISCXVPN2016/nlp_output_llmonly_200_6000/test.jsonl \
        --output_dir       /root/autodl-tmp/eval_out/ISCXVPN2016_llmonly \
        --samples_per_class 99999999 --batch_size 10 --max_new_tokens 500

    # 多卡
    python eval_llmonly_mGPU.py \
        --checkpoint_path  ... --vision_tower_path ... \
        --eval_data_path   ... --output_dir ... \
        --samples_per_class 99999999 --batch_size 10 --max_new_tokens 500 \
        --num_gpus 4
"""

import argparse
import json
import math
import os
import random
import re
import shutil
import tempfile
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer

from tinyllava.model import TinyLlavaForConditionalGeneration
from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from tinyllava.utils.constants import IGNORE_INDEX
from tinyllava.data.template import TemplateFactory
from tinyllava.data.template.base import Template


# ============================================================
# 工具函数
# ============================================================

def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def strip_think(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'</?think>', '', text)
    return text.strip()


# ============================================================
# 分层采样（不需要 npy 校验，所有样本都有效）
# ============================================================

def stratified_sample(data: List[Dict], samples_per_class: int) -> List[Dict]:
    groups = defaultdict(list)
    for item in data:
        cls = item.get("class", "unknown")
        groups[cls].append(item)

    print("=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)
    for cls in sorted(groups):
        print(f"  {cls:<30}: {len(groups[cls]):>6} samples")
    print("=" * 70 + "\n")

    sampled = []
    for cls in sorted(groups):
        items = groups[cls]
        n = min(samples_per_class, len(items))
        sampled.extend(random.sample(items, n))

    random.shuffle(sampled)
    print(f"Sampled {len(sampled)} samples total ({samples_per_class} per class max)\n")
    return sampled


# ============================================================
# Dataset（纯文本，不加载 npy）
# ============================================================

class EvalDatasetLLMOnly(Dataset):
    def __init__(
        self,
        data:      List[Dict],
        tokenizer: transformers.PreTrainedTokenizer,
        template:  Template,
    ):
        super().__init__()
        self.data      = data
        self.tokenizer = tokenizer
        self.template  = template

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, Any]:
        sample    = self.data[i]
        user_text = sample.get("user_text", "").strip()

        if not user_text:
            raise KeyError(f"Sample missing user_text: {sample.get('sample_id', i)}")

        messages = [
            {"from": "human", "value": user_text},
            {"from": "gpt",   "value": ""},
        ]
        encoded   = self.template.encode(messages, self.tokenizer, mode="eval")
        input_ids = encoded["input_ids"]

        return {
            "input_ids":      input_ids,
            "ground_truth":   sample.get("target", ""),
            "gt_class":       sample.get("class", "unknown"),
            "sample_relpath": sample.get("sample_relpath", ""),
        }


# ============================================================
# CollateFn（无 images）
# ============================================================

class CollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids       = [item["input_ids"]      for item in batch]
        ground_truths   = [item["ground_truth"]   for item in batch]
        sample_relpaths = [item["sample_relpath"] for item in batch]
        gt_classes      = [item["gt_class"]       for item in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return {
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
            "ground_truths":   ground_truths,
            "sample_relpaths": sample_relpaths,
            "gt_classes":      gt_classes,
        }


# ============================================================
# 模型加载
# ============================================================

def load_model(
    checkpoint_path:   str,
    device:            str = "cuda",
    vision_tower_path: str = None,
) -> TinyLlavaForConditionalGeneration:
    print(f"[{device}] Loading model from: {checkpoint_path}")

    parent_dir  = os.path.dirname(checkpoint_path) if "checkpoint-" in os.path.basename(checkpoint_path) else checkpoint_path
    config_path = checkpoint_path if os.path.exists(os.path.join(checkpoint_path, "config.json")) else parent_dir
    config      = TinyLlavaConfig.from_pretrained(config_path)
    model       = TinyLlavaForConditionalGeneration(config)

    # ── LLM ──
    llm_path = config.llm_model_name_or_path
    if os.path.exists(llm_path):
        model.language_model = model.language_model.from_pretrained(llm_path)
    else:
        weight_path = os.path.join(parent_dir, "language_model", "pytorch_model.bin")
        if os.path.exists(weight_path):
            state   = torch.load(weight_path, map_location="cpu", weights_only=False)
            cleaned = {k.replace(".base_layer.", "."): v for k, v in state.items()}
            model.language_model.load_state_dict(cleaned, strict=False)
            print(f"  [{device}] LLM loaded from saved weights")

    # ── Vision Tower（加载但推理时不使用）──
    if vision_tower_path is not None:
        try:
            model.vision_tower._load_model(
                vision_tower_name=None,
                pretrained_vision_tower_path=vision_tower_path,
            )
            print(f"  [{device}] vision_tower loaded (not used in inference)")
        except Exception as e:
            print(f"  [{device}] WARNING: vision_tower load failed: {e} (ignored)")
    else:
        print(f"  [{device}] --vision_tower_path not provided, skipping vision tower load")

    # ── Connector ──
    conn_path = os.path.join(parent_dir, "connector", "pytorch_model.bin")
    if os.path.exists(conn_path):
        state = torch.load(conn_path, map_location="cpu", weights_only=False)
        model.connector.load_state_dict(state, strict=False)
        print(f"  [{device}] connector loaded (not used in inference)")

    # ── LoRA Adapter ──
    adapter_cfg   = os.path.join(checkpoint_path, "adapter_config.json")
    adapter_model = os.path.join(checkpoint_path, "adapter_model.safetensors")
    if os.path.exists(adapter_cfg) and os.path.exists(adapter_model):
        size_mb = os.path.getsize(adapter_model) / 1024 / 1024
        print(f"  [{device}] Adapter size: {size_mb:.1f} MB")
        if size_mb < 0.001:
            print(f"  [{device}] WARNING: Adapter file too small, skipping")
        else:
            model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
            try:
                model = model.merge_and_unload()
                print(f"  [{device}] LoRA merged")
            except Exception as e:
                print(f"  [{device}] LoRA merge failed: {e}")
    else:
        print(f"  [{device}] WARNING: No adapter found, using base weights only")

    model = model.to(device)
    model.eval()

    # ── Tokenizer ──
    ckpt_tokenizer   = AutoTokenizer.from_pretrained(parent_dir)
    model.tokenizer  = ckpt_tokenizer
    ckpt_tokenizer   = AutoTokenizer.from_pretrained(parent_dir)
    model.tokenizer  = ckpt_tokenizer
    print(f"  [{device}] tokenizer replaced, vocab_size={len(ckpt_tokenizer)}")
    print(f"[{device}] Model loaded successfully\n")
    return model


# ============================================================
# 推理（纯文本路径，images=None）
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader:     DataLoader,
    device:         str,
    max_new_tokens: int,
    temperature:    float,
    top_p:          float,
) -> tuple:
    tokenizer = model.tokenizer

    im_end_id     = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_token_ids = list({t for t in [tokenizer.eos_token_id, im_end_id] if t is not None})
    print(f"[{device}] eos_token_ids: {eos_token_ids}")

    predictions     = []
    ground_truths   = []
    gt_classes_all  = []
    sample_relpaths = []

    for batch in tqdm(dataloader, desc=f"[{device}] Evaluating"):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # ── 纯文本推理：直接获取 token embeddings，不经过 vision tower ──
        inputs_embeds = model.language_model.get_input_embeddings()(input_ids)

        gen_kwargs = dict(
            inputs_embeds   = inputs_embeds,
            attention_mask  = attention_mask,
            max_new_tokens  = max_new_tokens,
            use_cache       = True,
            eos_token_id    = eos_token_ids,
            repetition_penalty = 1.1,
        )
        if temperature > 0:
            gen_kwargs.update(temperature=temperature, top_p=top_p, do_sample=True)
        else:
            gen_kwargs["do_sample"] = False

        outputs         = model.language_model.generate(**gen_kwargs)
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        final_texts = []
        for text in generated_texts:
            text = strip_think(text)
            obj  = parse_json_output(text)
            if obj is not None and isinstance(obj, dict):
                final_texts.append(json.dumps(obj, ensure_ascii=False))
            else:
                final_texts.append(json.dumps({"class": None, "raw": text}, ensure_ascii=False))

        predictions.extend([{"text": t} for t in final_texts])
        ground_truths.extend(batch["ground_truths"])
        sample_relpaths.extend(batch["sample_relpaths"])
        gt_classes_all.extend(batch["gt_classes"])

    return predictions, ground_truths, gt_classes_all, sample_relpaths


# ============================================================
# 子进程（多卡）
# ============================================================

def worker_process(rank: int, args, sampled_data: List[Dict], tmp_dir: str):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    chunk_size = math.ceil(len(sampled_data) / args.num_gpus)
    start = rank * chunk_size
    end   = min(start + chunk_size, len(sampled_data))
    chunk = sampled_data[start:end]

    if not chunk:
        print(f"[rank {rank}] 无数据，退出")
        return

    model     = load_model(args.checkpoint_path, device=device,
                           vision_tower_path=args.vision_tower_path)
    tokenizer = model.tokenizer
    template  = TemplateFactory(args.conv_version)()

    dataset = EvalDatasetLLMOnly(
        data      = chunk,
        tokenizer = tokenizer,
        template  = template,
    )
    dataloader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        collate_fn  = CollateFn(tokenizer),
        num_workers = 0,
        pin_memory  = False,
    )

    predictions, ground_truths, gt_classes_all, sample_relpaths = evaluate(
        model          = model,
        dataloader     = dataloader,
        device         = device,
        max_new_tokens = args.max_new_tokens,
        temperature    = args.temperature,
        top_p          = args.top_p,
    )

    tmp_file = os.path.join(tmp_dir, f"rank_{rank}.jsonl")
    with open(tmp_file, "w", encoding="utf-8") as f:
        for pred, gt, cls, rel in zip(predictions, ground_truths, gt_classes_all, sample_relpaths):
            f.write(json.dumps({
                "prediction":     pred["text"],
                "ground_truth":   gt,
                "gt_class":       cls,
                "sample_relpath": rel,
            }, ensure_ascii=False) + "\n")

    print(f"[rank {rank}] Done. {len(predictions)} samples → {tmp_file}")


# ============================================================
# 指标计算
# ============================================================

def parse_json_output(text: str) -> Optional[Dict]:
    text = strip_think(text)
    for marker in ("```json", "```"):
        if marker in text:
            start = text.find(marker) + len(marker)
            end   = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s = text.find("{")
        e = text.rfind("}") + 1
        if s != -1 and e > s:
            try:
                return json.loads(text[s:e])
            except Exception:
                pass
    return None


def calculate_metrics(
    predictions:   List[Dict],
    ground_truths: List[str],
    gt_classes:    List[str],
) -> Dict:
    required_fields = {"class", "traits", "evidence", "description", "notes"}
    class_metrics   = defaultdict(lambda: {"total": 0, "correct": 0, "valid_json": 0})

    total = len(predictions)
    valid_json_count = complete_fields_count = exact_match_count = 0

    for pred, gt_str, gt_class in zip(predictions, ground_truths, gt_classes):
        pred_json = parse_json_output(pred.get("text", ""))
        class_metrics[gt_class]["total"] += 1

        if pred_json is not None and isinstance(pred_json, dict):
            valid_json_count += 1
            class_metrics[gt_class]["valid_json"] += 1
            if required_fields.issubset(pred_json.keys()):
                complete_fields_count += 1
            if pred_json.get("class") == gt_class:
                exact_match_count += 1
                class_metrics[gt_class]["correct"] += 1

    per_class = {}
    for cls, cm in class_metrics.items():
        t = cm["total"]
        per_class[cls] = {
            "total_samples":    t,
            "json_parse_rate":  cm["valid_json"] / t * 100 if t else 0,
            "class_match_rate": cm["correct"]    / t * 100 if t else 0,
        }

    return {
        "total_samples":           total,
        "valid_json_count":        valid_json_count,
        "complete_fields_count":   complete_fields_count,
        "json_parse_rate":         valid_json_count      / total * 100 if total else 0,
        "field_completeness_rate": complete_fields_count / total * 100 if total else 0,
        "class_exact_match_rate":  exact_match_count     / total * 100 if total else 0,
        "per_class_metrics":       per_class,
    }


# ============================================================
# 结果保存
# ============================================================

def save_results(predictions, ground_truths, metrics, output_dir, sample_relpaths=None):
    os.makedirs(output_dir, exist_ok=True)

    pred_file = os.path.join(output_dir, "predictions.jsonl")
    with open(pred_file, "w", encoding="utf-8") as f:
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            row = {"sample_id": i, "prediction": pred["text"], "ground_truth": gt}
            if sample_relpaths:
                row["sample_relpath"] = sample_relpaths[i]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Predictions → {pred_file}")

    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics     → {metrics_file}")

    print("\n" + "=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)
    for k, v in metrics.items():
        if k not in ("per_class_metrics",):
            print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("PER-CLASS METRICS")
    print("=" * 70)
    print(f"  {'Class':<30} {'Samples':>8} {'JSON%':>8} {'Match%':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    for cls in sorted(metrics["per_class_metrics"]):
        cm = metrics["per_class_metrics"][cls]
        print(f"  {cls:<30} {cm['total_samples']:>8} "
              f"{cm['json_parse_rate']:>7.1f}% {cm['class_match_rate']:>7.1f}%")
    print("=" * 70 + "\n")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path",   required=True)
    parser.add_argument("--vision_tower_path", default=None,
                        help="NetMamba checkpoint 路径（可选，LLM-only 推理不使用）")
    parser.add_argument("--eval_data_path",    required=True,
                        help="LLM-only test.jsonl 路径")
    parser.add_argument("--output_dir",        default="./eval_results")
    parser.add_argument("--samples_per_class", type=int,   default=10)
    parser.add_argument("--batch_size",        type=int,   default=4)
    parser.add_argument("--max_new_tokens",    type=int,   default=500)
    parser.add_argument("--temperature",       type=float, default=0.0)
    parser.add_argument("--top_p",             type=float, default=0.9)
    parser.add_argument("--conv_version",      default="qwen3_instruct")
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--num_gpus",          type=int,   default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading dataset...")
    full_data = read_jsonl(args.eval_data_path)
    print(f"Total samples: {len(full_data)}")

    sampled_data = stratified_sample(full_data, args.samples_per_class)

    if args.num_gpus <= 1:
        device    = "cuda:0"
        model     = load_model(args.checkpoint_path, device=device,
                               vision_tower_path=args.vision_tower_path)
        tokenizer = model.tokenizer
        template  = TemplateFactory(args.conv_version)()

        dataset = EvalDatasetLLMOnly(
            data      = sampled_data,
            tokenizer = tokenizer,
            template  = template,
        )
        dataloader = DataLoader(
            dataset,
            batch_size  = args.batch_size,
            shuffle     = False,
            collate_fn  = CollateFn(tokenizer),
            num_workers = 4,
            pin_memory  = True,
        )
        predictions, ground_truths, gt_classes_all, sample_relpaths = evaluate(
            model          = model,
            dataloader     = dataloader,
            device         = device,
            max_new_tokens = args.max_new_tokens,
            temperature    = args.temperature,
            top_p          = args.top_p,
        )

    else:
        import torch.multiprocessing as mp
        num_gpus = min(args.num_gpus, torch.cuda.device_count())
        if num_gpus < args.num_gpus:
            print(f"WARNING: 只检测到 {num_gpus} 张 GPU")
        args.num_gpus = num_gpus
        print(f"[INFO] 多卡推理：{num_gpus} 张 GPU，共 {len(sampled_data)} 样本")

        tmp_dir = tempfile.mkdtemp(prefix="eval_llmonly_tmp_")
        mp.set_start_method("spawn", force=True)

        processes = []
        for rank in range(num_gpus):
            p = mp.Process(target=worker_process, args=(rank, args, sampled_data, tmp_dir))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        predictions     = []
        ground_truths   = []
        gt_classes_all  = []
        sample_relpaths = []

        for rank in range(num_gpus):
            tmp_file = os.path.join(tmp_dir, f"rank_{rank}.jsonl")
            if not os.path.exists(tmp_file):
                print(f"WARNING: rank {rank} 结果文件不存在")
                continue
            with open(tmp_file, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    predictions.append({"text": row["prediction"]})
                    ground_truths.append(row["ground_truth"])
                    gt_classes_all.append(row["gt_class"])
                    sample_relpaths.append(row["sample_relpath"])

        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[INFO] 合并完成，共 {len(predictions)} 条结果")

    metrics = calculate_metrics(predictions, ground_truths, gt_classes_all)
    metrics["eval_info"] = {
        "total_dataset":     len(full_data),
        "sampled":           len(sampled_data),
        "samples_per_class": args.samples_per_class,
        "checkpoint":        args.checkpoint_path,
        "num_gpus":          args.num_gpus,
        "seed":              args.seed,
    }

    save_results(predictions, ground_truths, metrics, args.output_dir, sample_relpaths)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
