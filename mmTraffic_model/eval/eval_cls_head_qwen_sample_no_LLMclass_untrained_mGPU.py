#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stratified sampling evaluation for TinyLLaVA traffic classification.
Untrained variant (ablation baseline):
  - Connector: random initialization (weights NOT loaded)
  - LLM:       base model only, NO LoRA adapter loaded
  - NetMamba:  pretrained weights loaded as usual
  - Class injection: NetMamba CLS head prediction injected into prompt (no_LLMclass mode)

v4 改动：
1. 多卡推理：--num_gpus N，自动将样本切分给多个进程，每个进程独占一张卡
2. num_workers=4，数据加载并行化
3. 去掉 CUDA_LAUNCH_BLOCKING 依赖
4. 子进程结果写临时 jsonl，主进程合并后统一计算指标

Usage:
    # 单卡（默认）
    python eval_cls_head_qwen_sample_no_LLMclass_untrained.py \
        --checkpoint_path /root/autodl-tmp/train_out/xxx/checkpoint-288 \
        --vision_tower_path /root/autodl-tmp/.../checkpoint-best.pth \
        --eval_data_path  /root/autodl-tmp/Datasets/.../test.jsonl \
        --image_folder    /root/autodl-tmp/Datasets/.../npy \
        --output_dir      /root/autodl-tmp/eval_out/xxx \
        --samples_per_class 20 --batch_size 4 --max_new_tokens 400

    # 多卡（例如4卡）
    python eval_cls_head_qwen_sample_no_LLMclass_untrained.py \
        --checkpoint_path ... --vision_tower_path ... \
        --eval_data_path ... --image_folder ... --output_dir ... \
        --samples_per_class 20 --batch_size 4 --max_new_tokens 400 \
        --num_gpus 4
"""
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional
import argparse
import json
import os
import random
import re
import tempfile
import math
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer

from tinyllava.model import TinyLlavaForConditionalGeneration
from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from tinyllava.model.modeling_tinyllava import NETMAMBA_IDX_TO_CLASS
from tinyllava.utils.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, IGNORE_INDEX
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


def pcap_rel_to_npy_rel(sample_relpath: str) -> str:
    base, ext = os.path.splitext(sample_relpath)
    return sample_relpath if ext.lower() == ".npy" else (base + ".npy")


def check_npy_exists(sample: Dict, image_folder: str) -> bool:
    rel = sample.get("sample_relpath")
    if not rel:
        return False
    npy_rel = pcap_rel_to_npy_rel(rel)
    return os.path.exists(os.path.join(image_folder, npy_rel))


def strip_think(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'</?think>', '', text)
    return text.strip()


# ============================================================
# 文件验证 & 分层采样
# ============================================================

def filter_valid_samples(data: List[Dict], image_folder: str):
    valid, invalid = [], []
    class_stats = defaultdict(lambda: {"total": 0, "valid": 0, "invalid": 0})

    print("\nChecking NPY file existence...")
    for sample in tqdm(data, desc="Validating files"):
        cls = sample.get("class", "unknown")
        class_stats[cls]["total"] += 1
        if check_npy_exists(sample, image_folder):
            valid.append(sample)
            class_stats[cls]["valid"] += 1
        else:
            invalid.append(sample)
            class_stats[cls]["invalid"] += 1

    print("\n" + "=" * 70)
    print("FILE VALIDATION REPORT")
    print("=" * 70)
    print(f"{'Class':<20} {'Total':>8} {'Valid':>8} {'Invalid':>8} {'Valid%':>8}")
    print("-" * 70)
    for cls in sorted(class_stats):
        s = class_stats[cls]
        pct = s["valid"] / s["total"] * 100 if s["total"] else 0
        print(f"{cls:<20} {s['total']:>8} {s['valid']:>8} {s['invalid']:>8} {pct:>7.1f}%")
    print("-" * 70)
    total_pct = len(valid) / len(data) * 100 if data else 0
    print(f"{'TOTAL':<20} {len(data):>8} {len(valid):>8} {len(invalid):>8} {total_pct:>7.1f}%")
    print("=" * 70 + "\n")

    no_valid = [c for c, s in class_stats.items() if s["valid"] == 0]
    if no_valid:
        print("WARNING: Classes with NO valid NPY files:", no_valid)

    return valid, {
        "total": len(data),
        "valid": len(valid),
        "invalid": len(invalid),
        "class_stats": dict(class_stats),
    }


def stratified_sample(data: List[Dict], samples_per_class: int) -> List[Dict]:
    groups = defaultdict(list)
    for item in data:
        cls = item.get("class", "unknown")
        groups[cls].append(item)

    print("=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)
    for cls in sorted(groups):
        print(f"  {cls:<20}: {len(groups[cls]):>6} samples")
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
# Dataset
# ============================================================

class EvalDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        template: Template,
        byte_length: int = 1600,
    ):
        super().__init__()
        self.data         = data
        self.image_folder = image_folder
        self.tokenizer    = tokenizer
        self.template     = template
        self.byte_length  = byte_length

    def __len__(self):
        return len(self.data)

    def _load_npy(self, sample) -> torch.Tensor:
        rel     = sample["sample_relpath"]
        npy_rel = pcap_rel_to_npy_rel(rel)
        path    = os.path.join(self.image_folder, npy_rel)

        arr = np.load(path)
        arr = np.asarray(arr)

        if arr.ndim == 2 and arr.shape[0] == 1:
            pass
        elif arr.ndim == 1:
            arr = arr[None, :]
        else:
            arr = arr.reshape(1, -1)

        L = self.byte_length
        if arr.shape[1] < L:
            arr = np.concatenate([arr, np.zeros((1, L - arr.shape[1]), dtype=arr.dtype)], axis=1)
        elif arr.shape[1] > L:
            arr = arr[:, :L]

        return torch.from_numpy(arr.astype(np.float32)) / 255.0

    def __getitem__(self, i) -> Dict[str, Any]:
        sample = self.data[i]

        user_text = sample.get("user_text", "").strip()
        if not user_text:
            user_text = (
                f"{DEFAULT_IMAGE_TOKEN}\n"
                "Above are the raw traffic byte features extracted from a network flow.\n\n"
                "The traffic has been classified as: <cls_placeholder>.\n\n"
                "Based on the traffic byte features and the above classification result, "
                "return a single JSON object with the following keys:\n"
                "- traits: an object describing byte-level characteristics\n"
                "- evidence: a list of 2~4 concrete byte-level observations\n"
                "- description: a paragraph of 2~3 sentences summarizing the traffic\n"
                "- notes: a single security-relevant observation or recommendation"
            )

        messages = [
            {"from": "human", "value": user_text},
            {"from": "gpt",   "value": ""},
        ]
        encoded   = self.template.encode(messages, self.tokenizer, mode="eval")
        input_ids = encoded["input_ids"]
        image     = self._load_npy(sample)

        return {
            "input_ids":      input_ids,
            "image":          image,
            "ground_truth":   sample.get("target", ""),
            "gt_class":       sample.get("class", "unknown"),
            "sample_relpath": sample.get("sample_relpath", ""),
        }


def collate_fn(batch, tokenizer):
    input_ids       = [item["input_ids"]      for item in batch]
    images          = [item["image"]          for item in batch]
    ground_truths   = [item["ground_truth"]   for item in batch]
    gt_classes      = [item["gt_class"]       for item in batch]
    sample_relpaths = [item["sample_relpath"] for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    images         = torch.stack(images, dim=0)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return {
        "input_ids":       input_ids,
        "attention_mask":  attention_mask,
        "images":          images,
        "ground_truths":   ground_truths,
        "gt_classes":      gt_classes,
        "sample_relpaths": sample_relpaths,
    }


# ============================================================
# 模型加载
# ============================================================

def load_model(checkpoint_path: str, device: str = "cuda", vision_tower_path: str = None):
    print(f"[{device}] Loading model from: {checkpoint_path}")

    parent_dir  = os.path.dirname(checkpoint_path) if "checkpoint-" in os.path.basename(checkpoint_path) else checkpoint_path
    config_path = checkpoint_path if os.path.exists(os.path.join(checkpoint_path, "config.json")) else parent_dir
    config      = TinyLlavaConfig.from_pretrained(config_path)
    model       = TinyLlavaForConditionalGeneration(config)

    llm_path = config.llm_model_name_or_path
    if os.path.exists(llm_path):
        model.language_model = model.language_model.from_pretrained(llm_path)
    else:
        weight_path = os.path.join(parent_dir, "language_model", "pytorch_model.bin")
        if os.path.exists(weight_path):
            state   = torch.load(weight_path, map_location="cpu", weights_only=False)
            cleaned = {k.replace(".base_layer.", "."): v for k, v in state.items()}
            model.language_model.load_state_dict(cleaned, strict=False)

    model.vision_tower._load_model(
        vision_tower_name=None,
        pretrained_vision_tower_path=vision_tower_path
    )

    # [Untrained variant] Connector weight loading skipped — random initialization retained.
    # [Untrained variant] LoRA adapter loading skipped — base LLM weights used as-is.

    ckpt_tokenizer = AutoTokenizer.from_pretrained(parent_dir)
    model.tokenizer = ckpt_tokenizer

    model = model.to(device)
    model.eval()

    # 再次替换确保 merge 后不被覆盖
    ckpt_tokenizer = AutoTokenizer.from_pretrained(parent_dir)
    model.tokenizer = ckpt_tokenizer
    pid = ckpt_tokenizer.convert_tokens_to_ids("<cls_placeholder>")
    assert isinstance(pid, int), f"<cls_placeholder> not in tokenizer! pid={pid}"

    print(f"[{device}] Model loaded. vocab_size={len(ckpt_tokenizer)}, <cls_placeholder> id={pid}")
    return model


# ============================================================
# 推理（单进程）
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple:
    tokenizer = model.tokenizer

    im_end_id     = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_token_ids = list({t for t in [tokenizer.eos_token_id, im_end_id] if t is not None})

    predictions     = []
    ground_truths   = []
    gt_classes_all  = []
    sample_relpaths = []

    for batch in tqdm(dataloader, desc=f"[{device}] Evaluating"):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images         = batch["images"].to(device, dtype=model.dtype)
        gt_classes     = batch["gt_classes"]

        _, logits = model.vision_tower(
            images.to(device=device, dtype=model.dtype)
        )
        probs        = F.softmax(logits, dim=-1)
        pred_classes = [
            NETMAMBA_IDX_TO_CLASS[i.item()] if i.item() < len(NETMAMBA_IDX_TO_CLASS)
            else f"Unknown-{i.item()}"
            for i in probs.argmax(dim=-1)
        ]

        (_, position_ids, attention_mask_new, _, inputs_embeds, _) = \
            model.prepare_inputs_labels_for_multimodal(
                input_ids       = input_ids,
                position_ids    = None,
                attention_mask  = attention_mask,
                past_key_values = None,
                labels          = None,
                images          = images,
                image_sizes     = None,
                gt_classes      = None,
            )

        gen_kwargs = dict(
            inputs_embeds        = inputs_embeds,
            attention_mask       = attention_mask_new,
            position_ids         = position_ids,
            max_new_tokens       = max_new_tokens,
            use_cache            = True,
            eos_token_id         = eos_token_ids,
            no_repeat_ngram_size = 5,
            repetition_penalty   = 1.1,
        )
        if temperature > 0:
            gen_kwargs.update(temperature=temperature, top_p=top_p, do_sample=True)
        else:
            gen_kwargs["do_sample"] = False

        outputs         = model.language_model.generate(**gen_kwargs)
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        final_texts = []
        for text, pred_cls in zip(generated_texts, pred_classes):
            text = strip_think(text)
            obj  = parse_json_output(text)
            if obj is not None and isinstance(obj, dict):
                result = OrderedDict(
                    [("class", pred_cls)] + [(k, v) for k, v in obj.items() if k != "class"]
                )
                final_texts.append(json.dumps(result, ensure_ascii=False))
            else:
                final_texts.append(json.dumps({"class": pred_cls, "raw": text}, ensure_ascii=False))

        predictions.extend([{"text": t} for t in final_texts])
        ground_truths.extend(batch["ground_truths"])
        gt_classes_all.extend(gt_classes)
        sample_relpaths.extend(batch["sample_relpaths"])

    return predictions, ground_truths, gt_classes_all, sample_relpaths


# ============================================================
# 子进程入口（多卡模式）
# ============================================================

def worker_process(rank: int, args, sampled_data: List[Dict], tmp_dir: str):
    """每个子进程独占 GPU rank，处理分配给它的数据切片，结果写到临时文件。"""
    # 子进程必须显式设置 CUDA 设备，否则 Triton/mamba 内核找不到正确的 GPU 上下文
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # 按 rank 切分数据
    chunk_size = math.ceil(len(sampled_data) / args.num_gpus)
    start = rank * chunk_size
    end   = min(start + chunk_size, len(sampled_data))
    chunk = sampled_data[start:end]

    if not chunk:
        return

    model     = load_model(args.checkpoint_path, device=device, vision_tower_path=args.vision_tower_path)
    tokenizer = model.tokenizer
    template  = TemplateFactory(args.conv_version)()

    dataset = EvalDataset(
        data         = chunk,
        image_folder = args.image_folder,
        tokenizer    = tokenizer,
        template     = template,
        byte_length  = args.byte_length,
    )
    # 子进程内不能嵌套多进程 DataLoader，num_workers 必须为 0
    # collate_fn 不能用 lambda，改用 CollateFn 可调用对象
    class CollateFn:
        def __init__(self, tok): self.tok = tok
        def __call__(self, batch): return collate_fn(batch, self.tok)

    dataloader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        collate_fn  = CollateFn(tokenizer),
        num_workers = 0,
        pin_memory  = False,  # 子进程里 pin_memory 会导致设备上下文问题
    )

    predictions, ground_truths, gt_classes_all, sample_relpaths = evaluate(
        model          = model,
        dataloader     = dataloader,
        device         = device,
        max_new_tokens = args.max_new_tokens,
        temperature    = args.temperature,
        top_p          = args.top_p,
    )

    # 结果写到临时文件
    tmp_file = os.path.join(tmp_dir, f"rank_{rank}.jsonl")
    with open(tmp_file, "w", encoding="utf-8") as f:
        for pred, gt, cls, rel in zip(predictions, ground_truths, gt_classes_all, sample_relpaths):
            f.write(json.dumps({
                "prediction":   pred["text"],
                "ground_truth": gt,
                "gt_class":     cls,
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


def calculate_metrics(predictions, ground_truths, gt_classes) -> Dict:
    required_fields = {"traits", "evidence", "description", "notes"}
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
    print(f"  {'Class':<22} {'Samples':>8} {'JSON%':>8} {'Match%':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")
    for cls in sorted(metrics["per_class_metrics"]):
        cm = metrics["per_class_metrics"][cls]
        print(f"  {cls:<22} {cm['total_samples']:>8} "
              f"{cm['json_parse_rate']:>7.1f}% {cm['class_match_rate']:>7.1f}%")
    print("=" * 70 + "\n")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path",   required=True)
    parser.add_argument("--vision_tower_path", required=True)
    parser.add_argument("--eval_data_path",    required=True)
    parser.add_argument("--image_folder",      required=True)
    parser.add_argument("--output_dir",        default="./eval_results")
    parser.add_argument("--samples_per_class", type=int,   default=10)
    parser.add_argument("--batch_size",        type=int,   default=4)
    parser.add_argument("--max_new_tokens",    type=int,   default=400)
    parser.add_argument("--temperature",       type=float, default=0.0)
    parser.add_argument("--top_p",             type=float, default=0.9)
    parser.add_argument("--byte_length",       type=int,   default=1600)
    parser.add_argument("--conv_version",      default="qwen3_instruct")
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--num_gpus",          type=int,   default=1,
                        help="使用几张GPU并行推理（默认1，多卡时自动切分数据）")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── 数据准备（主进程统一完成）──
    print("Loading dataset...")
    full_data = read_jsonl(args.eval_data_path)
    print(f"Total samples: {len(full_data)}")

    valid_data, val_report = filter_valid_samples(full_data, args.image_folder)
    if not valid_data:
        print("ERROR: No valid samples found!")
        return

    sampled_data = stratified_sample(valid_data, args.samples_per_class)

    # ── 单卡模式 ──
    if args.num_gpus <= 1:
        device    = "cuda:0"
        model     = load_model(args.checkpoint_path, device=device, vision_tower_path=args.vision_tower_path)
        tokenizer = model.tokenizer
        template  = TemplateFactory(args.conv_version)()

        dataset = EvalDataset(
            data         = sampled_data,
            image_folder = args.image_folder,
            tokenizer    = tokenizer,
            template     = template,
            byte_length  = args.byte_length,
        )
        dataloader = DataLoader(
            dataset,
            batch_size  = args.batch_size,
            shuffle     = False,
            collate_fn  = lambda b: collate_fn(b, tokenizer),
            num_workers = 4,
            pin_memory  = True,
        )
        print(f"DEBUG: placeholder_id = {model.tokenizer.convert_tokens_to_ids('<cls_placeholder>')}")
        predictions, ground_truths, gt_classes_all, sample_relpaths = evaluate(
            model          = model,
            dataloader     = dataloader,
            device         = device,
            max_new_tokens = args.max_new_tokens,
            temperature    = args.temperature,
            top_p          = args.top_p,
        )

    # ── 多卡模式：spawn 子进程，每张卡处理一个切片 ──
    else:
        import torch.multiprocessing as mp
        num_gpus = min(args.num_gpus, torch.cuda.device_count())
        if num_gpus < args.num_gpus:
            print(f"WARNING: 只检测到 {num_gpus} 张 GPU，忽略 --num_gpus {args.num_gpus}")
        print(f"[INFO] 多卡推理：{num_gpus} 张 GPU，共 {len(sampled_data)} 样本")

        tmp_dir = tempfile.mkdtemp(prefix="eval_tmp_")
        mp.set_start_method("spawn", force=True)

        processes = []
        for rank in range(num_gpus):
            p = mp.Process(
                target=worker_process,
                args=(rank, args, sampled_data, tmp_dir),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # 合并所有子进程结果（保持原始顺序）
        predictions     = []
        ground_truths   = []
        gt_classes_all  = []
        sample_relpaths = []

        chunk_size = math.ceil(len(sampled_data) / num_gpus)
        for rank in range(num_gpus):
            tmp_file = os.path.join(tmp_dir, f"rank_{rank}.jsonl")
            if not os.path.exists(tmp_file):
                continue
            with open(tmp_file) as f:
                for line in f:
                    row = json.loads(line)
                    predictions.append({"text": row["prediction"]})
                    ground_truths.append(row["ground_truth"])
                    gt_classes_all.append(row["gt_class"])
                    sample_relpaths.append(row["sample_relpath"])

        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[INFO] 合并完成，共 {len(predictions)} 条结果")

    # ── 统一计算指标 ──
    metrics = calculate_metrics(predictions, ground_truths, gt_classes_all)
    metrics["eval_info"] = {
        "total_dataset":     len(full_data),
        "valid_samples":     len(valid_data),
        "sampled":           len(sampled_data),
        "samples_per_class": args.samples_per_class,
        "checkpoint":        args.checkpoint_path,
        "num_gpus":          args.num_gpus,
        "seed":              args.seed,
    }
    metrics["validation_report"] = val_report

    save_results(predictions, ground_truths, metrics, args.output_dir, sample_relpaths)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()