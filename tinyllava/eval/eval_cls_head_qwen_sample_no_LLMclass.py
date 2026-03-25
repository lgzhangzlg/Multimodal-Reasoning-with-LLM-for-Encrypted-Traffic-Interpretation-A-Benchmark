
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stratified sampling evaluation for TinyLLaVA traffic classification.

v3 改动：
1. load_model 里强制用 parent_dir 的 tokenizer 替换模型内部 tokenizer
   （config.tokenizer_name_or_path 为空时，模型初始化加载原始基座 tokenizer，缺少 special tokens）
2. _replace_cls_token 标量 tensor 保护已在 modeling_tinyllava.py 里修复

Usage:
    python eval_cls_head_qwen_sample_no_LLMclass.py \
        --checkpoint_path /root/autodl-tmp/train_out/xxx/checkpoint-438 \
        --eval_data_path  /root/autodl-tmp/Datasets/.../test.jsonl \
        --image_folder    /root/autodl-tmp/Datasets/.../npy \
        --output_dir      /root/autodl-tmp/eval_out/xxx \
        --samples_per_class 10 \
        --batch_size 4 \
        --temperature 0.0 \
        --max_new_tokens 1024
"""

import argparse
import json
import os
import random
import re
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

        # 从 jsonl 读取 user_text（含 <cls_placeholder>，在 prepare_inputs_labels 里在线替换）
        user_text = sample.get("user_text", "").strip()
        if not user_text:
            # 兜底：旧格式 jsonl，同步为新格式
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

def load_model(checkpoint_path: str, device: str = "cuda", vision_tower_path: str = None) -> TinyLlavaForConditionalGeneration:
    print(f"Loading model from: {checkpoint_path}")

    parent_dir  = os.path.dirname(checkpoint_path) if "checkpoint-" in os.path.basename(checkpoint_path) else checkpoint_path
    config_path = checkpoint_path if os.path.exists(os.path.join(checkpoint_path, "config.json")) else parent_dir
    config      = TinyLlavaConfig.from_pretrained(config_path)
    model       = TinyLlavaForConditionalGeneration(config)

    llm_path = config.llm_model_name_or_path
    print(f"  Loading LLM from: {llm_path}")
    if os.path.exists(llm_path):
        model.language_model = model.language_model.from_pretrained(llm_path)
    else:
        weight_path = os.path.join(parent_dir, "language_model", "pytorch_model.bin")
        if os.path.exists(weight_path):
            state   = torch.load(weight_path, map_location="cpu", weights_only=False)
            cleaned = {k.replace(".base_layer.", "."): v for k, v in state.items()}
            model.language_model.load_state_dict(cleaned, strict=False)
            print("  LLM loaded from saved weights (key cleanup applied)")

    # 直接调用 _load_model，让它加载 checkpoint-best.pth
    # vision_tower 冻结训练时 pytorch_model.bin 不含正确权重
    model.vision_tower._load_model(
    vision_tower_name=None,
    pretrained_vision_tower_path=vision_tower_path
)
    print("  vision_tower loaded from checkpoint-best.pth")

    conn_path = os.path.join(parent_dir, "connector", "pytorch_model.bin")
    if os.path.exists(conn_path):
        state = torch.load(conn_path, map_location="cpu", weights_only=False)
        model.connector.load_state_dict(state, strict=False)
        print("  connector loaded")

    adapter_cfg   = os.path.join(checkpoint_path, "adapter_config.json")
    adapter_model = os.path.join(checkpoint_path, "adapter_model.safetensors")
    if os.path.exists(adapter_cfg) and os.path.exists(adapter_model):
        size_mb = os.path.getsize(adapter_model) / 1024 / 1024
        print(f"  Adapter size: {size_mb:.1f} MB")
        if size_mb < 0.001:
            print("  WARNING: Adapter file too small!")
        else:
            model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
            try:
                model = model.merge_and_unload()
                print("  LoRA merged")
            except Exception as e:
                print(f"  LoRA merge failed: {e}")
    else:
        print("  WARNING: No adapter found, using base weights only")

    # ── 强制用训练输出目录里的 tokenizer 替换模型内部的 tokenizer ──
    # config.tokenizer_name_or_path 为空时，模型初始化加载的是原始基座 tokenizer，
    # 没有 <cls_placeholder> 等 special tokens，placeholder_id 会解析错误导致崩溃
    ckpt_tokenizer = AutoTokenizer.from_pretrained(parent_dir)
    model.tokenizer = ckpt_tokenizer
    print(f"  tokenizer replaced from {parent_dir}, vocab_size={len(ckpt_tokenizer)}")

    # 确认关键 token 正确注册
    pid = ckpt_tokenizer.convert_tokens_to_ids("<cls_placeholder>")
    print(f"  <cls_placeholder> id={pid}")

    # LoRA merge 之后再替换 tokenizer
    model = model.to(device)
    model.eval()

    # ── 最后替换 tokenizer，确保不被 LoRA merge 覆盖 ──
    ckpt_tokenizer = AutoTokenizer.from_pretrained(parent_dir)
    model.tokenizer = ckpt_tokenizer
    pid = ckpt_tokenizer.convert_tokens_to_ids("<cls_placeholder>")
    print(f"  tokenizer replaced, vocab_size={len(ckpt_tokenizer)}, <cls_placeholder> id={pid}")
    assert isinstance(pid, int), f"<cls_placeholder> not in tokenizer! pid={pid}"

    print("Model loaded successfully\n")
    return model


# ============================================================
# 推理
# ============================================================

@torch.no_grad()
def evaluate(
    model: TinyLlavaForConditionalGeneration,
    dataloader: DataLoader,
    device: str = "cuda",
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> tuple:
    tokenizer = model.tokenizer

    im_end_id     = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_token_ids = list({t for t in [tokenizer.eos_token_id, im_end_id] if t is not None})
    print(f"eos_token_ids: {eos_token_ids}  (im_end_id={im_end_id})")

    predictions     = []
    ground_truths   = []
    gt_classes_all  = []
    sample_relpaths = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images         = batch["images"].to(device, dtype=model.dtype)
        gt_classes     = batch["gt_classes"]  # List[str]

        # ── 1. 单独跑 NetMamba 拿到 top-1 预测类别，用于最终 JSON 拼接 ──
        _, logits = model.vision_tower(
            images.to(device=device, dtype=model.dtype)
        )
        probs        = F.softmax(logits, dim=-1)
        pred_classes = [
            NETMAMBA_IDX_TO_CLASS[i.item()] if i.item() < len(NETMAMBA_IDX_TO_CLASS)
            else f"Unknown-{i.item()}"
            for i in probs.argmax(dim=-1)
        ]

        # ── 2. prepare_inputs_labels_for_multimodal（推理模式，gt_classes=None）──
        # <cls_placeholder> 自动被替换为 NetMamba 预测类别的 special token embedding
        (
            _,
            position_ids,
            attention_mask_new,
            _,
            inputs_embeds,
            _,
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids       = input_ids,
            position_ids    = None,
            attention_mask  = attention_mask,
            past_key_values = None,
            labels          = None,
            images          = images,
            image_sizes     = None,
            gt_classes      = None,  # 推理时不传，自动用 NetMamba 预测
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

        assert inputs_embeds.shape[1] == attention_mask_new.shape[1], \
            f"shape mismatch: embeds={inputs_embeds.shape}, mask={attention_mask_new.shape}"

        outputs         = model.language_model.generate(**gen_kwargs)
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # ── 3. 去掉 think 标签，把 class 字段拼到 JSON 最前面 ──
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
                # JSON 解析失败或返回非 dict，至少保留 class 和原始文本
                final_texts.append(json.dumps({"class": pred_cls, "raw": text}, ensure_ascii=False))

        predictions.extend([{"text": t} for t in final_texts])
        ground_truths.extend(batch["ground_truths"])
        gt_classes_all.extend(gt_classes)
        sample_relpaths.extend(batch["sample_relpaths"])

    return predictions, ground_truths, gt_classes_all, sample_relpaths


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
    predictions: List[Dict],
    ground_truths: List[str],
    gt_classes: List[str],
) -> Dict:
    # class 由代码拼接，required_fields 里不再包含 class
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
            # class 字段来自 NetMamba 预测，直接与 ground truth 比较
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
    parser.add_argument("--eval_data_path",    required=True)
    parser.add_argument("--image_folder",      required=True)
    parser.add_argument("--output_dir",        default="./eval_results")
    parser.add_argument("--samples_per_class", type=int,   default=10)
    parser.add_argument("--batch_size",        type=int,   default=4)
    parser.add_argument("--max_new_tokens",    type=int,   default=1024)
    parser.add_argument("--temperature",       type=float, default=0.0)
    parser.add_argument("--top_p",             type=float, default=0.9)
    parser.add_argument("--device",            default="cuda")
    parser.add_argument("--byte_length",       type=int,   default=1600)
    parser.add_argument("--conv_version",      default="qwen3_instruct")
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--vision_tower_path", required=True, help="Path to NetMamba checkpoint-best.pth")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading dataset...")
    full_data = read_jsonl(args.eval_data_path)
    print(f"Total samples: {len(full_data)}")

    valid_data, val_report = filter_valid_samples(full_data, args.image_folder)
    if not valid_data:
        print("ERROR: No valid samples found!")
        return

    sampled_data = stratified_sample(valid_data, args.samples_per_class)

    model = load_model(args.checkpoint_path, device=args.device, vision_tower_path=args.vision_tower_path)
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
        num_workers = 0,
    )
    print(f"DEBUG: placeholder_id = {model.tokenizer.convert_tokens_to_ids('<cls_placeholder>')}")
    predictions, ground_truths, gt_classes_all, sample_relpaths = evaluate(
        model          = model,
        dataloader     = dataloader,
        device         = args.device,
        max_new_tokens = args.max_new_tokens,
        temperature    = args.temperature,
        top_p          = args.top_p,
    )

    metrics = calculate_metrics(predictions, ground_truths, gt_classes_all)
    metrics["eval_info"] = {
        "total_dataset":     len(full_data),
        "valid_samples":     len(valid_data),
        "sampled":           len(sampled_data),
        "samples_per_class": args.samples_per_class,
        "checkpoint":        args.checkpoint_path,
        "seed":              args.seed,
    }
    metrics["validation_report"] = val_report

    save_results(predictions, ground_truths, metrics, args.output_dir, sample_relpaths)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()