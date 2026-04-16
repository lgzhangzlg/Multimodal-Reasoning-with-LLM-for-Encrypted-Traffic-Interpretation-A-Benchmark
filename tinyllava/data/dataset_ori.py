import copy
from dataclasses import dataclass
import json
from typing import Dict, Sequence
import os
import random
import numpy as np
import sys
from PIL import Image, ImageFile

from .text_preprocess import TextPreprocess
from .image_preprocess import ImagePreprocess
from tinyllava.utils.prompt_utils import (
    build_user_text,
    extract_class_from_label_sentence,
    remove_leading_class_sentence,
    strip_path_leak,
)
from ..utils.arguments import DataArguments
from ..utils.constants import *

import transformers
import torch
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()

        # 1) read jsonl
        self.list_data_dict = self._load_jsonl(data_path)

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)

        # image_processor is optional for npy traffic
        if hasattr(data_args, "image_processor") and data_args.image_processor is not None:
            self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)
        else:
            self.image_preprocess = None

        # hint probability: default 0.0 (no leakage)
        self.hint_prob = float(getattr(data_args, "hint_prob", 0.0))

    def _load_jsonl(self, path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data

    # def _build_messages_from_sample(self, sample):
    #     """
    #     Train target format:
    #       Line 1: This is <CLASS> traffic.
    #       Line 2+: natural language description (close to nl_description)
    #     Prompt:
    #       unified build_user_text(sample, use_hint=Bernoulli(hint_prob))
    #     """
    #
    #     # decide whether to include category hint (leak) — typically keep 0.0~0.3
    #     use_hint = (random.random() < self.hint_prob)
    #
    #     # unified prompt used by BOTH train and eval (eval should set use_hint=False)
    #     user_text = build_user_text(sample, use_hint=use_hint)
    #
    #     # class name extracted from label_sentence (e.g., "BitTorrent" / "EVSE-A-Charging-Benign")
    #     cls = extract_class_from_label_sentence(sample.get("label_sentence", ""))
    #
    #     # reference description: prefer nl_description; fallback sample_description
    #     ref_nl = sample.get("nl_description", "") or ""
    #     if ref_nl:
    #         # remove leading "This is xxx traffic." to avoid duplication (we generate our own first line)
    #         ref_nl = remove_leading_class_sentence(ref_nl).strip()
    #     else:
    #         # fallback: use sample_description but strip Path leakage
    #         ref_nl = strip_path_leak(sample.get("sample_description", "") or "").strip()
    #
    #     # final assistant target: first line is classification, then description
    #     if ref_nl:
    #         assistant_text = f"This is {cls} traffic.\n{ref_nl}"
    #     else:
    #         assistant_text = f"This is {cls} traffic."
    #
    #     messages = [
    #         {
    #             "from": "human",
    #             "value": DEFAULT_IMAGE_TOKEN + "\n" + user_text,
    #         },
    #         {
    #             "from": "gpt",
    #             "value": assistant_text,
    #         },
    #     ]
    #     return messages

    # def _build_messages_from_sample(self, sample):
    #     cls = extract_class_from_label_sentence(sample.get("label_sentence", ""))
    #     # print(f"DEBUG CLS: {cls}")
    #     # sys.stdout.flush()  # 强制立即输出缓冲区内容
    #     return [
    #         {"from": "human", "value": DEFAULT_IMAGE_TOKEN + "\nWhat is the category of this traffic?"},
    #         {"from": "gpt", "value": cls}
    #     ]

    def _build_messages_from_sample(self, sample):
        """
        针对 Stage 2 指令微调优化：
        利用 nl_description 或 stats 构建更丰富的回复，
        使模型学会分析流量，而不仅仅是吐出一个标签。
        """
        # 1. 提取标签 (例如: BitTorrent)
        cls = extract_class_from_label_sentence(sample.get("label_sentence", ""))

        # 2. 准备 Assistant 的回复内容
        # 优先使用 nl_description，因为它包含了安全背景和统计信息
        ref_nl = sample.get("nl_description", "")
        if ref_nl:
            # 简单清洗：去掉路径泄露，让句子更自然
            ref_nl = remove_leading_class_sentence(ref_nl).strip()
            # 合成一个有逻辑的回答
            assistant_text = f"This traffic is identified as {cls}. {ref_nl}"
        else:
            # 如果没有描述，至少给出一个完整的句子
            assistant_text = f"The observed data belongs to {cls} traffic based on its communication patterns."

        # 3. 随机化 Human 的提问方式 (增加泛化性)
        questions = [
            f"{DEFAULT_IMAGE_TOKEN}\nWhat type of network traffic is shown in this data?",
            f"{DEFAULT_IMAGE_TOKEN}\nPlease analyze this traffic and provide a detailed identification.",
            f"{DEFAULT_IMAGE_TOKEN}\nIdentify the category of this traffic and describe its characteristics.",
            f"{DEFAULT_IMAGE_TOKEN}\nCan you tell me what this traffic capture represents?"
        ]
        user_text = random.choice(questions)

        return [
            {"from": "human", "value": user_text},
            {"from": "gpt", "value": assistant_text}
        ]

    def __len__(self):
        return len(self.list_data_dict)

    # @property
    # def lengths(self):
    #     length_list = []
    #     for sample in self.list_data_dict:
    #         conv = self._build_messages_from_sample(sample)
    #         img_tokens = 128
    #         length_list.append(sum(len(c["value"].split()) for c in conv) + img_tokens)
    #     return length_list
    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # 基于新的对话逻辑估算长度
            conv = self._build_messages_from_sample(sample)
            text_len = sum(len(c["value"].split()) for c in conv)
            length_list.append(text_len + 128)  # 128 是视觉 token 预估
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            conv = self._build_messages_from_sample(sample)
            cur_len = sum(len(c["value"].split()) for c in conv)
            length_list.append(cur_len)
        return length_list

    def _load_npy_image(self, sample):
        """
        Load traffic npy as (C, L) float32 tensor for NetMamba.
        Collator will stack into (B, C, L).
        """
        rel = sample["sample_relpath"]  # e.g. xxx/yyy.pcap
        base, _ = os.path.splitext(rel)
        npy_rel = base + ".npy"

        root = self.data_args.image_folder
        npy_path = os.path.join(root, npy_rel)

        arr = np.load(npy_path)
        x = torch.from_numpy(arr).float()

        # Normalize shape to (C, L)
        if x.ndim == 1:
            x = x.unsqueeze(0)  # (1, L)
        elif x.ndim == 2:
            # Heuristic: if shape is (L, C) where C is small, transpose
            if x.shape[0] > x.shape[1] and x.shape[1] <= 64:
                x = x.t().contiguous()  # (C, L)
        elif x.ndim == 3:
            # (C, H, W) -> (C, H*W)
            if x.shape[0] <= 64:
                x = x.flatten(1)
            else:
                x = x.permute(2, 0, 1).contiguous().flatten(1)
        else:
            x = x.reshape(x.shape[0], -1)

        return x

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self.list_data_dict[i]

        # 1) text
        conversations = self._build_messages_from_sample(sample)
        data_dict = self.text_preprocess(conversations)

        # 2) traffic "image" from npy
        image = self._load_npy_image(sample)
        data_dict["image"] = image

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels")
        )

        # handle pad_token_id == eos_token_id
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, : self.tokenizer.model_max_length]

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
