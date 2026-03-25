#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_llmonly.py — Text-only dataset for LLM-only ablation experiment.

Drop-in replacement for dataset.py. Key differences:
  - No npy loading, no image field in output
  - user_text already contains hex byte sequence (built by convert_to_llmonly_jsonl.py)
  - Compatible with existing TextPreprocess / conv_version pipeline

Usage: replace data/dataset.py with this file when running LLM-only training,
or import LLMOnlySupervisedDataset directly in train_llmonly.py.
"""

import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset

from .text_preprocess import TextPreprocess
from ..utils.arguments import DataArguments
from ..utils.constants import IGNORE_INDEX


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class LLMOnlySupervisedDataset(Dataset):
    """
    Text-only dataset for LLM-only ablation.
    user_text contains the hex byte sequence; no image token, no npy loading.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super().__init__()
        self.list_data_dict = read_jsonl(data_path)
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)

    def __len__(self):
        return len(self.list_data_dict)

    def _build_conversations(self, sample):
        user_text = sample.get("user_text", "").strip()
        if not user_text:
            raise KeyError(f"Sample missing user_text: {sample.get('sample_id', '?')}")

        assistant_text = (sample.get("target") or "").strip()
        if not assistant_text:
            raise KeyError(f"Sample missing target: {sample.get('sample_id', '?')}")

        return [
            {"from": "human",  "value": user_text},
            {"from": "gpt",    "value": assistant_text},
        ]

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self.list_data_dict[i]
        conversations = self._build_conversations(sample)

        # Text only — no image field
        data_dict = self.text_preprocess(copy.deepcopy(conversations), mode="train")

        # Remove any residual labels_cls if text_preprocess adds it
        data_dict.pop("labels_cls", None)

        # Explicitly no image — forward() will skip vision tower
        # (images=None triggers pure-text path in prepare_inputs_labels_for_multimodal)
        return data_dict


@dataclass
class DataCollatorForLLMOnlyDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[k] for instance in instances] for k in ("input_ids", "labels")
        )

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels    = labels[:, : self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        # No images key — model receives images=None → pure text path
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


def make_llmonly_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
) -> Dict:
    train_dataset = LLMOnlySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
    )
    data_collator = DataCollatorForLLMOnlyDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
