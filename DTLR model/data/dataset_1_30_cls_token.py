# dataset.py (Stage-1 classification)

import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Sequence, Optional, List

import numpy as np
import torch
import transformers
from torch.utils.data import Dataset

from .text_preprocess import TextPreprocess
from ..utils.arguments import DataArguments
from ..utils.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class LazySupervisedDataset(Dataset):
    """
    Stage-1 (Classification) Dataset:
    - Input: jsonl + traffic npy (as "image")
    - Output: input_ids/labels/attention_mask/images + labels_cls

    Key points:
    - prompt MUST include DEFAULT_IMAGE_TOKEN ("<image>") for multimodal.
    - Do NOT put class name into prompt (avoid leakage).
    - assistant message is empty => LM labels mostly IGNORE; model uses labels_cls.
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

        # ✅ Use existing conv_version system
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)

        # For safety: expected byte_length for StrideEmbed
        self.byte_length = int(getattr(data_args, "byte_length", 1600))

        # =========================================================
        # [FIX] Ensure tokenizer has pad_token & <image> token id
        # =========================================================
        # pad token (Qwen family often has no pad_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ensure DEFAULT_IMAGE_TOKEN exists in tokenizer vocab
        img_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        if img_id is None:
            # add as additional special token
            self.tokenizer.add_special_tokens({"additional_special_tokens": [DEFAULT_IMAGE_TOKEN]})
            img_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)

        if img_id is None:
            raise RuntimeError(
                f"Failed to resolve image token id for DEFAULT_IMAGE_TOKEN={DEFAULT_IMAGE_TOKEN}. "
                f"tokenizer.convert_tokens_to_ids returned None."
            )

        self.image_token_id = int(img_id)

        if int(os.getenv("RANK", "0")) == 0:
            print(f"[DATASET][IMAGE] DEFAULT_IMAGE_TOKEN={DEFAULT_IMAGE_TOKEN} "
                  f"image_token_id={self.image_token_id} pad_id={self.tokenizer.pad_token_id}")

        # =========================================================
        # ### [MOD START] Build class2id / id2class for Stage-1
        # =========================================================
        self.class_key_candidates = getattr(
            data_args,
            "class_key_candidates",
            ["class", "gt_class", "label", "category"]
        )

        classes = []
        for s in self.list_data_dict:
            c = self._get_class_name(s)
            if c is not None:
                classes.append(c)

        if len(classes) == 0:
            raise KeyError(
                "No class labels found in jsonl. Expected one of keys: "
                f"{self.class_key_candidates}. (e.g., sample['class'])"
            )

        uniq = sorted(set(classes))
        self.class2id = {c: i for i, c in enumerate(uniq)}
        self.id2class = {i: c for c, i in self.class2id.items()}
        self.num_labels = len(self.class2id)
        # ### [MOD END]
        # =========================================================

    def __len__(self):
        return len(self.list_data_dict)

    def _pcap_rel_to_npy_rel(self, sample_relpath: str) -> str:
        base, ext = os.path.splitext(sample_relpath)
        return sample_relpath if ext.lower() == ".npy" else (base + ".npy")

    def _load_npy_image(self, sample) -> torch.Tensor:
        """
        Load npy as (1, 1600) float32 in [0,1] to match:
          StrideEmbed(byte_length=1600, in_chans=1) -> Conv1d expects (B,1,1600)
        """
        rel = sample.get("sample_relpath")
        if not rel:
            raise KeyError("jsonl sample missing required field: sample_relpath")

        npy_rel = self._pcap_rel_to_npy_rel(rel)
        npy_path = os.path.join(self.data_args.image_folder, npy_rel)

        arr = np.load(npy_path)
        arr = np.asarray(arr)

        # Normalize shape to (1, L)
        if arr.ndim == 2 and arr.shape[0] == 1:
            pass  # (1, L)
        elif arr.ndim == 1:
            arr = arr[None, :]  # (1, L)
        else:
            arr = arr.reshape(1, -1)  # fallback

        # Force length = byte_length (default 1600)
        L = self.byte_length
        if arr.shape[1] < L:
            pad = np.zeros((1, L - arr.shape[1]), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[1] > L:
            arr = arr[:, :L]

        # float32 + normalize bytes 0..255 -> 0..1
        x = torch.from_numpy(arr.astype(np.float32)) / 255.0  # (1, L)
        return x

    # =========================================================
    # helpers to read class name/id robustly
    # =========================================================
    def _get_class_name(self, sample) -> Optional[str]:
        for k in self.class_key_candidates:
            v = sample.get(k, None)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    def _get_class_id(self, sample) -> int:
        c = self._get_class_name(sample)
        if c is None:
            raise KeyError(
                f"sample missing class label. Expected one of keys: {self.class_key_candidates}"
            )
        if c not in self.class2id:
            raise KeyError(f"unknown class '{c}' not in class2id mapping.")
        return self.class2id[c]

    # =========================================================
    # Stage-1 prompt (classification only)
    # - keep DEFAULT_IMAGE_TOKEN
    # - do NOT include class names
    # - assistant empty
    # =========================================================
    def _build_conversations(self, sample):
        user_text = (
            f"{DEFAULT_IMAGE_TOKEN}\n"
            "Classify this network traffic sample."
        )
        assistant_text = ""
        return [
            {"from": "human", "value": user_text},
            {"from": "gpt", "value": assistant_text},
        ]

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self.list_data_dict[i]
        conversations = self._build_conversations(sample)
        data_dict = self.text_preprocess(copy.deepcopy(conversations), mode="train")

        # =========================================================
        # [FIX] Replace negative image placeholder (-200) -> real <image> token id
        # This prevents embedding() from seeing -200 and crashing.
        # =========================================================
        input_ids = data_dict["input_ids"]
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        # If template/preprocess emitted IMAGE_TOKEN_INDEX (-200), map it to real token id
        if (input_ids == IMAGE_TOKEN_INDEX).any():
            input_ids = input_ids.clone()
            input_ids[input_ids == IMAGE_TOKEN_INDEX] = self.image_token_id

        # hard check: no negative ids should remain
        if int(input_ids.min().item()) < 0:
            raise RuntimeError(
                f"[DATASET] input_ids still contains negative values after patch. "
                f"min={int(input_ids.min().item())}"
            )
        data_dict["input_ids"] = input_ids

        # =========================================================
        # Dynamic insert CLS token at the beginning
        # =========================================================
        cls_token_id = self.tokenizer.cls_token_id
        if cls_token_id is None:
            cls_token_id = self.tokenizer.bos_token_id
        if cls_token_id is None:
            cls_token_id = 101

        labels = data_dict["labels"]
        if labels.dtype != torch.long:
            labels = labels.long()

        data_dict["input_ids"] = torch.cat([
            torch.tensor([cls_token_id], dtype=data_dict["input_ids"].dtype),
            data_dict["input_ids"]
        ])
        data_dict["labels"] = torch.cat([
            torch.tensor([IGNORE_INDEX], dtype=labels.dtype),
            labels
        ])

        data_dict["image"] = self._load_npy_image(sample)
        data_dict["labels_cls"] = torch.tensor(self._get_class_id(sample), dtype=torch.long)
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[k] for instance in instances] for k in ("input_ids", "labels"))

        # handle pad_token_id == eos_token_id:
        # keep attention_mask correct by temporarily remapping eos to a sentinel value
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300  # CPU-side sentinel

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

        # stack images to (B, 1, 1600)
        if "image" in instances[0]:
            images = [ins["image"] for ins in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images, dim=0)
            else:
                batch["images"] = images

        # collate labels_cls (Stage-1)
        if "labels_cls" in instances[0]:
            batch["labels_cls"] = torch.stack([ins["labels_cls"] for ins in instances], dim=0)

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
