import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
import transformers
from torch.utils.data import Dataset

from .text_preprocess import TextPreprocess
from ..utils.arguments import DataArguments
from ..utils.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN


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
    Dataset for supervised fine-tuning (jsonl + traffic npy), using conv_version templates.

    Key points for your setting:
    - Use existing conv_version -> TemplateFactory -> template.encode (TextPreprocess).
    - Human message MUST include DEFAULT_IMAGE_TOKEN so Template inserts IMAGE_TOKEN_INDEX.
    - Assistant target MUST use the offline-built pure JSON string: sample["target"].
    - NetMamba StrideEmbed expects input (B, 1, 1600) float32 -> we return (1, 1600) float32 in [0,1].
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

        # ── 构建类别名 → 整数索引映射（从 JSONL 中自动提取）──
        class_names = set()
        for sample in self.list_data_dict:
            cls = sample.get("class", "")
            if cls:
                class_names.add(cls)
        self.class_list = sorted(class_names)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_list)}
        self.num_classes = len(self.class_list)
        print(f"  [Dataset] Extracted {self.num_classes} classes from {data_path}")

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
        cls_name = sample.get("class", "")  # 获取语义化类别名，比如 "TheStore"

        if not rel:
            raise KeyError("jsonl sample missing required field: sample_relpath")

        npy_rel = self._pcap_rel_to_npy_rel(rel)

        # 🚀【核心修复逻辑】：强制使用语义化的 class 名作为文件夹名
        filename = os.path.basename(npy_rel)  # 提取纯文件名: com.thestore.main-xxx.npy
        if cls_name:
            # 拼接新路径: image_folder / TheStore / com.thestore.main-xxx.npy
            npy_path = os.path.join(self.data_args.image_folder, cls_name, filename)
        else:
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

        # float32 + normalize bytes 0..255 -> 0..1 (Conv1d-friendly)
        x = torch.from_numpy(arr.astype(np.float32)) / 255.0  # (1, L)
        return x

    def _build_conversations(self, sample):
        """
        Human: instruction with <image> placeholder + category list + key explanations.
        Assistant: offline-built target JSON string in sample["target"].

        拼接后 LLM 实际看到的完整输入顺序:
          [系统 prompt]
          [image features: visual tokens]  ← 来自 <image> 占位符位置
          [<cls_placeholder> 替换为类别 special token embedding]  ← 在线替换
          [下面这段 prompt 文本]
          [助手回答: JSON]
        """
        user_text = sample.get("user_text", "").strip()

        # 兜底：旧格式 jsonl 没有 user_text 字段时
        if not user_text:
            user_text = (
                f"{DEFAULT_IMAGE_TOKEN}\n"
                "Above are the raw traffic byte features extracted from a network flow.\n\n"
                "Based on the traffic byte features and the above classification result, "
                "return a single JSON object with the following keys:\n"
                "- traits: an object describing byte-level characteristics\n"
                "- evidence: a list of 2~4 concrete byte-level observations\n"
                "- description: a paragraph of 2~3 sentences summarizing the traffic\n"
                "- notes: a single security-relevant observation or recommendation"
            )

        assistant_text = (sample.get("target") or "").strip()
        if not assistant_text:
            raise KeyError("jsonl sample missing required field: target.")

        return [
            {"from": "human", "value": user_text},
            {"from": "gpt", "value": assistant_text},
        ]

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self.list_data_dict[i]

        conversations = self._build_conversations(sample)

        # ✅ Text -> template.encode (conv_version applies here)
        data_dict = self.text_preprocess(copy.deepcopy(conversations), mode="train")
        # --- 清理逻辑 ---
        if "labels_cls" in data_dict:
            del data_dict["labels_cls"]
        # ------------------
        # ✅ Traffic npy as "image" (1,1600) float32
        data_dict["image"] = self._load_npy_image(sample)

        # ── 类别标签索引（用于 H_align 分类头 Loss）──
        cls_name = sample.get("class", "")
        if cls_name in self.class_to_idx:
            data_dict["class_label"] = self.class_to_idx[cls_name]
        else:
            data_dict["class_label"] = IGNORE_INDEX  # 未知类别，训练时跳过

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[k] for instance in instances] for k in ("input_ids", "labels"))

        # handle pad_token_id == eos_token_id
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

        # ── 类别标签（用于 H_align 分类头 Loss）──
        if "class_label" in instances[0]:
            batch["class_labels"] = torch.tensor(
                [ins["class_label"] for ins in instances], dtype=torch.long
            )

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)