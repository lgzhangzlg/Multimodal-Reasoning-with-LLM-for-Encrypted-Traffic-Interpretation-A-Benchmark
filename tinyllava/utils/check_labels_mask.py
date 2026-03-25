import json
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from tinyllava.data.dataset import LazySupervisedDataset
from tinyllava.utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


# ====== 你需要改成自己的路径/参数 ======
MODEL_DIR = "/root/autodl-tmp/Qwen3-8B"
DATA_PATH = "/root/autodl-tmp/Datasets/USTC-TFC-2016/nlp_description_output/train.jsonl"
IMAGE_FOLDER = "/root/autodl-tmp/Datasets/USTC-TFC-2016/USTC-TFC-2016_npy"
CONV_VERSION = "qwen3_instruct"   # 或你训练时实际用的
# =====================================


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        use_fast=False,
    )

    # 用一个“最小 data_args”对象即可（dataset 只用到这些字段）
    data_args = SimpleNamespace(
        data_path=DATA_PATH,
        image_folder=IMAGE_FOLDER,
        is_multimodal=True,
        conv_version=CONV_VERSION,
    )

    ds = LazySupervisedDataset(
        data_path=DATA_PATH,
        tokenizer=tokenizer,
        data_args=data_args,
    )

    counts = []
    img_counts = []

    for i in range(50):  # 抽查 50 条
        item = ds[i]
        labels = item["labels"]
        input_ids = item["input_ids"]

        num_supervised = int((labels != IGNORE_INDEX).sum().item())
        num_img_tok = int((input_ids == IMAGE_TOKEN_INDEX).sum().item())

        counts.append(num_supervised)
        img_counts.append(num_img_tok)

    print("supervised tokens over 50 samples:")
    print("min =", min(counts), "mean =", sum(counts) / len(counts), "max =", max(counts))
    print("first 10:", counts[:10])

    print("\nIMAGE_TOKEN_INDEX count over 50 samples:")
    print("min =", min(img_counts), "mean =", sum(img_counts) / len(img_counts), "max =", max(img_counts))
    print("first 10:", img_counts[:10])


if __name__ == "__main__":
    main()
