import json
from transformers import AutoTokenizer

from tinyllava.data.text_preprocess import TextPreprocess
from tinyllava.utils.constants import DEFAULT_IMAGE_TOKEN

# ====== 你需要改成自己的路径 ======
MODEL_DIR = "/root/autodl-tmp/Tiny_LLaVA/tinyllava/model/llm/Qwen3-1.7B"
JSONL = "/root/autodl-tmp/Datasets/USTC-TFC-2016/nlp_description_output/train.jsonl"
CONV_VERSION = "qwen3_instruct"   # 或你实际在训练里用的
# =================================

# 1) 取一条样本
with open(JSONL, "r", encoding="utf-8") as f:
    sample = json.loads(next(f).strip())

# 2) 构造 conversations（按你 dataset 的逻辑）
user_text = (
    f"{DEFAULT_IMAGE_TOKEN}\n"
    "Please identify the traffic category and describe the communication behavior "
    "based on the provided traffic representation."
)

assistant_text = sample.get("nl_description", "") or sample.get("label_sentence", "")

messages = [
    {"from": "human", "value": user_text},
    {"from": "gpt", "value": assistant_text},
]

# 3) 直接加载 tokenizer（不建 TinyLLaVA 模型）
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    use_fast=False,
)

tp = TextPreprocess(tokenizer, CONV_VERSION)

out = tp(messages, mode="eval")
print("======= PROMPT BEGIN =======")
print(out["prompt"][:2000])
print("======= PROMPT END   =======")
