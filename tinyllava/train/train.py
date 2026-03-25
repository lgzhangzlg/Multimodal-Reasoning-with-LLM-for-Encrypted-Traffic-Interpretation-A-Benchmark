#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_llmonly.py — LLM-only ablation training script.

Differences from train.py:
  - Uses LLMOnlySupervisedDataset (no image loading, no vision tower active)
  - vision_tower is still loaded (to keep model config compatible) but images=None
    so prepare_inputs_labels_for_multimodal falls through to pure-text path
  - All other settings (LoRA, optimizer, DeepSpeed) remain identical to DTLR

Usage (example for ISCXVPN2016):
    deepspeed --num_gpus=5 train_llmonly.py \
        --model_name_or_path /root/autodl-tmp/third_party/Qwen3-1.7B \
        --vision_tower /root/autodl-tmp/third_party/NetMamba \
        --connector_type mlp2x_gelu \
        --data_path /root/autodl-tmp/Datasets/ISCXVPN2016/nlp_output_llmonly_200_6000/train.jsonl \
        --image_folder /root/autodl-tmp/Datasets/ISCXVPN2016/ISCXVPN2016_npy_split_npy_v3_balacned_200_6000 \
        --output_dir /root/autodl-tmp/checkpoints/ISCXVPN2016_llmonly_qwen3-1.7B \
        --conv_version qwen3_instruct \
        --tune_type_llm lora \
        --tune_type_connector frozen \
        --tune_type_vision_tower frozen \
        --lora_r 16 \
        --lora_alpha 16 \
        --lora_dropout 0.1 \
        --bf16 True \
        --num_train_epochs 3 \
        --per_device_train_batch_size 6 \
        --gradient_accumulation_steps 16 \
        --learning_rate 2e-5 \
        --weight_decay 0.01 \
        --warmup_ratio 0.1 \
        --max_grad_norm 1.0 \
        --deepspeed ds_config_zero2.json \
        --training_recipe lora
"""

from packaging import version
import pathlib

import tokenizers
import transformers
import os
from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *

# ── 替换为 LLM-only dataset ──────────────────────────────────────────
from tinyllava.data.dataset import make_llmonly_data_module

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

    model_args = {}
    model_args['llm'] = {
        'model_name_or_path': model_arguments.model_name_or_path,
        'cache_dir': model_arguments.cache_dir,
        'attn_implementation': model_arguments.attn_implementation,
    }
    model_args['vision_tower'] = {
        'model_name_or_path': model_arguments.vision_tower.split(':')[-1],
    }
    if model_arguments.vision_tower2 != '':
        model_args['vision_tower']['model_name_or_path2'] = model_arguments.vision_tower2.split(':')[-1]

    model_args['connector'] = {
        'connector_type': model_arguments.connector_type,
    }
    return model_args


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()

    logger_setting(getattr(training_arguments, 'output_dir', None))

    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments)
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    model_args = training_recipe.add_args(model_args)

    model_config = TinyLlavaConfig()
    model_config.load_from_config(model_arguments)
    model = TinyLlavaForConditionalGeneration(model_config)

    if training_arguments.pretrained_model_path is not None:
        model = training_recipe.load(model, model_args)
    else:
        model.load_llm(**model_args['llm'])
        model.load_vision_tower(**model_args['vision_tower'])
        model.load_connector(**model_args['connector'])

    model = training_recipe(model)

    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio

    tokenizer = model.tokenizer

    if '<image>' not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<image>']})
        model.resize_token_embeddings(len(tokenizer))

    # ── 使用 LLM-only data module（不加载 npy，不传 images）──────────
    data_arguments.is_multimodal = False  # 关键：告知 dataset 不需要图像
    data_module = make_llmonly_data_module(tokenizer=tokenizer, data_args=data_arguments)

    log_trainable_params(model)

    trainer = LLaVATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        **data_module,
    )

    trainer.train()
    training_recipe.save(model, trainer)


if __name__ == "__main__":
    train()
