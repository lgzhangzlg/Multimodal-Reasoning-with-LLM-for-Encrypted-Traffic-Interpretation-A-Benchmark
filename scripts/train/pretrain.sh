#!/bin/bash

# 若传入参数数量不是 9 个，则打印用法并退出
if [ $# -ne 9 ]; then
    echo "Usage: $0 <DATA_PATH> <IMAGE_PATH> <LLM_VERSION> <VT_VERSION> <VT_VERSION2> <CN_VERSION> <VERSION> <TRAIN_RECIPE> <MODEL_MAX_LENGTH>"
    exit 1
fi

DATA_PATH="$1"          # jsonl 标注文件
IMAGE_PATH="$2"         # 流量 .npy 根目录
LLM_VERSION="$3"        # TinyLlama 模型
VT_VERSION="$4"         # 视觉塔，这里就是 netmamba
VT_VERSION2="$5"        # 第二视觉塔，用不到传 ""
CN_VERSION="$6"         # connector 类型，例如 linear
VERSION="$7"            # 版本号，用于命名输出目录
TRAIN_RECIPE="$8"       # 训练 recipe，比如 connector_only
MODEL_MAX_LENGTH="$9"   # 最大 token 长度

VT_VARIANT="${VT_VERSION#*/}"
LLM_VARIANT="${LLM_VERSION#*/}"

deepspeed --include localhost:0 --master_port 29501 /root/autodl-tmp/Tiny_LLaVA/tinyllava/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_PATH" \
    --is_multimodal True \
    --conv_version pretrain \
    --model_name_or_path "$LLM_VERSION" \
    --vision_tower "$VT_VERSION" \
    --vision_tower2 "$VT_VERSION2" \
    --connector_type "$CN_VERSION" \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation flash_attention_2 \
    --fp16 True \
    --training_recipe "$TRAIN_RECIPE" \
    --tune_type_llm frozen \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --output_dir /mnt/data/sata/yinghu/checkpoints/llava_factory/tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-pretrain
