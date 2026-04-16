#!/bin/bash
# batch_train.sh
# 功能：对多个数据集批量训练 TinyLLaVA + NetMamba + LoRA 分类生成模型
# 使用方法：bash batch_train.sh

# GPU 配置
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
NUM_GPUS=5

# 公共参数
MODEL_PATH="/root/autodl-tmp/LLM_models/Qwen3-1.7B"
VISION_TOWER="netmamba"
CONNECTOR_TYPE="mlp2x_gelu"
CONV_VERSION="qwen3_instruct"
MM_VISION_LAYER=-2
IMAGE_ASPECT_RATIO="square"
FP16="False"
BF16="True"
TRAINING_RECIPE="lora"
TUNE_TYPE_LLM="lora"
TUNE_TYPE_VISION="full"
TUNE_TYPE_CONNECTOR="full"
LORA_R=32
LORA_ALPHA=64
LORA_DROPOUT=0.1
LORA_BIAS="none"
BATCH_SIZE=3
GRAD_ACC=8
LR=5e-5
WARMUP=0.1
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
EPOCHS=10
LOGGING_STEPS=10
SAVE_STEPS=500
REPORT_TO="none"
BOOST_WEIGHT=5
LAMBDA_CLS=0.3

# 数据集列表：每行包含 dataset_name data_json image_folder NetMamba预训练权重
DATASETS=(
"ISCX-Tor-2016 /root/autodl-tmp/Datasets/ISCX-Tor-2016/nlp_output_LLMclass_3000_10000/train.jsonl /root/autodl-tmp/Datasets/ISCX-Tor-2016/Tor_split_pcap_merged_npy_v3_balacned_3000_10000 /root/autodl-tmp/Tiny_LLaVA/third_model/NetMamba/finetune_train_out/ISCX-Tor-2016/Tor_split_pcap_merged_npy_v3_balacned_3000_10000_train_test_splited_epochs120/checkpoint-best.pth"
"CrossPlatform_android /root/autodl-tmp/Datasets/CrossPlatform_android/nlp_output_LLMclass_50_2000_class_loss/train.jsonl /root/autodl-tmp/Datasets/CrossPlatform_android/CrossPlatform_android_npy_balacned_50_2000_class_loss /root/autodl-tmp/Tiny_LLaVA/third_model/NetMamba/finetune_train_out/CrossPlatform_android/CrossPlatform_android_npy_balacned_50_2000_class_loss_epochs_120/checkpoint-best.pth"
"CrossPlatform_ios /root/autodl-tmp/Datasets/CrossPlatform_ios/nlp_output_LLMclass_50_2000_class_loss/train.jsonl /root/autodl-tmp/Datasets/CrossPlatform_ios/CrossPlatform_ios_npy_balacned_50_2000_class_loss /root/autodl-tmp/Tiny_LLaVA/third_model/NetMamba/finetune_train_out/CrossPlatform_ios/CrossPlatform_ios_npy_balacned_50_3000_class_loss_epochs_120/checkpoint-best.pth"
"USTC-TFC-2016 /root/autodl-tmp/Datasets/USTC-TFC-2016/nlp_output_LLMclass_3000_6000_class_loss/train.jsonl /root/autodl-tmp/Datasets/USTC-TFC-2016/USTC-TFC-2016_npy_balacned_3000_6000_class_loss /root/autodl-tmp/Tiny_LLaVA/third_model/NetMamba/finetune_train_out/USTC-TFC-2016/USTC-TFC-2016_npy_v3_balacned_3000_6000_train_test_splited_epochs60/checkpoint-best.pth"
"CSTNet-TLS1.3 /root/autodl-tmp/Datasets/CSTNet-TLS1.3/nlp_output_LLMclass_0_6000_class_loss/train.jsonl /root/autodl-tmp/Datasets/CSTNet-TLS1.3/CSTNet-TLS1.3_npy_balacned_0_6000_class_loss /root/autodl-tmp/Tiny_LLaVA/third_model/NetMamba/finetune_train_out/CSTNet-TLS1.3/CSTNet-TLS1.3_npy_balacned_0_6000_class_loss_epochs_120/checkpoint-best.pth"
"ISCXVPN2016 /root/autodl-tmp/Datasets/ISCXVPN2016/nlp_output_LLMclass_200_6000_class_loss/train.jsonl /root/autodl-tmp/Datasets/ISCXVPN2016/ISCXVPN2016_npy_balacned_200_6000_class_loss /root/autodl-tmp/Tiny_LLaVA/third_model/NetMamba/finetune_train_out/ISCXVPN2016/ISCXVPN2016_npy_split_npy_v3_balacned_200_6000_train_test_splited_epochs60/checkpoint-best.pth"
)

# 循环训练
for ds in "${DATASETS[@]}"; do
    IFS=' ' read -r DATASET_NAME DATA_JSON IMAGE_FOLDER_PATH VISION_CKPT <<< "$ds"
    OUTPUT_DIR="/root/autodl-tmp/train_out/$DATASET_NAME/${DATASET_NAME}_lora_qwen3-1.7B_DTLR_class_loss_lambda${LAMBDA_CLS}_boost_weight_${BOOST_WEIGHT}"
    mkdir -p "$OUTPUT_DIR"

    echo ">>> Starting training on dataset: $DATASET_NAME"
    deepspeed --num_gpus $NUM_GPUS \
      /root/autodl-tmp/Tiny_LLaVA/tinyllava/train/train.py \
      --deepspeed /root/autodl-tmp/Tiny_LLaVA/scripts/zero2.json \
      --model_name_or_path $MODEL_PATH \
      --vision_tower $VISION_TOWER \
      --pretrained_vision_tower_path $VISION_CKPT \
      --connector_type $CONNECTOR_TYPE \
      --data_path $DATA_JSON \
      --image_folder $IMAGE_FOLDER_PATH \
      --is_multimodal True \
      --conv_version $CONV_VERSION \
      --mm_vision_select_layer $MM_VISION_LAYER \
      --image_aspect_ratio $IMAGE_ASPECT_RATIO \
      --fp16 $FP16 \
      --bf16 $BF16 \
      --training_recipe $TRAINING_RECIPE \
      --tune_type_llm $TUNE_TYPE_LLM \
      --tune_type_vision_tower $TUNE_TYPE_VISION \
      --tune_type_connector $TUNE_TYPE_CONNECTOR \
      --lora_r $LORA_R \
      --lora_alpha $LORA_ALPHA \
      --lora_dropout $LORA_DROPOUT \
      --lora_bias $LORA_BIAS \
      --per_device_train_batch_size $BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACC \
      --learning_rate $LR \
      --warmup_ratio $WARMUP \
      --weight_decay $WEIGHT_DECAY \
      --max_grad_norm $MAX_GRAD_NORM \
      --num_train_epochs $EPOCHS \
      --logging_steps $LOGGING_STEPS \
      --save_steps $SAVE_STEPS \
      --output_dir $OUTPUT_DIR \
      --report_to $REPORT_TO

done