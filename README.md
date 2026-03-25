# DTLR: Deep Traffic Language Reasoning

A multimodal framework for encrypted network traffic classification and forensic report generation, combining a frozen **NetMamba** encoder with a **Qwen3-1.7B** LLM under a **"Perception-before-Cognition"** architecture.

> This repository contains the official code for the paper: *[Paper Title]* (coming soon)

---

## Overview

DTLR decouples traffic understanding into two stages:
1. **Perception**: A frozen NetMamba encoder extracts deep representations from raw traffic byte sequences (stored as `.npy` files).
2. **Cognition**: A Qwen3-1.7B LLM with LoRA fine-tuning performs traffic classification and generates structured forensic reports.

---

## Datasets

This project is evaluated on six benchmark datasets:

| Dataset | Domain |
|---|---|
| ISCXVPN2016 | VPN traffic classification |
| ISCX-Tor-2016 | Tor anonymous traffic |
| CSTNET-TLS1.3 | TLS 1.3 encrypted traffic |
| USTC-TFC-2016 | Malicious & normal traffic |
| CrossPlatform (Android) | Cross-platform app traffic |
| CrossPlatform (iOS) | Cross-platform app traffic |

> Dataset download: [Coming soon — will be provided via BaiduNetdisk / Google Drive]

---

## Environment

```bash
conda create -n dtlr python=3.10
conda activate dtlr
pip install -r requirements.txt
```

> **Note**: Requires CUDA-capable GPU(s). Training uses 4 GPUs with DeepSpeed ZeRO-2. Inference uses 2 GPUs.

---

## Training

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus 4 \
  DTLR_model/tinyllava/train/train.py \
  --deepspeed scripts/zero2.json \
  --model_name_or_path /path/to/Qwen3-1.7B \
  --vision_tower netmamba \
  --vision_tower2 "" \
  --connector_type mlp2x_gelu \
  --data_path /path/to/dataset/train.jsonl \
  --image_folder /path/to/dataset/npy \
  --is_multimodal True \
  --conv_version qwen3_instruct \
  --mm_vision_select_layer -2 \
  --image_aspect_ratio square \
  --fp16 False \
  --bf16 True \
  --training_recipe lora \
  --tune_type_llm lora \
  --tune_type_vision_tower frozen \
  --tune_type_connector full \
  --lora_r 16 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --lora_bias none \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --num_train_epochs 3 \
  --logging_steps 10 \
  --save_steps 300 \
  --output_dir /path/to/output \
  --report_to none
```

---

## Inference

```bash
python DTLR_model/tinyllava/eval/eval_cls_head_qwen_sample_no_LLMclass_mGPU.py \
  --checkpoint_path /path/to/lora/checkpoint \
  --vision_tower_path /path/to/netmamba/checkpoint-best.pth \
  --eval_data_path /path/to/dataset/test.jsonl \
  --image_folder /path/to/dataset/npy \
  --output_dir /path/to/eval_output \
  --samples_per_class 9999999 \
  --batch_size 24 \
  --max_new_tokens 500 \
  --num_gpus 2
```

---

## Results

Experimental results are reported in the paper. *(Coming soon)*

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{dtlr2025,
  title={},
  author={},
  journal={},
  year={2025}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
