# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Guidelines

- 始终使用中文与用户交流

## Project Overview

TinyLLaVA Factory is a modular PyTorch/Hugging Face framework for building small-scale Large Multimodal Models (LMMs) ranging from 0.89B to 3.1B parameters. It combines language models, vision encoders, and connectors into customizable vision-language models.

## Common Commands

### Installation
```bash
conda create -n tinyllava_factory python=3.10 -y
conda activate tinyllava_factory
pip install -e .
pip install flash-attn==2.5.7 --no-build-isolation  # optional
```

### Training
```bash
# Full training pipeline (pretrain + finetune)
bash scripts/train/train_phi.sh

# Individual stages
bash scripts/train/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" ...
bash scripts/train/finetune.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" ...

# Custom fine-tuning
bash scripts/train/custom_finetune.sh

# LoRA fine-tuning
bash scripts/train/lora/train_phi_lora.sh
```

### Inference
```bash
# Gradio web UI
python tinyllava/serve/app.py --model-path tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B

# CLI
python -m tinyllava.serve.cli --model-path <model_path> --image-file <image_path>
```

### Evaluation
```bash
python tinyllava/train/eval_all.py                # Batch evaluation
python tinyllava/eval/model_vqa.py                # VQA-v2
python tinyllava/eval/model_vqa_pope.py           # POPE
python tinyllava/eval/model_vqa_mmmu.py           # MMMU
```

## Architecture

### Three-Component Model Design
```
TinyLlavaForConditionalGeneration
├── Language Model (LLM)     → Phi, Qwen2/3, TinyLlama, Gemma, StableLM
├── Vision Tower             → CLIP, SigLIP, DinoV2, NetMamba
└── Connector                → MLP, Qformer, Resampler, Linear
```

### Factory Pattern for Component Registration
All major components use decorator-based factory registration:

```python
# Register new LLM
@register_llm('my_llm')
def return_my_llm_class(): ...

# Register new vision tower
@register_vision_tower('my_vision')
class MyVisionTower(VisionTower): ...

# Register new connector
@register_connector('my_connector')
class MyConnector(Connector): ...

# Register conversation template
@register_template('my_template')
@dataclass
class MyTemplate(Template): ...
```

### Key Directories
- `model/llm/` - Language model adapters with lazy loading
- `model/vision_tower/` - Vision encoders (CLIP, SigLIP, DinoV2, NetMamba)
- `model/connector/` - Multimodal bridges between vision and language
- `data/template/` - Conversation templates per LLM type
- `training_recipe/` - Training strategies (frozen, full, LoRA, QLoRA)
- `eval/` - Benchmark evaluation scripts

### Training Recipes
- **Frozen**: Train connector only, freeze LLM and vision tower
- **Full**: Train all components
- **LoRA/QLoRA**: Parameter-efficient fine-tuning
- **ConnectorOnly**: Similar to frozen but explicit recipe

### Data Format
JSONL with conversation format:
```json
{"id": "unique_id", "image": "path/to/image.jpg", "conversations": [
  {"from": "human", "value": "<image>\nDescribe this image."},
  {"from": "gpt", "value": "This image shows..."}
]}
```

### Data Flow
```
JSONL + Images → LazySupervisedDataset → Template.encode() → LLaVATrainer
                                                    ↓
                           Vision Tower → Connector → LLM → Loss
```

## Key Files

- `model/modeling_tinyllava.py` - Main model class `TinyLlavaForConditionalGeneration`
- `model/configuration_tinyllava.py` - `TinyLlavaConfig` for model configuration
- `model/load_model.py` - Model loading utilities
- `train/train.py` - Main training entry point
- `train/tinyllava_trainer.py` - Custom trainer extending HuggingFace Trainer
- `utils/arguments.py` - `ModelArguments`, `DataArguments`, `TrainingArguments`
- `data/dataset.py` - `LazySupervisedDataset` for data loading

## Adding New Components

### New Language Model
1. Create `model/llm/my_llm.py`
2. Register with `@register_llm('my_llm')`
3. Create corresponding template in `data/template/my_llm_template.py`

### New Vision Tower
1. Create `model/vision_tower/my_vision.py`
2. Inherit from `VisionTower` base class
3. Register with `@register_vision_tower('my_vision')`

### New Connector
1. Create `model/connector/my_connector.py`
2. Inherit from `Connector` base class
3. Register with `@register_connector('my_connector')`

## Dependencies

- Python 3.9+
- PyTorch 2.0.1
- Transformers 4.40.1
- flash-attn 2.5.7 (optional, for efficiency)
