import os
import torch

from peft import LoraConfig, get_peft_model, PeftModel

from .base import BaseTrainingRecipe
from . import register_training_recipe
from ..utils.train_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
)
from ..utils import log


def find_all_linear_names_suffix(model, skip_keywords=("connector", "vision_tower")):
    """
    返回 Linear 层的“后缀名”(name.split('.')[-1])，更符合 PEFT 的 target_modules 匹配方式。
    注意：会跨模块去重，因此同名后缀会被合并（这是 PEFT 的常用用法）。
    """
    cls = torch.nn.Linear
    names = set()
    for name, module in model.named_modules():
        if any(k in name for k in skip_keywords):
            continue
        # 排除输出头
        if any(x in name for x in ("lm_head", "output_layer", "head")):
            continue
        if isinstance(module, cls):
            names.add(name.split(".")[-1])
    return sorted(names)


@register_training_recipe("lora")
class LoRATrainingRecipe(BaseTrainingRecipe):
    def __init__(self, training_arguments):
        super().__init__(training_arguments)
        self.training_arguments = training_arguments

        # 默认：都跳过（不加 LoRA），然后按 tune_type_xxx == 'lora' 再移除
        self.lora_skip_module = ["connector", "vision_tower", "language_model"]

    def training_model_converse(self, model):
        # 哪些模块要加 LoRA：tune_type_xxx == 'lora'
        if self.training_arguments.tune_type_connector == "lora" and "connector" in self.lora_skip_module:
            self.lora_skip_module.remove("connector")
        if self.training_arguments.tune_type_llm == "lora" and "language_model" in self.lora_skip_module:
            self.lora_skip_module.remove("language_model")
        if self.training_arguments.tune_type_vision_tower == "lora" and "vision_tower" in self.lora_skip_module:
            self.lora_skip_module.remove("vision_tower")

        # 生成 target_modules（推荐用 suffix）
        targets = find_all_linear_names_suffix(model, skip_keywords=tuple(self.lora_skip_module))
        log(f"[LoRA] skip_keywords={self.lora_skip_module}")
        log(f"[LoRA] target_modules(suffix)={targets[:50]} ... total={len(targets)}")
        if len(targets) == 0:
            raise RuntimeError(
                "[LoRA] target_modules is empty. "
                "Check module naming / skip_keywords / model structure."
            )

        lora_config = LoraConfig(
            r=self.training_arguments.lora_r,
            lora_alpha=self.training_arguments.lora_alpha,
            target_modules=targets,
            lora_dropout=self.training_arguments.lora_dropout,
            bias=self.training_arguments.lora_bias,
            task_type="CAUSAL_LM",
        )

        # dtype 控制（保持与你原先一致）
        # if getattr(self.training_arguments, "bits", 16) == 16:
        #     if getattr(self.training_arguments, "bf16", False):
        #         model.to(torch.bfloat16)
        #     if getattr(self.training_arguments, "fp16", False):
        #         model.to(torch.float16)

        # 关键修复：不要访问 model.peft_config
        if not isinstance(model, PeftModel):
            log("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)
        else:
            log("LoRA already attached, skip get_peft_model().")

        return model

    def save(self, model, trainer):
        """
        保存策略（修改后支持 cls_head）：
        1) tokenizer + config
        2) base 参数（language_model / vision_tower / connector / cls_head）——非 LoRA
        3) LoRA 参数：model.save_pretrained(...)
        """
        model.config.use_cache = True

        # save tokenizer
        if hasattr(model, "tokenizer") and model.tokenizer is not None:
            model.tokenizer.save_pretrained(self.training_arguments.output_dir)

        # save whole config
        model.config.save_pretrained(self.training_arguments.output_dir, from_pt=True)

        # save trainer state
        trainer.save_state()

        # 取 base model，避免 PeftModel 包装后属性路径变化
        base = model.get_base_model() if isinstance(model, PeftModel) else model

        # save base params (non-lora)
        language_model_state_dict = get_peft_state_non_lora_maybe_zero_3(
            base.language_model.named_parameters(), False
        )
        vision_tower_state_dict = get_peft_state_non_lora_maybe_zero_3(
            base.vision_tower._vision_tower.named_parameters(), False
        )
        connector_state_dict = get_peft_state_non_lora_maybe_zero_3(
            base.connector.named_parameters(), False
        )

        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            # language_model
            language_model_output_dir = os.path.join(self.training_arguments.output_dir, "language_model")
            os.makedirs(language_model_output_dir, exist_ok=True)
            torch.save(language_model_state_dict, os.path.join(language_model_output_dir, "pytorch_model.bin"))
            base.config.text_config.save_pretrained(language_model_output_dir, from_pt=True)

            # vision_tower
            vision_tower_output_dir = os.path.join(self.training_arguments.output_dir, "vision_tower")
            os.makedirs(vision_tower_output_dir, exist_ok=True)
            torch.save(vision_tower_state_dict, os.path.join(vision_tower_output_dir, "pytorch_model.bin"))
            base.config.vision_config.save_pretrained(vision_tower_output_dir, from_pt=True)

            # connector
            connector_output_dir = os.path.join(self.training_arguments.output_dir, "connector")
            os.makedirs(connector_output_dir, exist_ok=True)
            torch.save(connector_state_dict, os.path.join(connector_output_dir, "pytorch_model.bin"))

            # --- cls_head ---
            if hasattr(base, 'cls_head') and base.cls_head is not None:
                cls_head_output_dir = os.path.join(self.training_arguments.output_dir, "cls_head")
                os.makedirs(cls_head_output_dir, exist_ok=True)
                torch.save(base.cls_head.state_dict(), os.path.join(cls_head_output_dir, "pytorch_model.bin"))
                # 保存 class_list（idx → class name 映射，评估时需要）
                if hasattr(base, 'class_list') and base.class_list is not None:
                    import json as _json
                    with open(os.path.join(cls_head_output_dir, "class_list.json"), "w") as f:
                        _json.dump(base.class_list, f, ensure_ascii=False)
                print(f"[CLS_HEAD] Saved cls_head weights to {cls_head_output_dir}")

        # save lora params
        lora_state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), self.training_arguments.lora_bias
        )
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            model.save_pretrained(self.training_arguments.output_dir, state_dict=lora_state_dict)
