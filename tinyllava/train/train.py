from packaging import version
import pathlib

import tokenizers
import transformers
import os
from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data.dataset import make_supervised_data_module

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

    model_args = {}
    model_args['llm'] = _load_llm_settings(model_arguments)
    model_args['vision_tower'] = _load_vision_settings(model_arguments)
    model_args['connector'] = _load_connector_settings(model_arguments)
    return model_args


def _load_llm_settings(model_arguments):
    llm_args = {}
    llm_args['model_name_or_path'] = model_arguments.model_name_or_path
    llm_args['cache_dir'] = model_arguments.cache_dir
    llm_args['attn_implementation'] = model_arguments.attn_implementation
    return llm_args


def _load_vision_settings(model_arguments):
    vision_args = {}
    vision_args['model_name_or_path'] = model_arguments.vision_tower.split(':')[-1]
    if model_arguments.vision_tower2 != '':
        vision_args['model_name_or_path2'] = model_arguments.vision_tower2.split(':')[-1]
    return vision_args


def _load_connector_settings(model_arguments):
    connector_args = {}
    connector_args['connector_type'] = model_arguments.connector_type
    return connector_args


# =========================
# DEBUG: grad probe callback
# =========================
class GradDebugCallback(transformers.TrainerCallback):
    """
    Print connector grad stats for first few steps to prove image path participates in loss.
    Enable with:
      TINYLLAVA_DEBUG_GRAD=1
    """
    def __init__(self, max_steps=3):
        import os
        self.enabled = os.getenv("TINYLLAVA_DEBUG_GRAD", "0") == "1"
        self.max_steps = max_steps

    def on_step_end(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        if state.global_step >= self.max_steps:
            return
        model = kwargs.get("model", None)
        if model is None:
            return

        target = None
        for n, p in model.named_parameters():
            if "connector" in n and p.requires_grad:
                target = (n, p)
                break

        if target is None:
            print(f"[MM-DEBUG][GRAD] step {state.global_step}: no trainable connector params found.")
            return

        n, p = target
        g = p.grad
        if g is None:
            print(f"[MM-DEBUG][GRAD] step {state.global_step}: {n} grad=None")
        else:
            try:
                v = float(g.abs().mean().detach().cpu())
            except Exception:
                v = None
            print(f"[MM-DEBUG][GRAD] step {state.global_step}: {n} grad_abs_mean={v}")


class ClsHeadAccCallback(transformers.TrainerCallback):
    """
    每隔 log_every_steps 步打印 H_align 分类头的累计精度和 cls_loss。
    """
    def __init__(self, log_every_steps=10):
        self.log_every_steps = log_every_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every_steps != 0:
            return
        model = kwargs.get("model", None)
        if model is None:
            return

        # 解包 DeepSpeed / PeftModel，找到 TinyLlavaForConditionalGeneration
        base = model.module if hasattr(model, 'module') else model  # DeepSpeed
        if hasattr(base, 'get_base_model'):
            base = base.get_base_model()  # PeftModel → 原始模型

        correct = getattr(base, '_cls_acc_correct', 0)
        total = getattr(base, '_cls_acc_total', 0)
        cls_loss = getattr(base, '_cls_loss_cache', None)

        if total > 0:
            acc = correct / total * 100
            loss_str = f"{cls_loss:.4f}" if cls_loss is not None else "N/A"
            print(f"[CLS_HEAD] step={state.global_step} | acc={acc:.1f}% ({correct}/{total}) | cls_loss={loss_str}")
            # 重置累计计数
            base._cls_acc_correct = 0
            base._cls_acc_total = 0


def install_one_shot_grad_hook(model, keyword="connector"):
    """
    One-shot grad hook: prints exactly once when the first trainable param's grad is computed.
    Enable by env: TINYLLAVA_ONESHOT_GRAD=1
    """
    if os.getenv("TINYLLAVA_ONESHOT_GRAD", "0") != "1":
        return

    rank = int(os.getenv("RANK", "0"))
    only_rank0 = os.getenv("TINYLLAVA_ONESHOT_GRAD_RANK0", "1") == "1"

    target = None
    for name, p in model.named_parameters():
        if keyword in name and p.requires_grad:
            target = (name, p)
            break

    if target is None:
        if (not only_rank0) or rank == 0:
            print(f"[ONE-SHOT-GRAD][WARN] No trainable param found with keyword='{keyword}'.")
        return

    name, p = target
    fired = {"done": False}

    def _hook(grad):
        if fired["done"]:
            return grad
        fired["done"] = True
        if (not only_rank0) or rank == 0:
            gmean = float(grad.abs().mean().detach().cpu())
            gmax  = float(grad.abs().max().detach().cpu())
            print(f"[ONE-SHOT-GRAD] param={name} | shape={tuple(p.shape)} | grad_abs_mean={gmean:.6g} | grad_abs_max={gmax:.6g}")
        return grad

    p.register_hook(_hook)

    if (not only_rank0) or rank == 0:
        print(f"[ONE-SHOT-GRAD] installed on: {name} | requires_grad={p.requires_grad} | dtype={p.dtype} | device={p.device}")


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

    # load pretrained checkpoint
    if training_arguments.pretrained_model_path is not None:
        model = training_recipe.load(model, model_args)
    else:
        model.load_llm(**model_args['llm'])
        model.load_vision_tower(**model_args['vision_tower'])
        model.load_connector(**model_args['connector'])

    model = training_recipe(model)
    install_one_shot_grad_hook(model, keyword="connector")

    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio

    tokenizer = model.tokenizer

    # ── 注册 <image> special token（原有逻辑不变）──
    if '<image>' not in tokenizer.get_vocab():
        print("发现字典里没这个词，正在手动添加...")
        tokenizer.add_special_tokens({'additional_special_tokens': ['<image>']})
        model.resize_token_embeddings(len(tokenizer))

    data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_arguments)

    # ── 初始化 H_align 分类头 ──
    num_classes = data_module["train_dataset"].num_classes
    lambda_cls = getattr(training_arguments, 'lambda_cls', 0.1)
    if num_classes > 0:
        model.init_cls_head(num_classes=num_classes, lambda_cls=lambda_cls)
    else:
        print("  [WARN] No classes found in dataset, cls_head not initialized.")

    log_trainable_params(model)  # not work well with zero3

    print(">>> train_dataset class:", type(data_module["train_dataset"]))
    print(">>> train_dataset file:", type(data_module["train_dataset"]).__module__)
    print(">>> train_dataset has __getitem__:", hasattr(type(data_module["train_dataset"]), "__getitem__"))
    print(">>> train_dataset __getitem__:", type(data_module["train_dataset"]).__getitem__)

    trainer = LLaVATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        **data_module
    )

    trainer.add_callback(GradDebugCallback(max_steps=3))
    trainer.add_callback(ClsHeadAccCallback(log_every_steps=10))

    trainer.train()
    training_recipe.save(model, trainer)


if __name__ == "__main__":
    train()