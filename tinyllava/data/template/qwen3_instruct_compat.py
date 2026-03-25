from dataclasses import dataclass
from .base import Template
from .formatter import EmptyFormatter, StringFormatter, Formatter
from . import register_template
# 
# @register_template("qwen3_instruct")
# @dataclass
# class Qwen3InstructCompatTemplate(Template):
#     # 你现有 encode() 会在 question 里看到 DEFAULT_IMAGE_TOKEN 后调用 format_image_token
#     format_image_token: Formatter = StringFormatter(slot="<image>\n{{content}}")
# 
#     # Qwen3 官方说明：一般不需要默认 system prompt（可选）:contentReference[oaicite:2]{index=2}
#     system: Formatter = EmptyFormatter(slot="")
# 
#     # 关键：user 不加 <|im_end|>，避免 make_labels 的 split 被打碎
#     format_user: Formatter = StringFormatter(slot="<|im_start|>user\n{{content}}\n")
# 
#     # 关键：assistant 末尾加 <|im_end|>，维持你现有的 rounds 切分假设
#     format_assistant: Formatter = StringFormatter(slot="<|im_start|>assistant\n{{content}}<|im_end|>")
# 
#     # sep 用来在 make_labels 里 split rou.split(sep)
#     separator: Formatter = EmptyFormatter(slot=["<|im_start|>assistant\n", "<|im_end|>"])
@register_template("qwen3_instruct")
@dataclass
class Qwen3InstructCompatTemplate(Template):
    format_image_token: Formatter = StringFormatter(slot="<image>\n{{content}}")

    # system 设为空，避免干扰 rounds 切分
    system: Formatter = EmptyFormatter(
        slot="<|im_start|>system\n/no_think<|im_end|>\n"
    )

    # user 不加 <|im_end|>，让 eos_token 只出现在 assistant 结尾
    format_user: Formatter = StringFormatter(
        slot="<|im_start|>user\n{{content}}\n"
    )

    # assistant 结尾加 <|im_end|>\n
    format_assistant: Formatter = StringFormatter(
        slot="<|im_start|>assistant\n{{content}}<|im_end|>\n"
    )

    # eos_token = "<|im_end|>\n"，只出现在 assistant 结尾
    # sep = "<|im_start|>assistant\n"，用于定位 assistant 回答起始位置
    separator: Formatter = EmptyFormatter(
        slot=["<|im_start|>assistant\n", "<|im_end|>\n"]
    )