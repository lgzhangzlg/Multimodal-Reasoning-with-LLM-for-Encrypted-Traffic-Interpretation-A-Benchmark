from transformers import AutoTokenizer
from . import register_llm

@register_llm("qwen3")
def return_qwen3class():
    # 1) 导入真实类（如果 transformers 版本支持）
    try:
        from transformers import Qwen3ForCausalLM
        model_cls = Qwen3ForCausalLM
    except Exception:
        # 2) 如果没有这个类，就只能走 trust_remote_code 的实现
        # 但 TinyLLaVA 这里要求“可用 config 构造的类”，因此必须失败得更明确
        raise ImportError(
            "Your Transformers does not provide Qwen3ForCausalLM. "
            "Please upgrade transformers to a version that includes Qwen3ForCausalLM."
        )

    def tokenizer_and_post_load(tokenizer):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    return model_cls, (AutoTokenizer, tokenizer_and_post_load)
