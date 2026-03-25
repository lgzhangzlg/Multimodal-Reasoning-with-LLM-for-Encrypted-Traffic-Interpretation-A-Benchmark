# from typing import Any
#
# from .template import TemplateFactory
#
#
# class TextPreprocess:
#     def __init__(self, tokenizer, version):
#         self.tokenizer = tokenizer
#         self.template = TemplateFactory(version)()
#
#     def __call__(self, messages, mode='train'):
#         return self.template.encode(messages, self.tokenizer, mode)
# tinyllava/data/text_preprocess.py

# tinyllava/data/text_preprocess.py

from typing import Any, Dict
import torch

from .template import TemplateFactory
from ..utils.constants import DEFAULT_IMAGE_TOKEN


# LLaVA legacy image placeholder (很多实现固定用 -200)
LEGACY_IMAGE_TOKEN_INDEX = -200


class TextPreprocess:
    """
    Wrap template.encode() and enforce:
      - <image> must be a real tokenizer token id (>=0)
      - Replace legacy placeholder -200 with real <image> token id
      - Hard guard: no negative ids remain in input_ids
    """

    def __init__(self, tokenizer, version):
        self.tokenizer = tokenizer
        self.template = TemplateFactory(version)()
        self.image_token_id = None  # resolved lazily (tokenizer may be modified after init)

    def _resolve_image_token_id(self) -> int:
        img_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        if img_id is None:
            raise RuntimeError(
                f"[TextPreprocess] tokenizer does not recognize DEFAULT_IMAGE_TOKEN={DEFAULT_IMAGE_TOKEN}. "
                f"Fix in train.py: tokenizer.add_special_tokens({{'additional_special_tokens': ['{DEFAULT_IMAGE_TOKEN}']}})"
            )
        img_id = int(img_id)
        if img_id < 0:
            raise RuntimeError(f"[TextPreprocess] resolved <image> id is negative: {img_id}")
        return img_id

    def __call__(self, messages, mode: str = "train") -> Dict[str, Any]:
        out = self.template.encode(messages, self.tokenizer, mode)

        input_ids = out.get("input_ids", None)
        if input_ids is None:
            return out

        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        # resolve real <image> token id
        if self.image_token_id is None:
            self.image_token_id = self._resolve_image_token_id()

        # Replace legacy -200 no matter what IMAGE_TOKEN_INDEX was patched to elsewhere
        if (input_ids == LEGACY_IMAGE_TOKEN_INDEX).any():
            input_ids = input_ids.clone()
            input_ids[input_ids == LEGACY_IMAGE_TOKEN_INDEX] = self.image_token_id

        # Hard guard
        mn = int(input_ids.min().item())
        if mn < 0:
            # print unique negatives for debugging
            neg_vals = torch.unique(input_ids[input_ids < 0]).tolist()
            raise RuntimeError(
                f"[TextPreprocess] input_ids contains negative token id(s) after patch: "
                f"min={mn}, negatives={neg_vals}. This will crash embedding()."
            )

        out["input_ids"] = input_ids
        return out
