import torch

from .base import BaseTrainingRecipe
from . import register_training_recipe
from ..utils import log


@register_training_recipe('connector_only')
class ConnectorOnlyTrainingRecipe(BaseTrainingRecipe):
    """
    只训练 connector：
    - 不做 LoRA / QLoRA / 量化等额外操作
    - 直接沿用 BaseTrainingRecipe 的 add_args / save / load
    - 通过 tune_type_xxx 控制哪些模块参与梯度更新
    """

    def __init__(self, training_arguments):
        super().__init__(training_arguments)
        self.training_arguments = training_arguments

    # 不额外改 model_args，直接用 BaseTrainingRecipe 的逻辑
    # 里面只会设置 llm 的 dtype 和可选的 pretrained_xxx_path
    # def add_args(self, model_args):
    #     return super().add_args(model_args)

    # 不做模型结构变换，不挂 LoRA，也不做 8bit 量化
    def training_model_converse(self, model):
        log("ConnectorOnlyTrainingRecipe: no quantization / no LoRA, only tune_type_xxx controls trainable params.")
        return model

    # save / load 直接走 BaseTrainingRecipe 的默认实现即可
    # def save(self, model, trainer):
    #     return super().save(model, trainer)
    #
    # def load(self, model, model_args={}):
    #     return super().load(model, model_args)
