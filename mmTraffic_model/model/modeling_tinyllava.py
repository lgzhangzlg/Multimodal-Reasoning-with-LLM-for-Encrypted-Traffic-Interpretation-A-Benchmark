from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import ast

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from . import LLMFactory, ConnectorFactory, VisionTowerFactory
from .configuration_tinyllava import TinyLlavaConfig
from ..utils.constants import *


def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None


class TinyLlavaPreTrainedModel(PreTrainedModel):
    config_class = TinyLlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        return self.language_model._supports_sdpa


class TinyLlavaForConditionalGeneration(TinyLlavaPreTrainedModel):
    def __init__(self, config: TinyLlavaConfig):

        super().__init__(config)

        self.language_model = LLMFactory(config.llm_model_name_or_path)[0](config.text_config)
        self.vision_tower = VisionTowerFactory(config.vision_model_name_or_path)(config.vision_config)
        self.connector = ConnectorFactory(config.connector_type)(config)

        (Tokenizer, post_load) = LLMFactory(config.llm_model_name_or_path)[1]
        self.tokenizer = post_load(Tokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            cache_dir=config.cache_dir,
            model_max_length=config.tokenizer_model_max_length,
            padding_side=config.tokenizer_padding_side,
            use_fast=config.tokenizer_use_fast,
        ))
        self.post_init()

        # ── 分类头相关属性（训练前通过 init_cls_head 初始化）──
        self.cls_head = None
        self.num_traffic_classes = 0
        self.lambda_cls = 0.3      # todo
        self._cls_logits_cache = None   # forward 中暂存，用于计算 cls_loss

    def init_cls_head(self, num_classes: int, lambda_cls: float = 0.1):
        """
        在训练脚本中、数据加载后调用，动态初始化 H_align 分类头。
        分类头结构：H_align global avg pool -> Linear -> num_classes
        """
        hidden_size = self.config.text_config.hidden_size  # 2048 for Qwen3-1.7B
        self.num_traffic_classes = num_classes
        self.lambda_cls = lambda_cls
        self.cls_head = nn.Linear(hidden_size, num_classes)
        # 初始化权重
        nn.init.xavier_uniform_(self.cls_head.weight)
        nn.init.zeros_(self.cls_head.bias)
        # 移到模型所在设备和精度
        self.cls_head = self.cls_head.to(device=self.device, dtype=self.dtype)
        print(f"  [CLS_HEAD] Initialized: hidden_size={hidden_size}, "
              f"num_classes={num_classes}, lambda_cls={lambda_cls}")

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds



    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            image_sizes: Optional[List[List[int]]] = None,
            return_dict: Optional[bool] = None,
            class_labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
            )

        output = self.language_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # 🚀 措施 C：对 JSON 输出前 20 个 Token（即类别字段）施加动态加权 Loss
        if self.training and labels is not None and output.logits is not None:
            shift_logits = output.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Hugging Face 默认的 ignore_index 是 -100
            valid_mask = shift_labels != -100

            extra_loss = 0.0
            boost_weight = 5.0  # 额外增加 5 倍 Loss (加上原本的 1 倍，共 6 倍权重)  # todo
            M = 15  # 锁定助手回答的前 15 个 Token (完美覆盖 {"class": "类别名"...)

            loss_fct = nn.CrossEntropyLoss(reduction='sum', ignore_index=-100)

            for b in range(shift_labels.size(0)):
                # 找出这个 batch 中，有效 Token（即助手生成的文本）的起始位置
                valid_indices = valid_mask[b].nonzero(as_tuple=True)[0]
                if len(valid_indices) > 0:
                    start_idx = valid_indices[0]
                    # 只取前 M 个 Token，如果总长度不足 M，就取到最后
                    end_idx = min(start_idx + M, valid_indices[-1] + 1)

                    target_logits = shift_logits[b, start_idx:end_idx]
                    target_labels = shift_labels[b, start_idx:end_idx]

                    # 累加这几个特定 Token 的 Loss
                    extra_loss += loss_fct(target_logits, target_labels)

            # 将额外的 Loss 分摊到整个 Batch 的有效 Token 数量上，保持与原 Loss 尺度一致
            total_valid_tokens = valid_mask.sum()
            if total_valid_tokens > 0:
                extra_loss = (extra_loss * boost_weight) / total_valid_tokens
                output.loss += extra_loss

        # 🚀 措施 A：添加 H_align 的辅助分类 Loss + 精度监控
        if (self.training
                and self.cls_head is not None
                and self._cls_logits_cache is not None
                and class_labels is not None):
            cls_logits = self._cls_logits_cache
            loss_cls = F.cross_entropy(cls_logits, class_labels)

            if output.loss is not None:
                output.loss = output.loss + self.lambda_cls * loss_cls

            # 计算 cls_head 精度并缓存，供 Callback 读取
            with torch.no_grad():
                preds = cls_logits.argmax(dim=-1)
                correct = (preds == class_labels).sum().item()
                total = class_labels.size(0)
                self._cls_acc_correct = getattr(self, '_cls_acc_correct', 0) + correct
                self._cls_acc_total = getattr(self, '_cls_acc_total', 0) + total
                self._cls_loss_cache = loss_cls.item()

            self._cls_logits_cache = None

        return output

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            images: Optional[torch.Tensor] = None,
            image_sizes: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
            )
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(inputs)

        return self.language_model.generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def encode_images(self, images):
        """
        提取流量特征 + H_align 分类 logits。
        """
        kwargs = {}
        kwargs['vision_feature_layer'] = self.config.vision_feature_layer
        kwargs['vision_feature_select_strategy'] = self.config.vision_feature_select_strategy

        images = images.to(device=self.device, dtype=self.dtype)

        # vision_tower 返回 (features, logits)
        image_features_raw, logits = self.vision_tower(images, **kwargs)

        # 原始特征 -> Connector -> LLM 维度 (H_align)
        image_features = self.connector(image_features_raw)

        # H_align 分类头：计算并缓存 cls_logits
        if self.cls_head is not None:
            # image_features: (B, L, D) -> global avg pool -> (B, D) -> cls_logits
            pooled = image_features.mean(dim=1)   # (B, D)
            self._cls_logits_cache = self.cls_head(pooled)  # (B, num_classes)
        else:
            self._cls_logits_cache = None

        return image_features, logits

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = self.language_model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

    def _replace_conf_token(self, token_ids, embeds, labels, conf_placeholder_id, confidence: float):
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        if token_ids.dim() == 0:
            token_ids = token_ids.unsqueeze(0)

        pos = (token_ids == conf_placeholder_id).nonzero(as_tuple=True)[0]
        if len(pos) == 0:
            return embeds, labels

        # 把置信度编码为文字 embedding 的均值
        conf_text = f"{confidence:.1f}%"
        conf_token_ids = self.tokenizer.encode(conf_text, add_special_tokens=False)
        conf_id_tensor = torch.tensor(conf_token_ids, dtype=torch.long, device=self.device)
        conf_embs = self.language_model.get_input_embeddings()(conf_id_tensor)
        conf_emb_mean = conf_embs.mean(dim=0).to(dtype=self.dtype)

        new_embeds = embeds.clone()
        new_embeds[pos[0].item()] = conf_emb_mean

        new_labels = labels.clone()
        new_labels[pos[0].item()] = IGNORE_INDEX

        return new_embeds, new_labels

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels,
            images, image_sizes=None
    ):
        import torch.nn.functional as F

        vision_tower = self.vision_tower
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # --- 1. 暂存原始状态 ---
        _input_ids      = input_ids
        _labels         = labels
        _position_ids   = position_ids
        _attention_mask = attention_mask

        # --- 2. Token 校准 ---
        actual_image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        input_ids[input_ids == actual_image_token_id] = IMAGE_TOKEN_INDEX

        # --- 3. 提取图像特征和分类 logits ---
        image_features, _ = self.encode_images(images)


        # --- 5. 规范化 Mask, Position 和 Labels ---
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels    = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in
                     zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels       = []
        cur_image_idx    = 0

        # --- 6. 核心缝合循环 ---
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images     = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            cur_labels     = labels[batch_idx]


            # Case A: 纯文本样本
            if num_images == 0:
                cur_input_embeds = self.language_model.get_input_embeddings()(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(cur_labels)
                continue

            # Case B: 多模态拼接
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels_noim    = []

            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1: image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1: image_token_indices[i + 1]])

            split_sizes            = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds       = self.language_model.get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            cur_new_input_embeds = []
            cur_new_labels       = []

            for i in range(num_images + 1):
                chunk_embeds = cur_input_embeds_no_im[i]
                chunk_ids    = cur_input_ids_noim[i]
                chunk_labels = cur_labels_noim[i]


                cur_new_input_embeds.append(chunk_embeds)
                cur_new_labels.append(chunk_labels)

                if i < num_images:
                    # 插入图像/流量特征（visual tokens），不再插入 hint embedding
                    cur_image_features = image_features[cur_image_idx]
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full((cur_image_features.shape[0],), IGNORE_INDEX,
                                   device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_image_idx += 1

            new_input_embeds.append(torch.cat(cur_new_input_embeds, dim=0))
            new_labels.append(torch.cat(cur_new_labels, dim=0))

        # --- 7. 截断与 Padding ---
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels       = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len    = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded       = torch.full((batch_size, max_len), IGNORE_INDEX,
                                              dtype=new_labels[0].dtype, device=new_labels[0].device)
        final_attention_mask    = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)
        final_position_ids      = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:]    = cur_new_labels
                    final_attention_mask[i, -cur_len:] = True
                    final_position_ids[i, -cur_len:]   = torch.arange(0, cur_len, dtype=torch.long, device=self.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len]    = cur_new_labels
                    final_attention_mask[i, :cur_len] = True
                    final_position_ids[i, :cur_len]   = torch.arange(0, cur_len, dtype=torch.long, device=self.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # --- 8. 状态恢复与类型转换 ---
        new_labels_result = new_labels_padded if _labels is not None else None
        final_attention_mask_result = final_attention_mask.to(dtype=_attention_mask.dtype) if _attention_mask is not None else None
        final_position_ids_result = final_position_ids if _position_ids is not None else None

        return None, final_position_ids_result, final_attention_mask_result, past_key_values, new_input_embeds, new_labels_result

    def load_llm(self, **kwargs):
        language_model_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        pretrained_llm_path = get_value_from_kwargs(kwargs, 'pretrained_llm_path')
        if pretrained_llm_path is not None:
            language_model_name = pretrained_llm_path
        if language_model_name is not None:
            self.language_model = self.language_model.from_pretrained(
                language_model_name, **kwargs
            )
        print('loading language model from ', language_model_name)
        self.language_model.requires_grad_(False)

        self.config.text_config.torch_dtype = kwargs.get('torch_dtype', None)
        self.config.pad_token = getattr(self.tokenizer, 'pad_token', None)
        self.config.pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)

    def load_vision_tower(self, **kwargs):
        vision_tower_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        self.vision_tower.load_model(vision_tower_name, **kwargs)

    def load_connector(self, **kwargs):
        self.connector.load_model(**kwargs)

