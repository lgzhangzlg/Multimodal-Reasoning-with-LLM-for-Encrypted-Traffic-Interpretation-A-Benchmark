from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
import copy

from .formatter import EmptyFormatter, StringFormatter
from .formatter import Formatter
from ...utils.constants import *

from transformers import PreTrainedTokenizer
import torch


@dataclass
class Template:
    format_image_token: "Formatter"
    format_user: "Formatter"
    format_assistant: "Formatter"
    system: "Formatter"
    separator: "Formatter"

    def encode(self, messages, tokenizer, mode='train'):
        """
        1. get list form messages(conversations:[{from:human, value:message}, {from:gpt, value:message}])
            ===>  human_list, value_list
        2. prompt two list
        3. tokenize prompt
        4. make target
        """
        question_list, answer_list = self.get_list_from_message(messages)
        prompt = self.prompt(question_list, answer_list)
        input_ids = self.tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
        if mode == 'train':
            labels = self.make_labels(input_ids, prompt, tokenizer)
            return dict(
                input_ids=input_ids,
                labels=labels
            )
        else:
            return dict(input_ids=input_ids, prompt=prompt)

    def get_list_from_message(self, messages):
        return self._get_list_from_message(messages)

    def _get_list_from_message(self, messages):
        """
        messages  ====>  [{from:human, value:message}, {from:gpt, value:message}]
        """
        question_list = []
        answer_list = []
        first_is_not_question = 0
        for i, message in enumerate(messages):
            if i == 0 and message['from'] != 'human':
                first_is_not_question = 1
                continue
            if i % 2 == first_is_not_question:
                question_list.append(message['value'])
            else:
                answer_list.append(message['value'])

        assert len(question_list) == len(answer_list), \
            f"qa is not match : length_q:{len(question_list)} vs length_a:{len(answer_list)}"
        return question_list, answer_list

    def prompt(self, question_list, answer_list):
        if type(question_list) is str:
            question_list = [question_list]
        if type(answer_list) is str:
            answer_list = [answer_list]
        msg = self._prompt(question_list, answer_list)
        return msg

    def _prompt(self, question_list, answer_list):
        msg = ""
        for i, (question, answer) in enumerate(zip(question_list, answer_list)):
            if i == 0:
                msg += self.system.apply()
            if DEFAULT_IMAGE_TOKEN in question:
                question = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
                question = self.format_image_token.apply(content=question).strip()
            msg += self.format_user.apply(content=question)
            msg += self.format_assistant.apply(content=answer)
        return msg

    def make_labels(self, input_ids, prompt, tokenizer):
        labels = copy.deepcopy(input_ids)
        sep, eos_token = self.separator.apply()

        # =========================
        # [MODIFIED] 根治 mismatch：total_len 一律以“最终 input_ids 的真实长度”为准
        # =========================
        # ---- 原始代码（保留注释） ----
        # total_len = int(labels.ne(tokenizer.pad_token_id).sum())
        # if tokenizer.pad_token_id == tokenizer.eos_token_id:
        #     total_len += prompt.count(eos_token)

        # ---- 修改后 ----
        # input_ids 在你这里没有 padding（tokenizer_image_token 没做 pad），
        # 所以 total_len 直接等于 token 序列长度就是“真值”，不会再受 eos/pad 特殊情况影响
        if isinstance(labels, torch.Tensor):
            total_len = int(labels.numel())
        else:
            total_len = len(labels)
        # =========================

        rounds = prompt.split(eos_token)

        # =========================
        # [MODIFIED] eos_token_length 也用同一套 tokenizer_image_token 口径，避免 encode() 带 special tokens
        # =========================
        # ---- 原始代码（保留注释） ----
        # eos_token_length = len(tokenizer.encode(eos_token))

        # ---- 修改后 ----
        eos_token_length = len(self.tokenizer_image_token(eos_token, tokenizer))
        # =========================

        labels, cur_len = self._make_masks(labels, tokenizer, sep, eos_token_length, rounds)

        # =========================
        # [MODIFIED] mismatch 不再用“刷屏+sleep+清空 labels”这种方式处理
        # 你要么直接 assert/raise（严格），要么只 warning 不破坏训练数据
        # =========================
        if cur_len < tokenizer.model_max_length and cur_len != total_len:
            # ---- 原始代码（保留注释） ----
            # import time
            # print(
            #     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
            #     f" (ignored)"
            # )
            # print("number of rounds: ", len(rounds) - 1)
            # print("rounds: ", rounds[:-1])
            # print("prompt: ", prompt)
            # print(labels)
            # print(input_ids)
            # time.sleep(5)
            # labels[:] = IGNORE_INDEX

            # ---- 修改后（严格模式：直接报错，保证你第一时间发现问题）----
            raise ValueError(
                f"tokenization mismatch: cur_len={cur_len}, total_len={total_len}. "
                f"Please check template/eos/image token handling."
            )
        # =========================

        return labels

    def _make_masks(self, labels, tokenizer, sep, eos_token_length, rounds):
        cur_len = 0
        for rou in rounds:
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                # system block 或其他非 QA round，只推进 cur_len，不做 mask
                if rou.strip():
                    round_len = len(self.tokenizer_image_token(rou, tokenizer)) + eos_token_length
                    labels[cur_len: cur_len + round_len] = IGNORE_INDEX
                    cur_len += round_len
                continue
            parts[0] += sep
            round_len = len(self.tokenizer_image_token(rou, tokenizer)) + eos_token_length
            instruction_len = len(self.tokenizer_image_token(parts[0], tokenizer)) - 1
            labels[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        labels[cur_len:] = IGNORE_INDEX
        return labels, cur_len

    @classmethod
    def tokenizer_image_token(cls, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        def _insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        # =========================
        # [MODIFIED] 根治 mismatch：禁止 tokenizer(chunk) 默认加 special tokens
        # =========================
        # ---- 原始代码（保留注释） ----
        # prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        # ---- 修改后 ----
        # 强制 add_special_tokens=False，避免 chunk 前面被自动加 BOS、末尾被自动加 EOS 等导致长度飘
        prompt_chunks = [
            tokenizer(chunk, add_special_tokens=False).input_ids
            for chunk in prompt.split('<image>')
        ]
        # =========================

        input_ids = []
        offset = 0

        # =========================
        # 这里保留你原来的 BOS 处理逻辑（如果第一个 chunk 自己带了 BOS 就保留一个）
        # 但由于上面 add_special_tokens=False，一般不会再出现 BOS，
        # 所以这段通常不会触发，也就更稳定。
        # =========================
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in _insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids
