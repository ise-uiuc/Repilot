import os
from pathlib import Path
from typing import Any, Optional, TypeVar, Union, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5Config,
    T5ForConditionalGeneration,
)
from transformers.models.xglm.modeling_xglm import XGLMForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer

from . import utils

Self = TypeVar("Self")


class CodeT5ForRepilot(T5ForConditionalGeneration):
    def __init__(
        self,
        config: T5Config,
        tokenizer: PreTrainedTokenizer,
        *model_args,
    ):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.vocab_size: int = self.config.vocab_size  # type: ignore # noqa
        self.token_map: list[str] = utils.load_and_cache_data(
            Path(f"codet5_token_map.pkl"),
            lambda: [
                tokenizer.decode(
                    id, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                for id in range(self.vocab_size)
            ],
        )
        self.max_tokens: int = self.config.to_dict()["n_positions"]  # type: ignore # noqa
        self.model_args = model_args

    def tpl_encode(self, string: str) -> torch.LongTensor:
        return self.tokenizer.encode(
            string, return_tensors="pt", add_special_tokens=False
        )[0]

    # def tpl_decode(self, tokens: torch.LongTensor) -> str:
    #     return self.tokenizer.decode(tokens)

    @property
    def end_id(self) -> int:
        raise NotImplementedError

    @property
    def end_ids(self) -> list[int]:
        raise NotImplementedError

    @property
    def sep_id(self) -> int:
        raise NotImplementedError

    @property
    def sep(self) -> str:
        return self.token_map[self.sep_id]

    @classmethod
    def init(cls: type[Self]) -> Self:
        """The real init method"""
        return cls.t5_from_pretrained(cls.model_name())  # type: ignore # noqa

    @classmethod
    def model_name(cls) -> str:
        raise NotImplementedError

    @classmethod
    def is_special_token(cls, token: str) -> bool:
        raise NotImplementedError

    @classmethod
    def t5_from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs,
    ):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        model = cls.from_pretrained(
            pretrained_model_name_or_path, tokenizer, *model_args, **kwargs
        )
        return model

    def context(self, prefix: str, suffix: str) -> str:
        return prefix + self.sep + suffix

    def encode(self, prefix: str, suffix: str) -> torch.Tensor:
        context = self.tokenizer.encode(
            self.context(prefix, suffix), return_tensors="pt"
        ).to(utils.DEVICE)
        index = (context[0] == self.sep_id).nonzero()[0]
        half_token_limit = self.max_tokens // 2
        prefix_tensor = context[:, :index]
        prefix_len = prefix_tensor.shape[1]
        suffix_tensor = context[:, index:]
        suffix_len = suffix_tensor.shape[1]
        if prefix_len < half_token_limit and suffix_len < half_token_limit:
            result = torch.cat((prefix_tensor, suffix_tensor), dim=1)
        elif prefix_len >= half_token_limit and suffix_len >= half_token_limit:
            result = torch.cat(
                (
                    prefix_tensor[:, -half_token_limit:],
                    suffix_tensor[:, :half_token_limit],
                ),
                dim=1,
            )
        elif prefix_len < half_token_limit and suffix_len >= half_token_limit:
            n_more = min(half_token_limit - prefix_len, suffix_len - half_token_limit)
            result = torch.cat(
                (
                    prefix_tensor[:, -half_token_limit:],
                    suffix_tensor[:, : half_token_limit + n_more],
                ),
                dim=1,
            )
        elif prefix_len >= half_token_limit and suffix_len < half_token_limit:
            n_more = min(half_token_limit - suffix_len, prefix_len - half_token_limit)
            result = torch.cat(
                (
                    prefix_tensor[:, -half_token_limit - n_more :],
                    suffix_tensor[:, :half_token_limit],
                ),
                dim=1,
            )
        # print(CODET5_TOKENIZER.batch_decode(result))
        # breakpoint()
        assert result.shape[1] <= self.max_tokens
        return result


CODET5_LARGE_SPECIAL_TOKENS = {"<s>", "</s>", "<pad>", "<mask>", "<unk>"}
CODET5_LARGE_END_IDS = [2, 32098]


class CodeT5Large(CodeT5ForRepilot):
    @classmethod
    def model_name(cls) -> str:
        return "Salesforce/codet5-large"

    @property
    def end_id(self) -> int:
        return 2

    @property
    def end_ids(self) -> list[int]:
        return CODET5_LARGE_END_IDS

    @property
    def sep_id(self) -> int:
        return 32099

    @classmethod
    def is_special_token(cls, token: str) -> bool:
        return (token in CODET5_LARGE_SPECIAL_TOKENS) or token.startswith("<extra_id_")

    def pre_allocate(self):
        """https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#pre-allocate-memory-in-case-of-variable-input-length"""
        if torch.cuda.is_available():
            free_memory, _ = cast(tuple[int, int], torch.cuda.mem_get_info())
            free_memory -= 5 * 1024**3
            one_batch_increase = 120 * (1024**2)
            repeat = free_memory // one_batch_increase
            self.generate(  # type: ignore # noqa
                torch.zeros(self.max_tokens, dtype=torch.long).cuda().repeat(repeat, 1),
                max_new_tokens=50,
            )
            # No need to do backward


class Incoder(XGLMForCausalLM):
    def __init__(self, config, model_name: str):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.max_tokens = self.model.config.to_dict()["max_position_embeddings"]
        self.max_tokens = 1500
        self.vocab_size = self.model.config.to_dict()["vocab_size"]
        self.token_map: list[str] = utils.load_and_cache_data(
            Path(f"incoder_token_map.pkl"),
            lambda: [
                self.tokenizer.decode(
                    id, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                for id in range(self.vocab_size)
            ],
        )
        self.infill_ph = "<|mask:0|>"
        self.extra_end = "<|mask:1|><|mask:0|>"
        # signals the end of a generated infill
        self.EOM = "<|endofmask|>"
        self.EOM_ID = 50517
        self.end_id = self.EOM_ID
        self.BOS = "<|endoftext|>"
        self.META_FILE = "<|/ file"

    @property
    def end_ids(self) -> list[int]:
        return [self.EOM_ID]

    @classmethod
    def is_special_token(cls, token: str) -> bool:
        return (
            token.startswith("<|endof")
            or token.startswith("<|/")
            or token.startswith("<|mask")
            or token in ["<s>", "<pad>", "<|endoftext|>", "<unk>"]
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Any, *model_args, **kwargs
    ):
        return super().from_pretrained(
            pretrained_model_name_or_path,
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )

    def context(self, prefix: str, suffix: str) -> str:
        return prefix + self.infill_ph + suffix

    def encode(self, prefix: str, suffix: str) -> torch.Tensor:
        context = (
            self.tokenizer(self.context(prefix, suffix), return_tensors="pt")
            .to(utils.DEVICE)
            .input_ids
        )
        context1 = self.tokenizer.encode(
            self.context(prefix, suffix), return_tensors="pt"
        ).to(utils.DEVICE)
        if not torch.equal(context, context1):
            breakpoint()
        # <|mask:0|>
        index = (context[0] == 50261).nonzero()[0]
        half_token_limit = (self.max_tokens - 2) // 2
        assert len(context) == 1
        prefix_tensor = context[:, :index]
        prefix_len = prefix_tensor.shape[1]
        suffix_tensor = context[:, index:]
        suffix_len = suffix_tensor.shape[1]
        if prefix_len < half_token_limit and suffix_len < half_token_limit:
            result = torch.cat((prefix_tensor, suffix_tensor), dim=1)
        elif prefix_len >= half_token_limit and suffix_len >= half_token_limit:
            result = torch.cat(
                (
                    prefix_tensor[:, -half_token_limit:],
                    suffix_tensor[:, :half_token_limit],
                ),
                dim=1,
            )
        elif prefix_len < half_token_limit and suffix_len >= half_token_limit:
            n_more = min(half_token_limit - prefix_len, suffix_len - half_token_limit)
            result = torch.cat(
                (
                    prefix_tensor[:, -half_token_limit:],
                    suffix_tensor[:, : half_token_limit + n_more],
                ),
                dim=1,
            )
        elif prefix_len >= half_token_limit and suffix_len < half_token_limit:
            n_more = min(half_token_limit - suffix_len, prefix_len - half_token_limit)
            result = torch.cat(
                (
                    prefix_tensor[:, -half_token_limit - n_more :],
                    suffix_tensor[:, :half_token_limit],
                ),
                dim=1,
            )
        # print(CODET5_TOKENIZER.batch_decode(result))
        result = torch.cat(
            (result, torch.tensor([[50262, 50261]]).to(utils.DEVICE)), dim=1
        )
        assert result.shape[1] <= self.max_tokens
        return result

    def pre_allocate(self):
        pass

    def tpl_encode(self, string: str) -> torch.LongTensor:
        return self.tokenizer.encode(
            string, return_tensors="pt", add_special_tokens=False
        )[0]


ModelType = CodeT5Large | Incoder

# model.generate()
