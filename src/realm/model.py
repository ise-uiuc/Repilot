from transformers import T5ForConditionalGeneration, T5Config, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import List, Optional, Union, TypeVar
from realm import utils
from pathlib import Path
import torch
import os

Self = TypeVar('Self')


class CodeT5ForRealm(T5ForConditionalGeneration):
    def __init__(
        self,
        config: T5Config,
        tokenizer: PreTrainedTokenizer,
        *model_args,
    ):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.vocab_size: int = self.config.vocab_size  # type: ignore # noqa
        self.token_map: List[str] = utils.load_and_cache_data(Path('codet5_token_map.pkl'), [
            tokenizer.decode(
                id,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            ) for id in range(self.vocab_size)
        ])
        self.sep_id = 32099
        self.sep = self.token_map[self.sep_id]
        assert self.sep == "<extra_id_0>"
        self.max_tokens: int = self.config.to_dict()['n_positions']  # type: ignore # noqa
        self.model_args = model_args

    @classmethod
    def init(cls: type[Self]) -> Self:
        """The real init method"""
        return cls.t5_from_pretrained(cls.model_name()) # type: ignore # noqa

    @classmethod
    def model_name(cls) -> str:
        raise NotImplementedError

    @classmethod
    def is_special_token(cls, token: str) -> bool:
        raise NotImplementedError

    @classmethod
    def t5_from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path)
        model = cls.from_pretrained(
            pretrained_model_name_or_path, tokenizer, *model_args, **kwargs)
        return model

    def context(self, prefix: str, suffix: str) -> str:
        return prefix + self.sep + suffix

    def is_end(self, id: int) -> bool:
        return id == 32098 or id == 2

    def encode(self, prefix: str, suffix: str) -> torch.Tensor:
        context = self.tokenizer.encode(self.context(
            prefix, suffix), return_tensors='pt').to(utils.DEVICE)
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
                (prefix_tensor[:, -half_token_limit:], suffix_tensor[:, :half_token_limit]), dim=1)
        elif prefix_len < half_token_limit and suffix_len >= half_token_limit:
            n_more = min(half_token_limit - prefix_len,
                         suffix_len - half_token_limit)
            result = torch.cat(
                (prefix_tensor[:, -half_token_limit:], suffix_tensor[:, :half_token_limit + n_more]), dim=1)
        elif prefix_len >= half_token_limit and suffix_len < half_token_limit:
            n_more = min(half_token_limit - suffix_len,
                         prefix_len - half_token_limit)
            result = torch.cat(
                (prefix_tensor[:, -half_token_limit - n_more:], suffix_tensor[:, :half_token_limit]), dim=1)
        # print(CODET5_TOKENIZER.batch_decode(result))
        # breakpoint()
        assert result.shape[1] <= self.max_tokens
        return result


class CodeT5Large(CodeT5ForRealm):
    @classmethod
    def model_name(cls) -> str:
        return 'Salesforce/codet5-large'

    @classmethod
    def is_special_token(cls, token: str) -> bool:
        return (token in {'<s>', '</s>', '<pad>', '<mask>', '<unk>'}) or token.startswith('<extra_id_')