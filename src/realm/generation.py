"""Copied and rewritted from transformers.generation_utils"""
from logging import warning
import pickle
import random
from datasets import d4j
from realm.lsp import client, spec
import inspect
import typing
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Protocol, Tuple, Union, cast
import regex
from pathlib import Path
import time
from joblib import Parallel, delayed
import torch
import torch.distributed as dist
from torch import nn
import multiprocessing as mp
from transformers import T5ForConditionalGeneration
from transformers.generation_utils import GenerationMixin, GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, logger, SampleEncoderDecoderOutput, SampleDecoderOnlyOutput
from transformers.generation_beam_constraints import Constraint, DisjunctiveConstraint, PhrasalConstraint
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.pytorch_utils import torch_int_div
from transformers.utils import ModelOutput, logging
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizers import AddedToken
from realm.analyze.jdt_lsp import JdtLspAnalyzer
from realm.lsp.text import TextDocument, TextFile
from realm import utils


# class IncoderEOM(StoppingCriteria):
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         return True if input_ids[0, -1] == Repairer.EOM_ID else False


MODEL = 'Salesforce/codet5-large'
DEVICE = 'cuda'

JAVA_KEYWORDS = ['abstract', 'continue', 'for', 'new', 'switch',
                 'assert', 'default', 'goto', 'package', 'synchronized',
                 'boolean', 'do', 'if', 'private', 'this',
                 'break', 'double', 'implements', 'protected', 'throw',
                 'byte', 'else', 'import', 'public', 'throws',
                 'case', 'enum', 'instanceof', 'return', 'transient',
                 'catch', 'extends', 'int', 'short', 'try',
                 'char', 'final', 'interface', 'static', 'void',
                 'class', 'finally', 'long', 'strictfp', 'volatile',
                 'const' 'float', 'native', 'super', 'while']

PARTIAL_MEMOIZED: Dict[bytes, List[bool]] = {}
COMPLETE_MEMOIZED: Dict[bytes, List[int]] = {}

class IDTokenError(Exception):
    pass


Generation = Tuple[List[float], List[int], List[str]]

@dataclass
class GenerationContext:
    inside_line_comment: bool
    inside_block_comment: bool
    generated_tokens: List[str]
    generated_ids: List[int]

class LMInferenceConfig(NamedTuple):
    temperature: float
    top_k: int
    max_new_tokens: int
    end_id: int


class LMContext(NamedTuple):
    # for LM
    model: GenerationMixin
    tokenizer: PreTrainedTokenizer
    input_token_ids: torch.Tensor
    inference_config: LMInferenceConfig


class LspContext(NamedTuple):
    # for analysis
    text_file: TextFile
    analyzer: JdtLspAnalyzer

    def copy(self) -> 'LspContext':
        return LspContext(
            self.text_file.copy(),
            self.analyzer.copy()
        )


CODET5 = T5ForConditionalGeneration.from_pretrained(MODEL).to(DEVICE)
CODET5_TOKENIZER: PreTrainedTokenizer = AutoTokenizer.from_pretrained(MODEL)
CODET5_INFERENCE_CONFIG = LMInferenceConfig(0.8, 50, 70, 2)
CODET5_VOC_SIZE: int = CODET5.config.vocab_size
CODET5_TOKEN_MAP: List[str] = utils.load_and_cache_data(Path('codet5_token_map.pkl'), [
    CODET5_TOKENIZER.decode(
        id,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    ) for id in range(CODET5_VOC_SIZE)
])


def codet5_context(prefix: str, suffix: str) -> str:
    return prefix + "<extra_id_0>" + suffix

def codet5_tokenize(prefix: str, suffix: str) -> torch.Tensor:
    context = codet5_context(prefix, suffix)
    return CODET5_TOKENIZER.encode(context, return_tensors='pt').to(DEVICE)


def repair(
    lm_context: LMContext,
    lsp_context: LspContext,
) -> Generation:
    return generate(
        inputs=lm_context.input_token_ids,
        lm_context=lm_context,
        lsp_context=lsp_context,
        model=lm_context.model,
        do_sample=True,
        max_new_tokens=lm_context.inference_config.max_new_tokens,
        top_k=lm_context.inference_config.top_k,
        temperature=lm_context.inference_config.temperature,
    )


def feasible(
    analyzer: JdtLspAnalyzer,
    uri: str,
    token: str,
    pos: spec.Position,
    completion_overhead: List[float],
) -> bool:
    """Returns whether `token` is feasible at `pos` of the file located at `uri`"""
    if regex.match('^([a-zA-Z_][a-zA-Z\\d_]*)$', token) is None:
        warning(
            f'Cannot recognize {token} as an identifier, probabily unicode.')
    if token in JAVA_KEYWORDS:
        return True
    start_time = time.time()
    completion_result = analyzer.client.textDocument_completion({
        'textDocument': {
            'uri': uri
        },
        'position': pos,
    })
    completion_overhead.append(time.time() - start_time)
    # if 'result' in completion_result:
    completions: List[str] = [
        item['textEdit']['newText']  # type: ignore # noqa
        if 'textEdit' in item
        else item['insertText']  # type: ignore # noqa
        for item in completion_result['result']['items']  # type: ignore # noqa
    ]
    # else:
    #     print(uri)
    #     print(completion_result['error'])
    #     print(completion_result['error']['data'])
    #     raise RuntimeError
    if any(filter(
        lambda completion: completion.startswith(token), # type: ignore # noqa
        completions
    )):
        print('Accepted:', token)#, token, completions)
        # if 'lastFraction' in completions:
        #     breakpoint()
        # print("Yes!")
        # print(completions)
        return True
    else:
        print('Denied', token)#, token, completions)
        if token == 'null':
            breakpoint()
        return False

def is_special_token(token: str) -> bool:
    return (token.strip() in ['<s>', '</s>', '<pad>', '<mask>', '<unk>']) or token.startswith('<extra_id_')

# Get the exact identifier token position


def get_id_token(generated_tokens: List[str]) -> str:
    def get():
        for token in reversed(generated_tokens):
            for c in reversed(token):
                if c.isdigit() or c.isalpha() or c == '_':
                    yield c
                else:
                    return
    result = ''.join(reversed(list(get())))

    if len(result) > 0 and not result[0].isdigit():
        return result
    else:
        raise IDTokenError


def get_feasible_token_ids(
    memoized_result: Optional[List[bool]],
    generated_tokens: List[str],
    lsp_context: LspContext,
    # Should be ranked (higher probability first)
    considered_token_ids: List[int],
    top_k: int,
    completion_overhead: List[float]
):
    print("RUNNING feasibility check")
    # Trick to mitigate false positives of the langauge server (e.g., Chart-11, PathIterator)
    # NOTE: do not do it now. Just evaluate on both
    # if idx == 0 and random.random() < 0.1:
    #     exists_satsified_token = True
    #     new_index = idx
    #     break
    analyzer = lsp_context.analyzer
    text_file = lsp_context.text_file
    # analyzer.client.stop()
    result: List[int] = []
    for token_id in considered_token_ids:
        if len(result) == top_k:
            break
        token = CODET5_TOKEN_MAP[token_id]
        token_rstrip = token.rstrip()
        rspace_len = len(token) - len(token_rstrip)
        try:
            id_token = get_id_token(generated_tokens + [token_rstrip])
            text_file.add(token)
            analyzer.change(text_file)
            # analyzer.client.try_recv()
            # No exception afterwards
            pos = text_file.get_position(
                text_file.cursor - rspace_len)
            
            if (memoized_result is not None and memoized_result[token_id]
            ) or feasible(analyzer, text_file.path.as_uri(), id_token, pos, completion_overhead):
                result.append(token_id)
            text_file.delete(len(token))
            analyzer.change(text_file)
        except IDTokenError:
            result.append(token_id)
    length = len(result)
    # TODO: move this logic outer this function
    if length == 0:
        result = considered_token_ids[:top_k]
    elif length != top_k:
        result.extend(result[0] for _ in range(top_k - length))
    return result


@typing.no_type_check
@torch.no_grad()
def generate(
    model: GenerationMixin,
    lm_context: LMContext,
    lsp_context: LspContext,
    inputs: Optional[torch.Tensor] = None,
    max_length: Optional[int] = None,
    min_length: Optional[int] = None,
    do_sample: Optional[bool] = None,
    early_stopping: Optional[bool] = None,
    num_beams: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    typical_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    bad_words_ids: Optional[Iterable[int]] = None,
    force_words_ids: Optional[Union[Iterable[int],
                                    Iterable[Iterable[int]]]] = None,
    bos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    length_penalty: Optional[float] = None,
    no_repeat_ngram_size: Optional[int] = None,
    encoder_no_repeat_ngram_size: Optional[int] = None,
    num_return_sequences: Optional[int] = None,
    max_time: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    decoder_start_token_id: Optional[int] = None,
    use_cache: Optional[bool] = None,
    num_beam_groups: Optional[int] = None,
    diversity_penalty: Optional[float] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[
        int, torch.Tensor], List[int]]] = None,
    logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(
    ),
    renormalize_logits: Optional[bool] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(
    ),
    constraints: Optional[List[Constraint]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    forced_bos_token_id: Optional[int] = None,
    forced_eos_token_id: Optional[int] = None,
    remove_invalid_values: Optional[bool] = None,
    synced_gpus: Optional[bool] = False,
    exponential_decay_length_penalty: Optional[Tuple[Union[int, float]]] = None,
    **model_kwargs,
) -> Generation:
    # 1. Set generation parameters if not already defined
    bos_token_id = bos_token_id if bos_token_id is not None else model.config.bos_token_id
    num_beams = num_beams if num_beams is not None else model.config.num_beams
    length_penalty = length_penalty if length_penalty is not None else model.config.length_penalty
    early_stopping = early_stopping if early_stopping is not None else model.config.early_stopping
    num_beam_groups = num_beam_groups if num_beam_groups is not None else model.config.num_beam_groups
    do_sample = do_sample if do_sample is not None else model.config.do_sample
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else model.config.num_return_sequences
    )

    pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id

    if eos_token_id is None and hasattr(model.config, "decoder"):
        eos_token_id = model.config.decoder.eos_token_id

    if pad_token_id is None and eos_token_id is not None:
        if model_kwargs.get("attention_mask", None) is None:
            logger.warning(
                "The attention mask and the pad token id were not set. As a consequence, you may observe "
                "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
            )
        logger.warning(
            f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        pad_token_id = eos_token_id

    output_scores = output_scores if output_scores is not None else model.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else model.config.return_dict_in_generate
    )

    # 2. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
        inputs, bos_token_id, model_kwargs)
    batch_size = inputs_tensor.shape[0]
    assert batch_size == 1

    # 3. Define other model kwargs
    model_kwargs["output_attentions"] = output_attentions
    model_kwargs["output_hidden_states"] = output_hidden_states
    model_kwargs["use_cache"] = use_cache

    accepts_attention_mask = "attention_mask" in set(
        inspect.signature(model.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
            inputs_tensor, pad_token_id, eos_token_id
        )

    if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

    # 4. Prepare `input_ids` which will be used for auto-regressive generation
    if model.config.is_encoder_decoder:
        input_ids = model._prepare_decoder_input_ids_for_generation(
            batch_size,
            decoder_start_token_id=decoder_start_token_id,
            bos_token_id=bos_token_id,
            model_kwargs=model_kwargs,
            device=inputs_tensor.device,
        )
    else:
        # if decoder-only then inputs_tensor has to be `input_ids`
        input_ids = inputs_tensor

    # 5. Prepare `max_length` depending on other stopping criteria.
    input_ids_seq_length = input_ids.shape[-1]
    if max_length is None and max_new_tokens is None:
        warnings.warn(
            "Neither `max_length` nor `max_new_tokens` have been set, `max_length` will default to "
            f"{model.config.max_length} (`model.config.max_length`). Controlling `max_length` via the config is "
            "deprecated and `max_length` will be removed from the config in v5 of Transformers -- we recommend "
            "using `max_new_tokens` to control the maximum length of the generation.",
            UserWarning,
        )
    elif max_length is None and max_new_tokens is not None:
        max_length = max_new_tokens + input_ids_seq_length
    elif max_length is not None and max_new_tokens is not None:
        raise ValueError(
            "Both `max_new_tokens` and `max_length` have been set but they serve the same purpose -- setting a"
            " limit to the generated output length. Remove one of those arguments. Please refer to the"
            " documentation for more information. "
            "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
        )
    # default to config if still None
    max_length = max_length if max_length is not None else model.config.max_length
    min_length = min_length if min_length is not None else model.config.min_length

    if min_length is not None and min_length > max_length:
        raise ValueError(
            f"Unfeasable length constraints: the minimum length ({min_length}) is larger than the maximum "
            f"length ({max_length})"
        )
    if input_ids_seq_length >= max_length:
        input_ids_string = "decoder_input_ids" if model.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
            f" {max_length}. This can lead to unexpected behavior. You should consider increasing "
            "`max_new_tokens`."
        )

    # 7. prepare distribution pre_processing samplers
    logits_processor = model._get_logits_processor(
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=inputs_tensor,
        bad_words_ids=bad_words_ids,
        min_length=min_length,
        max_length=max_length,
        eos_token_id=eos_token_id,
        forced_bos_token_id=forced_bos_token_id,
        forced_eos_token_id=forced_eos_token_id,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        remove_invalid_values=remove_invalid_values,
        exponential_decay_length_penalty=exponential_decay_length_penalty,
        logits_processor=logits_processor,
        renormalize_logits=renormalize_logits,
    )

    # 11. expand input_ids with `num_return_sequences` additional sequences per batch
    input_ids, model_kwargs = model._expand_inputs_for_generation(
        input_ids,
        expand_size=num_return_sequences,
        is_encoder_decoder=model.config.is_encoder_decoder,
        **model_kwargs,
    )

    # 12. run sample
    return sample(
        model,
        input_ids,
        lsp_context=lsp_context,
        end_id=lm_context.inference_config.end_id,
        top_k=top_k,
        logits_processor=logits_processor,
        stopping_criteria=stopping_criteria,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        output_scores=output_scores,
        max_length=max_length,
        **model_kwargs,
    )


@typing.no_type_check
def sample(
    model: GenerationMixin,
    input_ids: torch.LongTensor,
    lsp_context: LspContext,
    top_k: int,
    end_id: int,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    **model_kwargs,
) -> Generation:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id
    output_scores = output_scores if output_scores is not None else model.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )

    completion_overhead: List[float] = []

    # Context variable controling when to do analysis
    # TODO(future): use AST for more accurate analysis (e.g., inside string literal, etc.)
    gen_context = GenerationContext(
        inside_line_comment=False,
        inside_block_comment=False,
        generated_tokens=[],
        generated_ids=[],
    )

    # auto-regressive generation
    while True:
        # prepare model inputs
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # -1 because decoder's self-attention produces n outputs given n inputs
        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = logits_processor(input_ids, next_token_logits)

        probs = nn.functional.softmax(next_token_scores, dim=-1)
        assert len(probs) == 1
        probs = probs[0]

        # TODO: justify this
        _, considered_next_token_ids = torch.topk(
            probs, k=2*top_k, dim=-1)

        assert len(input_ids) == 1

        input_ids_list = input_ids[0].tolist()
        considered_next_token_ids_list = considered_next_token_ids.tolist()

        state_bytes = pickle.dumps((input_ids_list, considered_next_token_ids_list))

        if not gen_context.inside_line_comment and not gen_context.inside_block_comment:
            if state_bytes in COMPLETE_MEMOIZED:
                print("HUGE HIT")
                # breakpoint()
                feasible_next_token_ids = torch.LongTensor(COMPLETE_MEMOIZED[state_bytes]).to(DEVICE)
            else:
                lsp_context.analyzer.change(lsp_context.text_file)
                input_ids_bytes = pickle.dumps(input_ids_list)
                # TODO: delete this
                partial_memoized_result = PARTIAL_MEMOIZED.get(input_ids_bytes)
                if partial_memoized_result is not None:
                    print("SMALL HIT")
                    # breakpoint()
                feasible_next_token_ids_list = get_feasible_token_ids(
                    partial_memoized_result,
                    gen_context.generated_tokens,
                    lsp_context,
                    considered_next_token_ids_list,
                    top_k,
                    completion_overhead,
                )
                COMPLETE_MEMOIZED[state_bytes] = feasible_next_token_ids_list
                if partial_memoized_result is None:
                    PARTIAL_MEMOIZED[input_ids_bytes] = [False] * CODET5_VOC_SIZE
                for idx in feasible_next_token_ids_list:
                    PARTIAL_MEMOIZED[input_ids_bytes][idx] = True
                feasible_next_token_ids = torch.LongTensor(feasible_next_token_ids_list).to(DEVICE)

            # rewrite probabilities
            mask = torch.ones(
                probs.shape,
                dtype=torch.bool
            ).to(DEVICE)
            mask.index_fill_(0, feasible_next_token_ids, False)
            probs.masked_fill_(mask, 0.)

        # shape: (1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        next_token_id_item = next_token_id.item()
        # except RuntimeError:
        #     # TODO: fix
        #     print("RUNTIME ERROR")
        #     probs = nn.functional.softmax(next_token_scores, dim=-1)
        #     next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        next_token = CODET5_TOKEN_MAP[next_token_id]
        if next_token.strip().startswith('//'):
            gen_context.inside_line_comment = True
        elif next_token.endswith('\n'):
            # breakpoint()
            gen_context.inside_line_comment = False
        elif next_token.strip().startswith('/*'):
            gen_context.inside_block_comment = True
        elif next_token.endswith('*/'):
            # breakpoint()
            gen_context.inside_block_comment = False

        if not is_special_token(next_token):
            lsp_context.text_file.add(next_token)
            gen_context.generated_ids.append(next_token_id)
            gen_context.generated_tokens.append(next_token)


        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_token_id[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )

        # stop when each sentence is finished, or if we exceed the maximum length
        # IMPORTANT: codet5 output format: <mask0>....<mask1>....<mask2>...
        # Mask ids are 32099, 32098, 32097...
        if next_token_id_item == 32098 or next_token_id_item == end_id or len(input_ids[0]) == max_length:
            break
    return completion_overhead, gen_context.generated_ids, gen_context.generated_tokens
