"""Copied and rewritted from transformers.generation_utils"""
from logging import warning
import random
from datasets import d4j
from realm.lsp import spec
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


class IDTokenError(Exception):
    pass


Generation = Tuple[List[float], List[int], List[str]]


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


LspContextList = List[LspContext]


CODET5 = T5ForConditionalGeneration.from_pretrained(MODEL).to(DEVICE)
CODET5_TOKENIZER: PreTrainedTokenizer = AutoTokenizer.from_pretrained(MODEL)
CODET5_INFERENCE_CONFIG = LMInferenceConfig(0.8, 50, 70, 2)
CODET5_TOKEN_MAP: Dict[int, str] = utils.load_and_cache_data(Path('codet5_token_map.pkl'), {
    int(id): CODET5_TOKENIZER.decode(
        id,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    ) for id in range(CODET5.config.vocab_size)
})


def codet5_context(prefix: str, suffix: str) -> str:
    return prefix + "<extra_id_0>" + suffix


def codet5_tokenize(prefix: str, suffix: str) -> torch.Tensor:
    context = codet5_context(prefix, suffix)
    return CODET5_TOKENIZER.encode(context, return_tensors='pt').to(DEVICE)


def repair(
    lm_context: LMContext,
    lsp_context: LspContextList,
    do_analysis: bool,
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
        config_do_analysis=do_analysis,
    )


def feasible(
    analyzer: JdtLspAnalyzer,
    uri: str,
    token: str,
    pos: spec.Position
) -> bool:
    """Returns whether `token` is feasible at `pos` of the file located at `uri`"""
    if regex.match('^([a-zA-Z_][a-zA-Z\\d_]*)$', token) is None:
        warning(
            f'Cannot recognize {token} as an identifier, probabily unicode.')
    if token in JAVA_KEYWORDS:
        return True
    # start_time = time.time()
    completion_result = analyzer.client.textDocument_completion({
        'textDocument': {
            'uri': uri
        },
        'position': pos,
    })
    # completion_overhead.append(time.time() - start_time)
    completions: List[str] = [
        item['textEdit']['newText']  # type: ignore # noqa
        if 'textEdit' in item
        else item['insertText']  # type: ignore # noqa
        for item in completion_result['result']['items']  # type: ignore # noqa
    ]
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


def feasible_token_ids(
    queue: mp.Queue,
    generated_tokens: List[str],
    lsp_context: LspContext,
    token_map: Dict[int, str],
    considered_token_ids: List[int],
    top_k: int,
):
    # Trick to mitigate false positives of the langauge server (e.g., Chart-11, PathIterator)
    # NOTE: do not do it now. Just evaluate on both
    # if idx == 0 and random.random() < 0.1:
    #     exists_satsified_token = True
    #     new_index = idx
    #     break
    analyzer = lsp_context.analyzer
    text_file = lsp_context.text_file
    satisfied_token_ids: List[int] = []
    for token_id in considered_token_ids:
        if len(satisfied_token_ids) == top_k:
            break
        token = token_map[token_id]
        token_rstrip = token.rstrip()
        rspace_len = len(token) - len(token_rstrip)
        try:
            id_token = get_id_token(generated_tokens + [token_rstrip])
            text_file.add(token)
            analyzer.change(text_file)
            analyzer.client.try_recv()
            # No exception afterwards
            pos = text_file.get_position(
                text_file.cursor - rspace_len)
            if feasible(analyzer, text_file.path.as_uri(), id_token, pos):
                satisfied_token_ids.append(token_id)
            text_file.delete(len(token))
            analyzer.change(text_file)
        except IDTokenError:
            satisfied_token_ids.append(token_id)
    queue.put(satisfied_token_ids)
    # Zeroing out unsatisfied tokens
    # return next_probabilities.masked_fill(torch.zeros(
    #     next_probabilities.shape.numel(),
    #     dtype=torch.bool
    # ).index_fill(0, torch.tensor(satisfied_token_ids), True), 0.)

# Modified from GenerationMixin.generate


@typing.no_type_check
@torch.no_grad()
def generate(
    model: GenerationMixin,
    lm_context: LMContext,
    lsp_context: LspContextList,
    config_do_analysis: bool,
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
    # Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
    r"""

    Generates sequences of token ids for models with a language modeling head. The method supports the following
    generation methods for text-decoder, text-to-text, speech-to-text, and vision-to-text models:

        - *greedy decoding* by calling [`~generation_utils.GenerationMixin.greedy_search`] if `num_beams=1` and
            `do_sample=False`.
        - *multinomial sampling* by calling [`~generation_utils.GenerationMixin.sample`] if `num_beams=1` and
            `do_sample=True`.
        - *beam-search decoding* by calling [`~generation_utils.GenerationMixin.beam_search`] if `num_beams>1` and
            `do_sample=False`.
        - *beam-search multinomial sampling* by calling [`~generation_utils.GenerationMixin.beam_sample`] if
            `num_beams>1` and `do_sample=True`.
        - *diverse beam-search decoding* by calling [`~generation_utils.GenerationMixin.group_beam_search`], if
            `num_beams>1` and `num_beam_groups>1`.
        - *constrained beam-search decoding* by calling
            [`~generation_utils.GenerationMixin.constrained_beam_search`], if `constraints!=None` or
            `force_words_ids!=None`.

    <Tip warning={true}>

    Apart from `inputs`, all the arguments below will default to the value of the attribute of the same name as
    defined in the model's config (`config.json`) which in turn defaults to the
    [`~modeling_utils.PretrainedConfig`] of the model.

    </Tip>

    Most of these parameters are explained in more detail in [this blog
    post](https://huggingface.co/blog/how-to-generate).

    Parameters:
        inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
            should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
            `input_ids`, `input_values`, `input_features`, or `pixel_values`.
        max_length (`int`, *optional*, defaults to `model.config.max_length`):
            The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
            `max_new_tokens`. In general, prefer the use of `max_new_tokens`, which ignores the number of tokens in
            the prompt.
        max_new_tokens (`int`, *optional*):
            The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        min_length (`int`, *optional*, defaults to 10):
            The minimum length of the sequence to be generated.
        do_sample (`bool`, *optional*, defaults to `False`):
            Whether or not to use sampling ; use greedy decoding otherwise.
        early_stopping (`bool`, *optional*, defaults to `False`):
            Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
        num_beams (`int`, *optional*, defaults to 1):
            Number of beams for beam search. 1 means no beam search.
        temperature (`float`, *optional*, defaults to 1.0):
            The value used to module the next token probabilities.
        top_k (`int`, *optional*, defaults to 50):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`, *optional*, defaults to 1.0):
            If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or higher
            are kept for generation.
        typical_p (`float`, *optional*, defaults to 1.0):
            The amount of probability mass from the original distribution to be considered in typical decoding. If
            set to 1.0 it takes no effect. See [this paper](https://arxiv.org/pdf/2202.00666.pdf) for more details.
        repetition_penalty (`float`, *optional*, defaults to 1.0):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        bos_token_id (`int`, *optional*):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token.
        length_penalty (`float`, *optional*, defaults to 1.0):
                Exponential penalty to the length. 1.0 means that the beam score is penalized by the sequence length.
                0.0 means no penalty. Set to values < 0.0 in order to encourage the model to generate longer
                sequences, to a value > 0.0 in order to encourage the model to produce shorter sequences.
        no_repeat_ngram_size (`int`, *optional*, defaults to 0):
            If set to int > 0, all ngrams of that size can only occur once.
        encoder_no_repeat_ngram_size (`int`, *optional*, defaults to 0):
            If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the
            `decoder_input_ids`.
        bad_words_ids(`List[List[int]]`, *optional*):
            List of token ids that are not allowed to be generated. In order to get the token ids of the words that
            should not appear in the generated text, use `tokenizer(bad_words, add_prefix_space=True,
            add_special_tokens=False).input_ids`.
        force_words_ids(`List[List[int]]` or `List[List[List[int]]]`, *optional*):
            List of token ids that must be generated. If given a `List[List[int]]`, this is treated as a simple
            list of words that must be included, the opposite to `bad_words_ids`. If given `List[List[List[int]]]`,
            this triggers a [disjunctive constraint](https://github.com/huggingface/transformers/issues/14081),
            where one can allow different forms of each word.
        num_return_sequences(`int`, *optional*, defaults to 1):
            The number of independently computed returned sequences for each element in the batch.
        max_time(`float`, *optional*):
            The maximum amount of time you allow the computation to run for in seconds. generation will still
            finish the current pass after allocated time has been passed.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values are in `[0, 1]`, 1 for tokens
            that are not masked, and 0 for masked tokens. If not provided, will default to a tensor the same shape
            as `input_ids` that masks the pad token. [What are attention masks?](../glossary#attention-mask)
        decoder_start_token_id (`int`, *optional*):
            If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
        use_cache: (`bool`, *optional*, defaults to `True`):
            Whether or not the model should use the past last key/values attentions (if applicable to the model) to
            speed up decoding.
        num_beam_groups (`int`, *optional*, defaults to 1):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of
            beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
        diversity_penalty (`float`, *optional*, defaults to 0.0):
            This value is subtracted from a beam's score if it generates a token same as any beam from other group
            at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is
            enabled.
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
            If provided, this function constraints the beam search to allowed tokens only at each step. If not
            provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
            `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
            on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
            for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
            Retrieval](https://arxiv.org/abs/2010.00904).
        logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and a
                model's config. If a logit processor is passed that is already created with the arguments or a model's
                config an error is thrown. This feature is intended for advanced users.
        renormalize_logits: (`bool`, *optional*, defaults to `False`):
            Whether to renormalize the logits after applying all the logits processors or warpers (including the
            custom ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the
            score logits are normalized but some logit processors or warpers break the normalization.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                model's config. If a stopping criteria is passed that is already created with the arguments or a
                model's config an error is thrown. This feature is intended for advanced users.
        constraints (`List[Constraint]`, *optional*):
                Custom constraints that can be added to the generation to ensure that the output will contain the use
                of certain tokens as defined by `Constraint` objects, in the most sensible way possible.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        forced_bos_token_id (`int`, *optional*):
            The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful
            for multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be
            the target language token.
        forced_eos_token_id (`int`, *optional*):
            The id of the token to force as the last generated token when `max_length` is reached.
        remove_invalid_values (`bool`, *optional*):
            Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to
            crash. Note that using `remove_invalid_values` can slow down generation.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        exponential_decay_length_penalty (`tuple(int, float)`, *optional*):
            This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been
            generated. The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates
            where penalty starts and `decay_factor` represents the factor of exponential decay

        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If the model
            is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific kwargs
            should be prefixed with *decoder_*.

    Return:
        [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
        or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

            If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation_utils.GreedySearchDecoderOnlyOutput`],
                - [`~generation_utils.SampleDecoderOnlyOutput`],
                - [`~generation_utils.BeamSearchDecoderOnlyOutput`],
                - [`~generation_utils.BeamSampleDecoderOnlyOutput`]

            If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation_utils.GreedySearchEncoderDecoderOutput`],
                - [`~generation_utils.SampleEncoderDecoderOutput`],
                - [`~generation_utils.BeamSearchEncoderDecoderOutput`],
                - [`~generation_utils.BeamSampleEncoderDecoderOutput`]
    """
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
    assert batch_size == len(lsp_context)

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

    # Prepare criteria
    # 8. prepare stopping criteria
    stopping_criteria = model._get_stopping_criteria(
        max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
    )

    # 10. prepare logits warper
    logits_warper = model._get_logits_warper(
        top_k=top_k,
        top_p=top_p,
        typical_p=typical_p,
        temperature=temperature,
        num_beams=num_beams,
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
        lsp_context_list=lsp_context,
        end_id=lm_context.inference_config.end_id,
        tokenizer=lm_context.tokenizer,
        top_k=top_k,
        config_do_analysis=config_do_analysis,
        logits_processor=logits_processor,
        logits_warper=logits_warper,
        stopping_criteria=stopping_criteria,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        output_scores=output_scores,
        return_dict_in_generate=return_dict_in_generate,
        synced_gpus=synced_gpus,
        max_length=max_length,
        **model_kwargs,
    )


@typing.no_type_check
def sample(
    model: GenerationMixin,
    input_ids: torch.LongTensor,
    lsp_context_list: LspContextList,
    top_k: int,
    config_do_analysis: bool,
    tokenizer: PreTrainedTokenizer,
    end_id: int,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: Optional[bool] = False,
    **model_kwargs,
) -> Generation:
    # Union[SampleOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        logits_warper (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
            to warp the prediction score distribution of the language modeling head applied before multinomial
            sampling at each generation step.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.
    """

    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        # Remove this max_length limitation
        # stopping_criteria = validate_stopping_criteria(
        #     stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id
    output_scores = output_scores if output_scores is not None else model.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else model.config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (
        return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get(
            "attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get(
                "hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    cur_len = input_ids.shape[-1]

    this_peer_finished = False  # used by synced_gpus only
    completion_overhead: List[float] = []
    generated_ids: List[int] = []
    generated_tokens: List[str] = []
    # Context variable controling when to do analysis
    # TODO(future): use AST for more accurate analysis (e.g., inside string literal, etc.)
    context = {
        'inside_line_comment': False,
        'inside_block_comment': False,
    }
    # auto-regressive generation
    while True:
        do_analysis = not context['inside_line_comment'] and not context['inside_block_comment']
        do_analysis = config_do_analysis and do_analysis
        # do_analysis = False
        if synced_gpus:
            assert False
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(
                0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

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

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        # -1 because decoder's self-attention produces n outputs given n inputs
        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (
                        outputs.attentions,)
                )
                if model.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if model.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # sample
        # probs = nn.functional.softmax(next_token_scores, dim=-1)
        # batch_size * TOP_K
        # all_next_token_ids = torch.topk(probs, k=top_k, dim=-1).indices
        # all_next_token_ids = torch.multinomial(
        #     probs, num_samples=TOP_K).squeeze(1).view(TOP_K, 1)
        # all_next_token_ids = torch.topk(next_token_scores, k=6, dim=-1).indices.view(6, 1)
        # exit()

        # finished sentences should have their next token be a padding token
        # print(next_token_ids)
        # if eos_token_id is not None:
        #     if pad_token_id is None:
        #         raise ValueError(
        #             "If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
        #     # print(pad_token_id)
        #     # print(unfinished_sequences)
        #     for idx, _ in enumerate(all_next_token_ids):
        #         all_next_token_ids[idx] = all_next_token_ids[idx] * unfinished_sequences + \
        #             pad_token_id * (1 - unfinished_sequences)
        #     # print(next_token_ids)
        #     # exit()

        # update generated ids, model inputs, and length for next step
        # for next_token_ids in all_next_token_ids:
        #     assert next_token_ids.shape == torch.Size([1]), next_token_ids.shape
        # Don't use `convert_ids_to_tokens`, which would give weird results
        # TODO: memorize decode
        # tokens = [{
        #     id.item(): token
        #     for (id, token) in
        #     zip(
        #         ids,
        #         (tokenizer.decode(
        #             id,
        #             skip_special_tokens=False,
        #             clean_up_tokenization_spaces=False
        #         ) for id in ids)
        #     )} for ids in all_next_token_ids
        # ]
        # breakpoint()
        # TODO: support parallel processing
        # if len(tokens) != 1:
        #     raise NotImplementedError
        # if len(all_next_token_ids) != 1:
        #     raise NotImplementedError
        # if len(probs) != 1:
        #     raise NotImplementedError
        # probs = probs[0]
        # all_next_token_ids = all_next_token_ids[0]
        # tokens = tokens[0]
        # tokens = self.tokenizer.batch_decode(
        # all_next_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        # assert len(tokens) == top_k

        if do_analysis:
            considered_next_token_ids_batch = torch.topk(
                next_token_scores, k=2*top_k, dim=-1).indices
            assert len(lsp_context_list) == len(considered_next_token_ids_batch), (len(
                lsp_context_list), len(considered_next_token_ids_batch))
            queue = mp.Queue()
            processes = [mp.Process(target=feasible_token_ids, args=(
                queue,
                generated_tokens,
                lsp_context,
                CODET5_TOKEN_MAP,
                considered_next_token_ids.tolist(),
                top_k,
            )) for considered_next_token_ids, lsp_context in zip(
                considered_next_token_ids_batch,
                lsp_context_list
            )]
            for p in processes:
                p.start()
            for p in processes:
                p.join()

            feasible_token_results: List[List[int]] = []
            while not queue.empty():
                feasible_token_results.append(queue.get())
            feasible_next_token_ids = torch.LongTensor(feasible_token_results).to(DEVICE)
            # rewrite probabilities
            mask = torch.ones(
                next_token_scores.shape,
                dtype=torch.bool
            ).to(DEVICE)
            mask.scatter_(1, feasible_next_token_ids, False)
            next_token_scores.masked_fill_(mask, -float("Inf"))

        # TODO
        # if not exists_satisfied_token and do_analysis:
        #     # Now all top k tokens are zeroed out
        #     # Default to the token with the highest probability
        #     # But if LSP finds a completion, use that.
        #     next_token_ids = all_next_token_ids[0].view(1)
        #     next_token = tokens[next_token_ids.item()]

        #     # Here we know in the above loop all tokens fail to pass the check,
        #     # so they are all identifiers (type, var, etc.)
        #     # So it is safe to call language server (but cannot prove w/o AST)
        #     # TODO: refactor the logic
        #     space_before = False
        #     if next_token.startswith(' '):
        #         text_file.add(' ')
        #         space_before = True
        #     pos = text_file.get_cursor_position()
        #     start_time = time.time()
        #     completion_result = analyzer.client.textDocument_completion({
        #         'textDocument': {
        #             'uri': text_file.path.as_uri()
        #         },
        #         'position': pos,
        #     })
        #     completion_overhead.append(time.time() - start_time)

        #     # TODO: opt
        #     completions = [
        #         item['textEdit']['newText']
        #         if 'textEdit' in item
        #         else item['insertText']
        #         for item in completion_result['result']['items']
        #     ]
        #     if not space_before:
        #         try:
        #             id_token = get_id_token_and_rspace_len(
        #                 generated_tokens)
        #             completions = []
        #             for item in completion_result['result']['items']:
        #                 if 'textEdit' in item:
        #                     result = item['textEdit']
        #                     insert = result['insert']
        #                     replace = result['replace']

        #                     # because we do not invoke it in the middle of a word
        #                     assert replace['end'] == pos
        #                     assert insert == replace
        #                     assert replace['end'] == pos, result
        #                     assert replace['end']['line'] == replace['start']['line']

        #                     start_index = replace['end']['character'] - \
        #                         replace['start']['character']
        #                     if result['newText'][:start_index] == id_token:
        #                         completions.append(
        #                             result['newText'][start_index:])
        #                 else:
        #                     assert 'insertText' in item
        #                     completions.append(item['insertText'])
        #         except IDTokenError:
        #             pass

        #     completions = (
        #         (c if '${' not in c else c[:c.index('${')]) for c in completions)
        #     completions = list(filter(lambda c: len(c) > 0, completions))
        #     if len(completions) > 0:
        #         # breakpoint()
        #         next_token = completions[0]
        #         next_token_ids = self.tokenizer.encode(
        #             next_token, return_tensors='pt', add_special_tokens=False).to(DEVICE)[0]
        #         # if len(next_token_ids) > 1:
        #         #     breakpoint()
        # else:
        #     # Either 1) not using LSP or 2) there is/are matched tokens
        #     next_token_ids = torch.multinomial(
        #         probs, num_samples=1).view(1)
        #     next_token = tokens[next_token_ids.item()]

        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_token_ids = torch.multinomial(
            probs, num_samples=1)
        breakpoint()
        next_token = tokens[next_token_ids.item()]

        if next_token.strip().startswith('//'):
            context['inside_line_comment'] = True
        elif next_token.endswith('\n'):
            # breakpoint()
            context['inside_line_comment'] = False
        elif next_token.strip().startswith('/*'):
            context['inside_block_comment'] = True
        elif next_token.endswith('*/'):
            # breakpoint()
            context['inside_block_comment'] = False

        # print(input_ids.shape)
        input_ids = torch.cat(
            [input_ids, next_token_ids.view(1, -1)], dim=-1)
        # print(repr(next_token), end=' ')
        if not is_special_token(next_token):
            text_file.add(next_token)
            if do_analysis:
                analyzer.change(text_file)
            generated_ids.extend(next_token_ids)
            generated_tokens.append(next_token)
        # print(input_ids.shape, 'new')
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
        # print(model_kwargs['past'][0][0].shape)
        cur_len = cur_len + 1

        # if eos_token was found in one sentence, set sentence to finished
        # if eos_token_id is not None:
        #     unfinished_sequences = unfinished_sequences.mul(
        #         (next_token_ids != eos_token_id).long())

        # stop when each sentence is finished, or if we exceed the maximum length
        # IMPORTANT: codet5 output format: <mask0>....<mask1>....<mask2>...
        # Mask ids are 32099, 32098, 32097...
        if next_token_ids[-1] == 32098 or next_token_ids[-1] == end_id or max_length == len(input_ids[0]) or stopping_criteria(input_ids, scores):
            # assert input_ids[0, -1] == self.EOM_ID
            # if unfinished_sequences.max() == 0 or max_length == len(input_ids) or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    if return_dict_in_generate:
        raise NotImplementedError
        if model.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return completion_overhead, generated_ids, generated_tokens
