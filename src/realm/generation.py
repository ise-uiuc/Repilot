"""Copied and rewritted from transformers.generation_utils"""
import random
from datasets import d4j
import inspect
import typing
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

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


# class IncoderEOM(StoppingCriteria):
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         return True if input_ids[0, -1] == Repairer.EOM_ID else False


MODEL = 'Salesforce/codet5-large'
DEVICE = 'cuda'
NUM_SAMPLES = 6


Generation = Tuple[List[int], List[str]]


class Repairer:
    model: GenerationMixin = T5ForConditionalGeneration.from_pretrained(
        MODEL).to(DEVICE)
    max_length: int = model.config.to_dict()['n_positions']  # type: ignore # noqa
    infill_ph = "<extra_id_0>"
    END_ID = 2
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        MODEL)

    def __init__(self, prefix: str, suffix: str) -> None:
        self.input_strings = self.inputs(prefix, suffix)
        # Must add a special BOS token (add_special_tokens=True). Maybe because this is
        # a decoder only model, BOS indicates the start of the generation.
        self.input_tokens = self.tokenizer.encode(
            self.input_strings, return_tensors='pt').to(DEVICE)

    def inputs(self, prefix: str, suffix: str) -> str:
        return prefix + self.infill_ph + suffix

    def repair(
        self,
        analyzer: JdtLspAnalyzer,
        text_file: TextFile,
        max_new_tokens: int = 30,
    ) -> Generation:
        # original_content = text_file.content
        # original_cursor = text_file.cursor
        # for idx in range(1):
        # text_file.content = original_content
        # text_file.cursor = original_cursor
        # text_file.sync()
        # text_file.write()
        return self.generate(
            analyzer,
            text_file,
            self.input_tokens,
            do_sample=True,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.95,
            eos_token_id=self.END_ID,
            # stopping_criteria=StoppingCriteriaList([IncoderEOM()])
        )
        # text_file.write()
        # outputs = outputs[:, len(inputs[0]):]
        # outputs = (output[:-1] if len(output) > 0 and output[-1]
        #            == self.EOM_ID else output for output in outputs)
        # output = self.tokenizer.batch_decode(
        #     outputs, clean_up_tokenization_spaces=False)[0]
        # if self.EOM in output:
        #     output = output[:output.index(self.EOM)]
        # if self.META_FILE in output:  # removes META file token that is sometimes generated
        #     assert False
        #     output = output[:output.index(self.META_FILE)]
        #     all_output.append(output)
        # for output in all_output:
        #     print(output)
        #     print("\n===========================================================\n")
        # print(len(all_output))
        # exit()

    @typing.no_type_check
    @torch.no_grad()
    def generate(
        self,
        analyzer: JdtLspAnalyzer,
        text_file: TextFile,
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

        Examples:

        Greedy Decoding:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> prompt = "Today I believe we can finally"
        >>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        >>> # generate up to 30 tokens
        >>> outputs = model.generate(input_ids, do_sample=False, max_length=30)
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Today I believe we can finally get to the point where we can make a difference in the lives of the people of the United States of America.\n']
        ```

        Multinomial Sampling:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> prompt = "Today I believe we can finally"
        >>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        >>> # sample up to 30 tokens
        >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
        >>> outputs = model.generate(input_ids, do_sample=True, max_length=30)
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Today I believe we can finally get rid of discrimination," said Rep. Mark Pocan (D-Wis.).\n\n"Just look at the']
        ```

        Beam-search decoding:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        >>> tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")

        >>> sentence = "Paris is one of the densest populated areas in Europe."
        >>> input_ids = tokenizer(sentence, return_tensors="pt").input_ids

        >>> outputs = model.generate(input_ids)
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Paris ist eines der dichtesten besiedelten Gebiete Europas.']
        ```"""
        # 1. Set generation parameters if not already defined
        bos_token_id = bos_token_id if bos_token_id is not None else self.model.config.bos_token_id
        num_beams = num_beams if num_beams is not None else self.model.config.num_beams
        length_penalty = length_penalty if length_penalty is not None else self.model.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.model.config.early_stopping
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.model.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.model.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.model.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.model.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.model.config.eos_token_id

        if eos_token_id is None and hasattr(self.model.config, "decoder"):
            eos_token_id = self.model.config.decoder.eos_token_id

        if pad_token_id is None and eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            logger.warning(
                f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        output_scores = output_scores if output_scores is not None else self.model.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.model.config.return_dict_in_generate
        )

        # 2. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self.model._prepare_model_inputs(
            inputs, bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]

        # 3. Define other model kwargs
        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states
        model_kwargs["use_cache"] = use_cache

        accepts_attention_mask = "attention_mask" in set(
            inspect.signature(self.model.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self.model._prepare_attention_mask_for_generation(
                inputs_tensor, pad_token_id, eos_token_id
            )

        if self.model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self.model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 4. Prepare `input_ids` which will be used for auto-regressive generation
        if self.model.config.is_encoder_decoder:
            input_ids = self.model._prepare_decoder_input_ids_for_generation(
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
                f"{self.model.config.max_length} (`self.model.config.max_length`). Controlling `max_length` via the config is "
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
        max_length = max_length if max_length is not None else self.model.config.max_length
        min_length = min_length if min_length is not None else self.model.config.min_length

        if min_length is not None and min_length > max_length:
            raise ValueError(
                f"Unfeasable length constraints: the minimum length ({min_length}) is larger than the maximum "
                f"length ({max_length})"
            )
        if input_ids_seq_length >= max_length:
            input_ids_string = "decoder_input_ids" if self.model.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {max_length}. This can lead to unexpected behavior. You should consider increasing "
                "`max_new_tokens`."
            )

        # 6. determine generation mode
        is_constraint_gen_mode = constraints is not None or force_words_ids is not None
        is_greedy_gen_mode = (
            (num_beams == 1) and (num_beam_groups ==
                                  1) and do_sample is False and not is_constraint_gen_mode
        )
        is_sample_gen_mode = (
            (num_beams == 1) and (num_beam_groups ==
                                  1) and do_sample is True and not is_constraint_gen_mode
        )
        is_beam_gen_mode = (
            (num_beams > 1) and (num_beam_groups ==
                                 1) and do_sample is False and not is_constraint_gen_mode
        )
        is_beam_sample_gen_mode = (
            (num_beams > 1) and (num_beam_groups ==
                                 1) and do_sample is True and not is_constraint_gen_mode
        )
        is_group_beam_gen_mode = (num_beams > 1) and (
            num_beam_groups > 1) and not is_constraint_gen_mode

        if num_beam_groups > num_beams:
            raise ValueError(
                "`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )
        if not is_sample_gen_mode:
            assert False

        # 7. prepare distribution pre_processing samplers
        logits_processor = self.model._get_logits_processor(
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

        # Do not prepare criteria
        # 8. prepare stopping criteria
        # stopping_criteria = self.model._get_stopping_criteria(
        #     max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
        # )

        # 9. go into different generation modes
        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # 10. run greedy search
            return self.model.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # 10. prepare logits warper
            logits_warper = self.model._get_logits_warper(
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                temperature=temperature,
                num_beams=num_beams,
                renormalize_logits=renormalize_logits,
            )

            # 11. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self.model._expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                is_encoder_decoder=self.model.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 12. run sample
            return self.sample(
                analyzer,
                text_file,
                input_ids,
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

        elif is_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now.")

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=inputs_tensor.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self.model._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.model.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            return self.model.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            # 10. prepare logits warper
            logits_warper = self.model._get_logits_warper(
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                temperature=temperature,
                num_beams=num_beams,
                renormalize_logits=renormalize_logits,
            )

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now.")
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size * num_return_sequences,
                num_beams=num_beams,
                device=inputs_tensor.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
            )

            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self.model._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams * num_return_sequences,
                is_encoder_decoder=self.model.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run beam sample
            return self.model.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if num_beams % num_beam_groups != 0:
                raise ValueError(
                    "`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now.")

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                max_length=stopping_criteria.max_length,
                device=inputs_tensor.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
                num_beam_groups=num_beam_groups,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self.model._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.model.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            return self.model.group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_constraint_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now.")

            if num_beams <= 1:
                raise ValueError(
                    "`num_beams` needs to be greater than 1 for constrained generation.")

            if do_sample:
                raise ValueError(
                    "`do_sample` needs to be false for constrained generation.")

            if num_beam_groups is not None and num_beam_groups > 1:
                raise ValueError(
                    "`num_beam_groups` not supported yet for constrained generation.")

            final_constraints = []
            if constraints is not None:
                final_constraints = constraints

            if force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]`"
                        f"of positive integers, but is {force_words_ids}."
                    )

                if not isinstance(force_words_ids, list) or len(force_words_ids) == 0:
                    typeerror()

                for word_ids in force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0)
                                for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 10. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=num_beams,
                device=inputs_tensor.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self.model._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.model.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            return self.model.constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

    @typing.no_type_check
    def sample(
        self,
        analyzer: JdtLspAnalyzer,
        text_file: TextFile,
        input_ids: torch.LongTensor,
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

        Return:
            [`~generation_utils.SampleDecoderOnlyOutput`], [`~generation_utils.SampleEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation_utils.SampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation_utils.SampleEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     TopKLogitsWarper,
        ...     TemperatureLogitsWarper,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        >>> model.config.pad_token_id = model.config.eos_token_id

        >>> input_prompt = "Today is a beautiful day, and"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )
        >>> # instantiate logits processors
        >>> logits_warper = LogitsProcessorList(
        ...     [
        ...         TopKLogitsWarper(50),
        ...         TemperatureLogitsWarper(0.7),
        ...     ]
        ... )

        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
        >>> outputs = model.sample(
        ...     input_ids,
        ...     logits_processor=logits_processor,
        ...     logits_warper=logits_warper,
        ...     stopping_criteria=stopping_criteria,
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Today is a beautiful day, and a wonderful day.\n\nI was lucky enough to meet the']
        ```"""

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
        pad_token_id = pad_token_id if pad_token_id is not None else self.model.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.model.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.model.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.model.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (
            return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get(
                "attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get(
                    "hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        this_peer_finished = False  # used by synced_gpus only
        generated_ids: List[int] = []
        generated_tokens: List[str] = []
        # auto-regressive generation
        while True:
            if synced_gpus:
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
            model_inputs = self.model.prepare_inputs_for_generation(
                input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self.model(
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
                        (outputs.decoder_attentions,) if self.model.config.is_encoder_decoder else (
                            outputs.attentions,)
                    )
                    if self.model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            all_next_tokens = torch.multinomial(
                probs, num_samples=NUM_SAMPLES).squeeze(1).view(NUM_SAMPLES, 1)
            new_index = 0
            # all_next_tokens = torch.topk(next_token_scores, k=6, dim=-1).indices.view(6, 1)
            # exit()

            # finished sentences should have their next token be a padding token
            # print(next_tokens)
            # if eos_token_id is not None:
            #     if pad_token_id is None:
            #         raise ValueError(
            #             "If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            #     # print(pad_token_id)
            #     # print(unfinished_sequences)
            #     for idx, _ in enumerate(all_next_tokens):
            #         all_next_tokens[idx] = all_next_tokens[idx] * unfinished_sequences + \
            #             pad_token_id * (1 - unfinished_sequences)
            #     # print(next_tokens)
            #     # exit()

            # update generated ids, model inputs, and length for next step
            for next_tokens in all_next_tokens:
                assert next_tokens.shape == torch.Size([1]), next_tokens.shape
            tokens = self.tokenizer.batch_decode(all_next_tokens, skip_special_tokens=False)
            assert len(tokens) == NUM_SAMPLES
            # [self.tokenizer.decode(
                # next_tokens.view(1)) for next_tokens in all_next_tokens]
            # exit()
            # if token.strip() in ['//', '/*', '*/', '/**']:
            #     continue
            # Force the model to complete the generation
            if True:
            # max_length - len(input_ids[0]) < 10 and self.EOM_ID in all_next_tokens:
            #     next_tokens = torch.tensor([self.EOM_ID]).to(DEVICE)
            # else:
                pos = text_file.get_position(text_file.cursor)
                if True:
                    # if not pos['character'] == 0 and not text_file.content[text_file.cursor - 1].strip() == '':
                    completion_result = analyzer.client.textDocument_completion({
                        'textDocument': {
                            'uri': text_file.path.as_uri()
                        },
                        'position': pos,
                        # {
                        #     'line': 302,
                        #     'character': 26,
                        # },
                    })
                    # completion_result1 = analyzer.client.textDocument_completion({
                    #     'textDocument': {
                    #         'uri': text_file.path.as_uri()
                    #     },
                    #     'position': {
                    #         'line': 302,
                    #         'character': 45,
                    #     },
                    # })

                    def completion_iter():
                        for item in completion_result['result']['items']:
                            if not 'textEdit' in item:
                                pass
                            else:
                                result = item['textEdit']
                                insert = result['insert']
                                replace = result['replace']
                                if replace['end'] == pos:
                                    # print("EQ")
                                    assert insert == replace
                                    assert replace['end'] == pos, result
                                    assert replace['end']['line'] == replace['start']['line']
                                    start_index = replace['end']['character'] - \
                                        replace['start']['character']
                                    yield result['newText'][start_index:]
                                else:
                                    assert False
                    completions = list(completion_iter())
                    # print(completions)
                    # print(completion_result)
                    # print(completion_result1)
                    # print(tokens)
                    # print("============================================")
                    # print(pos)
                    # breakpoint()
                    # print(completions)
                    # print(tokens)
                    # print("============================================")
                    completions = [
                        (c if '${' not in c else c[:c.index('${')]) for c in completions]
                    completions = list(filter(lambda c: c != '', completions))
                    # print(completions)
                    if random.random() > 0.2:
                        # if not pos['character'] == 0 and not text_file.content[text_file.cursor - 1].strip() == '' and len(completions) > 0:
                        for idx, token in random.sample(list(enumerate(tokens)), k=len(tokens)):
                            # t = token.rstrip()
                            # space_index = t.find(' ')
                            # t = t[:space_index] if space_index != -1 else t
                            try:
                                # if t == '':
                                #     break
                                # print('====================')
                                # print(completions)
                                # print(t)
                                # print('====================')
                                x = next(filter(lambda c: token.startswith(
                                    c) or c.startswith(token), completions))
                                new_index = idx
                                assert x != ''
                                assert token != ''
                                print("Log (match):", token,
                                      idx, x, sep=' === ')
                                print("All:", tokens)
                                break
                            except StopIteration:
                                break
                            # next_tokens = self.tokenizer.encode(token, add_special_tokens=False, return_tensors='pt')[0][:1]
                            # next_tokens = next_tokens.to(DEVICE)
                        # assert next_tokens.shape == torch.Size([1]), token
                    # print()
                    # print(token)
                    # print(text_file.content[text_file.cursor - 100:text_file.cursor])
                    # exit()
                next_tokens = all_next_tokens[new_index].view(1)
                if next_tokens.item() != self.END_ID:
                    text_file.add(tokens[new_index])
                    analyzer.change(text_file)
                    text_file.write()
                # text_file.write()
                # if random.random() > 0.9:
                #     text_file.write()
                #     exit()
                # analyzer.change(text_file)

            # print('*****************************************')
            # print(next_tokens)
            # print(self.tokenizer.decode(next_tokens, clean_up_tokenization_spaces=False))
            # print(tokens[new_index])
            # print('*****************************************')

            # print(input_ids.shape)
            input_ids = torch.cat([input_ids, next_tokens.view(1, -1)], dim=-1)
            generated_ids.append(next_tokens.item())
            generated_tokens.append(tokens[new_index])
            # print(input_ids.shape, 'new')
            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
            )
            # print(model_kwargs['past'][0][0].shape)
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    (next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if max_length == len(input_ids[0]) or stopping_criteria(input_ids, scores):
                # assert input_ids[0, -1] == self.EOM_ID
                # if unfinished_sequences.max() == 0 or max_length == len(input_ids) or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if return_dict_in_generate:
            raise NotImplementedError
            if self.model.config.is_encoder_decoder:
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
            return generated_ids, generated_tokens