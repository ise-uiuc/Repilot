from transformers.tokenization_utils import logger
from transformers.generation_beam_constraints import Constraint
from transformers.generation_logits_process import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper
from transformers.generation_stopping_criteria import StoppingCriteriaList
from typing import Dict, List, Optional, Tuple, NamedTuple, Union, Iterable, Callable, cast, Set
from dataclasses import dataclass
from realm import utils
from realm.lsp.text import TextFile
from realm.lsp import spec
from realm.model import CodeT5ForRealm, CodeT5Large
from realm.analyze.jdt_lsp import JdtLspAnalyzer
from torch import nn
import torch
import os
import inspect
import warnings
import pickle
import typing
import time
import random

JAVA_KEYWORDS = {'abstract', 'continue', 'for', 'new', 'switch',
                 'assert', 'default', 'goto', 'package', 'synchronized',
                 'boolean', 'do', 'if', 'private', 'this',
                 'break', 'double', 'implements', 'protected', 'throw',
                 'byte', 'else', 'import', 'public', 'throws',
                 'case', 'enum', 'instanceof', 'return', 'transient',
                 'catch', 'extends', 'int', 'short', 'try',
                 'char', 'final', 'interface', 'static', 'void',
                 'class', 'finally', 'long', 'strictfp', 'volatile',
                 'const' 'float', 'native', 'super', 'while'}

GenerationState = bytes


@dataclass
class Memorization:
    infeasible_token_ids: Dict[GenerationState, List[int]]
    feasible_token_ids: Dict[GenerationState, Set[int]]
    completions: Dict[GenerationState, Optional[List[dict]]]
    denied_tokens: Dict[GenerationState, utils.Trie]

    @staticmethod
    def init() -> 'Memorization':
        return Memorization({}, {}, {}, {})


# self.memorization.infeasible_token_ids: Dict[bytes, List[int]] = {}
# COMPLETE_MEMOIZED: Dict[bytes, List[int]] = {}
# self.memorization.completions: Dict[bytes, Optional[List[dict]]] = {}
# self.memorization.denied_tokens: Dict[bytes, utils.Trie] = {}
CHART_11 = True


GenerationLog = List[dict]
Generation = Tuple[List[float], List[int], Optional[List[str]], GenerationLog]


@dataclass
class GenerationContext:
    generated_tokens: List[str]
    generated_ids: List[int]


class LMInferenceConfig(NamedTuple):
    temperature: float
    top_k: int
    max_new_tokens: int
    end_id: int


class LMContext(NamedTuple):
    # for LM
    model: CodeT5ForRealm
    prefix: str
    suffix: str
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


MODEL = CodeT5Large.init().to(utils.DEVICE)  # type: ignore # noqa
INFERENCE_CONFIG = LMInferenceConfig(1.0, 50, 50, 2)


def get_completions(analyzer: JdtLspAnalyzer, uri: str, pos: spec.Position) -> Optional[List[dict]]:
    # completion_result = analyzer.client.textDocument_completion({
    #     'textDocument': {
    #         'uri': uri
    #     },
    #     'position': pos,
    # })
    # old_timeout = analyzer.client.timeout
    # analyzer.client.timeout = 0.5
    # try:
    new_completion_result = analyzer.client.newCompletion({
        'textDocument': {
            'uri': uri
        },
        'position': pos,
    })
    # except TimeoutError:
    #     return None
    # finally:
    #     analyzer.client.timeout = old_timeout
    # if new_completion_result['result'] is None:
    #     breakpoint()
    # breakpoint()
    # if 'result' in completion_result:
    # completions: Iterator[str] = (
    #     item['textEdit']['newText']  # type: ignore # noqa
    #     if 'textEdit' in item
    #     else item['insertText']  # type: ignore # noqa
    #     for item in new_completion_result['result']  # type: ignore # noqa
    # )
    return new_completion_result['result']


def char_may_trigger_completion(c: str) -> bool:
    assert len(c) == 1
    return c.isalnum() or (c in ['.', '_', '$'])


class Realm:
    def __init__(
        self,
        lm_context: LMContext,
        lsp_context: LspContext,
        use_memorization: bool = os.getenv('NO_MEM') is None,
    ) -> None:
        # Constants
        self.model = lm_context.model
        self.inference_config = lm_context.inference_config
        self.inputs = self.model.encode(lm_context.prefix, lm_context.suffix)
        self.analyzer = lsp_context.analyzer
        self.text_file = lsp_context.text_file

        # Memorizations
        self.use_mem = use_memorization
        if use_memorization:
            self.mem = Memorization.init()

    def repair(self) -> Generation:
        return self.generate(
            do_sample=True,
            max_new_tokens=self.inference_config.max_new_tokens,
            top_k=self.inference_config.top_k,
            temperature=self.inference_config.temperature,
        )

    # TODO: rewrite the logic
    def feasible(
        self,
        generation_log: GenerationLog,
        generated_ids: List[int],
        generated_tokens: List[str],
        uri: str,
        token_id: int,
        token: str,
        pos: spec.Position,
        completion_overhead: List[float],
    ) -> bool:
        """Returns whether `token` is feasible at `pos` of the file located at `uri`"""
        # if regex.match('^([a-zA-Z_][a-zA-Z\\d_]*)$', token) is None:
        #     warning(
        #         f'Cannot recognize {token} as an identifier, probabily unicode.')
        # assert token not in JAVA_KEYWORDS
        start_time = time.time()
        # if is_special_token(token) or token.strip() in JAVA_KEYWORDS:
        #     return True
        # if len(token) > 0 and not (
        #     token[-1].isalnum()
        #     or token[-1] in ['.', '_', '$']
        # ):
        #     return True
        # if input_state in self.memorization.infeasible_token_ids:
        #     print("HIT")
        #     return self.memorization.infeasible_token_ids[input_state]
        input_state = pickle.dumps(generated_ids + [token_id])
        if self.use_mem and input_state in self.mem.completions:
            # Due to memorization, each input_state be called on this function only once
            # => token_id in self.mem.(in)feasible_token_ids[state of generated_ids]
            assert False
            completions = self.mem.completions[input_state]
        else:
            completions = get_completions(self.analyzer, uri, pos)
            self.mem.completions[input_state] = completions
            completion_overhead.append(time.time() - start_time)
        context = {
            'ids': [id for id in generated_ids],
            'text': ''.join(generated_tokens),
            'new_token': token
        }
        print(context)
        if completions is None:
            generation_log.append({
                'context': context,
                'result': None,
            })
            print('UNKNOWN:', token)
            # breakpoint()
            return True
        filtered_completions = [
            c for c in completions if c['target'].startswith(c['source'])]
        # else:
        #     print(uri)
        #     print(completion_result['error'])
        #     print(completion_result['error']['data'])
        #     raise RuntimeError
        if len(filtered_completions) > 0:
            generation_log.append({
                'context': context,
                'result': filtered_completions,
            })
            # , token, completions)
            print('Accepted:', token,
                  f"{filtered_completions[0]['source']} -> {filtered_completions[0]['target']}")
            # if 'lastFraction' in completions:
            #     breakpoint()
            # print("Yes!")
            # print(completions)
            # self.memorization.infeasible_token_ids[input_state] = True
            return True
        else:
            generation_log.append({
                'context': context,
                'result': [],
            })
            # print('=======================DENIED============================')
            print('Denied', token)  # , token, completions)
            # self.memorization.infeasible_token_ids[input_state] = False
            return False

    def trivially_feasible(self, token: str) -> bool:
        if len(token) > 0 and not char_may_trigger_completion(token[-1]):
            return True
        elif token.strip() in JAVA_KEYWORDS:
            return True
        else:
            return False
    
    def plain_decode(self, gen_context: GenerationContext, probs: torch.Tensor) -> torch.LongTensor:
        next_token_id = cast(torch.LongTensor, torch.multinomial(probs, num_samples=1))
        assert next_token_id.dtype == torch.long
        next_token_id_item = cast(int, next_token_id.item())
        next_token = self.model.token_map[next_token_id_item]
        if not self.model.is_special_token(next_token):
            self.text_file.add(next_token)
            gen_context.generated_ids.append(next_token_id_item)
            gen_context.generated_tokens.append(next_token)
        return next_token_id

    def pruned_decode(
        self,
        gen_context: GenerationContext,
        probs: torch.Tensor,
    ) -> torch.LongTensor:
        """Stateful method that updates the generated token ids and tokens (excluding special
        tokens) and returns the 'real' generation"""
        generated_ids = gen_context.generated_ids
        input_state = pickle.dumps(generated_ids)
        if self.use_mem:
            denied_trie = self.mem.denied_tokens.setdefault(input_state, utils.Trie())
            feasible_indices = self.mem.feasible_token_ids.setdefault(input_state, set())
            infeasible_indices = self.mem.infeasible_token_ids.setdefault(input_state, [])
            # Ensures that all tokens tried are feasible
            probs[infeasible_indices] = 0.
        while True:
            # `probs` will change each iteration (some entries will be assigned 0.)
            try:
                trying_token_id = cast(torch.LongTensor, torch.multinomial(probs, num_samples=1))
                assert trying_token_id.dtype == torch.long
                trying_token_id_item = cast(int, trying_token_id.item())
                assert isinstance(trying_token_id_item, int)
            except RuntimeError as e:
                # Sum of probabilities < 0
                if self.use_mem and len(generated_ids) > 0:
                    prev_state = pickle.dumps(generated_ids[:-1])
                    last_token_id = generated_ids[-1]
                    self.mem.infeasible_token_ids[prev_state].append(last_token_id)
                raise e
            # All trying tokens are feasible
            if self.use_mem:
                assert trying_token_id_item not in infeasible_indices
            trying_token = self.model.token_map[trying_token_id_item]

            def update_gen():
                generated_ids.append(trying_token_id_item)
                gen_context.generated_tokens.append(trying_token)

            if self.model.is_special_token(trying_token):
                return trying_token_id
            elif self.trivially_feasible(trying_token):
                self.text_file.add(trying_token)
                update_gen()
                return trying_token_id
            elif self.use_mem and denied_trie.is_prefix_of(trying_token):
                pass
            elif self.use_mem and trying_token_id_item in feasible_indices:
                self.text_file.add(trying_token)
                update_gen()
                return trying_token_id
            else:
                self.text_file.add(trying_token)
                self.analyzer.change(self.text_file)
                pos = self.text_file.get_cursor_position()
                if self.feasible(
                    [],
                    generated_ids,
                    gen_context.generated_tokens,
                    self.text_file.path.as_uri(),
                    trying_token_id_item,
                    trying_token,
                    pos,
                    []
                ):
                    if self.use_mem:
                        feasible_indices.add(trying_token_id_item)
                    update_gen()
                    return trying_token_id
                else:
                    self.text_file.delete(len(trying_token))
            # Token is infeasible if the program runs to this line
            # By setting the probability to 0.0, this token will not be selected.
            probs[trying_token_id_item] = 0.
            if self.use_mem:
                infeasible_indices.append(trying_token_id_item)
                denied_trie.insert(trying_token)

    @typing.no_type_check
    def sample(
        self,
        input_ids: torch.LongTensor,
        top_k: int,
        temperature: float,
        logits_processor: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        **model_kwargs,
    ) -> Generation:
        model = self.model

        # init values
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
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
            generated_tokens=[],
            generated_ids=[],
        )
        # Whether the model generation is complete
        is_complete = True

        # Logging
        generation_log: GenerationLog = []

        # counter
        count = 0
        # auto-regressive generation
        while True:
            count += 1
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
            if temperature != 1.0:
                next_token_scores = TemperatureLogitsWarper(
                    temperature)(input_ids, next_token_scores)
            next_token_scores = TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1)(input_ids, next_token_scores)
            # next_token_scores = logits_warper(input_ids, next_token_scores)
            assert len(next_token_scores) == 1
            scores = next_token_scores[0]
            probs = nn.functional.softmax(scores, dim=-1)
            assert len(probs.shape) == 1

            if os.getenv('PLAIN') is not None or CHART_11:  # or count > 10:
                # shape: (1)
                next_token_ids = self.plain_decode(gen_context, probs)
            else:
                try:
                    next_token_ids = self.pruned_decode(gen_context, probs)
                except RuntimeError:
                    return completion_overhead, [], None, generation_log

            # update generated ids, model inputs, and length for next step
            # NOTE: originally next_token_ids[:, None] because for each batch it generate 'one' token;
            #  but here we do not consider batch and we can generate multiple tokens
            use_cache = len(next_token_ids) == 1
            # use_cache = False
            input_ids = torch.cat([input_ids, next_token_ids[None, :]], dim=-1)
            model_kwargs['use_cache'] = use_cache
            model_kwargs = model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
            )
            print(model_kwargs['use_cache'])
            # breakpoint()

            # stop when each sentence is finished, or if we exceed the maximum length
            # IMPORTANT: codet5 output format: <mask0>....<mask1>....<mask2>...
            # Mask ids are 32099, 32098, 32097...
            if model.is_end(next_token_ids[-1].item()):
                break
            elif len(input_ids[0]) == max_length:
                is_complete = False
                break
        return completion_overhead, gen_context.generated_ids, gen_context.generated_tokens if is_complete else None, generation_log

    # Rewritten from huggingface/transformers, not changed so much
    @typing.no_type_check
    @torch.no_grad()
    def generate(
        self,
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
        inputs = self.inputs
        model = self.model

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
        prev_use_cache = model_kwargs.get('use_cache')
        inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
            inputs, bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]
        assert prev_use_cache == model_kwargs.get('use_cache')
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
        return self.sample(
            input_ids,
            top_k=top_k,
            temperature=temperature,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            max_length=max_length,
            **model_kwargs,
        )
# Get the exact identifier token position

# def get_id_token(generated_tokens: List[str]) -> str:
#     result = ''
#     first_unmatched_char = None
#     for token in reversed(generated_tokens):
#         should_break = False
#         for c in reversed(token):
#             if c.isdigit() or c.isalpha() or c == '_':
#                 result = c + result
#             else:
#                 should_break = True
#                 break
#         if should_break:
#             break
#     # if first_unmatched_char == '.' and len(result) > 0:
#     #     breakpoint()

#     if len(result) > 0 and not result[0].isdigit() and result not in JAVA_KEYWORDS:
#         return result
#     else:
#         raise IDTokenError


# def get_feasible_token_ids(
#     token_map,
#     generation_log: GenerationLog,
#     memoized_result: Optional[List[bool]],
#     generated_tokens: List[str],
#     generated_ids: List[int],
#     lsp_context: LspContext,
#     # Should be ranked (higher probability first)
#     considered_token_ids: List[int],
#     top_k: int,
#     completion_overhead: List[float]
# ) -> List[int]:
#     print("RUNNING feasibility check")
#     # Trick to mitigate false positives of the langauge server (e.g., Chart-11, PathIterator)
#     # NOTE: do not do it now. Just evaluate on both
#     # if idx == 0 and random.random() < 0.1:
#     #     exists_satsified_token = True
#     #     new_index = idx
#     #     break
#     analyzer = lsp_context.analyzer
#     text_file = lsp_context.text_file
#     # analyzer.client.stop()
#     result: List[int] = []
#     denied = utils.Trie()
#     # all_feasible_tokens = Trie()
#     # all_possible_completions = Trie()
#     done = False
#     count = 0
#     for token_id in considered_token_ids:
#         count += 1
#         if len(result) == top_k:
#             break
#         if done:
#             result.append(token_id)
#             continue
#         # if token_id == 32098 or token_id == CODET5_INFERENCE_CONFIG.end_id:
#         #     try:
#         #         javalang.parse.parse(text_file.content)
#         #     except (javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError):
#         #         # breakpoint()
#         #         continue
#         token = token_map[token_id]
#         # token_rstrip = token.rstrip()
#         # rspace_len = len(token) - len(token_rstrip)
#         # dot_token = token_rstrip.endswith('.')
#         # print(CHART_11)
#         # try:
#         # if not dot_token:
#             # id_token = get_id_token(generated_tokens + [token_rstrip])
#         if CHART_11 and (token in 'PathIterator'):
#             result.append(token_id)
#         elif memoized_result is not None and memoized_result[token_id]:
#             result.append(token_id)
#         # # Opt: reduce analysis times (assuming id_token = some st
#         # # - if s is denied, st is denied (find s in `denied` using trie)
#         # # - if c is all possible completions for s, st should be a prefix of one completion in c
#         # #   (find s in `all_feasible_tokens` using trie; find st in c using trie)
#         # # - if id_token is one of the possible completion, then it is feasible!
#         # # TODO: make string search faster
#         # elif not dot_token and next(denied.prefixes(id_token), None) is not None:
#         # # any(filter(partial(str.startswith, id_token), denied)):
#         #     # breakpoint()
#         #     pass
#         # elif not dot_token and ((x := all_possible_completions.has_node(id_token)) == Trie.HAS_SUBTRIE or x == Trie.HAS_VALUE):
#         #     result.append(token_id)
#         # elif not dot_token and next(all_feasible_tokens.prefixes(id_token), None) is not None:
#         #     # Assertion!
#         #     # assert (x := kv[1].has_node(id_token)) != Trie.HAS_SUBTRIE and x != Trie.HAS_VALUE
#         #     denied[id_token] = None
#             # else:
#             # (x := kv[1].has_node(id_token)) != Trie.HAS_SUBTRIE and x != Trie.HAS_VALUE):
#         # any(filter(
#         #     lambda kv: id_token.startswith(kv[0]) and id_token not in kv[1], # type: ignore # noqa
#         #     all_feasible_tokens.items()
#         # )):
#         elif len(token) > 0 and not (
#             token[-1].isalnum()
#             or token[-1] in ['.', '_', '$']
#         ):
#             result.append(token_id)
#         elif denied.is_prefix_of(token):
#             pass
#         else:
#             # No exception afterwards
#             text_file.add(token)
#             analyzer.change(text_file)
#             pos = text_file.get_position(text_file.cursor)
#             # print(''.join(generated_tokens + [token]))
#             if feasible(
#                 generation_log,
#                 generated_ids,
#                 generated_tokens,
#                 analyzer,
#                 text_file.path.as_uri(),
#                 token_id,
#                 token,
#                 pos,
#                 completion_overhead
#             ):
#                 # or (dot_token
#                 # and any(map(lambda s: True if s not in ['cast', 'var'] else False,
#                 # (x := list(get_completions(analyzer, text_file.path.as_uri(), pos)))))):
#                 result.append(token_id)
#                 # if id_token == 'numerator':
#                 #     breakpoint()
#                 # if not dot_token:
#                 #     plausible_completions = Trie(filtered_completions)
#                 #     all_feasible_tokens[id_token] = plausible_completions
#                 #     all_possible_completions.merge(plausible_completions)
#             else:
#                 denied.insert(token)
#             #     # if dot_token:
#             #     #     breakpoint()
#             #     if not dot_token:
#             #         denied[id_token] = None
#             text_file.delete(len(token))
#             analyzer.change(text_file)
#         # if count == 20:
#         #     done = True
#         # except IDTokenError:
#         #     result.append(token_id)
#     return result

# OLD DECODING
# scores = next_token_scores[0]
# considered_next_token_ids: torch.Tensor = torch.argsort(scores, descending=True)
# assert len(considered_next_token_ids.shape) == 1
# considered_next_token_ids = considered_next_token_ids[
#     scores[considered_next_token_ids] > -float('inf')]

# assert len(input_ids) == 1
# input_ids_list = input_ids[0].tolist()

# considered_next_token_ids_list = considered_next_token_ids.tolist()[:top_k]
# state_bytes = pickle.dumps((input_ids_list, considered_next_token_ids_list))
# if state_bytes in COMPLETE_MEMOIZED:
#     print("HUGE HIT")
#     # breakpoint()
#     feasible_next_token_ids = COMPLETE_MEMOIZED[state_bytes]
# else:
#     lsp_context.analyzer.change(lsp_context.text_file)
#     input_ids_bytes = pickle.dumps(input_ids_list)
#     # TODO: delete this
#     partial_memoized_result = self.memorization.infeasible_token_ids.get(input_ids_bytes)
#     if partial_memoized_result is not None:
#         print("SMALL HIT")
#         # breakpoint()
#     feasible_next_token_ids = get_feasible_token_ids(
#         generation_log,
#         partial_memoized_result,
#         gen_context.generated_tokens,
#         gen_context.generated_ids,
#         lsp_context,
#         considered_next_token_ids_list,
#         top_k,
#         completion_overhead,
#     )
#     assert len(feasible_next_token_ids) <= top_k
#     COMPLETE_MEMOIZED[state_bytes] = feasible_next_token_ids
#     if partial_memoized_result is None:
#         self.memorization.infeasible_token_ids[input_ids_bytes] = [False] * CODET5_VOC_SIZE
#     for idx in feasible_next_token_ids:
#         self.memorization.infeasible_token_ids[input_ids_bytes][idx] = True

# if len(feasible_next_token_ids) != 0:
#     # rewrite scores
#     mask = torch.ones(
#         scores.shape,
#         dtype=torch.bool
#     ).to(utils.DEVICE)
#     mask.index_fill_(0, torch.LongTensor(feasible_next_token_ids).to(utils.DEVICE), False)
#     scores.masked_fill_(mask, -float('inf'))
# else:
#     is_complete = False
#     break

# ACTIVE
# if os.getenv('ACTIVE') is not None:
#     # Active completion
#     self.analyzer.change(self.text_file)
#     # assert text_file.content[text_file.cursor-idx] == '.'
#     # print(gen_tokens)
#     # completion_result = analyzer.client.textDocument_completion({
#     #     'textDocument': {
#     #         'uri': text_file.path.as_uri()
#     #     },
#     #     'position': text_file.get_position(text_file.cursor)
#     # })
#     # breakpoint()
#     input_ids_list = input_ids[0].tolist()
#     input_state = pickle.dumps(gen_context.generated_ids)
#     if input_state in self.mem.completions:
#         continuations = self.mem.completions[input_state]
#     else:
#         continuations = get_completions(self.analyzer, self.text_file.path.as_uri(
#         ), self.text_file.get_position(self.text_file.cursor))
#         self.mem.completions[input_state] = continuations
#     if continuations is not None:
#         continuations = [item if not (item := target[len(source):]).endswith(
#             '(') else item[:-1] for c in continuations if (target := c['target']).startswith(source := c['source'])]
#         # continuations = [item for item in continuations if len(item) > 0]
#         completion = utils.common_prefix(continuations)

#         # if completion is not None:
#         #     breakpoint()
#         # if len(continuations) == 1:
#         #     completion = continuations[0]
#         # elif len(continuations) < top_k:
#         #     warpers = LogitsProcessorList()
#         #     warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
#         #     next_token_scores = warpers(input_ids, next_token_scores)
#         #     assert len(next_token_scores) == 1
#         #     scores = next_token_scores[0]
#         #     considered_next_token_ids: torch.Tensor = torch.argsort(scores, descending=True)
#         #     assert len(considered_next_token_ids.shape) == 1
#         #     considered_next_token_ids = considered_next_token_ids[
#         #         scores[considered_next_token_ids] > -float('inf')]
#         #     considered_next_token_ids_list = considered_next_token_ids.tolist()[:top_k]
#         #     considered_next_tokens = [model.token_map[id] for id in considered_next_token_ids_list]
#         #     for possible_next_token in considered_next_tokens:
#         #         for continuation in continuations:
#         #             if continuation.startswith(possible_next_token):
#         #                 completion = continuation
#         #                 breakpoint()
#         #                 break
#         #         if completion is not None:
#         #             break
#     else:
#         completion = None
#     # Directly use completion to continue the code
#     if os.getenv('COMPLETION') is not None:
#         N_COMPLETION = 5
#         if completion is None and continuations is not None and len(continuations) > 0 and len(continuations) <= N_COMPLETION:
#             if os.getenv('NO_RAND') is None:
#                 completion = choice if (choice := random.choice(
#                     continuations)) != '' else None
#                 print(completion)
#             else:
#                 # if completion is not None:
#                 scores = next_token_scores[0]
#                 probs = nn.functional.softmax(scores, dim=-1)
#                 considered_next_token_ids: torch.Tensor = torch.argsort(
#                     scores, descending=True)
#                 assert len(
#                     considered_next_token_ids.shape) == 1
#                 considered_next_token_ids = considered_next_token_ids[
#                     scores[considered_next_token_ids] > -float('inf')]

#                 assert len(input_ids) == 1
#                 considered_next_token_ids_list = considered_next_token_ids.tolist()[
#                     :top_k]

#                 cont_probs = [
#                     0 for _ in range(len(continuations))]
#                 for idx, cont in enumerate(continuations):
#                     for id in considered_next_token_ids_list:
#                         token = model.token_map[id]
#                         prob = probs[id]
#                         if cont.startswith(token) or token.startswith(cont):
#                             cont_probs[idx] += prob.item()

#                 try:
#                     completion = choice if (choice := random.choices(
#                         continuations, weights=cont_probs, k=1)[0]) != '' else None
#                 except ValueError:
#                     completion = random.choice(continuations)
#                     if completion == '':
#                         completion = None
# if completion is not None:
#     pass
# else:
#     try:
#         next_token_ids = self.pruned_decode(gen_context, probs)
#     except RuntimeError:
#         return completion_overhead, [], None, generation_log
# else:

# if completion is None:
#     pass
# else:
#     next_token = completion
#     next_token_ids_list = model.tokenizer.encode(
#         completion, add_special_tokens=False)
#     next_token_ids = torch.LongTensor(
#         next_token_ids_list).to(utils.DEVICE)
#     next_token_id_item = -1