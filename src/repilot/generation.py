import inspect
import logging
import os
import pickle
import sys
import time
import typing
import warnings

LOG_FILE = os.getenv("LOG", "repilot.log")
logging.basicConfig(filename=LOG_FILE, encoding="utf-8", level=logging.INFO)
logging.info(str(sys.argv))
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Callable, Dict, Iterable, NamedTuple, Optional, Union, cast

import torch
from torch import nn
from transformers.generation_beam_constraints import Constraint
from transformers.generation_logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
)
from transformers.generation_stopping_criteria import StoppingCriteriaList
from transformers.tokenization_utils import logger

from . import utils
from .analyzer import JdtLspAnalyzer, Message
from .config import LMInferenceConfig, SynthesisMethod
from .generation_defs import GenerationContext, Memorization
from .lsp import TextFile, spec
from .model import ModelType
from .results import SynthesisResult, SynthesisResultBatch

JAVA_KEYWORDS = {
    "abstract",
    "continue",
    "for",
    "new",
    "switch",
    "assert",
    "default",
    "goto",
    "package",
    "synchronized",
    "boolean",
    "do",
    "if",
    "private",
    "this",
    "break",
    "double",
    "implements",
    "protected",
    "throw",
    "byte",
    "else",
    "import",
    "public",
    "throws",
    "case",
    "enum",
    "instanceof",
    "return",
    "transient",
    "catch",
    "extends",
    "int",
    "short",
    "try",
    "char",
    "final",
    "interface",
    "static",
    "void",
    "class",
    "finally",
    "long",
    "strictfp",
    "volatile",
    "const",
    "float",
    "native",
    "super",
    "while",
    "null",
}

ACTIVE = os.getenv("ACTIVE") is not None


@dataclass
class GenerationState:
    gen_contexts: list[GenerationContext]
    batch_is_failed: list[bool]
    batch_is_unfinished: list[bool]

    @staticmethod
    def init(batch_size: int) -> "GenerationState":
        return GenerationState(
            [GenerationContext([], []) for _ in range(batch_size)],
            [False] * batch_size,
            [False] * batch_size,
        )


class LMContext(NamedTuple):
    # for LM
    model: ModelType
    prefix: str
    suffix: str
    inference_config: LMInferenceConfig


class LspContext(NamedTuple):
    # for analysis
    text_file: TextFile
    analyzer: Connection


# INFERENCE_CONFIG = LMInferenceConfig(1, 1.0, 50, 50)


def get_completions(
    analyzer: Connection, uri: str, pos: spec.Position
) -> Optional[list[dict]]:
    # completion_result = analyzer.client.textDocument_completion({
    #     'textDocument': {
    #         'uri': uri
    #     },
    #     'position': pos,
    # })
    # old_timeout = analyzer.client.timeout
    # analyzer.client.timeout = 0.5
    # try:
    analyzer.send(
        Message(
            True,
            JdtLspAnalyzer.completion.__name__,
            {
                "textDocument": {"uri": uri},
                "position": pos,
            },
        )
    )
    new_completion_result: dict = analyzer.recv()

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
    return new_completion_result["result"]


def char_may_trigger_completion(c: str) -> bool:
    assert len(c) == 1
    return c.isalnum() or (c in [".", "_", "$"])


class Synthesizer:
    def __init__(
        self,
        lm_context: LMContext,
        connections: list[Connection],
        text_file: TextFile,
        gen_method: SynthesisMethod,
        buggy_hunk: str,
    ) -> None:
        # Constants
        self.init_lm(lm_context)
        self.text_file = text_file
        # self.hunk_start_cursor = self.text_file.cursor
        self.gen_method = gen_method
        self.connections = connections
        self.buggy_hunk = buggy_hunk
        assert len(connections) == self.batch_size

        # States that are initialized everytime calling `synthesize``
        self.lsp_contexts: Optional[list[LspContext]] = None
        self.mem: Optional[Memorization] = None
        self.gen_state: Optional[GenerationState] = None
        self.mem_active: dict[bytes, tuple[torch.Tensor, str]] | None = None

        if ACTIVE:
            assert self.use_mem

    def init_lm(self, lm_context: LMContext):
        self.model = lm_context.model
        self.end_ids_tensor = torch.tensor(self.model.end_ids).to(utils.DEVICE)
        self.inference_config = lm_context.inference_config
        self.inputs = self.model.encode(lm_context.prefix, lm_context.suffix).repeat(
            self.batch_size, 1
        )
        # self.all_ids = torch.tensor(range(self.model.vocab_size)).to(utils.DEVICE).repeat(self.batch_size, 1)

    def init_state(self):
        self.gen_state = GenerationState.init(self.batch_size)
        self.lsp_contexts = [
            LspContext(self.text_file.copy(), conn) for conn in self.connections
        ]
        if ACTIVE and self.mem_active is None:
            self.mem_active = {}
        if self.use_mem and self.mem is None:
            self.mem = Memorization.init()
            # state = pickle.dumps(
            #     self.model.tokenizer.tokenize(self.buggy_hunk, add_special_tokens=False)
            # )
            # state_with_newline = pickle.dumps(
            #     self.model.tokenizer.tokenize(
            #         self.buggy_hunk + "\n", add_special_tokens=False
            #     )
            # )
            # self.mem.infeasible_token_ids[state] = [id for id in self.model.end_ids]
            # self.mem.infeasible_token_ids[state_with_newline] = [
            #     id for id in self.model.end_ids
            # ]

    def tokenize(self, active_completion: str) -> list[int]:
        assert ACTIVE
        # mem = self.tok_mem.get(active_completion)
        # if mem is not None:
        #     return mem
        result = self.model.tokenizer.encode(
            active_completion, add_special_tokens=False
        )
        # mem[active_completion] = result
        return result

    @property
    def use_mem(self) -> bool:
        return self.gen_method.use_mem()

    @property
    def is_plain(self) -> bool:
        return self.gen_method.is_plain()

    @property
    def batch_size(self) -> int:
        return self.inference_config.batch_size

    def synthesize(self, t_prefix: str, t_suffix: str) -> SynthesisResultBatch:
        self.init_state()
        assert self.gen_state is not None
        assert self.lsp_contexts is not None
        assert len(self.lsp_contexts) == self.batch_size

        start_time = time.perf_counter()
        self.generate(
            do_sample=True,
            max_new_tokens=self.inference_config.max_new_tokens,
            top_k=self.inference_config.top_k,
            temperature=self.inference_config.temperature,
        )
        assert len(self.gen_state.gen_contexts) == self.batch_size
        assert len(self.gen_state.batch_is_failed) == self.batch_size
        results: list[SynthesisResult] = []
        for batch_idx in range(self.batch_size):
            if self.gen_state.batch_is_failed[batch_idx]:
                results.append(SynthesisResult(None, True, False))
            elif self.gen_state.batch_is_unfinished[batch_idx]:
                # print("Unfinished")
                # print("".join(self.gen_state.gen_contexts[batch_idx].generated_tokens))
                results.append(SynthesisResult(None, False, True))
            else:
                gen_context = self.gen_state.gen_contexts[batch_idx]
                # lsp_context = self.lsp_contexts[batch_idx]
                # patch_file = lsp_context.text_file
                # start_index = self.hunk_start_cursor
                # end_index = patch_file.cursor
                hunk = t_prefix + "".join(gen_context.generated_tokens) + t_suffix
                # assert patch_file.content[start_index:end_index] == hunk
                # success = SynthesisSuccessful(
                #     patch_file,
                #     start_index,
                #     end_index,
                #     hunk,
                # )
                results.append(SynthesisResult(hunk, False, False))
        cost = time.perf_counter() - start_time
        return SynthesisResultBatch(results, cost)

    def trivially_feasible(self, token: str) -> bool:
        if len(token) > 0 and not char_may_trigger_completion(token[-1]):
            return True
        # TODO: Can further optimize (preprocessing token_map)
        elif token.strip() in JAVA_KEYWORDS:
            return True
        else:
            return False

    def plain_decode(self, probs: torch.Tensor) -> torch.LongTensor:
        assert self.gen_state is not None
        assert self.lsp_contexts is not None
        gen_contexts = self.gen_state.gen_contexts
        next_token_ids = cast(
            torch.LongTensor, torch.multinomial(probs, num_samples=1).squeeze(1)
        )
        assert next_token_ids.dtype == torch.long
        assert next_token_ids.shape == torch.Size((self.batch_size,))
        for batch_idx, next_token_id in enumerate(next_token_ids):
            next_token_id_item = cast(int, next_token_id.item())
            assert isinstance(next_token_id_item, int)
            next_token = self.model.token_map[next_token_id_item]
            # breakpoint()
            if not self.model.is_special_token(next_token):
                self.lsp_contexts[batch_idx].text_file.add(next_token)
                gen_contexts[batch_idx].generated_ids.append(next_token_id_item)
                gen_contexts[batch_idx].generated_tokens.append(next_token)
        return next_token_ids

    # def active_decode(self, probs: torch.Tensor) -> torch.LongTensor:
    #     assert self.batch_size == 1
    #     assert self.gen_state is not None
    #     assert self.lsp_contexts is not None
    #     assert len(self.gen_state.gen_contexts) == 1
    #     assert len(self.lsp_contexts) == 1
    #     gen_context = self.gen_state.gen_contexts[0]
    #     generated_tokens = gen_context.generated_tokens
    #     lsp_context = self.lsp_contexts[0]
    #     # next_token_ids = self.pruned_decode(probs)
    #     if len(generated_tokens) == 0 or not char_may_trigger_completion(
    #         generated_tokens[-1][-1]
    #     ):
    #         return self.pruned_decode(probs)
    #     else:
    #         # TODO: memorize active completion and the encode(completion) result and try not to exclude the '('
    #         lsp_context.analyzer.send(
    #             (
    #                 Message(
    #                     True,
    #                     JdtLspAnalyzer.continuation.__name__,
    #                     gen_context.generated_ids,
    #                     lsp_context.text_file,
    #                 )
    #             )
    #         )
    #         continuation: str | None = lsp_context.analyzer.recv()
    #         if continuation is None or continuation == "":
    #             return self.pruned_decode(probs)
    #         else:
    #             next_token_ids_list = self.model.tokenizer.encode(
    #                 continuation, add_special_tokens=False
    #             )
    #             lsp_context.text_file.add(continuation)
    #             gen_context.generated_ids.append(next_token_ids_list)
    #             gen_context.generated_tokens.extend(
    #                 self.model.token_map[id] for id in next_token_ids_list
    #             )
    #             # logging.info(f"ACTIVE: {continuation}")
    #             return cast(
    #                 torch.LongTensor,
    #                 torch.tensor(
    #                     next_token_ids_list, dtype=torch.long, device=utils.DEVICE
    #                 ),
    #             )

    def pruned_decode(
        self, probs: torch.Tensor, active_completion: str | None
    ) -> tuple[torch.LongTensor, str | None]:
        """Stateful method that updates the generated token ids and tokens (excluding special
        tokens) and returns the 'real' generation"""
        assert self.gen_state is not None
        assert self.lsp_contexts is not None
        batch_is_failed = self.gen_state.batch_is_failed
        gen_contexts = self.gen_state.gen_contexts
        special_value = self.model.vocab_size
        next_token_ids = torch.full(
            (self.batch_size,), special_value, dtype=torch.long, device=utils.DEVICE
        )
        next_token_ids[batch_is_failed] = self.model.end_id
        # Keeps track of batches whose next token is not determined
        batch_needs_to_process = [not failed for failed in batch_is_failed]
        assert len(batch_needs_to_process) == self.batch_size

        # The next token of batch `batch_idx` is set to `next_token_id`
        def update_batch_state(batch_idx: int, next_token_id: int):
            assert self.lsp_contexts is not None
            batch_needs_to_process[batch_idx] = False
            next_token_ids[batch_idx] = next_token_id
            next_token = self.model.token_map[next_token_id]
            if not self.model.is_special_token(next_token):
                self.lsp_contexts[batch_idx].text_file.add(next_token)
                gen_contexts[batch_idx].generated_ids.append(next_token_id)
                gen_contexts[batch_idx].generated_tokens.append(next_token)

        # all_infeasible_indices: list[tuple[int, int]] = []
        start = time.perf_counter()
        probs_assign = 0.0
        if self.use_mem:
            assert self.mem is not None
            mem_infeasible_token_ids: list[list[int]] = []
            mem_feasible_token_ids: list[dict[int, str | None]] = []
            # mem_completions: list[Optional[list[dict]]] = []
            mem_denied_tokens: list[utils.Trie] = []
            for batch_idx in range(self.batch_size):
                gen_context = gen_contexts[batch_idx]
                input_state = pickle.dumps(gen_context.generated_ids)

                if ACTIVE:
                    assert batch_idx == 0
                    assert self.mem_active is not None

                denied_trie = self.mem.denied_tokens.setdefault(
                    input_state, utils.Trie()
                )
                feasible_indices = self.mem.feasible_token_ids.setdefault(
                    input_state, {}
                )
                infeasible_indices = self.mem.infeasible_token_ids.setdefault(
                    input_state, []
                )

                mem_denied_tokens.append(denied_trie)
                mem_feasible_token_ids.append(feasible_indices)
                mem_infeasible_token_ids.append(infeasible_indices)

                # Ensures that all tokens tried are feasible
                _start = time.perf_counter()
                probs[batch_idx, infeasible_indices] = 0.0
                probs_assign += time.perf_counter() - _start

        # Active completion constraint
        if ACTIVE:
            assert self.use_mem
            assert self.mem_active is not None
            if (mem_active_result := self.mem_active.get(input_state)) is not None:
                probs, active_completion = mem_active_result
                probs_to_process = probs
                logging.info(f"ACTIVE (HIT): {active_completion}")
            elif active_completion is not None:
                probs_to_process = probs
                # if the length is 0, it must be a new map because the same state will never be reached
                assert len(active_completion) > 0
                assert len(probs_to_process) == 1
                non_zeros = probs_to_process[0].nonzero()
                n_success = 0
                for tok_idx in non_zeros:
                    tok_idx_item = cast(int, tok_idx.item())
                    tok = self.model.token_map[tok_idx_item]
                    if tok.startswith(
                        active_completion
                    ) or active_completion.startswith(tok):
                        n_success += 1
                    else:
                        logging.info(
                            f"Active Pruning: {tok}, prob: {probs[0, tok_idx_item].item()}"
                        )
                        probs[0, tok_idx_item] = 0.0
                        infeasible_indices.append(tok_idx_item)
                        # probs_to_process[0, tok_idx_item] = 0.0
                if n_success == 0:
                    assert probs.sum().item() == 0
                else:
                    self.mem_active[input_state] = (probs, active_completion)

        # Invariant: not batch_needs_to_process[idx] === next_token_ids[idx] is determined
        # Active completion
        active_completion_ret: None | str = None
        while any(batch_needs_to_process):
            # print("Mem first preprocess:", time.perf_counter() - start)
            # print("Probability assigning:", probs_assign)

            start = time.perf_counter()
            assert probs.shape == torch.Size([self.batch_size, self.model.vocab_size])
            # The sampled ids corresponding to the batches that need to process
            zero_prob_batches = (probs.sum(dim=-1) == 0.0).nonzero().squeeze(1)
            # print("Batch zeroing:", time.perf_counter() - start)
            assert len(zero_prob_batches.shape) == 1
            for batch_idx_tensor in zero_prob_batches:
                batch_idx = batch_idx_tensor.item()
                assert isinstance(batch_idx, int)
                if not batch_is_failed[batch_idx]:
                    batch_is_failed[batch_idx] = True
                    if batch_needs_to_process[batch_idx]:
                        batch_needs_to_process[batch_idx] = False
                        generated_ids = gen_contexts[batch_idx].generated_ids
                        # Make the current state infeasible
                        if self.use_mem and len(generated_ids) > 0:
                            assert self.mem is not None
                            prev_state = pickle.dumps(generated_ids[:-1])
                            last_token_id = generated_ids[-1]
                            infeasible_token_ids = (
                                self.mem.infeasible_token_ids.setdefault(prev_state, [])
                            )
                            infeasible_token_ids.append(last_token_id)
                        next_token_ids[batch_idx] = self.model.end_id

            start = time.perf_counter()
            if not any(batch_needs_to_process):
                break
            if self.batch_size == 1:
                probs_to_process = probs
            else:
                assert False
                probs_to_process = probs[batch_needs_to_process]
            assert probs.shape == torch.Size([self.batch_size, self.model.vocab_size])

            trying_token_ids = torch.multinomial(
                probs_to_process, num_samples=1
            ).squeeze(1)
            assert trying_token_ids.dtype == torch.long
            assert len(trying_token_ids) == sum(batch_needs_to_process)

            if ACTIVE and active_completion is not None:
                # TODO: mem
                new_tok_idx = cast(int, trying_token_ids.item())
                new_tok = self.model.token_map[new_tok_idx]
                if active_completion.startswith(new_tok):
                    logging.info(
                        f"ACTIVE (MATCH): {new_tok}, prob: {probs[0, new_tok_idx].item()}"
                    )
                    update_batch_state(0, new_tok_idx)
                    # assert active_completion.startswith(new_tok) or new_tok.startswith(
                    #     active_completion
                    # )
                    active_completion = active_completion[len(new_tok) :]
                    active_completion_ret = active_completion
                    break
                else:
                    assert new_tok.startswith(active_completion)
                    active_completion = None
                    active_completion_ret = None
            assert active_completion is None

            # Batches denied by Trie
            if self.use_mem:
                batch_is_denied = [False] * self.batch_size

            # The map from batch_idx to trying_token_ids_idx
            trying_token_ids_idx_given_batch: list[int | bytes] = [
                b""
            ] * self.batch_size

            # print("Decoding:", time.perf_counter() - start)

            # Use memorization to do preprocessing and prepare for parallel checking
            # `trying_token_ids_idx`s are the rank position of each `batch_idx` where `batch_needs_to_process[batch_idx]`

            start = time.perf_counter()
            gpu_time = 0.0
            infeasible_time = 0.0
            batch_time = 0.0
            for trying_token_ids_idx, batch_idx in enumerate(
                filter(batch_needs_to_process.__getitem__, range(self.batch_size))
            ):
                trying_token_ids_idx_given_batch[batch_idx] = trying_token_ids_idx
                assert batch_needs_to_process[batch_idx]
                _start = time.perf_counter()
                trying_token_id = trying_token_ids[trying_token_ids_idx]
                trying_token_id_item = cast(int, trying_token_id.item())
                trying_token = self.model.token_map[trying_token_id_item]
                gpu_time += time.perf_counter() - _start
                if (
                    ACTIVE
                    and active_completion is not None
                    and active_completion.startswith(trying_token)
                ):
                    # WRONG
                    # trying_token.startswith(
                    #     active_completion
                    # ) or
                    update_batch_state(batch_idx, trying_token_id_item)
                    # print(active_completion)
                    active_completion = active_completion[len(trying_token) :]
                    active_completion_ret = active_completion
                    # print(active_completion)
                elif (
                    ACTIVE
                    and active_completion is not None
                    and not trying_token.startswith(active_completion)
                ):
                    probs[batch_idx, trying_token_id_item] = 0.0
                    mem_infeasible_token_ids[batch_idx].append(trying_token_id_item)
                    batch_is_denied[batch_idx] = True
                elif (
                    self.use_mem
                    and (
                        active_completion_ret := mem_feasible_token_ids[batch_idx].get(
                            trying_token_id_item
                        )
                    )
                    is not None
                ):
                    update_batch_state(batch_idx, trying_token_id_item)
                elif (
                    self.model.is_special_token(trying_token)
                    or self.trivially_feasible(trying_token)
                    or (
                        self.use_mem
                        and trying_token_id_item in mem_feasible_token_ids[batch_idx]
                    )
                ):
                    _start = time.perf_counter()
                    update_batch_state(batch_idx, trying_token_id_item)
                    batch_time += time.perf_counter() - _start
                elif self.use_mem and mem_denied_tokens[batch_idx].is_prefix_of(
                    trying_token
                ):
                    _start = time.perf_counter()
                    probs[batch_idx, trying_token_id_item] = 0.0
                    mem_infeasible_token_ids[batch_idx].append(trying_token_id_item)
                    batch_is_denied[batch_idx] = True
                    infeasible_time += time.perf_counter() - _start
                else:
                    lsp_context = self.lsp_contexts[batch_idx]
                    lsp_context.analyzer.send(
                        (
                            Message(
                                True,
                                JdtLspAnalyzer.pruned_decode.__name__,
                                lsp_context.text_file,
                                gen_contexts[batch_idx],
                                trying_token_id_item,
                                trying_token,
                            )
                        )
                    )
            # print("Real Initialization:", time.perf_counter() - start)
            # print("GPU time in real init:", gpu_time)
            # print("Infeasible checking time in real init:", infeasible_time)
            # print("Batch time in real init:", batch_time)
            start = time.perf_counter()
            for batch_idx in filter(
                batch_needs_to_process.__getitem__, range(self.batch_size)
            ):
                assert batch_needs_to_process[batch_idx]
                if self.use_mem and batch_is_denied[batch_idx]:
                    continue
                trying_token_ids_idx = cast(
                    int, trying_token_ids_idx_given_batch[batch_idx]
                )
                assert isinstance(trying_token_ids_idx, int)
                trying_token_id = trying_token_ids[trying_token_ids_idx]
                trying_token_id_item = cast(int, trying_token_id.item())
                success: bool | str = self.lsp_contexts[batch_idx].analyzer.recv()
                curr_trying_token = self.model.token_map[trying_token_id_item]
                if success == True or isinstance(success, str):
                    update_batch_state(batch_idx, trying_token_id_item)
                    if isinstance(success, str) and ACTIVE:
                        active_completion_ret = success
                    if self.use_mem:
                        mem_feasible_token_ids[batch_idx][trying_token_id_item] = (
                            success if isinstance(success, str) else None
                        )
                    token_prob = probs[batch_idx, trying_token_id_item].item()
                    logging.info(
                        f"Accepted: {curr_trying_token}, completion: {success}, prob: {token_prob}"
                    )
                else:
                    token_prob = probs[batch_idx, trying_token_id_item].item()
                    probs[batch_idx, trying_token_id_item] = 0.0
                    logging.info(f"Pruned: {curr_trying_token}, prob: {token_prob}")
                    if self.use_mem:
                        mem_infeasible_token_ids[batch_idx].append(trying_token_id_item)
                        mem_denied_tokens[batch_idx].insert(trying_token)
            # print("Checking:", time.perf_counter() - start)
        assert (next_token_ids == special_value).sum().item() == 0
        if active_completion_ret is not None and len(active_completion_ret) == 0:
            active_completion_ret = None
        if not ACTIVE:
            active_completion_ret = None
        return (
            cast(torch.LongTensor, next_token_ids),
            active_completion_ret,
        )

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
    ):
        model = self.model

        # init values
        logits_warper = (
            logits_warper if logits_warper is not None else LogitsProcessorList()
        )
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )
        pad_token_id = (
            pad_token_id if pad_token_id is not None else model.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id if eos_token_id is not None else model.config.eos_token_id
        )
        output_scores = (
            output_scores if output_scores is not None else model.config.output_scores
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else model.config.output_hidden_states
        )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

        # auto-regressive generation
        active_completion: str | None = None
        while True:
            assert not ACTIVE or (
                self.batch_size == 1
                and self.gen_state is not None
                and self.lsp_contexts is not None
                and len(self.gen_state.gen_contexts) == 1
                and len(self.lsp_contexts) == 1
            )

            # prepare model inputs
            model_inputs = model.prepare_inputs_for_generation(
                input_ids, **model_kwargs
            )
            # breakpoint()

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
                next_token_scores = TemperatureLogitsWarper(temperature)(
                    input_ids, next_token_scores
                )
            next_token_scores = TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1)(
                input_ids, next_token_scores
            )
            # next_token_scores = logits_warper(input_ids, next_token_scores)
            # assert len(next_token_scores) == 1
            # scores = next_token_scores[0]
            # BATCH_SIZE * TOKEN_MAP_SIZE
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            assert len(probs.shape) == 2
            assert probs.shape == torch.Size((self.batch_size, model.vocab_size))

            # # ACTIVE
            # if self.batch_size == 1 and os.getenv("ACTIVE") is not None:
            #     next_token_ids = self.active_decode(probs)

            if self.is_plain:  # or count > 10:
                # shape: (1)
                start = time.perf_counter()
                next_token_ids = self.plain_decode(probs)
            else:
                # try:
                start = time.perf_counter()
                next_token_ids, active_completion = self.pruned_decode(
                    probs, active_completion
                )
                assert ACTIVE or active_completion is None, active_completion
                # if active_completion is not None:
                failed_msg = "" if not self.gen_state.batch_is_failed[0] else "Failed"
                if ACTIVE:
                    assert self.batch_size == 1
                    if active_completion is not None:
                        logging.info(f"ACTIVE {failed_msg}: {active_completion}")
            generated_tokens = self.gen_state.gen_contexts[0].generated_tokens
            logging.info(
                f"       {generated_tokens}\n" f"       {''.join(generated_tokens)}\n"
            )
            # print("Time:", time.perf_counter() - start)
            # breakpoint()
            # except RuntimeError:
            #     return completion_overhead, [], None, generation_log

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                assert isinstance(pad_token_id, int)
                next_token_ids = (
                    next_token_ids * unfinished_sequences
                    + pad_token_id * (1 - unfinished_sequences)
                )
            # print(model_kwargs['use_cache'])
            # breakpoint()
            # cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            # if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul(
                torch.isin(
                    next_token_ids if self.batch_size != 1 else next_token_ids[-1:],
                    self.end_ids_tensor,
                    invert=True,
                ).long()
            )
            assert len(unfinished_sequences) == self.batch_size

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0:
                break
            if len(input_ids[0]) + 1 == max_length:
                break

            # if ACTIVE:
            #     input_state_after_pruning = pickle.dumps(
            #         self.gen_state.gen_contexts[0].generated_ids
            #     )
            #     assert self.mem_active is not None
            #     mem_result = self.mem_active.get(input_state_after_pruning)
            #     if mem_result is not None:
            #         (
            #             potential_next_token_ids,
            #             next_token_ids_list,
            #             active_completion_str,
            #         ) = mem_result
            #         logging.info(
            #             f"ACTIVE(HIT): {''.join(self.model.token_map[idx] for idx in potential_next_token_ids)}"
            #         )
            #         # assert isinstance(input_state_after_pruning, bytes)
            #         # potential_next_token_ids = self.mem_active.get(input_state_after_pruning)
            #         # if potential_next_token_ids is not None:
            #         assert potential_next_token_ids[0] == next_token_ids[0]
            #         next_token_ids = potential_next_token_ids
            #         lsp_context = self.lsp_contexts[0]
            #         gen_context = self.gen_state.gen_contexts[0]
            #         lsp_context.text_file.add(active_completion_str)
            #         gen_context.generated_ids.extend(next_token_ids_list)
            #         gen_context.generated_tokens.extend(
            #             self.model.token_map[id] for id in next_token_ids_list
            #         )
            #     elif isinstance(active_completion, str) and len(active_completion) > 0:
            #         # breakpoint()
            #         assert self.batch_size == 1
            #         next_token_ids_list = self.tokenize(active_completion)
            #         lsp_context = self.lsp_contexts[0]
            #         gen_context = self.gen_state.gen_contexts[0]
            #         lsp_context.text_file.add(active_completion)
            #         gen_context.generated_ids.extend(next_token_ids_list)
            #         gen_context.generated_tokens.extend(
            #             self.model.token_map[id] for id in next_token_ids_list
            #         )
            #         logging.info(f"ACTIVE: {active_completion}")
            #         active_completion_tensor = torch.tensor(
            #             next_token_ids_list, dtype=torch.long, device=utils.DEVICE
            #         )
            #         # assert active_completion_tensor.shape[0] == next_token_ids.shape[0]
            #         next_token_ids = torch.cat(
            #             (next_token_ids, active_completion_tensor)
            #         )
            #         self.mem_active[input_state_after_pruning] = (
            #             next_token_ids,
            #             next_token_ids_list,
            #             active_completion,
            #         )
            # update generated ids, model inputs, and length for next step
            use_cache = self.batch_size != 1 or len(next_token_ids) == 1
            assert use_cache
            # use_cache = True
            # An ad-hoc solution for active completion (batch size must equals 1)
            if self.batch_size == 1:
                input_ids = torch.cat([input_ids, next_token_ids[None, :]], dim=-1)
            else:
                input_ids = torch.cat([input_ids, next_token_ids[:, None]], dim=-1)
            model_kwargs["use_cache"] = use_cache
            model_kwargs = model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=model.config.is_encoder_decoder,
            )
            # IMPORTANT: codet5 output format: <mask0>....<mask1>....<mask2>...
            # Mask ids are 32099, 32098, 32097...
        for batch_idx in range(self.batch_size):
            is_unfinished = unfinished_sequences[batch_idx].item() == 1
            self.gen_state.batch_is_unfinished[batch_idx] = is_unfinished
            if os.getenv("TRY") is not None:
                generated_ids = self.gen_state.gen_contexts[batch_idx].generated_ids
                input_state = pickle.dumps(
                    generated_ids[:-1] if is_unfinished else generated_ids
                )
                infeasible = self.mem.infeasible_token_ids.setdefault(input_state, [])
                if is_unfinished and len(input_ids[0]) > 0:
                    infeasible.append(input_ids[batch_idx, -1])
                if not is_unfinished and len(input_ids[0]) > 0:
                    infeasible.extend(model.end_ids)
        return

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
        force_words_ids: Optional[Union[Iterable[int], Iterable[Iterable[int]]]] = None,
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
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], list[int]]
        ] = None,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
        renormalize_logits: Optional[bool] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
        constraints: Optional[list[Constraint]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        exponential_decay_length_penalty: Optional[tuple[Union[int, float]]] = None,
        **model_kwargs,
    ):
        inputs = self.inputs
        model = self.model

        # 1. set generation parameters if not already defined
        bos_token_id = (
            bos_token_id if bos_token_id is not None else model.config.bos_token_id
        )
        num_beams = num_beams if num_beams is not None else model.config.num_beams
        length_penalty = (
            length_penalty
            if length_penalty is not None
            else model.config.length_penalty
        )
        early_stopping = (
            early_stopping
            if early_stopping is not None
            else model.config.early_stopping
        )
        num_beam_groups = (
            num_beam_groups
            if num_beam_groups is not None
            else model.config.num_beam_groups
        )
        do_sample = do_sample if do_sample is not None else model.config.do_sample
        num_return_sequences = (
            num_return_sequences
            if num_return_sequences is not None
            else model.config.num_return_sequences
        )

        pad_token_id = (
            pad_token_id if pad_token_id is not None else model.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id if eos_token_id is not None else model.config.eos_token_id
        )

        if eos_token_id is None and hasattr(model.config, "decoder"):
            eos_token_id = model.config.decoder.eos_token_id

        if pad_token_id is None and eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            logger.warning(
                f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation."
            )
            pad_token_id = eos_token_id

        output_scores = (
            output_scores if output_scores is not None else model.config.output_scores
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else model.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else model.config.return_dict_in_generate
        )

        # 2. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        prev_use_cache = model_kwargs.get("use_cache")
        inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
            inputs, bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]
        assert prev_use_cache == model_kwargs.get("use_cache")
        # assert batch_size == 1

        # 3. Define other model kwargs
        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states
        model_kwargs["use_cache"] = use_cache

        accepts_attention_mask = "attention_mask" in set(
            inspect.signature(model.forward).parameters.keys()
        )
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if (
            model_kwargs.get("attention_mask", None) is None
            and requires_attention_mask
            and accepts_attention_mask
        ):
            model_kwargs[
                "attention_mask"
            ] = model._prepare_attention_mask_for_generation(
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
            input_ids_string = (
                "decoder_input_ids" if model.config.is_encoder_decoder else "input_ids"
            )
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
