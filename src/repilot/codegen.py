import json
import logging
import os
import random
import shutil
import time
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path
from typing import cast

import numpy
import regex as re
import torch

from . import generation as gen
from . import utils
from .analyzer import JdtLspAnalyzer, Message
from .config import MetaConfig, RepairConfig
from .d4j import Bug, Change, Defects4J
from .lsp import TextFile, spec
from .model import CodeT5Large, Incoder, ModelType
from .report import Report
from .template import generate_templates

DATA_DIR = Path(os.getenv("LSP", ".lsp_data"))

if utils.INCODER:
    INCODER_PREFIX_SUFFIX: dict = json.loads(
        Path("data/single-hunk-prefix-suffix.json").read_text()
    )


def wait_until_all_analyzers_free(
    tool_conns: list[Connection],
    max_waiting_time: float = 20,
    free_check_time: float = 1.0,
):
    batch_is_free = [False] * len(tool_conns)
    start_time = time.perf_counter()
    print("Waiting until all analyzers are free...")
    while time.perf_counter() < start_time + max_waiting_time:
        for idx, connection in enumerate(tool_conns):
            if not batch_is_free[idx]:
                connection.send(
                    Message(True, JdtLspAnalyzer.is_free.__name__, free_check_time)
                )

        for idx, connection in enumerate(tool_conns):
            if not batch_is_free[idx]:
                is_free = connection.recv()
                batch_is_free[idx] = is_free
        print("Elapsed:", time.perf_counter() - start_time)
        if all(batch_is_free):
            print("All analyzers are free:", time.perf_counter() - start_time)
            break


def get_buggy_hunk_start_end_indices_and_positions(
    text_file: TextFile, change: Change
) -> tuple[int, int, spec.Position, spec.Position]:
    start = change.start - 1
    end = start + len(change.removed_lines)
    start_pos = text_file.refine_index(start, 0)
    end_pos = text_file.refine_index(end, 0)

    start_index = text_file.form_index(start, 0)
    end_index = text_file.form_index(end, 0)
    return start_index, end_index, start_pos, end_pos


def remove_buggy_hunk(text_file: TextFile, change: Change) -> tuple[str, str]:
    """Modifies `text_file` and returns the prefix and the suffix"""
    (
        start_index,
        end_index,
        _,
        _,
    ) = get_buggy_hunk_start_end_indices_and_positions(text_file, change)
    prefix_start = 0
    suffix_end = len(text_file.content)
    prefix = text_file.content[prefix_start:start_index]
    # "\n" is important as we need a blank place for generation
    suffix = "\n" + text_file.content[end_index:suffix_end]

    # prefix(\n)
    # insertion(\n)
    # <cursor:infill>
    # (\n)suffix
    text_file.modify(start_index, end_index, "\n")
    # text_file.change(
    #     [
    #         cast(
    #             spec.EntireDocumentChange,
    #             {"text": "\n", "range": {"start": start_pos, "end": end_pos}},
    #         )
    #     ]
    # )

    text_file.move_cursor(start_index)
    if start_index != 0:
        assert prefix.endswith("\n")
        assert text_file.content[text_file.cursor - 1] == "\n"
        assert text_file.content[text_file.cursor] == "\n"

    return prefix, suffix


class Codegen:
    def __init__(
        self,
        config: MetaConfig,
        repair_config: RepairConfig,
        model: ModelType,
        active_connection_analyzer_pairs: list[tuple[Connection, JdtLspAnalyzer]],
        proj_root: Path,
        file_path: Path,
        line: int,
        column: int,
        n: int,
    ) -> None:
        self.config = config
        self.repair_config = repair_config
        self.model = model
        self.active_connection_analyzer_pairs = active_connection_analyzer_pairs
        self.proj_root = proj_root
        self.file_path = file_path
        self.line = line
        self.column = column
        self.n = n

    @staticmethod
    def init(
        config: MetaConfig,
        repair_config: RepairConfig,
        codet5: bool,
        proj_root: Path,
        file_path: Path,
        line: int,
        column: int,
        n: int,
    ) -> "Codegen":
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)
        # report = Report.create(report_dir, config)
        if codet5:
            model: ModelType = CodeT5Large.init().to(utils.DEVICE)  # type: ignore # noqa
        else:
            model = Incoder.from_pretrained("facebook/incoder-6B").to(utils.DEVICE)
        return Codegen(
            config, repair_config, model, [], proj_root, file_path, line, column, n
        )

    def server_cmd_maker(self) -> list[str]:
        # IMPORTANT: -data dir should be DIFFERENT for different analyzers!!!
        return self.config.language_server_cmd + [
            "-data",
            str(DATA_DIR / str(next(utils.COUNTER))),
        ]

    def fix_seed(self):
        print("Fixing seed to", self.config.seed)
        torch.manual_seed(self.config.seed)
        numpy.random.seed(self.config.seed)
        random.seed(self.config.seed)

    def codegen(self):
        self.fix_seed()
        try:
            self.codegen_nocleanup()
        finally:
            self.clean_up()
            print("Cleaned up")

    def clean_up(self):
        for connection, _ in self.active_connection_analyzer_pairs:
            connection.send(None)
        for _, analyzer in self.active_connection_analyzer_pairs:
            analyzer.join()
        self.active_connection_analyzer_pairs.clear()

    def codegen_nocleanup(self):
        def init_analyzers() -> tuple[list[Connection], list[TextFile]]:
            connection_pairs = cast(
                list[tuple[Connection, Connection]],
                [Pipe(duplex=True) for _ in range(1)],
            )
            connection_analyzer_pairs = [
                (
                    client_conn,
                    JdtLspAnalyzer(
                        analyzer_conn,
                        self.server_cmd_maker(),
                        Path(self.proj_root),
                        self.model,
                        str(self.config.java8_home),
                    ),
                )
                for analyzer_conn, client_conn in connection_pairs
            ]
            connections = [connection for connection, _ in connection_analyzer_pairs]
            for connection, analyzer in connection_analyzer_pairs:
                analyzer.start()
                self.active_connection_analyzer_pairs.append((connection, analyzer))
            for connection in connections:
                connection.send(Message(True, JdtLspAnalyzer.init.__name__))
            for connection in connections:
                init_result = connection.recv()

            # Buggy text files
            base_path = Path(self.proj_root).parent.absolute()
            # proj_path = Path(self.proj_root).relative_to(base_path)
            text_file = TextFile.read(
                base_path, self.file_path.relative_to(base_path)
            )
            buggy_text_files = [text_file]

            # Initialize each buggy file for LSP
            assert isinstance(connections, list)
            for connection in connections:
                for buggy_text_file in buggy_text_files:
                    connection.send(
                        Message(False, JdtLspAnalyzer.open.__name__, buggy_text_file)
                    )
            wait_until_all_analyzers_free(connections)
            return connections, buggy_text_files

        # Ready to repair
        connections, text_files = init_analyzers()
        text_file = text_files[0]
        index = text_file.form_index(self.line - 1, self.column - 1)
        text_file.move_cursor(index)
        prefix = text_file.content[:index]
        suffix = text_file.content[index:]

        print("Prefix:")
        print(prefix)
        print("Suffix:")
        print(suffix)
        lm_context = gen.LMContext(
            self.model, prefix, suffix, self.repair_config.lm_inference_config
        )
        synthesizer = gen.Synthesizer(
            lm_context, connections, text_file, self.repair_config.method
        )
        for _ in range(self.n):
            synthesis_result_batch = synthesizer.synthesize("", "")
            assert self.repair_config.batch_size == 1
            assert len(synthesis_result_batch.results) == 1
            result = synthesis_result_batch.results[0]
            print("Result:")
            print(result.hunk)
