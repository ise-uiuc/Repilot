import difflib
import functools
import os
import subprocess
import time
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Callable, Iterable, Iterator, TypeVar, cast

import javalang
import regex as re
from joblib import Parallel, delayed

from . import utils
from .config import MetaConfig, RepairConfig, ValidationConfig
from .d4j import Defects4J
from .lsp import TextFile
from .repair import Repairer
from .report import Report
from .results import (
    AvgFilePatch,
    AvgPatch,
    AvgSynthesisResult,
    BuggyHunk,
    GenerationDatapoint,
    HunkRepairResult,
    Outcome,
    PatchValidationResult,
    RepairResult,
    RepairTransformedResult,
    ValidationCache,
    ValidationDatapoint,
    ValidationResult,
    concat_hunks,
)

PatchInfo = TypeVar("PatchInfo")
Datapoint = TypeVar("Datapoint")
EvaluationResult = TypeVar("EvaluationResult")


@dataclass
class Runner:
    """A `Report` converter"""

    report: Report

    def __post_init__(self):
        self.d4j = self.report.get_d4j()

    @staticmethod
    def create(root: Path, config: MetaConfig, repair_config: RepairConfig) -> "Runner":
        root.mkdir()
        print("Metadata will be saved to", root)
        return Runner(Report(root, config, RepairResult({}, repair_config), None, None))

    @staticmethod
    def load(root: Path) -> "Runner":
        print(f"Loading data from {root}")
        report = Report.load(root)
        print("Done")
        return Runner(report)

    def repair(self, repairer: Repairer):
        utils.disable_tokenizer_parallel()
        repairer.repair(self.report)

    def transform(self):
        """Incremental data transformation"""
        # BUG (maybe fixed): incremental update not working
        report = self.report
        assert isinstance(report.repair_result, RepairResult)
        # a_results = report.analysis_result.results
        # for repair_idx, result in enumerate(report.repair_result.results):
        # if repair_idx == len(a_results):
        #     a_results.append(RepairAnalysisResult({}, {}))
        if report.transformed_result is None:
            report.transformed_result = RepairTransformedResult({}, {})
        result_dict = report.transformed_result.result_dict
        all_appeared = report.transformed_result.all_appeared
        for bug_id, files in report.repair_result.result_dict.items():
            buggy_files, patches = result_dict.setdefault(bug_id, ([], []))
            appeared = all_appeared.setdefault(bug_id, [])
            for patch_idx, (buggy_text_files, patch) in enumerate(iter_files(files)):
                if patch_idx < len(patches):
                    continue
                if len(buggy_files) == 0:
                    buggy_files.extend(buggy_text_files)
                if any(
                    hunk.result.hunk is None for file in patch for hunk in file.hunks
                ):
                    is_duplicate = False
                else:
                    concat_hunk_str = concat_hunks(patch)
                    aggressive = os.getenv("AGGRESSIVE") is not None
                    if not aggressive and concat_hunk_str in appeared:
                        is_duplicate = True
                    elif aggressive and utils.remove_whitespace(
                        utils.remove_java_comments(concat_hunk_str)
                    ) in {
                        utils.remove_whitespace(utils.remove_java_comments(hunk_str))
                        for hunk_str in appeared
                    }:
                        is_duplicate = True
                    else:
                        appeared.append(concat_hunk_str)
                        is_duplicate = False
                patches.append(AvgPatch(patch, is_duplicate))
            # result_dict[bug_id] = patches
        # a_results.append(RepairAnalysisResult(result_dict))
        # report.analysis_result = RepairAnalysisResults(a_results)
        report.save()

    def transform_with_message(self):
        print("Doing data transformation first...")
        self.transform()
        print("Done")

    def validate(self, config: ValidationConfig, cache: ValidationCache | None):
        self.transform_with_message()
        """Recoverable validation"""
        bug_pattern = re.compile(config.bug_pattern)
        report = self.report
        d4j = self.d4j
        assert report.transformed_result is not None
        if report.validation_result is None:
            report.validation_result = ValidationResult([], {})
        report.validation_result.validation_configs.append(config)
        val_config_idx = len(report.validation_result.validation_configs) - 1
        transformed = report.transformed_result
        validation_result = report.validation_result
        validation_result_dict = validation_result.result_dict
        transformed_result_dict = transformed.result_dict
        unvalidated_analysis_results: list[
            list[tuple[str, int, list[TextFile], AvgPatch]]
        ] = []
        for bug_id, (buggy_text_files, patches) in transformed_result_dict.items():
            if bug_pattern.fullmatch(bug_id) is None:
                continue
            validated_patches = validation_result_dict.setdefault(bug_id, {})
            unvalidated_analysis_results.append([])
            for patch_idx, patch in enumerate(patches):
                if (
                    patch.is_duplicate
                    or patch.is_broken
                    or patch_idx in validated_patches
                ):
                    print(f"[{bug_id}, {patch_idx}] Skipped (dup/broken/validated)")
                    continue
                if cache is not None:
                    concat_hunk_str = concat_hunks(patch.file_patches)
                    bug_id_cache = cache.result_dict.setdefault(bug_id, {})
                    ws_removed_cache = {
                        utils.remove_whitespace(
                            utils.remove_java_comments(hunk_str)
                        ): result
                        for hunk_str, result in bug_id_cache.items()
                    }
                    ws_removed_concat_hunk_str = utils.remove_whitespace(
                        utils.remove_java_comments(concat_hunk_str)
                    )
                    bug_id_result = ws_removed_cache.get(ws_removed_concat_hunk_str)
                    if bug_id_result is not None:
                        print(f"[{bug_id}] Skipped (cache):")
                        print(concat_hunk_str)
                        print("WS REMOVED")
                        print(ws_removed_concat_hunk_str)
                        validated_patches[patch_idx] = (-1, bug_id_result)
                        continue
                unvalidated_analysis_results[-1].append(
                    (bug_id, patch_idx, buggy_text_files, patch)
                )
        n_unvalidated = sum(1 for xs in unvalidated_analysis_results for _ in xs)
        # Validate n_cores bugs with different bug_ids in parallel
        for zipped_result in zip_longest(*unvalidated_analysis_results):
            print("#Unvalidated:", n_unvalidated)
            zipped_results = filter(lambda r: r is not None, zipped_result)
            for result_batch in utils.chunked(config.n_cores, zipped_results):
                assert len(result_batch) == len(set(r[0] for r in result_batch))
                val_results: list[PatchValidationResult] = Parallel(
                    n_jobs=len(result_batch),
                    backend="multiprocessing",
                )(
                    delayed(validate_patch)(d4j, bug_id, buggy_text_files, avg_patch)
                    for (bug_id, _, buggy_text_files, avg_patch) in result_batch
                )
                for (bug_id, val_idx, _, _), val_result in zip(
                    result_batch, val_results
                ):
                    assert bug_id in validation_result_dict
                    assert val_idx not in validation_result_dict[bug_id]
                    validation_result_dict[bug_id][val_idx] = (
                        val_config_idx,
                        val_result,
                    )
                    n_unvalidated -= 1
                # Temporary method to prevent memory leakage
                if os.getenv("KILL") is not None:
                    os.system('pkill -SIGKILL -u $USER -f "javac1.7"')
                report.save()

    def get_transformed_items(
        self,
    ) -> Iterable[tuple[str, list[AvgPatch]]]:
        assert self.report.repair_result is not None
        # For speed of ploting, there is no warning if the transformed result is partial
        # self.transform_with_message()
        assert self.report.transformed_result is not None
        result = (
            (bug_id, patches)
            for bug_id, (
                _,
                patches,
            ) in self.report.transformed_result.result_dict.items()
        )
        if os.getenv("LINE") is not None:
            result = (
                (bug_id, patches)
                for bug_id, patches in result
                if bug_id in self.d4j.single_line_bugs
            )
        return result

    def get_validation_items(
        self,
    ) -> Iterable[tuple[str, Iterable[tuple[AvgPatch, PatchValidationResult]]]]:
        assert self.report.transformed_result is not None
        assert self.report.validation_result is not None
        result_dict = self.report.validation_result.result_dict
        for bug_id, patches in self.get_transformed_items():
            try:
                result = result_dict[bug_id]
            except KeyError:
                continue
            yield (
                bug_id,
                (
                    (patch, result[patch_idx][1])
                    for patch_idx, patch in enumerate(patches)
                    if patch_idx in result
                ),
            )

    # def evaluate_generation(self) -> dict[str, list[GenerationDatapoint]]:
    #     return self.evaluate(
    #         Runner.get_transformed_items,
    #         map_to_generation_datapoint,
    #         lambda points, point: points + [points[-1] + point],
    #         [GenerationDatapoint.zero()],
    #     )

    # def evaluate_validation(self) -> dict[str, list[ValidationDatapoint]]:
    #     self.transform_with_message()
    #     return self.evaluate(
    #         Runner.get_validation_items,
    #         map_to_validation_datapoint,
    #         lambda points, point: points + [points[-1] + point],
    #         [ValidationDatapoint.zero()],
    #     )

    def evaluate_generation_grouped(
        self,
    ) -> list[tuple[str, dict[str, GenerationDatapoint]]]:
        return Defects4J.group_by_project(self.evaluate_generation())

    def evaluate_validation_grouped(
        self,
    ) -> list[tuple[str, dict[str, ValidationDatapoint]]]:
        return Defects4J.group_by_project(self.evaluate_validation())

    def get_plausible_patches_grouped(
        self,
    ) -> dict[str, dict[str, list[list[AvgFilePatch]]]]:
        result: dict[str, dict[str, list[list[AvgFilePatch]]]] = {}
        validation_dict = {
            bug_id: list(patches) for bug_id, patches in self.get_validation_items()
        }
        validation_dict_grouped = Defects4J.group_by_project(validation_dict)
        for proj, proj_dict in validation_dict_grouped:
            proj_result = result.setdefault(proj, {})
            for bug_id, patches in proj_dict.items():
                results = []
                for patch, val_result in patches:
                    if val_result.outcome == Outcome.Success:
                        # Assertions
                        for file_patch in patch.file_patches:
                            for hunk in file_patch.hunks:
                                assert hunk.result.hunk is not None
                        results.append(patch.file_patches)
                if len(results) > 0:
                    proj_result[bug_id] = results
        return result

    def evaluate_generation(self) -> dict[str, GenerationDatapoint]:
        # self.transform_with_message()
        # assert self.report.transformed_result is not None
        return {
            bug_id: functools.reduce(
                GenerationDatapoint.__add__,
                map(map_to_generation_datapoint, patches),
                GenerationDatapoint.zero(),
            )
            for bug_id, patches in self.get_transformed_items()
        }

    def evaluate_validation(self) -> dict[str, ValidationDatapoint]:
        assert self.report.transformed_result is not None
        assert self.report.validation_result is not None
        return {
            bug_id: functools.reduce(
                ValidationDatapoint.__add__,
                map(
                    map_to_validation_datapoint,
                    filter(lambda patch: not patch[0].is_duplicate, patches),
                ),
                ValidationDatapoint.zero(),
            )
            for bug_id, patches in self.get_validation_items()
        }

    def evaluate_validation_summary(
        self,
    ) -> tuple[dict[str, ValidationDatapoint], ValidationDatapoint]:
        result_grouped = self.evaluate_validation_grouped()
        add = ValidationDatapoint.__add__
        zero = ValidationDatapoint.zero()
        proj_summary = {
            proj: functools.reduce(add, proj_dict.values(), zero)
            for proj, proj_dict in result_grouped
        }
        summary = functools.reduce(add, proj_summary.values(), zero)
        return proj_summary, summary

    def evaluate_unique_generation_summary(self) -> GenerationDatapoint:
        return functools.reduce(
            GenerationDatapoint.__add__,
            map(
                map_to_unique_generation_datapoint,
                (
                    patch
                    for _, patches in self.get_transformed_items()
                    for patch in patches
                ),
            ),
            GenerationDatapoint.zero(),
        )

    def evaluate_generation_summary(self) -> GenerationDatapoint:
        return functools.reduce(
            GenerationDatapoint.__add__,
            map(
                map_to_generation_datapoint,
                (
                    patch
                    for _, patches in self.get_transformed_items()
                    for patch in patches
                ),
            ),
            GenerationDatapoint.zero(),
        )

    # def evaluate_validation_first_one_grouped(
    #     self, get_number: Callable[[ValidationDatapoint], int]
    # ) -> list[tuple[str, dict[str, ValidationDatapoint]]]:
    #     return Defects4J.group_by_project(
    #         self.evaluate_validation_first_one(get_number)
    #     )

    # def evaluate_validation_first_one(
    #     self, get_number: Callable[[ValidationDatapoint], int]
    # ) -> dict[str, ValidationDatapoint]:
    #     assert self.report.validation_result is not None

    #     def reduce_fn(
    #         lhs: ValidationDatapoint, rhs: ValidationDatapoint
    #     ) -> ValidationDatapoint:
    #         if get_number(lhs) == 1:
    #             return lhs
    #         return lhs + rhs

    #     return {
    #         bug_id: functools.reduce(
    #             reduce_fn,
    #             map(map_to_validation_datapoint, patches),
    #             ValidationDatapoint.zero(),
    #         )
    #         for bug_id, patches in self.get_validation_items()
    #     }

    # def evaluate(
    #     self,
    #     patch_infos: "Callable[[Runner], Iterable[tuple[str, Iterable[PatchInfo]]]]",
    #     map_to_datapoint: Callable[[PatchInfo], Datapoint],
    #     add_reduce: Callable[[EvaluationResult, Datapoint], EvaluationResult],
    #     initial_result: EvaluationResult,
    # ) -> dict[str, EvaluationResult]:
    #     # assert self.report.repair_result is not None
    #     # self.transform_with_message()
    #     # transformed_result = self.report.transformed_result
    #     # assert transformed_result is not None
    #     # transformed_result_dict = transformed_result.result_dict
    #     result: dict[str, EvaluationResult] = {}
    #     for bug_id, patches in patch_infos(self):
    #         mapped_generation_datapoints = map(map_to_datapoint, patches)
    #         bug_id_result = functools.reduce(
    #             add_reduce, mapped_generation_datapoints, initial_result
    #         )
    #         result[bug_id] = bug_id_result
    #     return result

    # def evaluate_validation(self) -> dict[str, list[ValidationDatapoint]]:
    #     generation_result = self.evaluate_generation()
    #     assert self.report.repair_result is not None
    #     validation_result = self.report.validation_result
    #     assert validation_result is not None
    #     validation_result_dict = validation_result.result_dict
    #     result: dict[str, list[ValidationResult]] = {}
    #     for bug_id, patches in list(validation_result_dict.items()):
    #         generation_datapoints = generation_result[bug_id]
    #         mapped_validation_datapoints = map(
    #             lambda patch: ValidationDatapoint(
    #                 n_parse_success=,
    #                 n_comp_success=,
    #                 n_test_success=,
    #                 total_time_consumed=
    #                 gen_datapoint=
    #             ),
    #             generation_datapoints,
    #         )
    #         datapoints = functools.reduce(
    #             lambda points, point: points + [points[-1] + point],
    #             mapped_validation_datapoints,
    #             [GenerationDatapoint.zero()],
    #         )
    #         result[bug_id] = datapoints
    #     return result


def validate_patch(
    d4j: Defects4J, bug_id: str, bugs: list[TextFile], patch: AvgPatch
) -> PatchValidationResult:
    start_time = time.perf_counter()
    assert not patch.is_duplicate

    def cost():
        return time.perf_counter() - start_time

    patch_files: list[TextFile] = []
    assert len(patch.file_patches) == len(bugs)
    for patch_file, bug in zip(patch.file_patches, bugs):
        patch_text_file = patch_file.compute_patch(bug)
        # Convert unified_diff to string
        assert patch_text_file is not None
        unified_diff = difflib.unified_diff(
            bug.content.splitlines(), patch_text_file.content.splitlines()
        )
        print("\n".join(unified_diff))
        # assert (d4j.d4j_checkout_root / patch_text_file.path).exists()
        patch_files.append(patch_text_file)
    # assert len(set(p._path for p in patch_files)) == 1, [p._path for p in patch_files]
    # Checkout the fixed version and then apply patches b/c we do not consider test file changes
    d4j.checkout(bug_id, buggy=False)
    for patch_text_file in patch_files:
        try:
            javalang.parse.parse(patch_text_file.content)
        except (
            javalang.parser.JavaSyntaxError,
            javalang.tokenizer.LexerError,
        ) as e:
            return PatchValidationResult(Outcome.ParseError, cost(), "", str(e))
        except Exception as e:
            # TODO: add an InternalError record in `Report`
            with open("unexpected_exception", "a") as f:
                f.write(str(type(e)))
                f.write("\n")
                f.write(str(e))
        patch_text_file.root = d4j.d4j_checkout_root
        assert patch_text_file.path.exists()
        patch_text_file.write()
    comp_success, comp_stdout, comp_stderr = d4j.compile(bug_id)
    if not comp_success:
        return PatchValidationResult(
            Outcome.CompilationError, cost(), comp_stdout, comp_stderr
        )
    try:
        val_success, val_stdout, val_stderr = d4j.test(bug_id, timeout=180)
        return PatchValidationResult(
            Outcome.Success if val_success else Outcome.TestingError,
            cost(),
            comp_stdout + utils.RULE + val_stdout,
            comp_stderr + utils.RULE + val_stderr,
        )
    except subprocess.TimeoutExpired:
        return PatchValidationResult(
            Outcome.TestingError,
            cost(),
            comp_stdout + utils.RULE + "Timeout",
            comp_stderr + utils.RULE + "Timeout",
        )


_AvgResult = tuple[AvgSynthesisResult, BuggyHunk]


def iter_files(
    file_results: list[list[HunkRepairResult]],
) -> Iterable[tuple[list[TextFile], list[AvgFilePatch]]]:
    # For one bug
    # items = list(hunk_dict.items())
    # items.sort(key=lambda kv: kv[0])
    groups: list[list[list[_AvgResult]]] = []
    # TODO: maybe take a look at `itertools.groupby`
    # last_f: int | None = None
    for hunk_results in file_results:
        groups.append([])
        group = groups[-1]
        for hunk_result in hunk_results:
            avg_results = [
                (avg_result, hunk_result.buggy_hunk)
                for tagged_result in hunk_result.results
                for avg_result in tagged_result.synthesis_result_batch.to_average_results()
            ]
            # assert len(avg_results) > 0
            # assert f_idx == len(groups)
            # if last_f is None or f_idx != last_f:
            #     group: list[list[_AvgResult]] = []
            #     groups.append(group)
            # else:
            #     group = groups[-1]
            # assert h_idx == len(group)
            group.append(avg_results)
    for group in groups:
        assert len(group) > 0
        assert len(set(len(data) for data in group)) == 1
        assert len(set(t[1].file._path for data in group for t in data)) == 1

    for file_groups in zip(*(zip(*group) for group in groups)):
        assert len(file_groups) > 0
        buggy_files: list[TextFile] = []
        file_patches: list[AvgFilePatch] = []
        for file_group in file_groups:
            hunks: list[AvgSynthesisResult] = []
            buggy_hunk_indices: list[tuple[int, int]] = []
            assert len(file_group) > 0
            bug: TextFile | None = None
            for avg_result, buggy_hunk in file_group:
                bug = buggy_hunk.file
                buggy_hunk_indices.append((buggy_hunk.start, buggy_hunk.end))
                hunks.append(avg_result)
            assert bug is not None
            buggy_files.append(bug)
            file_patches.append(AvgFilePatch(hunks, buggy_hunk_indices))
        yield buggy_files, file_patches


def map_to_unique_generation_datapoint(patch: AvgPatch) -> GenerationDatapoint:
    return GenerationDatapoint(
        gen_time=0.0 if patch.is_duplicate else patch.total_gen_time,
        n_total=utils.binary_bool(not patch.is_duplicate),
        n_unique=utils.binary_bool(not patch.is_duplicate),
        n_unfinished=utils.binary_bool(patch.is_unfinished),
        n_pruned=utils.binary_bool(patch.is_pruned),
    )


def map_to_generation_datapoint(patch: AvgPatch) -> GenerationDatapoint:
    n_unfinshed = utils.binary_bool(patch.is_unfinished)
    return GenerationDatapoint(
        gen_time=patch.total_gen_time,
        n_total=1,
        n_unique=utils.binary_bool(not patch.is_duplicate),
        n_unfinished=n_unfinshed,
        n_pruned=utils.binary_bool(patch.is_pruned and not patch.is_unfinished),
    )


def map_to_validation_datapoint(
    patch_info: tuple[AvgPatch, PatchValidationResult]
) -> ValidationDatapoint:
    avg_patch, validation_result = patch_info
    assert not avg_patch.is_duplicate
    gen_datapoint = map_to_generation_datapoint(avg_patch)
    return ValidationDatapoint(
        n_parse_success=utils.binary_bool(
            validation_result.outcome != Outcome.ParseError
        ),
        n_comp_success=utils.binary_bool(
            validation_result.outcome
            not in [Outcome.ParseError, Outcome.CompilationError]
        ),
        n_test_success=utils.binary_bool(validation_result.outcome == Outcome.Success),
        total_time_consumed=gen_datapoint.gen_time + (validation_result.time_cost),
        gen_datapoint=gen_datapoint,
    )
