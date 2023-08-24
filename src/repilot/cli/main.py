import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar

from repilot.config import (
    LMInferenceConfig,
    MetaConfig,
    RepairConfig,
    SynthesisMethod,
    ValidationConfig,
)
from repilot.evaluation import evaluate_runners
from repilot.repair import Repairer
from repilot.results import ValidationCache, concat_hunks
from repilot.runner import Runner

parser = ArgumentParser("The Repilot Patch Generation Tool")
subparsers = parser.add_subparsers(title="Repilot runner options", dest="option")

# Repair parser
repair_parser = subparsers.add_parser("repair")
repair_parser.add_argument(
    "-d",
    "--dir",
    required=False,
    default=None,
    help="The directory to store generated data",
)
repair_parser.add_argument(
    "-b",
    "--bug-pattern",
    required=True,
    help="Regex representing the bugs to repair",
)
repair_parser.add_argument(
    "-n",
    "--n-samples",
    required=False,
    default=20,
    type=int,
    help="Number of batched samples to generate (#total samples = n_samples * batch_size)",
)
repair_parser.add_argument(
    "--method",
    required=True,
    choices=["pruned-nomem", "pruned-mem", "plain"],
    help="The method to use for patch synthesis",
)
repair_parser.add_argument(
    "--batch-size",
    required=False,
    default=1,
    type=int,
    help="The batch size of the language model",
)
repair_parser.add_argument(
    "--temperature",
    required=False,
    default=1.0,
    type=float,
    help="Temperature for sampling",
)
repair_parser.add_argument(
    "--top-k", required=False, default=50, type=int, help="Top-K value of the sampling"
)
repair_parser.add_argument(
    "--n-max-tokens",
    required=False,
    default=50,
    type=int,
    help="Max number of tokens to generate for each hunk",
)
repair_parser.add_argument(
    "--not-single-hunk-only",
    required=False,
    action="store_true",
    help="Whether enable multi-hunk multi-file repair",
)
repair_parser.add_argument(
    "--pre-allocate",
    required=False,
    action="store_true",
    help="Whether to do pre-allocation",
)

# The capability to resume a terminated repair process
resuming_parser = subparsers.add_parser("resume")
resuming_parser.add_argument(
    "-d",
    "--dir",
    required=True,
    help="The directory that stores the repair results",
)
resuming_parser.add_argument(
    "--pre-allocate",
    required=False,
    action="store_true",
    help="Whether to do pre-allocation",
)

# Analysis parser
transformation_parser = subparsers.add_parser(
    "transform", help="Transform the hunk-based repair result to a per bug based one"
)
transformation_parser.add_argument(
    "-d",
    "--dir",
    required=True,
    help="The directory that stores the repair results",
)

# Validation parser
validation_parser = subparsers.add_parser("validate")
validation_parser.add_argument(
    "-d",
    "--dir",
    required=True,
    help="The directory that stores the repair results",
)
validation_parser.add_argument(
    "-b",
    "--bug-pattern",
    required=False,
    default=".*",
    help="Regex representing the bug pattern to validate",
)
validation_parser.add_argument(
    "--n-cores",
    required=False,
    default=1,
    type=int,
    help="Number of cores to use for validation",
)
validation_parser.add_argument(
    "--cache-save-path",
    required=False,
)

# Evaluation parser
evaluation_parser = subparsers.add_parser("evaluate")
evaluation_parser.add_argument(
    "-d",
    "--dirs",
    required=True,
    nargs="+",
    help="The directories for comparison that store the repair results",
)

cache_parser = subparsers.add_parser("cache", help="Cache validation results")
cache_parser.add_argument(
    "-d",
    "--dirs",
    required=True,
    nargs="+",
    help="The directories for comparison that store the repair results",
)
cache_parser.add_argument(
    "--cache-save-path",
    required=True,
)

args = parser.parse_args()

META_CONFIG_PATH = os.getenv("META_CONFIG_PATH", "meta_config.json")
META_CONFIG = MetaConfig.from_json_file(Path(META_CONFIG_PATH))


def random_dir(config: RepairConfig) -> str:
    return datetime.now().strftime(
        f"results-%Y%m%d-%H%M%S-{config.method}-{config.batch_size}_{config.n_samples * config.batch_size}-{config.bug_pattern}"
    )


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def run(
    runner_creator: Callable[[U], V],
    action: Callable[[U, V], T],
    arg: U,
) -> T:
    runner = runner_creator(arg)
    return action(arg, runner)


def run_by_loading(action: Callable[[Namespace, Runner], T], args: Namespace) -> T:
    def load_runner(args: Namespace) -> Runner:
        return Runner.load(Path(args.dir), META_CONFIG)

    return run(load_runner, action, args)


def repair(args: Namespace) -> None:
    def create_repair_runner(args: Namespace) -> Runner:
        inference_config = LMInferenceConfig(
            args.batch_size, args.temperature, args.top_k, args.n_max_tokens
        )
        repair_config = RepairConfig(
            args.n_samples,
            inference_config,
            {
                "pruned-nomem": SynthesisMethod.PRUNED_NO_MEM,
                "pruned-mem": SynthesisMethod.PRUNED_MEM,
                "plain": SynthesisMethod.PLAIN,
            }[args.method],
            args.bug_pattern,
            not args.not_single_hunk_only,
        )
        result_dir = Path(
            args.dir if args.dir is not None else random_dir(repair_config)
        )
        runner = Runner.create(result_dir, META_CONFIG, repair_config)
        return runner

    def repair_action(args: Namespace, runner: Runner) -> None:
        repairer = Repairer.init(runner.report.config, args.pre_allocate)
        runner.repair(repairer)

    return run(create_repair_runner, repair_action, args)


def resume(args: Namespace) -> None:
    def action(args: Namespace, runner: Runner) -> None:
        repairer = Repairer.init(runner.report.config, args.pre_allocate)
        runner.repair(repairer)

    return run_by_loading(action, args)


def transform(args: Namespace) -> None:
    return run_by_loading(lambda _, runner: Runner.transform(runner), args)


def validate(args: Namespace) -> None:
    cache_save_path = args.cache_save_path
    if cache_save_path is not None:
        print("Loading cache")
        cache: ValidationCache | None = ValidationCache.from_json_file(
            Path(cache_save_path)
        )
        print("Done")
    else:
        cache = None
    return run_by_loading(
        lambda args, runner: Runner.validate(
            runner, ValidationConfig(args.n_cores, args.bug_pattern), cache
        ),
        args,
    )


def evaluate(args: Namespace):
    runners = [Runner.load(Path(dir), META_CONFIG) for dir in args.dirs]
    evaluate_runners(runners, [])


def save_validation_cache(args: Namespace):
    runners = [Runner.load(Path(dir), META_CONFIG) for dir in args.dirs]
    f_cache = Path(args.cache_save_path)
    if f_cache.exists():
        data = ValidationCache.from_json_file(f_cache)
    else:
        data = ValidationCache({})
    for runner in runners:
        for bug_id, patches in runner.get_validation_items():
            bug_id_dict = data.result_dict.setdefault(bug_id, {})
            for avg_patch, val_result in patches:
                if val_result is None:
                    continue
                concat_hunk_str = concat_hunks(avg_patch.file_patches)
                bug_id_dict.setdefault(concat_hunk_str, val_result)
    data.save_json(f_cache)


if __name__ == "__main__":
    if args.option == "repair":
        repair(args)
    elif args.option == "transform":
        transform(args)
    elif args.option == "validate":
        validate(args)
    elif args.option == "resume":
        resume(args)
    elif args.option == "evaluate":
        evaluate(args)
    elif args.option == "cache":
        save_validation_cache(args)
    else:
        assert False
