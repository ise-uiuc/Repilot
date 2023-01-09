import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, TypeVar

from realm.config import (
    LMInferenceConfig,
    MetaConfig,
    RepairConfig,
    SynthesisMethod,
    ValidationConfig,
)
from realm.repair import Repairer
from realm.runner import Runner

parser = ArgumentParser("REALM program repair")
subparsers = parser.add_subparsers(title="REALM runner options", dest="option")

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
    default=10,
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

# Evaluation parser
evaluation_parser = subparsers.add_parser("evaluate")
evaluation_parser.add_argument(
    "-d",
    "--dir",
    required=True,
    help="The directory that stores the repair results",
)


args = parser.parse_args()

META_CONFIG = MetaConfig.from_json_file(Path("meta_config.json"))


def random_dir(suffix: str) -> str:
    return datetime.now().strftime(f"results-%Y%m%d-%H%M%S-{suffix}")


def create_repair_runner(args: Namespace) -> Runner:
    result_dir = Path(args.dir if args.dir is not None else random_dir(args.method))
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
    runner = Runner.create(result_dir, META_CONFIG, repair_config)
    return runner


T = TypeVar("T")


def run_with_runner_action_and_args(
    runner_creator: Callable[[Namespace], Runner],
    action: Callable[[Namespace, Runner], T],
    args: Namespace,
) -> T:
    runner = runner_creator(args)
    return action(args, runner)


def load_runner_from_path(args: Namespace) -> Runner:
    return Runner.load(Path(args.dir))


run_with_action_and_args = partial(
    run_with_runner_action_and_args, load_runner_from_path
)


repair = partial(
    run_with_runner_action_and_args,
    create_repair_runner,
    lambda args, runner: Runner.repair(
        runner, Repairer.init(META_CONFIG, args.pre_allocate)
    ),
)

evaluate_generation = partial(
    run_with_action_and_args, lambda _, runner: Runner.evaluate_generation(runner)
)

transform = partial(
    run_with_action_and_args, lambda runner, _: Runner.transform(runner)
)

validate = partial(
    run_with_action_and_args,
    lambda args, runner: Runner.validate(
        runner, ValidationConfig(args.n_cores, args.bug_pattern)
    ),
)

resume = partial(
    run_with_action_and_args,
    lambda args, runner: Runner.repair(
        runner, Repairer.init(runner.report.config, args.pre_allocate)
    ),
)


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
        import json

        result = evaluate_generation(args)

        # print(
        #     json.dumps(
        #         {
        #             bug_id: [point.to_json() for point in points]
        #             for bug_id, points in result.items()
        #         },
        #         indent=2,
        #     )
        # )
        # print(result["Chart-1"])
    else:
        assert False
