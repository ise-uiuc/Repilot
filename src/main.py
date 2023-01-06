from realm.repair import Repairer
from realm.config import MetaConfig, RepairConfig, LMInferenceConfig, SynthesisMethod
from pathlib import Path
import os
import argparse
from datetime import datetime
import torch

if torch.cuda.is_available():
    _EMPTY = torch.empty((1,)).cuda()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# def str_hash(s: str) -> int:
#     return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**8

# def compress(patch_group: List[TextFile]) -> int:
#     return str_hash(''.join(re.sub(r'\s+', '', t.content) for t in patch_group))

META_CONFIG = MetaConfig.from_json_file(Path("meta_config.json"))

parser = argparse.ArgumentParser("REALM program repair")
parser.add_argument(
    "-d",
    "--dir",
    required=False,
    default=None,
    help="The directory to store generated data",
)
parser.add_argument(
    "-b",
    "--bug",
    required=True,
    help="Regex representing the bugs to repair",
)
parser.add_argument(
    "-n",
    "--n-samples",
    required=False,
    default=20,
    type=int,
    help="Number of batched samples to generate (#total samples = n_samples * batch_size)",
)
parser.add_argument(
    "--method",
    required=True,
    choices=["pruned-nomem", "pruned-mem", "plain"],
    help="The method to use for patch synthesis",
)
parser.add_argument(
    "--batch-size",
    required=False,
    default=10,
    type=int,
    help="The batch size of the language model",
)
parser.add_argument(
    "--temperature",
    required=False,
    default=1.0,
    type=float,
    help="Temperature for sampling",
)
parser.add_argument(
    "--top-k", required=False, default=50, type=int, help="Top-K value of the sampling"
)
parser.add_argument(
    "--n-max-tokens",
    required=False,
    default=50,
    type=int,
    help="Max number of tokens to generate for each hunk",
)
parser.add_argument(
    "--pre-allocate",
    required=False,
    action="store_true",
    help="Whether to do pre-allocation",
)
args = parser.parse_args()

INFERENCE_CONFIG = LMInferenceConfig(
    args.batch_size, args.temperature, args.top_k, args.n_max_tokens
)

SYNTHESIS_CONFIG = RepairConfig(
    args.n_samples,
    INFERENCE_CONFIG,
    {
        "pruned-nomem": SynthesisMethod.PRUNED_NO_MEM,
        "pruned-mem": SynthesisMethod.PRUNED_MEM,
        "plain": SynthesisMethod.PLAIN,
    }[args.method],
    args.bug,
)


def random_dir(suffix: str) -> str:
    return datetime.now().strftime(f"results-%Y%m%d-%H%M%S-{suffix}")


if __name__ == "__main__":
    result_dir = Path(args.dir if args.dir is not None else random_dir(args.method))
    repairer = Repairer.init(META_CONFIG, result_dir, args.pre_allocate)
    repairer.repair(SYNTHESIS_CONFIG, args.bug)
    if os.getenv("KEEP") is not None:
        print("Type anything to exit")
        input()
