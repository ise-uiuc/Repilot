from realm.repair import Repairer
from realm.config import MetaConfig, SynthesisConfig, LMInferenceConfig, SynthesisMethod
from realm.model import CodeT5Large
from realm import utils
from pathlib import Path
import os
import shutil
import argparse
import uuid


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Checks
assert shutil.which("defects4j")
assert os.getenv("JAVA8_HOME")

# def str_hash(s: str) -> int:
#     return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**8

# def compress(patch_group: List[TextFile]) -> int:
#     return str_hash(''.join(re.sub(r'\s+', '', t.content) for t in patch_group))

META_CONFIG = MetaConfig(
    d4j_home=Path("/home/yuxiang/Developer/defects4j"),
    d4j_checkout_root=Path("/JawTitan/yuxiang-data/Developer/d4j-checkout"),
    jdt_ls_repo=Path("/home/yuxiang/fastd/Developer/eclipse.jdt.ls/"),
    seed=0,
)


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
args = parser.parse_args()

INFERENCE_CONFIG = LMInferenceConfig(
    args.batch_size, args.temperature, args.top_k, args.n_max_tokens
)

SYNTHESIS_CONFIG = SynthesisConfig(
    args.n_samples,
    INFERENCE_CONFIG,
    {
        "pruned-nomem": SynthesisMethod.PRUNED_NO_MEM,
        "pruned-mem": SynthesisMethod.PRUNED_MEM,
        "plain": SynthesisMethod.PLAIN,
    }[args.method],
)

if __name__ == "__main__":
    result_dir = Path(f"results-{uuid.uuid4()}" if args.dir is None else args.dir)
    if os.getenv("DEBUG") is not None:
        result_dir = Path("../results") / "temp" / result_dir
    result_dir.parent.mkdir(exist_ok=True, parents=True)

    repairer = Repairer.init(META_CONFIG, result_dir)
    repairer.repair(SYNTHESIS_CONFIG, args.bug)
