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

INFERENCE_CONFIG = LMInferenceConfig(1, 1.0, 50, 50)

SYNTHESIS_CONFIG = SynthesisConfig(
    10,
    INFERENCE_CONFIG,
    SynthesisMethod.PRUNED_MEM,
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
    "-b", "--bug", required=False, default=None, help="The bug to repair"
)
args = parser.parse_args()

if __name__ == "__main__":
    result_dir = Path(f"results-{uuid.uuid4()}" if args.dir is None else args.dir)
    if os.getenv("DEBUG") is not None:
        result_dir = Path("../results") / "temp" / result_dir
    result_dir.parent.mkdir(exist_ok=True, parents=True)

    model = CodeT5Large.init().to(utils.DEVICE) # type: ignore # noqa
    repairer = Repairer.init(META_CONFIG, result_dir)
    repairer.repair(SYNTHESIS_CONFIG, args.bug)
    