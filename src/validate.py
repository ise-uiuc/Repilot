from realm.report import Reporter

# from realm.analyze_result import repair_analysis
# from realm.validate import Validator
from pathlib import Path
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("REALM: validate repair results")
    parser.add_argument(
        "-d",
        "--dir",
        required=False,
        default=None,
        help="The directory that stores the repair results",
    )
    parser.add_argument(
        "-b",
        "--bug",
        required=False,
        default=".*",
        help="Regex representing the bug pattern to validate",
    )
    parser.add_argument(
        "--repair-idx",
        required=False,
        default=".*",
        help="Regex representing which repair collection to validate",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = Path(args.dir)
    # validator = Validator.init(path)
    # validator.validate(args.bug, args.repair_idx)
    reporter = Reporter.load(path)
    reporter.analyze()
    reporter.save()
    reporter.save()
    reporter.save()
    path = Path("results-test")
    path.mkdir(exist_ok=True)
    for r in reporter.repair_result.results:
        print(type(r))
        for _, v in r.items():
            for x in (c for b in v for c in b):
                for t in x.results:
                    t.is_dumpped = False
    reporter.dump(path)
    # print(f"RepairReporter loaded from {args.dir}")
    # a_reporter = repair_analysis(reporter)
    # a_reporter.save()

# class ValResult(Enum):
#     ParseError = 0
#     CompilationError = 1
#     TestingError = 2
#     Success = 3


# def validate_proj(bug_id: str, bug: d4j.Bug, patch_group: List[TextFile]) -> Tuple[ValResult, str, str]:
#     proj, id_str = bug_id.split('-')
#     java8_home = cast(str, os.getenv('JAVA8_HOME'))
#     env = dict(os.environ, JAVA_HOME=java8_home)
#     repo = git.Repo(bug.proj_path)
#     repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
#     subprocess.run(['defects4j', 'checkout', '-p', proj,
#                     f'-v{id_str}f', '-w', bug.proj_path])
#     repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
#     repo.git.execute(['git', 'clean', '-xfd'])

#     for patch_file in patch_group:
#         try:
#             javalang.parse.parse(patch_file.content)
#         except (javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError) as e:
#             return ValResult.ParseError, '', str(e)
#         except Exception as e:
#             with open('unexpected_exception', 'a') as f:
#                 f.write(str(type(e)))
#                 f.write('\n')
#                 f.write(str(e))
#         patch_file.write()
#     result = subprocess.run(['defects4j', 'compile'],
#                             env=env, cwd=bug.proj_path, text=True, capture_output=True)
#     if result.returncode != 0:
#         return ValResult.CompilationError, result.stdout, result.stderr
#     # Filter out candidate patches (not even plausible)
#     try:
#         subprocess.run(['defects4j', 'test'], env=env,
#                        cwd=bug.proj_path, timeout=120)
#         failing_tests = Path(bug.proj_path) / 'failing_tests'
#         assert failing_tests.exists()
#         with open(failing_tests) as f:
#             return (ValResult.Success if f.read().strip() == '' else ValResult.TestingError), '', ''
#     except subprocess.TimeoutExpired:
#         return ValResult.TestingError, '', ''


# def get_patch_groups(bug_dir: Path) -> List[Tuple[str, List[TextFile]]]:
#     result: List[Tuple[str, List[TextFile]]] = []
#     for patch_group_dir in sorted(list(filter(Path.is_dir, bug_dir.iterdir())), key=lambda p: int(str(p.stem))):
#         files: List[TextFile] = []
#         for json_file in filter(Path.is_file, patch_group_dir.iterdir()):
#             assert json_file.name.endswith('.json')
#             with open(json_file) as f:
#                 data = json.load(f)
#             text_file = TextFile(data['path'], data['content'])
#             text_file.move_cursor(data['cursor'])
#             files.append(text_file)
#         result.append((patch_group_dir.name, files))
#     return result


# def do_validation(bug_dir: Path, bug_id: str, bug: d4j.Bug) -> dict:
#     parse_failed_patches: Set[int] = set()
#     comp_failed_patches: Set[int] = set()
#     test_failed_patches: Set[int] = set()
#     succeeded_patches: Set[int] = set()
#     patch_groups = get_patch_groups(bug_dir)
#     result: dict = {
#         # 'with_lsp': {
#         'succeeded': [],
#         'parse_failed': [],
#         'comp_failed': [],
#         'test_failed': [],
#         'dup_succeeded': [],
#         'dup_parse_failed': [],
#         'dup_test_failed': [],
#         'dup_comp_failed': [],
#         'times': {},
#         'stdout': {},
#         'stderr': {},
#         # },
#         # 'without_lsp': {
#         #     'succeeded': [],
#         #     'comp_failed': [],
#         #     'test_failed': [],
#         #     'dup_succeeded': [],
#         #     'dup_test_failed': [],
#         #     'dup_comp_failed': [],
#         # }
#     }
#     # half = len(patch_groups) // 2
#     for idx, patch_group in patch_groups:
#         hash = compress(patch_group)
#         if hash in comp_failed_patches:
#             # print(bug_id, idx, 'is duplicated')
#             result['dup_comp_failed'].append(int(idx))
#             continue
#         if hash in test_failed_patches:
#             # print(bug_id, idx, 'is duplicated')
#             result['dup_test_failed'].append(int(idx))
#             continue
#         if hash in succeeded_patches:
#             # print(bug_id, idx, 'is duplicated')
#             result['dup_succeeded'].append(int(idx))
#             continue
#         start_time = time.perf_counter()
#         val_result, stdout, stderr = validate_proj(bug_id, bug, patch_group)
#         cost = time.perf_counter() - start_time
#         result['times'][idx] = cost
#         result['stdout'][idx] = stdout
#         result['stderr'][idx] = stderr
#         match val_result:
#             case ValResult.ParseError:
#                 result['parse_failed'].append(int(idx))
#                 parse_failed_patches.add(hash)
#             case ValResult.CompilationError:
#                 result['comp_failed'].append(int(idx))
#                 comp_failed_patches.add(hash)
#             case ValResult.TestingError:
#                 result['test_failed'].append(int(idx))
#                 test_failed_patches.add(hash)
#             case ValResult.Success:
#                 result['succeeded'].append(int(idx))
#                 succeeded_patches.add(hash)

#     # # Do it again
#     # comp_failed_patches = set()
#     # test_failed_patches = set()
#     # succeeded_patches = set()
#     # for idx, patch_group in patch_groups[half:]:
#     #     hash = compress(patch_group)
#     #     if hash in comp_failed_patches:
#     #         # print(bug_id, idx, 'is duplicated')
#     #         result['without_lsp']['dup_comp_failed'].append(int(idx) - half)
#     #         comp_failed_patches.add(hash)
#     #         continue
#     #     if hash in test_failed_patches:
#     #         # print(bug_id, idx, 'is duplicated')
#     #         result['without_lsp']['dup_test_failed'].append(int(idx) - half)
#     #         test_failed_patches.add(hash)
#     #         continue
#     #     if hash in succeeded_patches:
#     #         # print(bug_id, idx, 'is duplicated')
#     #         result['without_lsp']['dup_succeeded'].append(int(idx) - half)
#     #         succeeded_patches.add(hash)
#     #         continue
#     #     val_result = validate_proj(bug_id, bug, patch_group)
#     #     match val_result:
#     #         case ValResult.CompilationError:
#     #             result['without_lsp']['comp_failed'].append(int(idx) - half)
#     #             comp_failed_patches.add(hash)
#     #         case ValResult.TestingError:
#     #             result['without_lsp']['test_failed'].append(int(idx) - half)
#     #             test_failed_patches.add(hash)
#     #         case ValResult.Success:
#     #             result['without_lsp']['succeeded'].append(int(idx) - half)
#     #             succeeded_patches.add(hash)
#     return result


# VAL_BATCH_SIZE = int(os.getenv('VAL_BATCH_SIZE', 3))
# GEN_BATCH_SIZE = int(os.getenv('GEN_BATCH_SIZE', 1))


# def validate_all_bugs(all_bugs: dict, proj_dir: Path):
#     ret: dict = {}
#     bug_dirs: List[Path] = sorted(
#         list(filter(Path.is_dir, proj_dir.iterdir())), key=lambda p: int(str(p.stem)))
#     for bug_dir_batch in utils.chunked(VAL_BATCH_SIZE, bug_dirs):
#         params_list = [(bug_dir, (bug_id := proj_dir.stem + '-' + bug_dir.stem),
#                         all_bugs[bug_id]) for bug_dir in bug_dir_batch]
#         results: List[dict] = Parallel(n_jobs=VAL_BATCH_SIZE)(
#             delayed(do_validation)(*params) for params in params_list)
#         for (_, bug_id, _), result in zip(params_list, results):
#             ret[bug_id] = result
#         with open(proj_dir / proj_dir.with_suffix('.json').name, 'w') as f:
#             json.dump(ret, f, indent=2)

# if os.getenv('VAL') is not None:
#     assert args.dir is not None, "Directory should be explicitly specified"
#     result_dir = Path(args.dir)
#     assert result_dir.exists()
#     with open(result_dir / 'val_meta.txt', 'w') as f:
#         log_repo(f, 'Defects4J', git.Repo(Path(D4J_REPO)))
#     all_bugs = dataset.all_bugs()
#     all_results: dict = {}
#     for proj_dir in filter(Path.is_dir, result_dir.iterdir()):
#         validate_all_bugs(all_bugs, proj_dir)
#         # all_results = dict(all_results, **result)
#     exit()
