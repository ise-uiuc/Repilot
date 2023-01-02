import argparse
import difflib
import hashlib
import io
import json
import os
import random
import shlex
import shutil
import subprocess
import time
import uuid
from enum import Enum
import multiprocessing as mp
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Dict, List, Set, Tuple, cast

import git
import javalang
import regex as re
import torch
from joblib import Parallel, delayed

from datasets import d4j
from init import data
from realm import utils
from realm.jdt_lsp import JdtLspAnalyzer, Message
from realm.lsp import spec
from realm.lsp.text import TextFile

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
assert shutil.which('defects4j')

# print(Repairer.tokenizer.encode("<s> </s> <pad> <extra_id_0> <extra_id_1> <unk> <mask>", add_special_tokens=False))
# print(Repairer.tokenizer.decode(torch.tensor(range(0, 10)).to('cuda'), clean_up_tokenization_spaces = False))
# exit()
# print(Repairer.tokenizer.tokenize("def long_function(): <extra_id_0> ", padding=True))
# tokens = Repairer.tokenizer.encode("def long_function(): <extra_id_0>", return_tensors='pt', add_special_tokens=True).to('cuda')
# print(tokens)
# tokens = torch.cat((tokens, torch.tensor([[Repairer.END_ID]]).to('cuda')), dim=-1)
# print(Repairer.tokenizer.batch_decode(Repairer.model.generate(tokens, max_new_tokens=25), skip_special_tokens=False))
# exit()


def str_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**8


JDT_LS_REPO = '/home/yuxiang/fastd/Developer/eclipse.jdt.ls/'
D4J_REPO = '/home/yuxiang/Developer/defects4j'

dataset = d4j.Defects4J(D4J_REPO, data)
# model = SpanLM('facebook/incoder-1B', batch_size=N_SAMPLE)
# model = None


def server_cmd(bug_id: str) -> List[str]:
    JDT_REPO = f'{JDT_LS_REPO}/org.eclipse.jdt.ls.product/target/repository'
    return shlex.split(f"java -Declipse.application=org.eclipse.jdt.ls.core.id1 \
        -Dosgi.bundles.defaultStartLevel=4 \
        -Declipse.product=org.eclipse.jdt.ls.core.product \
        -Dlog.level=ALL \
        -noverify \
        -Xmx1G \
        --add-modules=ALL-SYSTEM \
        --add-opens java.base/java.util=ALL-UNNAMED \
        --add-opens java.base/java.lang=ALL-UNNAMED \
        -jar {JDT_REPO}/plugins/org.eclipse.equinox.launcher_1.6.400.v20210924-0641.jar \
        -configuration {JDT_REPO}/config_linux \
        -data .lsp_data/{uuid.uuid4()}")


def wait_until_all_analyzers_free(realm_conns: List[Connection], max_waiting_time: float = 20, free_check_time: float = 1.0):
    batch_is_free = [False] * len(realm_conns)
    start_time = time.perf_counter()
    print('Waiting until all analyzers are free...')
    while time.perf_counter() < start_time + max_waiting_time:
        for idx, realm_conn in enumerate(realm_conns):
            if not batch_is_free[idx]:
                realm_conn.send(Message(True, JdtLspAnalyzer.is_free.__name__, free_check_time))
        
        for idx, realm_conn in enumerate(realm_conns):
            if not batch_is_free[idx]:
                is_free = realm_conn.recv()
                batch_is_free[idx] = is_free
        if all(batch_is_free):
            print('All analyzers are free:', time.perf_counter() - start_time)
            break


def repair_proj(result_dir: Path, bug_id: str, bug: d4j.Bug, n_patch_groups: int = 1) -> List[List[TextFile]]:
    proj, id_str = bug_id.split('-')
    from git.exc import GitCommandError
    try:
        repo = git.Repo(bug.proj_path)
        repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
        subprocess.run(['defects4j', 'checkout', '-p', proj,
                        f'-v{id_str}b', '-w', bug.proj_path])
        repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
        repo.git.execute(['git', 'clean', '-xfd'])
        repo.close()
    except GitCommandError:
        pass

    if os.getenv('PLAIN') is None:
        connection_pairs = cast(
            list[tuple[Connection, Connection]],
            [Pipe(duplex=True) for _ in range(GEN_BATCH_SIZE)]
        )
    # analyzer_conn, realm_conn = cast(
    #     Tuple[Connection, Connection], Pipe(duplex=True))
    meaning_less = utils.Meaningless()
    connection_analyzer_pairs = [(realm_conn, JdtLspAnalyzer(
        analyzer_conn,
        server_cmd(bug_id),
        bug.proj_path,
        cast(str, os.getenv('JAVA8_HOME')),
        verbose=False
    )) for analyzer_conn, realm_conn in connection_pairs] if os.getenv('PLAIN') is None else [
        (cast(Connection, meaning_less), cast(JdtLspAnalyzer, meaning_less))
        for _ in range(GEN_BATCH_SIZE)
    ]
    connections = [_realm_conn for _realm_conn, _ in connection_analyzer_pairs]

    for _, analyzer in connection_analyzer_pairs:
        analyzer.start()

    for _realm_conn, _ in connection_analyzer_pairs:
        _realm_conn.send(Message(False, JdtLspAnalyzer.init.__name__))

    patch_groups: List[List[TextFile]] = []
    time_completion: List[float] = []
    times: List[float] = []
    base_dir = result_dir / proj / id_str
    base_dir.mkdir(exist_ok=False, parents=True)

    # gen.PARTIAL_MEMOIZED.clear()
    # gen.COMPLETE_MEMOIZED.clear()
    # Clear memoization
    memoized: Dict[Tuple[int, int], gen.Memorization] = dict()
    for idx in range(n_patch_groups):
        print('Repair:', idx)
        # There is no writing, so it's not needed
        # if idx != 0:
        #     wait_until_all_analyzers_free(connections, 5, 0.5)
        #     # repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
        #     # TODO: refactor textfile
        #     for buggy_file in bug.buggy_files:
        #         text_file = TextFile(Path(bug.proj_path) / buggy_file.path)
        #         realm_conn.send(
        #             Message(False, JdtLspAnalyzer.change.__name__, text_file))
        #     # wait_until_analyzer_is_free(realm_conn)
        text_files: List[TextFile] = []
        # For each file, generated a patch for each change (in reversed order relative to change)

        logs: List[gen.GenerationLog] = []
        generation_successful = True
        for buggy_file_idx, buggy_file in enumerate(bug.buggy_files):
            text_file = TextFile(Path(bug.proj_path) / buggy_file.path)
            original_text_file = text_file.copy()
            if idx == 0 and os.getenv('PLAIN') is None:
                for _realm_conn, _ in connection_analyzer_pairs:
                    _realm_conn.send(
                        Message(False, JdtLspAnalyzer.open.__name__, text_file))
                wait_until_all_analyzers_free(connections)
            print(len(buggy_file.changes))
            print(buggy_file.path)
            print([(c.start, len(c.removed_lines))
                   for c in reversed(buggy_file.changes)])

            for (change_idx, change) in enumerate(reversed(buggy_file.changes)):
                mem_id = (buggy_file_idx, change_idx)
                mem = memoized.setdefault(mem_id, gen.Memorization.init())

                start = change.start - 1
                end = start + len(change.removed_lines)
                start_pos = text_file.refine_index(start, 0)
                end_pos = text_file.refine_index(end, 0)

                start_index = text_file.form_index(start, 0)
                end_index = text_file.form_index(end, 0)

                # TODO: justify 25
                # prefix_start = text_file.form_index(
                #     max(0, start_pos['line'] - 25), 0)
                prefix_start = 0
                # suffix_end = text_file.form_index(end_pos['line'] + 25, 0)
                suffix_end = len(text_file.content)
                prefix = text_file.content[prefix_start:start_index]
                suffix = '\n' + text_file.content[end_index:suffix_end]

                # prefix(\n)
                # insertion(\n)
                # <cursor:infill>
                # (\n)suffix
                text_file.change([cast(spec.EntireDocumentChange, {
                    'text': '\n',
                    'range': {
                        'start': start_pos,
                        'end': end_pos
                    }
                })])

                text_file.move_cursor(start_index)
                start_cursor = text_file.cursor
                assert prefix.endswith('\n')
                assert text_file.content[text_file.cursor - 1] == '\n'
                assert text_file.content[text_file.cursor] == '\n'

                text_file_batch = [text_file.copy()
                              for _ in range(GEN_BATCH_SIZE - 1)] + [text_file]
                # lm_context = gen.LMContext(
                #     gen.CODET5,
                #     gen.CODET5_TOKENIZER,
                #     gen.codet5_tokenize(prefix, suffix),
                #     gen.CODET5_INFERENCE_CONFIG,
                # )
                lm_context = gen.LMContext(
                    MODEL,
                    prefix,
                    suffix,
                    gen.INFERENCE_CONFIG
                )

                lsp_contexts = [gen.LspContext(
                    text_file,
                    realm_conn 
                ) for text_file, (realm_conn, _) in zip(
                    text_file_batch,
                    connection_analyzer_pairs
                )]

                if os.getenv('DRY') is not None:
                    # How to calculate batch size?
                    samples = MODEL.generate(MODEL.encode(prefix, suffix).repeat(
                        GEN_BATCH_SIZE, 1), max_new_tokens=50, temperature=1.0, )
                    for sample in samples:
                        print(sample)
                    generation_successful = False
                    continue

                repairer = gen.Realm(lm_context, lsp_contexts, GEN_BATCH_SIZE)
                repairer.mem = mem
                start_time = time.perf_counter()
                try:
                    completion_overhead, contexts, generation_log = repairer.repair()
                except TimeoutError:
                    print('Fatal timeout error')
                    with open('timeout-error', 'a') as f:
                        f.write("TIMEOUT\n")
                    completion_overhead = []
                    continue
                # if output is None:
                #     generation_successful = False
                #     break
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                time_completion.extend(completion_overhead)
                # TODO: make this inside the repairer
                memoized[mem_id] = repairer.mem
                # This is always True

                # TODO: change
                # assert (lhs := ''.join(output)) == (
                #     rhs := text_file.content[start_cursor:text_file.cursor]), (lhs, rhs)
                print('Success')
                assert len(contexts) == GEN_BATCH_SIZE
                for context in contexts:
                    print(''.join(context.generated_tokens))
                    print()
                # print(''.join(output))
                # breakpoint()
                print([f'{t:.2f}' for t in times])
                for _realm_conn, _ in connection_analyzer_pairs:
                    _realm_conn.send(
                        Message(False, JdtLspAnalyzer.change.__name__, original_text_file))
            # text_file.write()
            if not generation_successful:
                break
            # TODO: change
            text_files.append(text_file)
            logs.append(generation_log)
        if not generation_successful:
            continue
        patch_groups.append(text_files)
        # TODO: opt
        save_dir = base_dir / str(len(patch_groups))
        save_dir.mkdir(exist_ok=False, parents=True)
        debug_dir = save_dir / 'debug'
        debug_dir.mkdir()
        assert len(text_files) == len(bug.buggy_files)
        for text_file, log, buggy_file_path in zip(
            text_files,
            logs,
            map(lambda b: Path(bug.proj_path) / b.path, bug.buggy_files)
        ):
            with open(save_dir / text_file.path.with_suffix('.json').name, 'w') as f:
                json.dump({
                    'path': str(text_file.path.absolute()),
                    'cursor': text_file.cursor,
                    'content': text_file.content,
                }, f, indent=2)
            with open(buggy_file_path) as f:
                buggy_file_lines = f.readlines()
            unified_diff = difflib.unified_diff(
                buggy_file_lines,
                text_file.content.splitlines(keepends=True),
                fromfile='bug',
                tofile='patch'
            )
            with open(debug_dir / buggy_file_path.with_suffix('.diff').name, 'w') as f:
                f.writelines(unified_diff)
            with open(debug_dir / buggy_file_path.with_suffix('.log.json').name, 'w') as f:
                json.dump(log, f, indent=2)
            # with open()
    with open(base_dir / 'time.json', 'w') as f:
        json.dump({
            'completion': time_completion,
            'times': times,
        }, f, indent=2)

    for realm_conn, analyzer in connection_analyzer_pairs:
        realm_conn.send(None)
        analyzer.join()
    return patch_groups

    # repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
    # repo.git.execute(['defects4j', 'checkout', '-p', proj,
    #                  f'-v{id_str}f', '-w', bug.proj_path])
    # repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
    # for change, patch in best_group:
    #     patch.change([change])
    #     patch.write()
    # repo.git.execute(['git', 'clean', '-dfx'])
    # repo.close()
    # # for text_file in text_files:
    #     text_file.write()


def compress(patch_group: List[TextFile]) -> int:
    return str_hash(''.join(re.sub(r'\s+', '', t.content) for t in patch_group))


class ValResult(Enum):
    ParseError = 0
    CompilationError = 1
    TestingError = 2
    Success = 3


def validate_proj(bug_id: str, bug: d4j.Bug, patch_group: List[TextFile]) -> Tuple[ValResult, str, str]:
    proj, id_str = bug_id.split('-')
    java8_home = cast(str, os.getenv('JAVA8_HOME'))
    env = dict(os.environ, JAVA_HOME=java8_home)
    repo = git.Repo(bug.proj_path)
    repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
    subprocess.run(['defects4j', 'checkout', '-p', proj,
                    f'-v{id_str}f', '-w', bug.proj_path])
    repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
    repo.git.execute(['git', 'clean', '-xfd'])

    for patch_file in patch_group:
        try:
            javalang.parse.parse(patch_file.content)
        except (javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError) as e:
            return ValResult.ParseError, '', str(e)
        except Exception as e:
            with open('unexpected_exception', 'a') as f:
                f.write(str(type(e)))
                f.write('\n')
                f.write(str(e))
        patch_file.write()
    result = subprocess.run(['defects4j', 'compile'],
                            env=env, cwd=bug.proj_path, text=True, capture_output=True)
    if result.returncode != 0:
        return ValResult.CompilationError, result.stdout, result.stderr
    # Filter out candidate patches (not even plausible)
    try:
        subprocess.run(['defects4j', 'test'], env=env,
                       cwd=bug.proj_path, timeout=120)
        failing_tests = Path(bug.proj_path) / 'failing_tests'
        assert failing_tests.exists()
        with open(failing_tests) as f:
            return (ValResult.Success if f.read().strip() == '' else ValResult.TestingError), '', ''
    except subprocess.TimeoutExpired:
        return ValResult.TestingError, '', ''


def get_patch_groups(bug_dir: Path) -> List[Tuple[str, List[TextFile]]]:
    result: List[Tuple[str, List[TextFile]]] = []
    for patch_group_dir in sorted(list(filter(Path.is_dir, bug_dir.iterdir())), key=lambda p: int(str(p.stem))):
        files: List[TextFile] = []
        for json_file in filter(Path.is_file, patch_group_dir.iterdir()):
            assert json_file.name.endswith('.json')
            with open(json_file) as f:
                data = json.load(f)
            text_file = TextFile(data['path'], data['content'])
            text_file.move_cursor(data['cursor'])
            files.append(text_file)
        result.append((patch_group_dir.name, files))
    return result


def do_validation(bug_dir: Path, bug_id: str, bug: d4j.Bug) -> dict:
    parse_failed_patches: Set[int] = set()
    comp_failed_patches: Set[int] = set()
    test_failed_patches: Set[int] = set()
    succeeded_patches: Set[int] = set()
    patch_groups = get_patch_groups(bug_dir)
    result: dict = {
        # 'with_lsp': {
        'succeeded': [],
        'parse_failed': [],
        'comp_failed': [],
        'test_failed': [],
        'dup_succeeded': [],
        'dup_parse_failed': [],
        'dup_test_failed': [],
        'dup_comp_failed': [],
        'times': {},
        'stdout': {},
        'stderr': {},
        # },
        # 'without_lsp': {
        #     'succeeded': [],
        #     'comp_failed': [],
        #     'test_failed': [],
        #     'dup_succeeded': [],
        #     'dup_test_failed': [],
        #     'dup_comp_failed': [],
        # }
    }
    # half = len(patch_groups) // 2
    for idx, patch_group in patch_groups:
        hash = compress(patch_group)
        if hash in comp_failed_patches:
            # print(bug_id, idx, 'is duplicated')
            result['dup_comp_failed'].append(int(idx))
            continue
        if hash in test_failed_patches:
            # print(bug_id, idx, 'is duplicated')
            result['dup_test_failed'].append(int(idx))
            continue
        if hash in succeeded_patches:
            # print(bug_id, idx, 'is duplicated')
            result['dup_succeeded'].append(int(idx))
            continue
        start_time = time.perf_counter()
        val_result, stdout, stderr = validate_proj(bug_id, bug, patch_group)
        cost = time.perf_counter() - start_time
        result['times'][idx] = cost
        result['stdout'][idx] = stdout
        result['stderr'][idx] = stderr
        match val_result:
            case ValResult.ParseError:
                result['parse_failed'].append(int(idx))
                parse_failed_patches.add(hash)
            case ValResult.CompilationError:
                result['comp_failed'].append(int(idx))
                comp_failed_patches.add(hash)
            case ValResult.TestingError:
                result['test_failed'].append(int(idx))
                test_failed_patches.add(hash)
            case ValResult.Success:
                result['succeeded'].append(int(idx))
                succeeded_patches.add(hash)

    # # Do it again
    # comp_failed_patches = set()
    # test_failed_patches = set()
    # succeeded_patches = set()
    # for idx, patch_group in patch_groups[half:]:
    #     hash = compress(patch_group)
    #     if hash in comp_failed_patches:
    #         # print(bug_id, idx, 'is duplicated')
    #         result['without_lsp']['dup_comp_failed'].append(int(idx) - half)
    #         comp_failed_patches.add(hash)
    #         continue
    #     if hash in test_failed_patches:
    #         # print(bug_id, idx, 'is duplicated')
    #         result['without_lsp']['dup_test_failed'].append(int(idx) - half)
    #         test_failed_patches.add(hash)
    #         continue
    #     if hash in succeeded_patches:
    #         # print(bug_id, idx, 'is duplicated')
    #         result['without_lsp']['dup_succeeded'].append(int(idx) - half)
    #         succeeded_patches.add(hash)
    #         continue
    #     val_result = validate_proj(bug_id, bug, patch_group)
    #     match val_result:
    #         case ValResult.CompilationError:
    #             result['without_lsp']['comp_failed'].append(int(idx) - half)
    #             comp_failed_patches.add(hash)
    #         case ValResult.TestingError:
    #             result['without_lsp']['test_failed'].append(int(idx) - half)
    #             test_failed_patches.add(hash)
    #         case ValResult.Success:
    #             result['without_lsp']['succeeded'].append(int(idx) - half)
    #             succeeded_patches.add(hash)
    return result


VAL_BATCH_SIZE = int(os.getenv('VAL_BATCH_SIZE', 3))
GEN_BATCH_SIZE = int(os.getenv('GEN_BATCH_SIZE', 1))


def validate_all_bugs(all_bugs: dict, proj_dir: Path):
    ret: dict = {}
    bug_dirs: List[Path] = sorted(
        list(filter(Path.is_dir, proj_dir.iterdir())), key=lambda p: int(str(p.stem)))
    for bug_dir_batch in utils.chunked(VAL_BATCH_SIZE, bug_dirs):
        params_list = [(bug_dir, (bug_id := proj_dir.stem + '-' + bug_dir.stem),
                        all_bugs[bug_id]) for bug_dir in bug_dir_batch]
        results: List[dict] = Parallel(n_jobs=VAL_BATCH_SIZE)(
            delayed(do_validation)(*params) for params in params_list)
        for (_, bug_id, _), result in zip(params_list, results):
            ret[bug_id] = result
        with open(proj_dir / proj_dir.with_suffix('.json').name, 'w') as f:
            json.dump(ret, f, indent=2)


def log_repo(f: io.TextIOBase, tag: str, repo: git.Repo):
    f.write(f'{tag} GIT HASH: {repo.head.object.hexsha}\n')
    f.write(f'{tag} GIT STATUS: ')
    f.write(
        '\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
    f.write(repo.git.status())
    f.write(
        '\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n')


parser = argparse.ArgumentParser('REALM program repair')
parser.add_argument('-d', '--dir', required=False, default=None,
                    help='The directory to store generated data')
parser.add_argument('-b', '--bug', required=False,
                    default=None, help='The bug to repair')
args = parser.parse_args()

if __name__ == '__main__':
    assert os.getenv('JAVA8_HOME')

    if os.getenv('VAL') is not None:
        assert args.dir is not None, "Directory should be explicitly specified"
        result_dir = Path(args.dir)
        assert result_dir.exists()
        with open(result_dir / 'val_meta.txt', 'w') as f:
            log_repo(f, 'Defects4J', git.Repo(Path(D4J_REPO)))
        all_bugs = dataset.all_bugs()
        all_results: dict = {}
        for proj_dir in filter(Path.is_dir, result_dir.iterdir()):
            validate_all_bugs(all_bugs, proj_dir)
            # all_results = dict(all_results, **result)
        exit()
    
    # mp.set_start_method('spawn')

    from realm import generation as gen
    from realm.generation_defs import MODEL
    torch.manual_seed(0)
    random.seed(0)
    result_dir = Path(
        f'results-{uuid.uuid4()}' if args.dir is None else args.dir)
    if os.getenv('DEBUG') is not None:
        result_dir = Path('../results') / 'temp' / result_dir
    result_dir.mkdir(exist_ok=False, parents=True)
    with open(result_dir / 'gen_meta.txt', 'w') as f:
        log_repo(f, 'Repair tool', git.Repo(Path('.')))
        log_repo(f, 'Defects4J', git.Repo(Path(D4J_REPO)))
        log_repo(f, 'Language server', git.Repo(Path(JDT_LS_REPO)))
    with open(result_dir / 'args.txt', 'w') as f:
        f.write(str(gen.INFERENCE_CONFIG))
    print(f'Metadata will be saved in {result_dir}')

    # def is_single_hunk(bug: d4j.Bug) -> bool:
    #     if len(bug.buggy_files) == 1 and len(changes := bug.buggy_files[0].changes) == 1:
    #         change = changes[0]
    #     else:
    #         return False

    # Get single hunk bugs
    all_bugs = dataset.all_bugs()
    single_hunk_bugs = {
        id: bug for (id, bug) in all_bugs.items()
        # TODO
        if len(bug.buggy_files) == 1 and len(bug.buggy_files[0].changes) == 1
    }

    assert args.bug is not None

    for bug_id, bug in single_hunk_bugs.items():
        proj = bug_id.split('-')[0]
        # Unicode error
        if bug_id == 'Lang-25':
            continue
        # if proj in proj_accessed or proj == 'Mockito':
        # TODO (DONE): IMPORTANT!!!!! Memorize multiple changes when doing repair
        if not bug_id.startswith(args.bug):
            continue
        # if int(bug_id.split('-')[1]) < 115:
        #     continue
        # if bug_id == 'Math-1':
        #     continue
        # proj_accessed.add(proj)
        # if bug_id != 'Mockito-1':
        #     continue
        print(bug_id)
        gen.CHART_11 = bug_id == 'Chart-11'
        n_samples = 200 if (n_samples_str := os.getenv(
            'N_SAMPLES')) is None else int(n_samples_str)
        patch_groups = repair_proj(result_dir, bug_id, bug, n_samples)
        # candidate_patch_groups: List[int] = []
        # for idx, patch_group in enumerate(patch_groups):
        #     if validate_proj(bug_id, bug, patch_group):
        #         candidate_patch_groups.append(idx)
        # with open('result.log', 'a') as f:
        #     f.writelines(
        #         [str(bug_id), ' : ', f'{len(candidate_patch_groups)} / {len(patch_groups)}'])

    # file_path = Path(
    #     '/home/yuxiang/Developer/d4j-checkout/Lang-1-buggy/src/main/java/org/apache/commons/lang3/Validate.java')
    # with open(file_path) as f:
    #     content = TextDocument(f.read())
