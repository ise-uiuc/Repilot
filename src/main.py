from realm.lsp import TextDocument
from realm.syntax import reduce

# doc = TextDocument(r'System.out.println("Hello " + "world")')
# reduce(doc, raise_unexpected_exc=True)
# print(doc)
# exit()

import difflib
import multiprocessing as mp
from enum import Enum
import uuid
from joblib import Parallel, delayed
import regex as re
import time
import torch
import random
import os
import shlex
from pathlib import Path
import shutil
import subprocess
from realm import generation, utils
from realm.analyze.jdt_lsp import JdtLspAnalyzer
from realm.lsp import TextDocument, client, spec
import json
from typing import Dict, Generator, List, Set, Tuple, cast
from init import data
from unidiff import PatchSet
import git
import hashlib

from realm.lsp.spec import TextChange
from realm.lsp.text import MutableTextDocument, TextFile
from datasets import d4j

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


dataset = d4j.Defects4J('/home/yuxiang/Developer/defects4j', data)
# model = SpanLM('facebook/incoder-1B', batch_size=N_SAMPLE)
model = None


def server_cmd(bug_id: str) -> List[str]:
    return shlex.split(f"/home/yuxiang/Developer/jdt-lsp/bin/jdtls \
        -configuration /home/yuxiang/.cache/jdtls \
        -data .lsp_data/{uuid.uuid4()}")


def repair_proj(result_dir: Path, bug_id: str, bug: d4j.Bug, n_patch_groups: int = 1) -> List[List[TextFile]]:
    proj, id_str = bug_id.split('-')
    repo = git.Repo(bug.proj_path)
    repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
    subprocess.run(['defects4j', 'checkout', '-p', proj,
                    f'-v{id_str}b', '-w', bug.proj_path])
    repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
    repo.git.execute(['git', 'clean', '-xfd'])

    analyzer = JdtLspAnalyzer(
        server_cmd(bug_id),
        bug.proj_path,
        cast(str, os.getenv('JAVA8_HOME')),
        verbose=False
    )

    analyzer.init_client()
    analyzer.init()

    patch_groups: List[List[TextFile]] = []
    time_completion: List[float] = []
    times: List[float] = []
    base_dir = result_dir / proj / id_str
    base_dir.mkdir(exist_ok=False, parents=True)

    # generation.PARTIAL_MEMOIZED.clear()
    # generation.COMPLETE_MEMOIZED.clear()
    # Clear memoization
    memoized: Dict[Tuple[int, int], Tuple[
        Dict[bytes, List[bool]],
        Dict[bytes, List[int]]
    ]] = dict()
    for idx in range(n_patch_groups):
        print('Repair:', idx)
        if idx != 0:
            repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
            # TODO: refactor textfile
            for buggy_file in bug.buggy_files:
                text_file = TextFile(Path(bug.proj_path) / buggy_file.path)
                analyzer.change(text_file)
        text_files: List[TextFile] = []
        # For each file, generated a patch for each change (in reversed order relative to change)

        for buggy_file_idx, buggy_file in enumerate(bug.buggy_files):
            text_file = TextFile(Path(bug.proj_path) / buggy_file.path)
            if idx == 0:
                analyzer.open(text_file)
            print(len(buggy_file.changes))
            print(buggy_file.path)
            print([(c.start, len(c.removed_lines))
                   for c in reversed(buggy_file.changes)])

            for (change_idx, change) in enumerate(reversed(buggy_file.changes)):
                if (mem_id := (buggy_file_idx, change_idx)) not in memoized:
                    memoized[mem_id] = ({}, {})
                generation.PARTIAL_MEMOIZED, generation.COMPLETE_MEMOIZED = memoized[mem_id]
                start = change.start - 1
                end = start + len(change.removed_lines)
                start_pos = text_file.refine_index(start, 0)
                end_pos = text_file.refine_index(end, 0)

                start_index = text_file.form_index(start, 0)
                end_index = text_file.form_index(end, 0)

                # TODO: justify 25
                prefix_start = text_file.form_index(
                    max(0, start_pos['line'] - 25), 0)
                suffix_end = text_file.form_index(end_pos['line'] + 25, 0)
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
                # start_cursor = text_file.cursor
                assert prefix.endswith('\n')
                assert text_file.content[text_file.cursor - 1] == '\n'
                assert text_file.content[text_file.cursor] == '\n'

                lm_context = gen.LMContext(
                    gen.CODET5,
                    gen.CODET5_TOKENIZER,
                    gen.codet5_tokenize(prefix, suffix),
                    gen.CODET5_INFERENCE_CONFIG,
                )

                lsp_context = gen.LspContext(
                    text_file,
                    analyzer
                )

                start_time = time.time()
                try:
                    completion_overhead, _, output = gen.repair(
                        lm_context, lsp_context)
                except TimeoutError:
                    print('Fatal timeout error')
                    with open('timeout-error', 'a') as f:
                        f.write("TIMEOUT\n")
                    output = ['']
                    completion_overhead = []
                end_time = time.time()
                times.append(end_time - start_time)
                time_completion.extend(completion_overhead)
                memoized[mem_id] = (generation.PARTIAL_MEMOIZED, generation.COMPLETE_MEMOIZED)
                # This is always True
                # assert ''.join(output) == text_file.content[start_cursor:text_file.cursor]
                print('Success')
                print(''.join(output))
                # breakpoint()
                print([f'{t:.2f}' for t in times])
            # text_file.write()
            text_files.append(text_file)
        patch_groups.append(text_files)
        # TODO: opt
        save_dir = base_dir / str(len(patch_groups))
        save_dir.mkdir(exist_ok=False, parents=True)
        debug_dir = save_dir / 'debug'
        debug_dir.mkdir()
        assert len(text_files) == len(bug.buggy_files)
        for text_file, buggy_file_path in zip(
            text_files,
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
            # with open()
    with open(base_dir / 'time.json', 'w') as f:
        json.dump({
            'completion': time_completion,
            'times': times,
        }, f, indent=2)

    # TODO: still buggy (because when analyzer.stop() the process terminates,
    # but the client is still in the while loop)
    analyzer.client.stop()
    analyzer.client.shutdown(None)
    analyzer.client.exit(None)
    analyzer.process.terminate()
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
    CompilationError = 0
    TestingError = 1
    Success = 2


def validate_proj(bug_id: str, bug: d4j.Bug, patch_group: List[TextFile]) -> ValResult:
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
        patch_file.write()
    result = subprocess.run(['defects4j', 'compile'],
                            env=env, cwd=bug.proj_path)
    if result.returncode != 0:
        return ValResult.CompilationError
    # Filter out candidate patches (not even plausible)
    try:
        subprocess.run(['defects4j', 'test'], env=env,
                       cwd=bug.proj_path, timeout=120)
        failing_tests = Path(bug.proj_path) / 'failing_tests'
        assert failing_tests.exists()
        with open(failing_tests) as f:
            return ValResult.Success if f.read().strip() == '' else ValResult.TestingError
    except subprocess.TimeoutExpired:
        return ValResult.TestingError


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
    comp_failed_patches: Set[int] = set()
    test_failed_patches: Set[int] = set()
    succeeded_patches: Set[int] = set()
    patch_groups = get_patch_groups(bug_dir)
    result: dict = {
        # 'with_lsp': {
        'succeeded': [],
        'comp_failed': [],
        'test_failed': [],
        'dup_succeeded': [],
        'dup_test_failed': [],
        'dup_comp_failed': [],
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
        val_result = validate_proj(bug_id, bug, patch_group)
        match val_result:
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


BATCH_SIZE = 16


def validate_all_bugs(all_bugs: dict, proj_dir: Path):
    ret: dict = {}
    bug_dirs: List[Path] = sorted(
        list(filter(Path.is_dir, proj_dir.iterdir())), key=lambda p: int(str(p.stem)))
    for bug_dir_batch in utils.chunked(BATCH_SIZE, bug_dirs):
        params_list = [(bug_dir, (bug_id := proj_dir.stem + '-' + bug_dir.stem),
                        all_bugs[bug_id]) for bug_dir in bug_dir_batch]
        results: List[dict] = Parallel(n_jobs=BATCH_SIZE)(
            delayed(do_validation)(*params) for params in params_list)
        for (_, bug_id, _), result in zip(params_list, results):
            ret[bug_id] = result
        with open(proj_dir / proj_dir.with_suffix('.json').name, 'w') as f:
            json.dump(ret, f, indent=2)


if __name__ == '__main__':
    if os.getenv('VAL') is not None:
        import sys
        result_dir = Path(sys.argv[1])
        all_bugs = dataset.all_bugs()
        assert result_dir.exists()
        all_results: dict = {}
        for proj_dir in filter(Path.is_dir, result_dir.iterdir()):
            validate_all_bugs(all_bugs, proj_dir)
            # all_results = dict(all_results, **result)
        exit()

    assert os.getenv('JAVA8_HOME')

    from realm import generation as gen
    torch.manual_seed(0)
    random.seed(0)
    result_dir = Path(f'results-{uuid.uuid4()}')
    if os.getenv('DEBUG') is not None:
        result_dir = Path('../results') / 'temp' / result_dir
    result_dir.mkdir(exist_ok=False, parents=True)
    print(f'Metadata will be saved in {result_dir}')
    for bug_id, bug in dataset.all_bugs().items():
        proj = bug_id.split('-')[0]
        # if proj in proj_accessed or proj == 'Mockito':
        # TODO (DONE): IMPORTANT!!!!! Memorize multiple changes when doing repair
        if not bug_id.startswith('Chart'):
            continue
        # if int(bug_id.split('-')[1]) < 115:
        #     continue
        # if bug_id == 'Math-1':
        #     continue
        # proj_accessed.add(proj)
        # if bug_id != 'Mockito-1':
        #     continue
        print(bug_id)
        patch_groups = repair_proj(result_dir, bug_id, bug, 20)
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
