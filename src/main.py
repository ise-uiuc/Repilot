import logging
import torch
import random
import os
import shlex
from pathlib import Path
import shutil
import subprocess
from time import sleep
from tkinter import E
from Repair.LM.model import SpanLM
from realm import utils
from realm.analyze import java_syntax
from realm.analyze.jdt_lsp import JdtLspAnalyzer
from realm.generation import Repairer
from realm.lsp import TextDocument, spec
import json
from typing import Generator, List, Set, Tuple, cast
from init import data
from unidiff import PatchSet
import git

from realm.lsp.spec import TextChange
from realm.lsp.text import TextFile
from datasets import d4j

assert shutil.which('defects4j')
assert os.getenv('JAVA8_HOME')

# print(Repairer.tokenizer.encode("<s> </s> <pad> <extra_id_0> <extra_id_1> <unk> <mask>", add_special_tokens=False))
# print(Repairer.tokenizer.decode(torch.tensor(range(0, 10)).to('cuda'), clean_up_tokenization_spaces = False))
# exit()
# print(Repairer.tokenizer.tokenize("def long_function(): <extra_id_0> ", padding=True))
# tokens = Repairer.tokenizer.encode("def long_function(): <extra_id_0>", return_tensors='pt', add_special_tokens=True).to('cuda')
# print(tokens)
# tokens = torch.cat((tokens, torch.tensor([[Repairer.END_ID]]).to('cuda')), dim=-1)
# print(Repairer.tokenizer.batch_decode(Repairer.model.generate(tokens, max_new_tokens=25), skip_special_tokens=False))
# exit()


N_SAMPLE = 1

dataset = d4j.Defects4J('/home/yuxiang/Developer/defects4j', data)
# model = SpanLM('facebook/incoder-1B', batch_size=N_SAMPLE)
model = None


def server_cmd(bug_id: str) -> List[str]:
    return shlex.split(f"/home/yuxiang/Developer/jdt-lsp/bin/jdtls \
        -configuration /home/yuxiang/.cache/jdtls \
        -data .lsp_data/{bug_id}")

dummy = torch.tensor([[    0, 32099,   203,  7734,   368,   203,  7734,   368,   333,   353,
           358,  4543,   333,    18,   203,  7734,   368, 32098,  7734, 32097,
           203,  5411,   289,   203,   203,  5411, 32096,   203, 32095,   203,
         32094,   203, 32093,   203,  3639, 32092,   203,   565,   289,  1044,
           261,  6385,   478,     2,     1]], device='cuda:0')

def repair_proj(bug_id: str, bug: d4j.Bug, n_patch_groups: int = 1) -> List[List[TextFile]]:
    proj, id_str = bug_id.split('-')
    repo = git.Repo(bug.proj_path)
    repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
    subprocess.run(['defects4j', 'checkout', '-p', proj,
                    f'-v{id_str}b', '-w', bug.proj_path])
    repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
    # repo.git.execute(['git', 'clean', '-xfd'])

    analyzer = JdtLspAnalyzer(server_cmd(bug_id), bug.proj_path, cast(
        str, os.getenv('JAVA8_HOME')), verbose=False)

    patch_groups: List[List[TextFile]] = []
    for idx in range(n_patch_groups):
        print('Repair:', idx)
        if idx != 0:
            repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
            # TODO: refactor textfile
            for buggy_file in bug.buggy_files:
                text_file = TextFile(Path(bug.proj_path) / buggy_file.path)
        text_files: List[TextFile] = []
        # For each file, generated a patch for each change (in reversed order relative to change)
        for buggy_file in bug.buggy_files:
            text_file = TextFile(Path(bug.proj_path) / buggy_file.path)
            if idx == 0:
                analyzer.open(text_file)
            print(len(buggy_file.changes))
            print(buggy_file.path)
            print([(c.start, len(c.removed_lines))
                   for c in reversed(buggy_file.changes)])

            for change in reversed(buggy_file.changes):
                original_content = text_file.content
                original_cursor = text_file.cursor
                while True:
                    start = change.start - 1
                    end = start + len(change.removed_lines)
                    start_pos = text_file.refine_index(start, 0)
                    end_pos = text_file.refine_index(end, 0)

                    start_index = text_file.form_index(start, 0)
                    end_index = text_file.form_index(end, 0)

                    def comment(line: str) -> str:
                        stripped = line.lstrip()
                        index = len(line) - len(stripped)
                        return line[:index] + '// Buggy: ' + stripped

                    insertion = ''.join(comment(line) for line in change.removed_lines)
                    insertion = ''
                    if not insertion.endswith('\n'):
                        insertion += '\n'
                    # if not insertion.endswith('\n'):
                    #     print('Warining:', insertion)
                    #     insertion += '\n'
                    # print(text_file.get_position(text_file.cursor))
                    # text_file.write()
                    # exit()

                    # prefix
                    # removed_lines
                    # suffix
                    prefix_start = text_file.form_index(
                        max(0, start_pos['line'] - 25), 0)
                    suffix_end = text_file.form_index(end_pos['line'] + 25, 0)
                    prefix = text_file.content[prefix_start:start_index] + insertion
                    suffix = '\n' + text_file.content[end_index:suffix_end]

                    # prefix(\n)
                    # insertion(\n)
                    # <cursor:infill>
                    # (\n)suffix
                    text_file.change([{
                        'text': insertion + '\n',
                        'range': {
                            'start': start_pos,
                            'end': end_pos
                        }
                    }])

                    text_file.move_cursor(start_index + len(insertion))
                    assert prefix.endswith('\n')
                    assert text_file.content[text_file.cursor - 1] == '\n'
                    assert text_file.content[text_file.cursor] == '\n'

                    repairer = Repairer(prefix, suffix)
                    # text_file.write()
                    # print(repairer.input_strings)
                    # exit()
                    # print(repairer.tokenizer.batch_decode(dummy))
                    # print(repairer.tokenizer.batch_decode(repairer.model.generate(repairer.input_tokens, decoder_input_ids=dummy, max_length=50), skip_special_tokens=True)[0])
                    # exit()
                    output_ids, output = repairer.repair(
                        analyzer, text_file, max_new_tokens=200)
                    # Keeps generation until EOM
                    if not output_ids[-1] == repairer.END_ID:
                        print('Failure')
                        print(''.join(output))
                        text_file.content = original_content
                        text_file.cursor = original_cursor
                        text_file.sync()
                        text_file.write()
                        analyzer.change(text_file)
                        continue
                    else:
                        print('Success')
                        print(''.join(output))
                        break
            text_files.append(text_file)
        patch_groups.append(text_files)
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

TIMEOUT = 60

def validate_proj(bug_id: str, bug: d4j.Bug, patch_group: List[TextFile]) -> bool:
    proj, id_str = bug_id.split('-')
    java8_home = cast(str, os.getenv('JAVA8_HOME'))
    env = dict(os.environ, JAVA_HOME=java8_home)
    repo = git.Repo(bug.proj_path)
    repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
    subprocess.run(['defects4j', 'checkout', '-p', proj,
                    f'-v{id_str}f', '-w', bug.proj_path])
    repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
    for patch_file in patch_group:
        patch_file.write()
    result = subprocess.run(['defects4j', 'compile'],
                            env=env, cwd=bug.proj_path)
    if result.returncode != 0:
        return False
    # Filter out candidate patches (not even plausible)
    try:
        subprocess.run(['defects4j', 'test'], env=env, cwd=bug.proj_path, timeout=60)
        failing_tests = Path(bug.proj_path) / 'failing_tests'
        assert failing_tests.exists()
        with open(failing_tests) as f:
            return f.read().strip() == ''
    except subprocess.TimeoutExpired:
        return False


torch.manual_seed(0)
random.seed(0)
for bug_id, bug in dataset.all_bugs().items():
    proj = bug_id.split('-')[0]
    # if proj in proj_accessed or proj == 'Mockito':
    if proj != 'Chart':
        continue
    # if bug_id == 'Math-1':
    #     continue
    # proj_accessed.add(proj)
    # if bug_id != 'Mockito-1':
    #     continue
    print(bug_id)
    patch_groups = repair_proj(bug_id, bug, 50)
    candidate_patch_groups: List[int] = []
    for idx, patch_group in enumerate(patch_groups):
        if validate_proj(bug_id, bug, patch_group):
            candidate_patch_groups.append(idx)
    with open('result.log', 'a') as f:
        f.writelines(
            [str(bug_id), ' : ', f'{len(candidate_patch_groups)} / {len(patch_groups)}'])

# file_path = Path(
#     '/home/yuxiang/Developer/d4j-checkout/Lang-1-buggy/src/main/java/org/apache/commons/lang3/Validate.java')
# with open(file_path) as f:
#     content = TextDocument(f.read())

# a = java_syntax.reduce(content)
# print(content.content)
# print(a.content)
# a.feed(';;')

# try:
#     a.feed('test')
# except java_syntax.TokenizeError:
#     print(a.parser.tokens.look())
#     pass
# # try:
# #     a.feed('')
# # except java_syntax.AnalysisError:
# #     pass
# # a.feed('package com.yourorganization.maven_sample')
# #     # print(a.parser.tokens.look())
# #     # a.feed(content[:10])
