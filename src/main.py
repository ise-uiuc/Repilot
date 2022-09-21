import os
import shlex
from pathlib import Path
import shutil
import subprocess
from time import sleep
from Repair.LM.model import SpanLM
from realm import utils
from realm.analyze import java_syntax
from realm.analyze.jdt_lsp import JdtLspAnalyzer
from realm.generation import Repairer
from realm.lsp import TextDocument, spec
import json
from typing import List, Set, Tuple, cast
from init import data
from unidiff import PatchSet
import git

from realm.lsp.spec import TextChange
from realm.lsp.text import TextFile
from datasets import d4j

assert shutil.which('defects4j')
assert os.getenv('JAVA8_HOME')

x, y = utils.take_while_two(lambda _: True,
                            lambda x, y: x == y - 1,
                            [1, 2, 3, 4, 6, 7, 8])
print(x)
print(list(y))


CONTEXT_SIZE = 1000
N_SAMPLE = 1

dataset = d4j.Defects4J('/home/yuxiang/Developer/defects4j', data)
# model = SpanLM('facebook/incoder-1B', batch_size=N_SAMPLE)
model = None


def server_cmd(bug_id: str) -> List[str]:
    return shlex.split(f"/home/yuxiang/Developer/jdt-lsp/bin/jdtls \
        -configuration /home/yuxiang/.cache/jdtls \
        -data .lsp_data/{bug_id}")


def repair_proj(model: SpanLM, bug_id: str, bug: d4j.Bug):
    proj, id_str = bug_id.split('-')
    repo = git.Repo(bug.proj_path)
    repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
    repo.git.execute(['defects4j', 'checkout', '-p', proj,
                     f'-v{id_str}b', '-w', bug.proj_path])
    repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
    # repo.git.execute(['git', 'clean', '-xfd'])

    analyzer = JdtLspAnalyzer(server_cmd(bug_id), bug.proj_path, cast(
        str, os.getenv('JAVA8_HOME')), verbose=False)
    text_files: List[TextFile] = []
    zipped_patches: List[List[Tuple[spec.EntireDocumentChange, TextFile]]] = []
    for buggy_file in bug.buggy_files:
        text_file = TextFile(Path(bug.proj_path) / buggy_file.path)
        text_files.append(text_file)
        print(len(buggy_file.changes))
        print(buggy_file.path)
        print([(c.start, len(c.removed_lines))
              for c in reversed(buggy_file.changes)])
        # Generate N samples for each change
        for change in reversed(buggy_file.changes):
            start = change.start - 1
            end = start + len(change.removed_lines)
            start_pos = text_file.refine_index(start, 0)
            end_pos = text_file.refine_index(end, 0)

            start_index = text_file.form_index(start, 0)
            end_index = text_file.form_index(end, 0)
            text_file.move_cursor(start_index)

            text_file.change([{
                'text': '',
                'range': {
                    'start': start_pos,
                    'end': end_pos
                }
            }])

            prefix = text_file.content[max(
                0, start_index - CONTEXT_SIZE):start_index]
            suffix = text_file.content[end_index:end_index + CONTEXT_SIZE]
            repairer = Repairer()
            output = repairer.repair(analyzer, text_file, prefix, suffix)
            print(output)
            # print()
            # print(text_file)
            exit()
            well, _, outputs, _ = model.model_predict(
                prefix, suffix, do_sample=True, strict=False)
            assert len(outputs) == N_SAMPLE
            assert well
            print(outputs[0])
            exit()

            def full_content(change: spec.TextChange) -> spec.EntireDocumentChange:
                text = TextDocument(text_file.content)
                text.change([change])
                return {
                    'text': text.content,
                }

            zipped_patches.append([(full_content({
                'text': output,
                'range': {
                    'start': start_pos,
                    'end': end_pos
                }
            }), text_file) for output in outputs])
        continue
    # Choose the best patch group according to the analysis
    # patch_group_iter = zip(*zipped_patches)
    # best_group, best_error = None, 999999999999
    # for patch_group in patch_group_iter:
    #     n_errors = 0
    #     for change, patch in patch_group:
    #         patch.change([change])
    #         patch.write()
    #         analyzer.sync(patch)
    #         diagnosis = analyzer.diagnose(2)
    #         if diagnosis is None:
    #             n_errors += 0
    #         else:
    #             n_errors += sum(
    #                 1 for d in diagnosis
    #                 if d['severity'] == 1
    #             )
    #     print(n_errors)
    #     if n_errors < best_error:
    #         best_error = n_errors
    #         best_group = patch_group
    # assert best_group is not None
    for change, patch in best_group:
        patch.change([change])
        patch.write()

    repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
    repo.git.execute(['defects4j', 'checkout', '-p', proj,
                     f'-v{id_str}f', '-w', bug.proj_path])
    repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
    for change, patch in best_group:
        patch.change([change])
        patch.write()
    repo.git.execute(['git', 'clean', '-dfx'])
    repo.close()
    # for text_file in text_files:
    #     text_file.write()


def validate_proj(bug_id: str, bug: d4j.Bug):
    # proj, id_str = bug_id.split('-')
    java8_home = cast(str, os.getenv('JAVA8_HOME'))
    env = dict(os.environ, JAVA_HOME=java8_home)
    subprocess.run(['defects4j', 'compile'], env=env, cwd=bug.proj_path)
    subprocess.run(['defects4j', 'test'], env=env, cwd=bug.proj_path)


proj_accessed: Set[str] = set()
for bug_id, bug in dataset.all_bugs().items():
    proj = bug_id.split('-')[0]
    if proj in proj_accessed or proj == 'Mockito':
        continue
    proj_accessed.add(proj)
    # if bug_id != 'Mockito-1':
    #     continue
    print(bug_id)
    repair_proj(model, bug_id, bug)
    validate_proj(bug_id, bug)


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
