import os
import shlex
from pathlib import Path
import shutil
from time import sleep
from Repair.LM.model import SpanLM
from realm.analyze import java_syntax
from realm.analyze.jdt_lsp import JdtLspAnalyzer
from realm.lsp import TextDocument
import json
from typing import List, cast
from init import data
from unidiff import PatchSet
import git

from realm.lsp.spec import TextChange
from realm.lsp.text import TextFile
from datasets import d4j

assert shutil.which('defects4j')
assert os.getenv('JAVA8_HOME')

dataset = d4j.Defects4J('/home/yuxiang/Developer/defects4j', data)
model = SpanLM('facebook/incoder-1B')

CONTEXT_SIZE = 1000

def server_cmd(bug_id: str) -> List[str]:
    return shlex.split(f"/home/yuxiang/Developer/jdt-lsp/bin/jdtls \
        -configuration /home/yuxiang/.cache/jdtls \
        -data .lsp_data/{bug_id}")

def repair_proj(model: SpanLM, bug_id: str, bug: d4j.Bug):
    proj, id_str = bug_id.split('-')
    repo = git.Repo(bug.proj_path)
    repo.git.execute(['defects4j', 'checkout', '-p', proj, f'-v{id_str}b', '-w', bug.proj_path])
    repo.git.execute(['git', 'checkout', 'HEAD', '-f', '.'])
    repo.git.execute(['git', 'clean', '-xfd'])

    # analyzer = JdtLspAnalyzer(server_cmd(bug_id), bug.proj_path, os.getenv('JAVA8_HOME'))
    for buggy_file in bug.buggy_files:
        text_file = TextFile(Path(bug.proj_path) / buggy_file.path)
        print(len(buggy_file.changes))
        print(buggy_file.path)
        for change in reversed(buggy_file.changes):
            start = change.start - 1
            end = start + len(change.removed_lines)
            start_pos = text_file.refine_index(start, 0)
            end_pos = text_file.refine_index(end, 0)

            start_index = text_file.form_index(start, 0)
            end_index = text_file.form_index(end, 0)
            prefix = text_file.content[max(0, start_index - CONTEXT_SIZE):start_index]
            suffix = text_file.content[end_index:end_index + CONTEXT_SIZE]
            well, _, [output], _ = model.model_predict(prefix, suffix, do_sample=True, strict=False)
            assert well

            text_file.change([{
                'text': output,
                'range': {
                    'start': start_pos,
                    'end': end_pos
                }
            }])
        text_file.write()
        # analyzer.sync(text_file)
        # analyzer.diagnose(5)


for bug_id, bug in dataset.all_bugs().items():
    if bug_id != 'Mockito-2':
        continue
    repair_proj(model, bug_id, bug)


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
