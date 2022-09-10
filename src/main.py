import os
import shlex
from pathlib import Path
from time import sleep
from realm.analyze import java_syntax
from realm.analyze.jdt_lsp import JdtLspAnalyzer
from realm.lsp import TextDocument
import json
from typing import cast
from init import data
from unidiff import PatchSet

from realm.lsp.spec import TextChange
from realm.lsp.text import TextFile
from datasets import d4j


dataset = d4j.Defects4J(data, '/home/yuxiang/Developer/realm/buggy_locations')
print([(p, b) for p, b in dataset.all_bugs() if 'Jsoup' in str(p)])
exit()

text = TextDocument("""def f(x: int, y: int) -> float:
    x = partial(operator.add, 1)
    y = x(10)
    return y
""")
text.change([cast(TextChange, {
    'range': {
        'start': {
            'line': 4,
            'character': 0,
        },
        'end': {
            'line': 4,
            'character': 0,
        }
    },
    'text': 'test '
})])
# print(text)
# exit()
java_server = shlex.split("/home/yuxiang/Developer/jdt-lsp/bin/jdtls \
        -configuration /home/yuxiang/.cache/jdtls \
        -data .path_to_dataa")

root = Path('/home/yuxiang/Developer/javasymbolsolver-maven-sample')
# root = Path('/tmp/lang_1_fixed')

file_path = root / 'src/main/java/com/yourorganization/maven_sample/MyAnalysis.java'
# file_path = root / 'src/main/java/org/apache/commons/lang3/math/NumberUtils.java'

# analyzer = JdtLspAnalyzer(java_server, root, str(os.getenv('JAVA_HOME')))
# analyzer.sync(TextFile(file_path))
# print(data)

from realm.repair import main
import git

def checkout(path: str, commit: str, buggy=True):
    repo = git.Repo(path) 
    repo.git.diff(buggy_commit, fixed_commit)

def diff(path: str, buggy_commit: str, fixed_commit: str):
    repo = git.Repo(path) 
    return repo.git.diff(buggy_commit, fixed_commit)

for d in filter(lambda d: d['fixed'], data):
    # print(d.keys())
    # print(d['path'])
    try:
        analyzer = JdtLspAnalyzer(java_server, d['path'], str(os.getenv('JAVA_HOME')))
        # analyzer.sync(TextFile(file_path))
        for patch in patches:
            p = Path(d['path']) / patch.source_file[2:]
            analyzer.sync(TextFile(p))
            print([d for d in analyzer.diagnose() if d['severity'] == 1])
        # exit()
    except git.BadName:
        print("?")
        continue

exit()

print("\nSTART")
print(analyzer.diagnose())


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
