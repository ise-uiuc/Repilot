import os
import shlex
from pathlib import Path
from time import sleep
from realm.analyze import java_syntax
from realm.analyze.jdt_lsp import JdtLspAnalyzer
from realm.lsp import TextDocument
import json
from typing import cast

from realm.lsp.spec import TextChange
from realm.lsp.text import TextFile


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
java_server = shlex.split('java  \
        -Declipse.application=org.eclipse.jdt.ls.core.id1 \
        -Dosgi.bundles.defaultStartLevel=4 \
        -Declipse.product=org.eclipse.jdt.ls.core.product \
        -Dlog.level=ALL \
        -noverify \
        -Xmx1G \
        --add-modules=ALL-SYSTEM \
        --add-opens java.base/java.util=ALL-UNNAMED \
        --add-opens java.base/java.lang=ALL-UNNAMED \
        -jar /Users/nole/Developer/eclipse.jdt.ls/org.eclipse.jdt.ls.product/target/repository/plugins/org.eclipse.equinox.launcher_1.6.400.v20210924-0641.jar\
        -configuration /Users/nole/Developer/eclipse.jdt.ls/org.eclipse.jdt.ls.product/target/repository/config_mac \
        -data .path_to_data')

root = Path('/Users/nole/Developer/javasymbolsolver-maven-sample')
# root = Path('/tmp/lang_1_fixed')

file_path = root / 'src/main/java/com/yourorganization/maven_sample/MyAnalysis.java'
# file_path = root / 'src/main/java/org/apache/commons/lang3/math/NumberUtils.java'

analyzer = JdtLspAnalyzer(java_server, root, str(os.getenv('JAVA_HOME')))
analyzer.sync(TextFile(file_path))

print("\nSTART")
print(analyzer.diagnose())


file_path = Path(
    '/Users/nole/Developer/d4j-checkout/Lang-1-buggy/src/main/java/org/apache/commons/lang3/Validate.java')
with open(file_path) as f:
    content = TextDocument(f.read())

a = java_syntax.reduce(content)
print(content.content)
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
