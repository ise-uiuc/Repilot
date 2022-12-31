import itertools
import pickle
import subprocess
from multiprocessing import Process
from multiprocessing.connection import Connection
import os
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import torch

from realm.generation_defs import GenerationContext, Memorization
from realm.lsp import LSPClient, TextFile, spec
from realm.model import CodeT5Large
from realm.generation_defs import MODEL
from realm import utils

JAVA_KEYWORDS = {'abstract', 'continue', 'for', 'new', 'switch',
                 'assert', 'default', 'goto', 'package', 'synchronized',
                 'boolean', 'do', 'if', 'private', 'this',
                 'break', 'double', 'implements', 'protected', 'throw',
                 'byte', 'else', 'import', 'public', 'throws',
                 'case', 'enum', 'instanceof', 'return', 'transient',
                 'catch', 'extends', 'int', 'short', 'try',
                 'char', 'final', 'interface', 'static', 'void',
                 'class', 'finally', 'long', 'strictfp', 'volatile',
                 'const' 'float', 'native', 'super', 'while'}

# def post_process(get_client: Callable[[C], LSPClient], rpc: Callable[Concatenate[LSPClient, P], Msg]) \
#         -> Callable[[Callable[[Msg], T]], Callable[Concatenate[C, P], T]]:
#     def decorator(f: Callable[[Msg], T]) -> Callable[Concatenate[C, P], T]:
#         @wraps(f)
#         def impl(self: C, *args: P.args, **kwargs: P.kwargs) -> T:
#             client = get_client(self)
#             return f(rpc(client, *args, **kwargs))
#         return impl
#     return decorator


def char_may_trigger_completion(c: str) -> bool:
    assert len(c) == 1
    return c.isalnum() or (c in ['.', '_', '$'])


class Message:
    def __init__(self, return_result: bool, method: str, *args: Any, **kwargs: Any) -> None:
        self.return_result = return_result
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def __str__(self) -> str:
        return str((self.return_result, self.method, self.args, self.kwargs))

    def __repr__(self) -> str:
        return repr((self.return_result, self.method, self.args, self.kwargs))


class JdtLspAnalyzer(Process):
    """Jdt LSP based Java program analyzer leveraging whole-project information.
    Now assume only one active file for diagnosis"""
    # counter = itertools.count(0)

    def __init__(
        self,
        conn: Connection,
        server_cmd: List[str],
        proj_path: PathLike,
        java_home: str,
        model: CodeT5Large = MODEL,
        n_hunks: int = 1,
        use_mem: bool = os.getenv('NO_MEM') is None,
        verbose: bool = False
    ) -> None:
        super().__init__()
        self.conn = conn
        self.server_cmd = server_cmd
        self.proj_path = proj_path
        self.java_home = java_home
        self.verbose = verbose
        self.counter = itertools.count(0)

        self.model = model
        self.use_mem = use_mem
        self.n_hunks = n_hunks
        self.mems = [Memorization.init() for _ in range(n_hunks)]
        self.active_hunk = 0

    @property
    def mem(self) -> Memorization:
        return self.mems[self.active_hunk]

    def set_active_hunk(self, new_hunk_idx: int):
        self.active_hunk = new_hunk_idx

    def init_lsp(self):
        self.process = subprocess.Popen(
            self.server_cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        assert self.process.stdin is not None and self.process.stdout is not None
        self.client = LSPClient(
            self.process.stdin, self.process.stdout, self.verbose, 120)
        self.client.start()

    def stop_lsp(self):
        self.process.terminate()
        self.client.stop()
        self.client.shutdown(None)
        self.client.exit(None)
        self.process.terminate()

    def run(self) -> None:
        # Start the thread
        self.init_lsp()
        while True:
            message: Optional[Message] = self.conn.recv()
            if message is None:
                break
            # print('RECEIVED:', message.method)
            assert isinstance(message, Message)
            result = getattr(self, message.method)(
                *message.args, **message.kwargs)
            if message.return_result:
                # print('RESULT:', result)
                self.conn.send(result)
        self.stop_lsp()
        print('Analyzer terminated')

    def init(self):
        # self.active_text: Optional[TextDocument] = None

        # Initialize the server
        path = Path(self.proj_path)
        # with open('log1.json', 'w') as f:
        self.client.initialize({
            "processId": self.process.pid,
            "clientInfo": {
                "name": path.name,
                "version": "0.0.0"
            },
            "locale": "en",
            "rootPath": str(path.absolute()),
            "rootUri": path.as_uri(),
            "capabilities": spec.ClientCapabilities({
                "workspace": {
                    "applyEdit": True,
                    "workspaceEdit": {
                        "documentChanges": True,
                        "resourceOperations": [
                            "create",
                            "rename",
                            "delete"
                        ],
                        "failureHandling": "textOnlyTransactional",
                        "normalizesLineEndings": True,
                        "changeAnnotationSupport": {
                            "groupsOnLabel": True
                        }
                    },
                    "didChangeConfiguration": {
                        "dynamicRegistration": True
                    },
                    "didChangeWatchedFiles": {
                        "dynamicRegistration": True
                    },
                    "symbol": {
                        "dynamicRegistration": True,
                        "symbolKind": {
                            "valueSet": [
                                1,
                                2,
                                3,
                                4,
                                5,
                                6,
                                7,
                                8,
                                9,
                                10,
                                11,
                                12,
                                13,
                                14,
                                15,
                                16,
                                17,
                                18,
                                19,
                                20,
                                21,
                                22,
                                23,
                                24,
                                25,
                                26
                            ]
                        },
                        "tagSupport": {
                            "valueSet": [
                                1
                            ]
                        }
                    },
                    "codeLens": {
                        "refreshSupport": True
                    },
                    "executeCommand": {
                        "dynamicRegistration": True
                    },
                    "configuration": True,
                    "workspaceFolders": True,
                    "semanticTokens": {
                        "refreshSupport": True
                    },
                    "fileOperations": {
                        "dynamicRegistration": True,
                        "didCreate": True,
                        "didRename": True,
                        "didDelete": True,
                        "willCreate": True,
                        "willRename": True,
                        "willDelete": True
                    }
                },
                "textDocument": {
                    "publishDiagnostics": {
                        "relatedInformation": True,
                        "versionSupport": False,
                        "tagSupport": {
                            "valueSet": [
                                1,
                                2
                            ]
                        },
                        "codeDescriptionSupport": True,
                        "dataSupport": True
                    },
                    "synchronization": {
                        "dynamicRegistration": True,
                        "willSave": True,
                        "willSaveWaitUntil": True,
                        "didSave": True
                    },
                    "completion": {
                        "dynamicRegistration": True,
                        "contextSupport": True,
                        "completionItem": {
                            "snippetSupport": True,
                            "commitCharactersSupport": True,
                            "documentationFormat": [
                                "markdown",
                                "plaintext"
                            ],
                            "deprecatedSupport": True,
                            "preselectSupport": True,
                            "tagSupport": {
                                "valueSet": [
                                    1
                                ]
                            },
                            "insertReplaceSupport": True,
                            "resolveSupport": {
                                "properties": [
                                    "documentation",
                                    "detail",
                                    "additionalTextEdits"
                                ]
                            },
                            "insertTextModeSupport": {
                                "valueSet": [
                                    1,
                                    2
                                ]
                            },
                            "labelDetailsSupport": True
                        },
                        "insertTextMode": 2,
                        "completionItemKind": {
                            "valueSet": [
                                1,
                                2,
                                3,
                                4,
                                5,
                                6,
                                7,
                                8,
                                9,
                                10,
                                11,
                                12,
                                13,
                                14,
                                15,
                                16,
                                17,
                                18,
                                19,
                                20,
                                21,
                                22,
                                23,
                                24,
                                25
                            ]
                        }
                    },
                    "hover": {
                        "dynamicRegistration": True,
                        "contentFormat": [
                            "markdown",
                            "plaintext"
                        ]
                    },
                    "signatureHelp": {
                        "dynamicRegistration": True,
                        "signatureInformation": {
                            "documentationFormat": [
                                "markdown",
                                "plaintext"
                            ],
                            "parameterInformation": {
                                "labelOffsetSupport": True
                            },
                            "activeParameterSupport": True
                        },
                        "contextSupport": True
                    },
                    "definition": {
                        "dynamicRegistration": True,
                        "linkSupport": True
                    },
                    "references": {
                        "dynamicRegistration": True
                    },
                    "documentHighlight": {
                        "dynamicRegistration": True
                    },
                    "documentSymbol": {
                        "dynamicRegistration": True,
                        "symbolKind": {
                            "valueSet": [
                                1,
                                2,
                                3,
                                4,
                                5,
                                6,
                                7,
                                8,
                                9,
                                10,
                                11,
                                12,
                                13,
                                14,
                                15,
                                16,
                                17,
                                18,
                                19,
                                20,
                                21,
                                22,
                                23,
                                24,
                                25,
                                26
                            ]
                        },
                        "hierarchicalDocumentSymbolSupport": True,
                        "tagSupport": {
                            "valueSet": [
                                1
                            ]
                        },
                        "labelSupport": True
                    },
                    "codeAction": {
                        "dynamicRegistration": True,
                        "isPreferredSupport": True,
                        "disabledSupport": True,
                        "dataSupport": True,
                        "resolveSupport": {
                            "properties": [
                                "edit"
                            ]
                        },
                        "codeActionLiteralSupport": {
                            "codeActionKind": {
                                "valueSet": [
                                    "",
                                    "quickfix",
                                    "refactor",
                                    "refactor.extract",
                                    "refactor.inline",
                                    "refactor.rewrite",
                                    "source",
                                    "source.organizeImports"
                                ]
                            }
                        },
                        "honorsChangeAnnotations": False
                    },
                    "codeLens": {
                        "dynamicRegistration": True
                    },
                    "formatting": {
                        "dynamicRegistration": True
                    },
                    "rangeFormatting": {
                        "dynamicRegistration": True
                    },
                    "onTypeFormatting": {
                        "dynamicRegistration": True
                    },
                    "rename": {
                        "dynamicRegistration": True,
                        "prepareSupport": True,
                        "prepareSupportDefaultBehavior": 1,
                        "honorsChangeAnnotations": True
                    },
                    "documentLink": {
                        "dynamicRegistration": True,
                        "tooltipSupport": True
                    },
                    "typeDefinition": {
                        "dynamicRegistration": True,
                        "linkSupport": True
                    },
                    "implementation": {
                        "dynamicRegistration": True,
                        "linkSupport": True
                    },
                    "colorProvider": {
                        "dynamicRegistration": True
                    },
                    "foldingRange": {
                        "dynamicRegistration": True,
                        "rangeLimit": 5000,
                        "lineFoldingOnly": True
                    },
                    "declaration": {
                        "dynamicRegistration": True,
                        "linkSupport": True
                    },
                    "selectionRange": {
                        "dynamicRegistration": True
                    },
                    "callHierarchy": {
                        "dynamicRegistration": True
                    },
                    "semanticTokens": {
                        "dynamicRegistration": True,
                        "tokenTypes": [
                            "namespace",
                            "type",
                            "class",
                            "enum",
                            "interface",
                            "struct",
                            "typeParameter",
                            "parameter",
                            "variable",
                            "property",
                            "enumMember",
                            "event",
                            "function",
                            "method",
                            "macro",
                            "keyword",
                            "modifier",
                            "comment",
                            "string",
                            "number",
                            "regexp",
                            "operator"
                        ],
                        "tokenModifiers": [
                            "declaration",
                            "definition",
                            "readonly",
                            "static",
                            "deprecated",
                            "abstract",
                            "async",
                            "modification",
                            "documentation",
                            "defaultLibrary"
                        ],
                        "formats": [
                            "relative"
                        ],
                        "requests": {
                            "range": True,
                            "full": {
                                "delta": True
                            }
                        },
                        "multilineTokenSupport": False,
                        "overlappingTokenSupport": False
                    },
                    "linkedEditingRange": {
                        "dynamicRegistration": True
                    }
                },
                "window": {
                    "showMessage": {
                        "messageActionItem": {
                            "additionalPropertiesSupport": True
                        }
                    },
                    "showDocument": {
                        "support": True
                    },
                    "workDoneProgress": True
                },
                "general": {
                    "staleRequestSupport": {
                        "cancel": True,
                        "retryOnContentModified": [
                            "textDocument/semanticTokens/full",
                            "textDocument/semanticTokens/range",
                            "textDocument/semanticTokens/full/delta"
                        ]
                    },
                    "regularExpressions": {
                        "engine": "ECMAScript",
                        "version": "ES2020"
                    },
                    "markdown": {
                        "parser": "marked",
                        "version": "1.1.0"
                    }
                }
            }),
            "initializationOptions": {
                "bundles": [],
                "workspaceFolders": [
                    path.as_uri()
                ],
                "settings": {
                    "java": {
                        # "home": java_home,
                        "jdt": {
                            "ls": {
                                "java": {
                                    "home": None,
                                },
                                "vmargs": "-XX:+UseParallelGC -XX:GCTimeRatio=4 -XX:AdaptiveSizePolicyWeight=90 -Dsun.zip.disableMemoryMapping=true -Xmx1G -Xms100m",
                                "lombokSupport": {
                                    "enabled": True
                                }
                            }
                        },
                        "errors": {
                            "incompleteClasspath": {
                                "severity": "warning"
                            }
                        },
                        "configuration": {
                            "checkProjectSettingsExclusions": False,
                            "updateBuildConfiguration": "interactive",
                            "maven": {
                                "userSettings": None,
                                "globalSettings": None,
                                "notCoveredPluginExecutionSeverity": "warning"
                            },
                            "workspaceCacheLimit": 90,
                            "runtimes": [
                                {
                                    "name": "JavaSE-1.8",
                                    "path": self.java_home,
                                    "default": True,
                                },
                            ]
                        },
                        "trace": {
                            "server": "verbose"
                        },
                        "import": {
                            "maven": {
                                "enabled": True
                            },
                            "gradle": {
                                "enabled": True,
                                "wrapper": {
                                    "enabled": True
                                },
                                "version": None,
                                "home": None,
                                "java": {
                                    "home": None,
                                },
                                "offline": {
                                    "enabled": False
                                },
                                "arguments": None,
                                "jvmArguments": None,
                                "user": {
                                    "home": None,
                                }
                            },
                            "exclusions": [
                                "**/node_modules/**",
                                "**/.metadata/**",
                                "**/archetype-resources/**",
                                "**/META-INF/maven/**"
                            ],
                            "generatesMetadataFilesAtProjectRoot": False
                        },
                        "maven": {
                            "downloadSources": False,
                            "updateSnapshots": False
                        },
                        "eclipse": {
                            "downloadSources": False
                        },
                        "referencesCodeLens": {
                            "enabled": False
                        },
                        "signatureHelp": {
                            "enabled": False,
                            "description": {
                                "enabled": False
                            }
                        },
                        "implementationsCodeLens": {
                            "enabled": False
                        },
                        "format": {
                            "enabled": True,
                            "settings": {
                                "url": None,
                                "profile": None,
                            },
                            "comments": {
                                "enabled": True
                            },
                            "onType": {
                                "enabled": True
                            },
                            "insertSpaces": True,
                            "tabSize": 4
                        },
                        "saveActions": {
                            "organizeImports": False
                        },
                        "project": {
                            "referencedLibraries": [
                                "lib/**/*.jar"
                            ],
                            "importOnFirstTimeStartup": "automatic",
                            "importHint": True,
                            "resourceFilters": [
                                "node_modules",
                                ".git"
                            ],
                            "encoding": "ignore"
                        },
                        "contentProvider": {
                            "preferred": None,
                        },
                        "autobuild": {
                            "enabled": True
                        },
                        "maxConcurrentBuilds": 1,
                        "recommendations": {
                            "dependency": {
                                "analytics": {
                                    "show": True
                                }
                            }
                        },
                        "completion": {
                            "maxResults": 0,
                            "enabled": True,
                            "guessMethodArguments": False,
                            "favoriteStaticMembers": [
                                "org.junit.Assert.*",
                                "org.junit.Assume.*",
                                "org.junit.jupiter.api.Assertions.*",
                                "org.junit.jupiter.api.Assumptions.*",
                                "org.junit.jupiter.api.DynamicContainer.*",
                                "org.junit.jupiter.api.DynamicTest.*",
                                "org.mockito.Mockito.*",
                                "org.mockito.ArgumentMatchers.*",
                                "org.mockito.Answers.*"
                            ],
                            "filteredTypes": [
                                "java.awt.*",
                                "com.sun.*",
                                "sun.*",
                                "jdk.*",
                                "org.graalvm.*",
                                "io.micrometer.shaded.*"
                            ],
                            "importOrder": [
                                "java",
                                "javax",
                                "org",
                                "com"
                            ]
                        },
                        "foldingRange": {
                            "enabled": True
                        },
                        "progressReports": {
                            "enabled": True
                        },
                        "codeGeneration": {
                            "hashCodeEquals": {
                                "useJava7Objects": False,
                                "useInstanceof": False
                            },
                            "useBlocks": False,
                            "generateComments": False,
                            "toString": {
                                "template": "${object.className} [${member.name()}=${member.value}, ${otherMembers}]",
                                "codeStyle": "STRING_CONCATENATION",
                                "SkipNullValues": False,
                                "listArrayContents": True,
                                "limitElements": 0
                            },
                            "insertionLocation": "afterCursor"
                        },
                        "selectionRange": {
                            "enabled": True
                        },
                        "showBuildStatusOnStart": {
                            "enabled": "notification"
                        },
                        "server": {
                            "launchMode": "Hybrid"
                        },
                        "sources": {
                            "organizeImports": {
                                "starThreshold": 99,
                                "staticStarThreshold": 99
                            }
                        },
                        "imports": {
                            "gradle": {
                                "wrapper": {
                                    "checksums": []
                                }
                            }
                        },
                        "templates": {
                            "fileHeader": [],
                            "typeComment": []
                        },
                        "references": {
                            "includeAccessors": True,
                            "includeDecompiledSources": True
                        },
                        "typeHierarchy": {
                            "lazyLoad": False
                        },
                        "settings": {
                            "url": None,
                        },
                        "symbols": {
                            "includeSourceMethodDeclarations": False
                        },
                        "quickfix": {
                            "showAt": "line"
                        },
                        "inlayHints": {
                            "parameterNames": {
                                "enabled": "literals",
                                "exclusions": []
                            }
                        }
                    }
                },
                "extendedClientCapabilities": {
                    "progressReportProvider": True,
                    "classFileContentsSupport": True,
                    "overrideMethodsPromptSupport": True,
                    "hashCodeEqualsPromptSupport": True,
                    "advancedOrganizeImportsSupport": True,
                    "generateToStringPromptSupport": True,
                    "advancedGenerateAccessorsSupport": True,
                    "generateConstructorsPromptSupport": True,
                    "generateDelegateMethodsPromptSupport": True,
                    "advancedExtractRefactoringSupport": True,
                    "inferSelectionSupport": [
                        "extractMethod",
                        "extractVariable",
                        "extractField"
                    ],
                    "moveRefactoringSupport": True,
                    "clientHoverProvider": True,
                    "clientDocumentSymbolProvider": True,
                    "gradleChecksumWrapperPromptSupport": True,
                    "resolveAdditionalTextEditsSupport": True,
                    "advancedIntroduceParameterRefactoringSupport": True,
                    "actionableRuntimeNotificationSupport": True,
                    "shouldLanguageServerExitOnShutdown": True,
                    "onCompletionItemSelectedCommand": "editor.action.triggerParameterHints"
                },
                "triggerFiles": [
                    # "file:///Users/nole/Developer/javasymbolsolver-maven-sample/src/main/java/com/yourorganization/maven_sample/MyAnalysis.java"
                ]
            },
            # "trace": "verbose",
            "workspaceFolders": [
                {
                    "uri": path.as_uri(),
                    "name": path.name,
                }
            ]
        })
        # with open('log') as f, open('config.json') as f1:
        #     d = json.load(f)
        #     d1 = json.load(f1)
        #     d['processId'] = self.process.pid
        #     self.client.initialize(d)
        self.client.initialized(cast(spec.InitializedParams, {}))
        # self.client.workspace_didChangeConfiguration(d1)

    def sync(self, text: TextFile):
        self.active_text = text

    def open(self, text: TextFile):
        self.sync(text)
        self.client.textDocument_didOpen(cast(spec.DidOpenTextDocumentParams, {
            'textDocument': {
                'uri': text.path.as_uri(),
                'languageId': 'java',
                'version': next(self.counter),
                'text': text.content,
            }}))

    def change(self, text: TextFile):
        self.sync(text)
        self.client.textDocument_didChange({
            'textDocument': {
                'uri': text.path.as_uri(),
                'version': next(self.counter)
            },
            'contentChanges': [{
                'text': text.content,
            }]
        })

    def completion(self, params: spec.CompletionParams) -> spec.ResponseMessage:
        return self.client.newCompletion(params)

    def save(self):
        self.client.textDocument_didSave({
            'textDocument': {
                'uri': self.active_text.path.as_uri()
            },
            'text': self.active_text.content
        })

    def is_free(self, timeout: float = 2.0) -> bool:
        return self.client.is_free(timeout)

    def pruned_decode(
        self,
        text_file: TextFile,
        gen_context: GenerationContext,
        trying_token_id: int,
        trying_token: str,
    ) -> int:
        """Stateful method that updates the generated token ids and tokens (excluding special
        tokens) and returns the 'real' generation"""
        # generated_ids = gen_context.generated_ids
        # input_state = pickle.dumps(generated_ids)
        # if self.use_mem:
        #     # TODO: this needs to be fixed.. DON't use mem for now
        #     denied_trie = self.mem.denied_tokens.setdefault(
        #         input_state, utils.Trie())
        #     feasible_indices = self.mem.feasible_token_ids.setdefault(
        #         input_state, set())
        #     infeasible_indices = self.mem.infeasible_token_ids.setdefault(
        #         input_state, [])
        #     # Ensures that all tokens tried are feasible
        #     considered_probs[infeasible_indices] = 0.
        # # print('Start')
        # while True:
        #     # `probs` will change each iteration (some entries will be assigned 0.)
        #     try:
        #         trying_token_indirect_id = cast(int, torch.multinomial(considered_probs, num_samples=1).item())
        #         trying_token_id = considered_token_ids[trying_token_indirect_id]
        #         assert trying_token_id.dtype == torch.long
        #         trying_token_id_item = cast(int, trying_token_id.item())
        #         assert isinstance(trying_token_id_item, int)
        #     except RuntimeError as e:
        #         return 0
        #         # Sum of probabilities < 0
        #         # if self.use_mem and len(generated_ids) > 0:
        #         #     prev_state = pickle.dumps(generated_ids[:-1])
        #         #     last_token_id = generated_ids[-1]
        #         #     self.mem.infeasible_token_ids[prev_state].append(
        #         #         last_token_id)
        #         # TODO: handle this exception
        #         # raise e
        #     # All trying tokens are feasible
        #     if self.use_mem:
        #         assert trying_token_id_item not in infeasible_indices
        #     trying_token = self.model.token_map[trying_token_id_item]
        #     # print(trying_token)

        if self.model.is_special_token(trying_token):
            assert False
        elif self.trivially_feasible(trying_token):
            assert False
        # elif self.use_mem and denied_trie.is_prefix_of(trying_token):
        #     pass
        # elif self.use_mem and trying_token_id_item in feasible_indices:
        #     return trying_token_id_item
        else:
            text_file.add(trying_token)
            self.change(text_file)
            pos = text_file.get_cursor_position()
            if self.feasible(
                gen_context.generated_ids,
                gen_context.generated_tokens,
                text_file.path.as_uri(),
                trying_token_id,
                trying_token,
                pos,
            ):
                return True
            #     if self.use_mem:
            #         feasible_indices.add(trying_token_id_item)
            #     return trying_token_id_item
            # else:
            #     text_file.delete(len(trying_token))
        return False
        # Token is infeasible if the program runs to this line
        # By setting the probability to 0.0, this token will not be selected.
        # considered_probs[trying_token_indirect_id] = 0.
        # if self.use_mem:
        #     infeasible_indices.append(trying_token_id_item)
        #     denied_trie.insert(trying_token)

    def trivially_feasible(self, token: str) -> bool:
        if len(token) > 0 and not char_may_trigger_completion(token[-1]):
            return True
        elif token.strip() in JAVA_KEYWORDS:
            return True
        else:
            return False

    def feasible(
        self,
        # generation_log: GenerationLog,
        generated_ids: List[int],
        generated_tokens: List[str],
        uri: str,
        token_id: int,
        token: str,
        pos: spec.Position,
        # completion_overhead: List[float],
    ) -> bool:
        """Returns whether `token` is feasible at `pos` of the file located at `uri`"""
        input_state = pickle.dumps(generated_ids + [token_id])
        if self.use_mem and input_state in self.mem.completions:
            # Due to memorization, each input_state be called on this function only once
            # => token_id in self.mem.(in)feasible_token_ids[state of generated_ids]
            assert False, (generated_ids, token_id, token)
            completions = self.mem.completions[input_state]
        else:
            completions = self.get_completions(uri, pos)
            self.mem.completions[input_state] = completions
            # completion_overhead.append(time.time() - start_time)
        context = {
            'ids': [id for id in generated_ids],
            'text': ''.join(generated_tokens),
            'new_token': token
        }
        print(context)
        if completions is None:
            # generation_log.append({
            #     'context': context,
            #     'result': None,
            # })
            print('UNKNOWN:', token)
            # breakpoint()
            return True
        filtered_completions = [
            c for c in completions if c['target'].startswith(c['source'])]
        # else:
        #     print(uri)
        #     print(completion_result['error'])
        #     print(completion_result['error']['data'])
        #     raise RuntimeError
        if len(filtered_completions) > 0:
            # generation_log.append({
            #     'context': context,
            #     'result': filtered_completions,
            # })
            # , token, completions)
            print('Accepted:', token,
                  f"{filtered_completions[0]['source']} -> {filtered_completions[0]['target']}")
            # if 'lastFraction' in completions:
            #     breakpoint()
            # print("Yes!")
            # print(completions)
            # self.memorization.infeasible_token_ids[input_state] = True
            return True
        else:
            # generation_log.append({
            #     'context': context,
            #     'result': [],
            # })
            # print('=======================DENIED============================')
            print('Denied', token)  # , token, completions)
            # self.memorization.infeasible_token_ids[input_state] = False
            return False

    def get_completions(self, uri: str, pos: spec.Position) -> Optional[List[dict]]:
        new_completion_result = self.completion({
            'textDocument': {
                'uri': uri
            },
            'position': pos,
        })
        return new_completion_result['result']  # type: ignore # noqa

    # # TODO: opt return type
    # # TODO: this implementation not gonna work
    # def diagnose(self, timeout: float = 0.5) -> List[dict]:
    #     self.save()
    #     while True:
    #         try:
    #             diagnostics = self.client.try_recv(timeout)
    #             if 'method' in diagnostics and (diagnostics := cast(
    #                 spec.RequestMessage,
    #                 diagnostics
    #             ))['method'] == 'textDocument/publishDiagnostics' and \
    #                     cast(Dict[str, str], diagnostics['params'])['uri'] == self.active_text.path.as_uri():
    #                 # TODO: diagnostics LSP spec type
    #                 return diagnostics['params']['diagnostics']  # type: ignore # noqa
    #             # print(diagnostics)
    #         except TimeoutError:
    #             return None  # type: ignore
