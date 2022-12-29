import itertools
import json
from os import PathLike
from pathlib import Path
from typing import Dict, List, cast
from realm.lsp import LSPClient, spec
import subprocess
from realm.lsp import TextFile


# def post_process(get_client: Callable[[C], LSPClient], rpc: Callable[Concatenate[LSPClient, P], Msg]) \
#         -> Callable[[Callable[[Msg], T]], Callable[Concatenate[C, P], T]]:
#     def decorator(f: Callable[[Msg], T]) -> Callable[Concatenate[C, P], T]:
#         @wraps(f)
#         def impl(self: C, *args: P.args, **kwargs: P.kwargs) -> T:
#             client = get_client(self)
#             return f(rpc(client, *args, **kwargs))
#         return impl
#     return decorator


class JdtLspAnalyzer:
    """Jdt LSP based Java program analyzer leveraging whole-project information.
    Now assume only one active file for diagnosis"""
    counter = itertools.count(0)

    def stop(self):
        self.process.terminate()
    
    def copy(self) -> 'JdtLspAnalyzer':
        return JdtLspAnalyzer(
            self.server_cmd,
            self.proj_path,
            self.java_home,
            self.verbose
        )

    def __init__(self, server_cmd: List[str], proj_path: PathLike, java_home: str, verbose: bool = False) -> None:
        self.server_cmd = server_cmd
        self.proj_path = proj_path
        self.java_home = java_home
        self.verbose = verbose
        self.process = subprocess.Popen(
            self.server_cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        assert self.process.stdin is not None and self.process.stdout is not None
    
    def init_client(self):
        self.client = LSPClient(
            self.process.stdin, self.process.stdout, self.verbose, 120)
        self.client.start()


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

    def save(self):
        self.client.textDocument_didSave({
            'textDocument': {
                'uri': self.active_text.path.as_uri()
            },
            'text': self.active_text.content
        })

    # TODO: opt return type
    # TODO: this implementation not gonna work
    def diagnose(self, timeout: float = 0.5) -> List[dict]:
        self.save()
        while True:
            try:
                diagnostics = self.client.try_recv(timeout)
                if 'method' in diagnostics and (diagnostics := cast(
                    spec.RequestMessage,
                    diagnostics
                ))['method'] == 'textDocument/publishDiagnostics' and \
                        cast(Dict[str, str], diagnostics['params'])['uri'] == self.active_text.path.as_uri():
                    # TODO: diagnostics LSP spec type
                    return diagnostics['params']['diagnostics']  # type: ignore # noqa
                # print(diagnostics)
            except TimeoutError:
                return None  # type: ignore
