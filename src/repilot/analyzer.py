import itertools
import subprocess
from multiprocessing import Process
from multiprocessing.connection import Connection
from os import PathLike
from pathlib import Path
from typing import Any, Optional, cast

from repilot import utils
from repilot.generation_defs import GenerationContext
from repilot.lsp import LSPClient, TextFile, spec
from repilot.model import ModelType

TIMEOUT_THRESHOULD = 300


class Message:
    def __init__(
        self, return_result: bool, method: str, *args: Any, **kwargs: Any
    ) -> None:
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
        server_cmd: list[str],
        proj_path: PathLike,
        model: ModelType,
        java8_home: str,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.conn = conn
        self.server_cmd = server_cmd
        self.proj_path = proj_path
        # TODO: change this using config
        self.java8_home = java8_home
        self.verbose = verbose
        self.counter = itertools.count(0)
        self.model = model
        self.mem: dict[bytes, Optional[list[dict]]] = {}

    def init_lsp(self):
        self.process = subprocess.Popen(
            self.server_cmd,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
        assert self.process.stdin is not None and self.process.stdout is not None
        self.client = LSPClient(
            self.process.stdin, self.process.stdout, self.verbose, TIMEOUT_THRESHOULD
        )
        self.client.start()

    def stop_lsp(self):
        self.client.shutdown(None)
        self.client.exit(None)
        self.client.join()
        # Faster
        # self.process.terminate()
        # The code below also works but is slower
        return_code = self.process.wait()
        assert return_code == 0

    def run(self) -> None:
        # Start the thread
        self.init_lsp()
        while True:
            message: Optional[Message] = self.conn.recv()
            if message is None:
                break
            # print('RECEIVED:', message.method)
            assert isinstance(message, Message)
            result = getattr(self, message.method)(*message.args, **message.kwargs)
            if message.return_result:
                # print('RESULT:', result)
                self.conn.send(result)
        self.stop_lsp()
        print("Analyzer terminated")

    def get_diagnostics(self) -> list[spec.ResponseMessage]:
        return self.client.current_diagnostics

    def clear_diagnostics(self):
        self.client.current_diagnostics.clear()

    def init(self) -> spec.ResponseMessage:
        # self.active_text: Optional[TextDocument] = None

        # Initialize the server
        path = Path(self.proj_path)
        # with open('log1.json', 'w') as f:
        msg = self.client.initialize(
            {
                "processId": self.process.pid,
                "clientInfo": {"name": path.name, "version": "0.0.0"},
                "locale": "en",
                "rootPath": str(path.absolute()),
                "rootUri": path.as_uri(),
                "capabilities": spec.ClientCapabilities(
                    {
                        # "workspace": {
                        #     "applyEdit": True,
                        #     "workspaceEdit": {
                        #         "documentChanges": True,
                        #         "resourceOperations": [
                        #             "create",
                        #             "rename",
                        #             "delete"
                        #         ],
                        #         "failureHandling": "textOnlyTransactional",
                        #         "normalizesLineEndings": True,
                        #         "changeAnnotationSupport": {
                        #             "groupsOnLabel": True
                        #         }
                        #     },
                        #     "didChangeConfiguration": {
                        #         "dynamicRegistration": True
                        #     },
                        #     "didChangeWatchedFiles": {
                        #         "dynamicRegistration": True
                        #     },
                        #     "symbol": {
                        #         "dynamicRegistration": True,
                        #         "symbolKind": {
                        #             "valueSet": [
                        #                 1,
                        #                 2,
                        #                 3,
                        #                 4,
                        #                 5,
                        #                 6,
                        #                 7,
                        #                 8,
                        #                 9,
                        #                 10,
                        #                 11,
                        #                 12,
                        #                 13,
                        #                 14,
                        #                 15,
                        #                 16,
                        #                 17,
                        #                 18,
                        #                 19,
                        #                 20,
                        #                 21,
                        #                 22,
                        #                 23,
                        #                 24,
                        #                 25,
                        #                 26
                        #             ]
                        #         },
                        #         "tagSupport": {
                        #             "valueSet": [
                        #                 1
                        #             ]
                        #         }
                        #     },
                        #     "codeLens": {
                        #         "refreshSupport": True
                        #     },
                        #     "executeCommand": {
                        #         "dynamicRegistration": True
                        #     },
                        #     "configuration": True,
                        #     "workspaceFolders": True,
                        #     "semanticTokens": {
                        #         "refreshSupport": True
                        #     },
                        #     "fileOperations": {
                        #         "dynamicRegistration": True,
                        #         "didCreate": True,
                        #         "didRename": True,
                        #         "didDelete": True,
                        #         "willCreate": True,
                        #         "willRename": True,
                        #         "willDelete": True
                        #     }
                        # },
                        "textDocument": {
                            # "publishDiagnostics": {
                            #     "relatedInformation": True,
                            #     "versionSupport": False,
                            #     "tagSupport": {
                            #         "valueSet": [
                            #             1,
                            #             2
                            #         ]
                            #     },
                            #     "codeDescriptionSupport": True,
                            #     "dataSupport": True
                            # },
                            "synchronization": {
                                "dynamicRegistration": True,
                                "willSave": True,
                                "willSaveWaitUntil": True,
                                "didSave": True,
                            },
                            "completion": {
                                "dynamicRegistration": True,
                                "contextSupport": True,
                                "completionItem": {
                                    "snippetSupport": True,
                                    "commitCharactersSupport": True,
                                    "documentationFormat": ["markdown", "plaintext"],
                                    "deprecatedSupport": True,
                                    "preselectSupport": True,
                                    "tagSupport": {"valueSet": [1]},
                                    "insertReplaceSupport": True,
                                    "resolveSupport": {
                                        "properties": [
                                            "documentation",
                                            "detail",
                                            "additionalTextEdits",
                                        ]
                                    },
                                    "insertTextModeSupport": {"valueSet": [1, 2]},
                                    "labelDetailsSupport": True,
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
                                        25,
                                    ]
                                },
                            },
                            # "hover": {
                            #     "dynamicRegistration": True,
                            #     "contentFormat": [
                            #         "markdown",
                            #         "plaintext"
                            #     ]
                            # },
                            # "signatureHelp": {
                            #     "dynamicRegistration": True,
                            #     "signatureInformation": {
                            #         "documentationFormat": [
                            #             "markdown",
                            #             "plaintext"
                            #         ],
                            #         "parameterInformation": {
                            #             "labelOffsetSupport": True
                            #         },
                            #         "activeParameterSupport": True
                            #     },
                            #     "contextSupport": True
                            # },
                            # "definition": {
                            #     "dynamicRegistration": True,
                            #     "linkSupport": True
                            # },
                            # "references": {
                            #     "dynamicRegistration": True
                            # },
                            # "documentHighlight": {
                            #     "dynamicRegistration": True
                            # },
                            # "documentSymbol": {
                            #     "dynamicRegistration": True,
                            #     "symbolKind": {
                            #         "valueSet": [
                            #             1,
                            #             2,
                            #             3,
                            #             4,
                            #             5,
                            #             6,
                            #             7,
                            #             8,
                            #             9,
                            #             10,
                            #             11,
                            #             12,
                            #             13,
                            #             14,
                            #             15,
                            #             16,
                            #             17,
                            #             18,
                            #             19,
                            #             20,
                            #             21,
                            #             22,
                            #             23,
                            #             24,
                            #             25,
                            #             26
                            #         ]
                            #     },
                            #     "hierarchicalDocumentSymbolSupport": True,
                            #     "tagSupport": {
                            #         "valueSet": [
                            #             1
                            #         ]
                            #     },
                            #     "labelSupport": True
                            # },
                            # "codeAction": {
                            #     "dynamicRegistration": True,
                            #     "isPreferredSupport": True,
                            #     "disabledSupport": True,
                            #     "dataSupport": True,
                            #     "resolveSupport": {
                            #         "properties": [
                            #             "edit"
                            #         ]
                            #     },
                            #     "codeActionLiteralSupport": {
                            #         "codeActionKind": {
                            #             "valueSet": [
                            #                 "",
                            #                 "quickfix",
                            #                 "refactor",
                            #                 "refactor.extract",
                            #                 "refactor.inline",
                            #                 "refactor.rewrite",
                            #                 "source",
                            #                 "source.organizeImports"
                            #             ]
                            #         }
                            #     },
                            #     "honorsChangeAnnotations": False
                            # },
                            # "codeLens": {
                            #     "dynamicRegistration": True
                            # },
                            # "formatting": {
                            #     "dynamicRegistration": True
                            # },
                            # "rangeFormatting": {
                            #     "dynamicRegistration": True
                            # },
                            # "onTypeFormatting": {
                            #     "dynamicRegistration": True
                            # },
                            # "rename": {
                            #     "dynamicRegistration": True,
                            #     "prepareSupport": True,
                            #     "prepareSupportDefaultBehavior": 1,
                            #     "honorsChangeAnnotations": True
                            # },
                            # "documentLink": {
                            #     "dynamicRegistration": True,
                            #     "tooltipSupport": True
                            # },
                            # "typeDefinition": {
                            #     "dynamicRegistration": True,
                            #     "linkSupport": True
                            # },
                            # "implementation": {
                            #     "dynamicRegistration": True,
                            #     "linkSupport": True
                            # },
                            # "colorProvider": {
                            #     "dynamicRegistration": True
                            # },
                            # "foldingRange": {
                            #     "dynamicRegistration": True,
                            #     "rangeLimit": 5000,
                            #     "lineFoldingOnly": True
                            # },
                            # "declaration": {
                            #     "dynamicRegistration": True,
                            #     "linkSupport": True
                            # },
                            # "selectionRange": {
                            #     "dynamicRegistration": True
                            # },
                            # "callHierarchy": {
                            #     "dynamicRegistration": True
                            # },
                            # "semanticTokens": {
                            #     "dynamicRegistration": True,
                            #     "tokenTypes": [
                            #         "namespace",
                            #         "type",
                            #         "class",
                            #         "enum",
                            #         "interface",
                            #         "struct",
                            #         "typeParameter",
                            #         "parameter",
                            #         "variable",
                            #         "property",
                            #         "enumMember",
                            #         "event",
                            #         "function",
                            #         "method",
                            #         "macro",
                            #         "keyword",
                            #         "modifier",
                            #         "comment",
                            #         "string",
                            #         "number",
                            #         "regexp",
                            #         "operator"
                            #     ],
                            #     "tokenModifiers": [
                            #         "declaration",
                            #         "definition",
                            #         "readonly",
                            #         "static",
                            #         "deprecated",
                            #         "abstract",
                            #         "async",
                            #         "modification",
                            #         "documentation",
                            #         "defaultLibrary"
                            #     ],
                            #     "formats": [
                            #         "relative"
                            #     ],
                            #     "requests": {
                            #         "range": True,
                            #         "full": {
                            #             "delta": True
                            #         }
                            #     },
                            #     "multilineTokenSupport": False,
                            #     "overlappingTokenSupport": False
                            # },
                            # "linkedEditingRange": {
                            #     "dynamicRegistration": True
                            # }
                        },
                        # "window": {
                        #     "showMessage": {
                        #         "messageActionItem": {
                        #             "additionalPropertiesSupport": True
                        #         }
                        #     },
                        #     "showDocument": {
                        #         "support": True
                        #     },
                        #     "workDoneProgress": True
                        # },
                        # "general": {
                        #     "staleRequestSupport": {
                        #         "cancel": True,
                        #         "retryOnContentModified": [
                        #             "textDocument/semanticTokens/full",
                        #             "textDocument/semanticTokens/range",
                        #             "textDocument/semanticTokens/full/delta"
                        #         ]
                        #     },
                        #     "regularExpressions": {
                        #         "engine": "ECMAScript",
                        #         "version": "ES2020"
                        #     },
                        #     "markdown": {
                        #         "parser": "marked",
                        #         "version": "1.1.0"
                        #     }
                        # }
                    }
                ),
                "initializationOptions": {
                    "bundles": [],
                    "workspaceFolders": [path.as_uri()],
                    "settings": {
                        "java": {
                            # "home": java8_home,
                            "jdt": {
                                "ls": {
                                    "java": {
                                        "home": None,
                                    },
                                    "vmargs": "-XX:+UseParallelGC -XX:GCTimeRatio=4 -XX:AdaptiveSizePolicyWeight=90 -Dsun.zip.disableMemoryMapping=true -Xmx1G -Xms100m",
                                    "lombokSupport": {"enabled": True},
                                }
                            },
                            "errors": {"incompleteClasspath": {"severity": "warning"}},
                            "configuration": {
                                "checkProjectSettingsExclusions": False,
                                "updateBuildConfiguration": "interactive",
                                "maven": {
                                    "userSettings": None,
                                    "globalSettings": None,
                                    "notCoveredPluginExecutionSeverity": "warning",
                                },
                                "workspaceCacheLimit": 90,
                                "runtimes": [
                                    {
                                        "name": "JavaSE-1.8",
                                        "path": self.java8_home,
                                        "default": True,
                                    },
                                ],
                            },
                            "trace": {"server": "off"},
                            "import": {
                                "maven": {"enabled": True},
                                "gradle": {
                                    "enabled": True,
                                    "wrapper": {"enabled": True},
                                    "version": None,
                                    "home": None,
                                    "java": {
                                        "home": None,
                                    },
                                    "offline": {"enabled": False},
                                    "arguments": None,
                                    "jvmArguments": None,
                                    "user": {
                                        "home": None,
                                    },
                                },
                                "exclusions": [
                                    "**/node_modules/**",
                                    "**/.metadata/**",
                                    "**/archetype-resources/**",
                                    "**/META-INF/maven/**",
                                ],
                                "generatesMetadataFilesAtProjectRoot": False,
                            },
                            "maven": {
                                "downloadSources": False,
                                "updateSnapshots": False,
                            },
                            "eclipse": {"downloadSources": False},
                            "referencesCodeLens": {"enabled": False},
                            "signatureHelp": {
                                "enabled": False,
                                "description": {"enabled": False},
                            },
                            "implementationsCodeLens": {"enabled": False},
                            "format": {
                                "enabled": True,
                                "settings": {
                                    "url": None,
                                    "profile": None,
                                },
                                "comments": {"enabled": True},
                                "onType": {"enabled": True},
                                "insertSpaces": True,
                                "tabSize": 4,
                            },
                            "saveActions": {"organizeImports": False},
                            "project": {
                                "referencedLibraries": ["lib/**/*.jar"],
                                "importOnFirstTimeStartup": "automatic",
                                "importHint": True,
                                "resourceFilters": ["node_modules", ".git"],
                                "encoding": "ignore",
                            },
                            "contentProvider": {
                                "preferred": None,
                            },
                            "autobuild": {"enabled": True},
                            "maxConcurrentBuilds": 1,
                            "recommendations": {
                                "dependency": {"analytics": {"show": True}}
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
                                    "org.mockito.Answers.*",
                                ],
                                "filteredTypes": [
                                    "java.awt.*",
                                    "com.sun.*",
                                    "sun.*",
                                    "jdk.*",
                                    "org.graalvm.*",
                                    "io.micrometer.shaded.*",
                                ],
                                "importOrder": ["java", "javax", "org", "com"],
                            },
                            "foldingRange": {"enabled": False},
                            "progressReports": {"enabled": False},
                            "codeGeneration": {
                                "hashCodeEquals": {
                                    "useJava7Objects": False,
                                    "useInstanceof": False,
                                },
                                "useBlocks": False,
                                "generateComments": False,
                                "toString": {
                                    "template": "${object.className} [${member.name()}=${member.value}, ${otherMembers}]",
                                    "codeStyle": "STRING_CONCATENATION",
                                    "SkipNullValues": False,
                                    "listArrayContents": True,
                                    "limitElements": 0,
                                },
                                "insertionLocation": "afterCursor",
                            },
                            "selectionRange": {"enabled": True},
                            "showBuildStatusOnStart": {"enabled": "off"},
                            "server": {"launchMode": "Hybrid"},
                            "sources": {
                                "organizeImports": {
                                    "starThreshold": 99,
                                    "staticStarThreshold": 99,
                                }
                            },
                            "imports": {"gradle": {"wrapper": {"checksums": []}}},
                            "templates": {"fileHeader": [], "typeComment": []},
                            "references": {
                                "includeAccessors": True,
                                "includeDecompiledSources": True,
                            },
                            "typeHierarchy": {"lazyLoad": False},
                            "settings": {
                                "url": None,
                            },
                            "symbols": {"includeSourceMethodDeclarations": False},
                            "quickfix": {"showAt": "line"},
                            "inlayHints": {
                                "parameterNames": {
                                    "enabled": "literals",
                                    "exclusions": [],
                                }
                            },
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
                            "extractField",
                        ],
                        "moveRefactoringSupport": True,
                        "clientHoverProvider": True,
                        "clientDocumentSymbolProvider": True,
                        "gradleChecksumWrapperPromptSupport": True,
                        "resolveAdditionalTextEditsSupport": True,
                        "advancedIntroduceParameterRefactoringSupport": True,
                        "actionableRuntimeNotificationSupport": True,
                        "shouldLanguageServerExitOnShutdown": True,
                        "onCompletionItemSelectedCommand": "editor.action.triggerParameterHints",
                    },
                    "triggerFiles": [
                        # "file:///Users/nole/Developer/javasymbolsolver-maven-sample/src/main/java/com/yourorganization/maven_sample/MyAnalysis.java"
                    ],
                },
                # "trace": "verbose",
                "workspaceFolders": [
                    {
                        "uri": path.as_uri(),
                        "name": path.name,
                    }
                ],
            }
        )
        # with open('log') as f, open('config.json') as f1:
        #     d = json.load(f)
        #     d1 = json.load(f1)
        #     d['processId'] = self.process.pid
        #     self.client.initialize(d)
        self.client.initialized(cast(spec.InitializedParams, {}))
        # self.client.workspace_didChangeConfiguration(d1)
        return msg

    def sync(self, text: TextFile):
        self.active_text = text

    def open(self, text: TextFile):
        self.sync(text)
        self.client.textDocument_didOpen(
            cast(
                spec.DidOpenTextDocumentParams,
                {
                    "textDocument": {
                        "uri": text.path.as_uri(),
                        "languageId": "java",
                        "version": next(self.counter),
                        "text": text.content,
                    }
                },
            )
        )

    def change(self, text: TextFile):
        self.sync(text)
        self.client.textDocument_didChange(
            {
                "textDocument": {
                    "uri": text.path.as_uri(),
                    "version": next(self.counter),
                },
                "contentChanges": [
                    {
                        "text": text.content,
                    }
                ],
            }
        )

    def completion(self, params: spec.CompletionParams) -> spec.ResponseMessage:
        return self.client.newCompletion(params)

    def save(self):
        self.client.textDocument_didSave(
            {
                "textDocument": {"uri": self.active_text.path.as_uri()},
                "text": self.active_text.content,
            }
        )

    def is_free(self, timeout) -> bool:
        return self.client.is_free(timeout)

    def pruned_decode(
        self,
        text_file: TextFile,
        gen_context: GenerationContext,
        trying_token_id: int,
        trying_token: str,
    ) -> bool | str:
        """Use language server to check validity"""
        text_file.add(trying_token)
        self.change(text_file)
        pos = text_file.get_cursor_position()
        result = self.feasible(
            gen_context.generated_ids,
            gen_context.generated_tokens,
            text_file.path.as_uri(),
            trying_token_id,
            trying_token,
            pos,
        )
        if result == True:
            return True
        elif result == False:
            return False
        else:
            assert isinstance(result, list)
            assert len(result) > 0
            return utils.longest_common_prefix(result)

    def feasible(
        self,
        # generation_log: GenerationLog,
        generated_ids: list[int],
        generated_tokens: list[str],
        uri: str,
        token_id: int,
        token: str,
        pos: spec.Position,
        # completion_overhead: list[float],
    ) -> bool | list[str]:
        """Returns whether `token` is feasible at `pos` of the file located at `uri`"""
        # if self.use_mem and input_state in self.mem.completions:
        #     # Due to memorization, each input_state be called on this function only once
        #     # => token_id in self.mem.(in)feasible_token_ids[state of generated_ids]
        #     assert False, (generated_ids, token_id, token)
        #     completions = self.mem.completions[input_state]
        # else:
        # TODO: memorize completions
        completions = self.get_completions(uri, pos)

        # Memorize
        # input_state = pickle.dumps(generated_ids + [token_id])
        # self.mem.setdefault(input_state, completions)

        # self.mem.completions[input_state] = completions
        # completion_overhead.append(time.time() - start_time)
        context = {
            "ids": [id for id in generated_ids],
            "text": "".join(generated_tokens),
            "new_token": token,
        }
        print(context)
        if completions is None:
            # generation_log.append({
            #     'context': context,
            #     'result': None,
            # })
            print("UNKNOWN:", token)
            # breakpoint()
            return True
        # filtered_completions = [
        #     c for c in completions if c["target"].startswith(c["source"])
        # ]
        continuations = [
            # item if not (item := target[len(source) :]).endswith("(") else item[:-1]
            target[len(source) :]
            for c in completions
            if (target := c["target"]).startswith(source := c["source"])
        ]
        # else:
        #     print(uri)
        #     print(completion_result['error'])
        #     print(completion_result['error']['data'])
        #     raise RuntimeError
        if len(continuations) > 0:
            # generation_log.append({
            #     'context': context,
            #     'result': filtered_completions,
            # })
            # , token, completions)
            print("Accepted:", token, f"Completions[0]: {continuations[0]}")
            # if 'lastFraction' in completions:
            #     breakpoint()
            # print("Yes!")
            # print(completions)
            # self.memorization.infeasible_token_ids[input_state] = True
            return continuations
        else:
            # generation_log.append({
            #     'context': context,
            #     'result': [],
            # })
            # print('=======================DENIED============================')
            print("Denied", token)  # , token, completions)
            # self.memorization.infeasible_token_ids[input_state] = False
            return False

    # def continuation(self, generated_ids: list[int], text_file: TextFile) -> str | None:
    #     state = pickle.dumps(generated_ids)
    #     if state in self.mem:
    #         completions = self.mem[state]
    #     else:
    #         self.change(text_file)
    #         completions = self.get_completions(
    #             text_file.path.as_uri(), text_file.get_cursor_position()
    #         )
    #     if completions is None:
    #         return None
    #     continuations = [
    #         item if not (item := target[len(source) :]).endswith("(") else item[:-1]
    #         for c in completions
    #         if (target := c["target"]).startswith(source := c["source"])
    #     ]
    #     # continuations: list[str] = [
    #     #     target[len(source) :]
    #     #     for c in completions
    #     #     if (target := c["target"]).startswith(source := c["source"])
    #     # ]
    #     completion = utils.longest_common_prefix(continuations)
    #     return completion

    def get_completions(self, uri: str, pos: spec.Position) -> Optional[list[dict]]:
        new_completion_result = self.completion(
            {
                "textDocument": {"uri": uri},
                "position": pos,
            }
        )
        return new_completion_result["result"]  # type: ignore # noqa
