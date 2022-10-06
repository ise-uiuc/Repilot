from select import select
import json
from threading import Condition, Lock, Thread
from typing import IO, Callable, Dict, Optional, Tuple, TypeVar, Type, cast
from typing import ParamSpec
from itertools import count
from . import spec

# TODO: support Content-Type
HEADER = 'Content-Length: '


def add_header(content_str: str) -> str:
    return f'Content-Length: {len(content_str)}\r\n\r\n{content_str}'


def request_getter(init_id: int):
    request_id_gen = count(init_id)

    def request(method: spec.string, params: spec.array | spec.object) -> spec.RequestMessage:
        return {
            'jsonrpc': '2.0',
            'method': method,
            'params': params,
            'id': next(request_id_gen)
        }
    return request


request = request_getter(0)


def notification(method: spec.string, params: spec.array | spec.object) -> spec.NotificationMessage:
    return {
        'jsonrpc': '2.0',
        'method': method,
        'params': params,
    }


def response(
    id: spec.integer | spec.string | spec.null,
    result: spec.string | spec.number | spec.boolean | spec.object | spec.null
) -> spec.ResponseMessage:
    return {
        'jsonrpc': '2.0',
        'id': id,
        'result': result,
    }


P = ParamSpec('P')
RPC = Tuple[str, dict]
T = TypeVar('T')


def d_call(method: str, _: Type[T]) -> Callable[['LSPClient', T], spec.ResponseMessage]:
    def impl(self: 'LSPClient', params: T) -> spec.ResponseMessage:
        return self.call(method, params)
    return impl


def d_notify(method: str, _: Type[T]) -> Callable[['LSPClient', T], None]:
    def impl(self: 'LSPClient', params: T) -> None:
        self.notify(method, params)
    return impl


class LSPClient:
    def __init__(self, stdin: IO[bytes], stdout: IO[bytes], verbose: bool, timeout: float):
        super().__init__()
        self.stdin = stdin
        self.stdout = stdout
        self.verbose = verbose
        self.responses: Dict[
            spec.integer | str | None, 
            spec.ResponseMessage
        ] = {}
        self.timeout = timeout
    
    # def run(self) -> None:
    #     while not self.stopped:
    #         server_response = self.recv()
    #         if 'method' in server_response and 'id' in server_response:
    #             # print(server_response)
    #             server_response = cast(spec.RequestMessage, server_response)
    #             self.send(response(server_response['id'], None))
    #         elif 'id' in server_response:
    #             server_response = cast(spec.ResponseMessage, server_response)
    #             id = server_response['id']
    #             self.responses[id] = server_response
    #             self.lock.acquire()
    #             self.lock.notify()
    #             self.lock.release()
    #         # else:
    #         #     assert False, server_response

    def call(self, method: str, params: spec.array | spec.object) -> spec.ResponseMessage:
        message = request(method, params)
        id = message['id']
        self.send(message)
        # return self.responses.pop(id)
        while True:
            server_response = self.recv()
            # and server_response['method'] == 'client/registerCapability':
            # print(server_response)
            if 'method' in server_response and 'id' in server_response:
                # print(server_response)
                server_response = cast(spec.RequestMessage, server_response)
                self.send(response(server_response['id'], None))
            if 'id' in server_response:
                server_response = cast(spec.ResponseMessage, server_response)
                if server_response['id'] == id:
                    return server_response

    def notify(self, method: str, params: spec.array | spec.object):
        # if 'textDocument/didChange' == method:
        #     print(notification(method, params))
        self.send(notification(method, params))
        # if method == 'textDocument/didSave':
        #     while True:
        #         self.recv()

    def send(self, message: spec.RequestMessage | spec.ResponseMessage | spec.NotificationMessage):
        content = json.dumps(message)
        content = add_header(content)
        if self.verbose:
            print('SEND:', message)
        self.stdin.write(content.encode())
        self.stdin.flush()

    def try_recv(self) -> spec.ResponseMessage | spec.RequestMessage | spec.NotificationMessage:
        reader, _, _ = select([self.stdout], [], [], self.timeout)
        if len(reader) == 0:
            raise TimeoutError
        return self.recv()

    def recv(self) -> spec.ResponseMessage | spec.RequestMessage | spec.NotificationMessage:
        # read header
        line = self.stdout.readline().decode()
        assert line.endswith('\r\n'), repr(line)
        assert line.startswith(HEADER) and line.endswith('\r\n'), line

        # get content length
        content_len = int(line[len(HEADER):].strip())
        line_breaks = self.stdout.readline().decode()
        assert line_breaks == '\r\n', line_breaks
        response = self.stdout.read(content_len).decode()
        # if 'textDocument/publishDiagnostics' in response:
        if self.verbose:
            print(response)
        return json.loads(response)

    initialize = d_call('initialize', spec.InitializeParams)
    initialized = d_notify('initialized', spec.InitializedParams)
    workspace_didChangeConfiguration = d_notify(
        'textDocument/didChangeConfiguration', dict)
    textDocument_didOpen = d_notify(
        'textDocument/didOpen', spec.DidOpenTextDocumentParams)
    textDocument_didSave = d_notify(
        'textDocument/didSave', spec.DidSaveTextDocumentParams)
    textDocument_didChange = d_notify(
        'textDocument/didChange', spec.DidChangeTextDocumentParams)
    textDocument_completion = d_call(
        'textDocument/completion', spec.CompletionParams)

    # @d_call
    # def _initialize(pid: int, proj_path: PathLike, java_home: str) -> RPC:  # type: ignore[misc] # noqa
    #     path = Path(proj_path)
    #     return 'initialize', {
    #         "processId": pid,
    #         "clientInfo": {
    #             "name": path.name,
    #             "version": "0.0.0"
    #         },
    #         "locale": "en-us",
    #         "rootPath": str(path.absolute()),
    #         "rootUri": path.as_uri(),
    #         "capabilities": {
    #             "workspace": {
    #                 "applyEdit": True,
    #                 "workspaceEdit": {
    #                     "documentChanges": True,
    #                     "resourceOperations": [
    #                         "create",
    #                         "rename",
    #                         "delete"
    #                     ],
    #                     "failureHandling": "textOnlyTransactional",
    #                     "normalizesLineEndings": True,
    #                     "changeAnnotationSupport": {
    #                         "groupsOnLabel": True
    #                     }
    #                 },
    #                 "didChangeConfiguration": {
    #                     "dynamicRegistration": True
    #                 },
    #                 "didChangeWatchedFiles": {
    #                     "dynamicRegistration": True
    #                 },
    #                 "symbol": {
    #                     "dynamicRegistration": True,
    #                     "symbolKind": {
    #                         "valueSet": [
    #                             1,
    #                             2,
    #                             3,
    #                             4,
    #                             5,
    #                             6,
    #                             7,
    #                             8,
    #                             9,
    #                             10,
    #                             11,
    #                             12,
    #                             13,
    #                             14,
    #                             15,
    #                             16,
    #                             17,
    #                             18,
    #                             19,
    #                             20,
    #                             21,
    #                             22,
    #                             23,
    #                             24,
    #                             25,
    #                             26
    #                         ]
    #                     },
    #                     "tagSupport": {
    #                         "valueSet": [
    #                             1
    #                         ]
    #                     }
    #                 },
    #                 "codeLens": {
    #                     "refreshSupport": True
    #                 },
    #                 "executeCommand": {
    #                     "dynamicRegistration": True
    #                 },
    #                 "configuration": True,
    #                 "workspaceFolders": True,
    #                 "semanticTokens": {
    #                     "refreshSupport": True
    #                 },
    #                 "fileOperations": {
    #                     "dynamicRegistration": True,
    #                     "didCreate": True,
    #                     "didRename": True,
    #                     "didDelete": True,
    #                     "willCreate": True,
    #                     "willRename": True,
    #                     "willDelete": True
    #                 }
    #             },
    #             "textDocument": {
    #                 "publishDiagnostics": {
    #                     "relatedInformation": True,
    #                     "versionSupport": False,
    #                     "tagSupport": {
    #                         "valueSet": [
    #                             1,
    #                             2
    #                         ]
    #                     },
    #                     "codeDescriptionSupport": True,
    #                     "dataSupport": True
    #                 },
    #                 "synchronization": {
    #                     "dynamicRegistration": True,
    #                     "willSave": True,
    #                     "willSaveWaitUntil": True,
    #                     "didSave": True
    #                 },
    #                 "completion": {
    #                     "dynamicRegistration": True,
    #                     "contextSupport": True,
    #                     "completionItem": {
    #                         "snippetSupport": True,
    #                         "commitCharactersSupport": True,
    #                         "documentationFormat": [
    #                             "markdown",
    #                             "plaintext"
    #                         ],
    #                         "deprecatedSupport": True,
    #                         "preselectSupport": True,
    #                         "tagSupport": {
    #                             "valueSet": [
    #                                 1
    #                             ]
    #                         },
    #                         "insertReplaceSupport": True,
    #                         "resolveSupport": {
    #                             "properties": [
    #                                 "documentation",
    #                                 "detail",
    #                                 "additionalTextEdits"
    #                             ]
    #                         },
    #                         "insertTextModeSupport": {
    #                             "valueSet": [
    #                                 1,
    #                                 2
    #                             ]
    #                         },
    #                         "labelDetailsSupport": True
    #                     },
    #                     "insertTextMode": 2,
    #                     "completionItemKind": {
    #                         "valueSet": [
    #                             1,
    #                             2,
    #                             3,
    #                             4,
    #                             5,
    #                             6,
    #                             7,
    #                             8,
    #                             9,
    #                             10,
    #                             11,
    #                             12,
    #                             13,
    #                             14,
    #                             15,
    #                             16,
    #                             17,
    #                             18,
    #                             19,
    #                             20,
    #                             21,
    #                             22,
    #                             23,
    #                             24,
    #                             25
    #                         ]
    #                     }
    #                 },
    #                 "hover": {
    #                     "dynamicRegistration": True,
    #                     "contentFormat": [
    #                         "markdown",
    #                         "plaintext"
    #                     ]
    #                 },
    #                 "signatureHelp": {
    #                     "dynamicRegistration": True,
    #                     "signatureInformation": {
    #                         "documentationFormat": [
    #                             "markdown",
    #                             "plaintext"
    #                         ],
    #                         "parameterInformation": {
    #                             "labelOffsetSupport": True
    #                         },
    #                         "activeParameterSupport": True
    #                     },
    #                     "contextSupport": True
    #                 },
    #                 "definition": {
    #                     "dynamicRegistration": True,
    #                     "linkSupport": True
    #                 },
    #                 "references": {
    #                     "dynamicRegistration": True
    #                 },
    #                 "documentHighlight": {
    #                     "dynamicRegistration": True
    #                 },
    #                 "documentSymbol": {
    #                     "dynamicRegistration": True,
    #                     "symbolKind": {
    #                         "valueSet": [
    #                             1,
    #                             2,
    #                             3,
    #                             4,
    #                             5,
    #                             6,
    #                             7,
    #                             8,
    #                             9,
    #                             10,
    #                             11,
    #                             12,
    #                             13,
    #                             14,
    #                             15,
    #                             16,
    #                             17,
    #                             18,
    #                             19,
    #                             20,
    #                             21,
    #                             22,
    #                             23,
    #                             24,
    #                             25,
    #                             26
    #                         ]
    #                     },
    #                     "hierarchicalDocumentSymbolSupport": True,
    #                     "tagSupport": {
    #                         "valueSet": [
    #                             1
    #                         ]
    #                     },
    #                     "labelSupport": True
    #                 },
    #                 "codeAction": {
    #                     "dynamicRegistration": True,
    #                     "isPreferredSupport": True,
    #                     "disabledSupport": True,
    #                     "dataSupport": True,
    #                     "resolveSupport": {
    #                         "properties": [
    #                             "edit"
    #                         ]
    #                     },
    #                     "codeActionLiteralSupport": {
    #                         "codeActionKind": {
    #                             "valueSet": [
    #                                 "",
    #                                 "quickfix",
    #                                 "refactor",
    #                                 "refactor.extract",
    #                                 "refactor.inline",
    #                                 "refactor.rewrite",
    #                                 "source",
    #                                 "source.organizeImports"
    #                             ]
    #                         }
    #                     },
    #                     "honorsChangeAnnotations": False
    #                 },
    #                 "codeLens": {
    #                     "dynamicRegistration": True
    #                 },
    #                 "formatting": {
    #                     "dynamicRegistration": True
    #                 },
    #                 "rangeFormatting": {
    #                     "dynamicRegistration": True
    #                 },
    #                 "onTypeFormatting": {
    #                     "dynamicRegistration": True
    #                 },
    #                 "rename": {
    #                     "dynamicRegistration": True,
    #                     "prepareSupport": True,
    #                     "prepareSupportDefaultBehavior": 1,
    #                     "honorsChangeAnnotations": True
    #                 },
    #                 "documentLink": {
    #                     "dynamicRegistration": True,
    #                     "tooltipSupport": True
    #                 },
    #                 "typeDefinition": {
    #                     "dynamicRegistration": True,
    #                     "linkSupport": True
    #                 },
    #                 "implementation": {
    #                     "dynamicRegistration": True,
    #                     "linkSupport": True
    #                 },
    #                 "colorProvider": {
    #                     "dynamicRegistration": True
    #                 },
    #                 "foldingRange": {
    #                     "dynamicRegistration": True,
    #                     "rangeLimit": 5000,
    #                     "lineFoldingOnly": True
    #                 },
    #                 "declaration": {
    #                     "dynamicRegistration": True,
    #                     "linkSupport": True
    #                 },
    #                 "selectionRange": {
    #                     "dynamicRegistration": True
    #                 },
    #                 "callHierarchy": {
    #                     "dynamicRegistration": True
    #                 },
    #                 "semanticTokens": {
    #                     "dynamicRegistration": True,
    #                     "tokenTypes": [
    #                         "namespace",
    #                         "type",
    #                         "class",
    #                         "enum",
    #                         "interface",
    #                         "struct",
    #                         "typeParameter",
    #                         "parameter",
    #                         "variable",
    #                         "property",
    #                         "enumMember",
    #                         "event",
    #                         "function",
    #                         "method",
    #                         "macro",
    #                         "keyword",
    #                         "modifier",
    #                         "comment",
    #                         "string",
    #                         "number",
    #                         "regexp",
    #                         "operator"
    #                     ],
    #                     "tokenModifiers": [
    #                         "declaration",
    #                         "definition",
    #                         "readonly",
    #                         "static",
    #                         "deprecated",
    #                         "abstract",
    #                         "async",
    #                         "modification",
    #                         "documentation",
    #                         "defaultLibrary"
    #                     ],
    #                     "formats": [
    #                         "relative"
    #                     ],
    #                     "requests": {
    #                         "range": True,
    #                         "full": {
    #                             "delta": True
    #                         }
    #                     },
    #                     "multilineTokenSupport": False,
    #                     "overlappingTokenSupport": False
    #                 },
    #                 "linkedEditingRange": {
    #                     "dynamicRegistration": True
    #                 }
    #             },
    #             "window": {
    #                 "showMessage": {
    #                     "messageActionItem": {
    #                         "additionalPropertiesSupport": True
    #                     }
    #                 },
    #                 "showDocument": {
    #                     "support": True
    #                 },
    #                 "workDoneProgress": True
    #             },
    #             "general": {
    #                 "staleRequestSupport": {
    #                     "cancel": True,
    #                     "retryOnContentModified": [
    #                         "textDocument/semanticTokens/full",
    #                         "textDocument/semanticTokens/range",
    #                         "textDocument/semanticTokens/full/delta"
    #                     ]
    #                 },
    #                 "regularExpressions": {
    #                     "engine": "ECMAScript",
    #                     "version": "ES2020"
    #                 },
    #                 "markdown": {
    #                     "parser": "marked",
    #                     "version": "1.1.0"
    #                 }
    #             }
    #         },
    #         "initializationOptions": {
    #             "bundles": [],
    #             "workspaceFolders": [
    #                 path.as_uri()
    #             ],
    #             "settings": {
    #                 "java": {
    #                     "home": java_home,
    #                     "jdt": {
    #                         "ls": {
    #                             "java": {
    #                                 "home": None,
    #                             },
    #                             "vmargs": "-XX:+UseParallelGC -XX:GCTimeRatio=4 -XX:AdaptiveSizePolicyWeight=90 -Dsun.zip.disableMemoryMapping=True -Xmx1G -Xms100m",
    #                             "lombokSupport": {
    #                                 "enabled": True
    #                             }
    #                         }
    #                     },
    #                     "errors": {
    #                         "incompleteClasspath": {
    #                             "severity": "warning"
    #                         }
    #                     },
    #                     "configuration": {
    #                         "checkProjectSettingsExclusions": False,
    #                         "updateBuildConfiguration": "interactive",
    #                         "maven": {
    #                             "userSettings": None,
    #                             "globalSettings": None,
    #                             "notCoveredPluginExecutionSeverity": "warning"
    #                         },
    #                         "workspaceCacheLimit": 90,
    #                         "runtimes": []
    #                     },
    #                     "trace": {
    #                         "server": "verbose"
    #                     },
    #                     "import": {
    #                         "maven": {
    #                             "enabled": True
    #                         },
    #                         "gradle": {
    #                             "enabled": True,
    #                             "wrapper": {
    #                                 "enabled": True
    #                             },
    #                             "version": None,
    #                             "home": None,
    #                             "java": {
    #                                 "home": None,
    #                             },
    #                             "offline": {
    #                                 "enabled": False
    #                             },
    #                             "arguments": None,
    #                             "jvmArguments": None,
    #                             "user": {
    #                                 "home": None,
    #                             }
    #                         },
    #                         "exclusions": [
    #                             "**/node_modules/**",
    #                             "**/.metadata/**",
    #                             "**/archetype-resources/**",
    #                             "**/META-INF/maven/**"
    #                         ],
    #                         "generatesMetadataFilesAtProjectRoot": False
    #                     },
    #                     "maven": {
    #                         "downloadSources": False,
    #                         "updateSnapshots": False
    #                     },
    #                     "eclipse": {
    #                         "downloadSources": False
    #                     },
    #                     "referencesCodeLens": {
    #                         "enabled": False
    #                     },
    #                     "signatureHelp": {
    #                         "enabled": False,
    #                         "description": {
    #                             "enabled": False
    #                         }
    #                     },
    #                     "implementationsCodeLens": {
    #                         "enabled": False
    #                     },
    #                     "format": {
    #                         "enabled": True,
    #                         "settings": {
    #                             "url": None,
    #                             "profile": None,
    #                         },
    #                         "comments": {
    #                             "enabled": True
    #                         },
    #                         "onType": {
    #                             "enabled": True
    #                         },
    #                         "insertSpaces": True,
    #                         "tabSize": 4
    #                     },
    #                     "saveActions": {
    #                         "organizeImports": False
    #                     },
    #                     "project": {
    #                         "referencedLibraries": [
    #                             "lib/**/*.jar"
    #                         ],
    #                         "importOnFirstTimeStartup": "automatic",
    #                         "importHint": True,
    #                         "resourceFilters": [
    #                             "node_modules",
    #                             ".git"
    #                         ],
    #                         "encoding": "ignore"
    #                     },
    #                     "contentProvider": {
    #                         "preferred": None,
    #                     },
    #                     "autobuild": {
    #                         "enabled": True
    #                     },
    #                     "maxConcurrentBuilds": 1,
    #                     "recommendations": {
    #                         "dependency": {
    #                             "analytics": {
    #                                 "show": True
    #                             }
    #                         }
    #                     },
    #                     "completion": {
    #                         "maxResults": 0,
    #                         "enabled": True,
    #                         "guessMethodArguments": False,
    #                         "favoriteStaticMembers": [
    #                             "org.junit.Assert.*",
    #                             "org.junit.Assume.*",
    #                             "org.junit.jupiter.api.Assertions.*",
    #                             "org.junit.jupiter.api.Assumptions.*",
    #                             "org.junit.jupiter.api.DynamicContainer.*",
    #                             "org.junit.jupiter.api.DynamicTest.*",
    #                             "org.mockito.Mockito.*",
    #                             "org.mockito.ArgumentMatchers.*",
    #                             "org.mockito.Answers.*"
    #                         ],
    #                         "filteredTypes": [
    #                             "java.awt.*",
    #                             "com.sun.*",
    #                             "sun.*",
    #                             "jdk.*",
    #                             "org.graalvm.*",
    #                             "io.micrometer.shaded.*"
    #                         ],
    #                         "importOrder": [
    #                             "java",
    #                             "javax",
    #                             "org",
    #                             "com"
    #                         ]
    #                     },
    #                     "foldingRange": {
    #                         "enabled": True
    #                     },
    #                     "progressReports": {
    #                         "enabled": True
    #                     },
    #                     "codeGeneration": {
    #                         "hashCodeEquals": {
    #                             "useJava7Objects": False,
    #                             "useInstanceof": False
    #                         },
    #                         "useBlocks": False,
    #                         "generateComments": False,
    #                         "toString": {
    #                             "template": "${object.className} [${member.name()}=${member.value}, ${otherMembers}]",
    #                             "codeStyle": "STRING_CONCATENATION",
    #                             "SkipNone,Values": False,
    #                             "listArrayContents": True,
    #                             "limitElements": 0
    #                         },
    #                         "insertionLocation": "afterCursor"
    #                     },
    #                     "selectionRange": {
    #                         "enabled": True
    #                     },
    #                     "showBuildStatusOnStart": {
    #                         "enabled": "notification"
    #                     },
    #                     "server": {
    #                         "launchMode": "Hybrid"
    #                     },
    #                     "sources": {
    #                         "organizeImports": {
    #                             "starThreshold": 99,
    #                             "staticStarThreshold": 99
    #                         }
    #                     },
    #                     "imports": {
    #                         "gradle": {
    #                             "wrapper": {
    #                                 "checksums": []
    #                             }
    #                         }
    #                     },
    #                     "templates": {
    #                         "fileHeader": [],
    #                         "typeComment": []
    #                     },
    #                     "references": {
    #                         "includeAccessors": True,
    #                         "includeDecompiledSources": True
    #                     },
    #                     "typeHierarchy": {
    #                         "lazyLoad": False
    #                     },
    #                     "settings": {
    #                         "url": None,
    #                     },
    #                     "symbols": {
    #                         "includeSourceMethodDeclarations": False
    #                     },
    #                     "quickfix": {
    #                         "showAt": "line"
    #                     },
    #                     "inlayHints": {
    #                         "parameterNames": {
    #                             "enabled": "literals",
    #                             "exclusions": []
    #                         }
    #                     }
    #                 }
    #             },
    #             "extendedClientCapabilities": {
    #                 "progressReportProvider": True,
    #                 "classFileContentsSupport": True,
    #                 "overrideMethodsPromptSupport": True,
    #                 "hashCodeEqualsPromptSupport": True,
    #                 "advancedOrganizeImportsSupport": True,
    #                 "generateToStringPromptSupport": True,
    #                 "advancedGenerateAccessorsSupport": True,
    #                 "generateConstructorsPromptSupport": True,
    #                 "generateDelegateMethodsPromptSupport": True,
    #                 "advancedExtractRefactoringSupport": True,
    #                 "inferSelectionSupport": [
    #                     "extractMethod",
    #                     "extractVariable",
    #                     "extractField"
    #                 ],
    #                 "moveRefactoringSupport": True,
    #                 "clientHoverProvider": True,
    #                 "clientDocumentSymbolProvider": True,
    #                 "gradleChecksumWrapperPromptSupport": True,
    #                 "resolveAdditionalTextEditsSupport": True,
    #                 "advancedIntroduceParameterRefactoringSupport": True,
    #                 "actionableRuntimeNotificationSupport": True,
    #                 "shouldLanguageServerExitOnShutdown": True,
    #                 "onCompletionItemSelectedCommand": "editor.action.triggerParameterHints"
    #             },
    #             "triggerFiles": [
    #                 # "file:///Users/nole/Developer/javasymbolsolver-maven-sample/src/main/java/com/yourorganization/maven_sample/MyAnalysis.java"
    #             ]
    #         },
    #         "trace": "verbose",
    #         "workspaceFolders": [
    #             {
    #                 "uri": path.as_uri(),
    #                 "name": path.name,
    #             }
    #         ]
    #     }

    # @d_notify
    # def _did_open(lang_id: str, file_path: PathLike) -> RPC:  # type: ignore[misc] # noqa
    #     with open(Path(file_path)) as f:
    #         text = f.read()
    #     return 'textDocument/didOpen', {
    #         'textDocument': {
    #             'uri': Path(file_path).as_uri(),
    #             'languageId': lang_id,
    #             'version': 0,
    #             'text': text,
    #         }
    #     }

    # @d_notify
    # def _did_save(file_path: PathLike) -> RPC:  # type: ignore[misc] # noqa
    #     return 'textDocument/didSave', {
    #         'textDocument': {
    #             'uri': Path(file_path).as_uri(),
    #         },
    #     }

    # @d_notify
    # def _did_change(
    #     file_path: PathLike,
    #     version: int,
    #     changes: List[spec.TextDocumentContentChangeEvent],
    # ) -> RPC:  # type: ignore[misc] # noqa
    #     return 'textDocument/didChange', {
    #         'textDocument': {
    #             'uri': Path(file_path).as_uri(),
    #             'version': version,
    #         },
    #         'contentChanges': changes,
    #     }

    # @d_notify
    # def _initialized() -> RPC:  # type: ignore[misc] # noqa
    #     return 'initialized', {}

    # @d_call
    # def _doc_symbol(path: PathLike) -> RPC:  # type: ignore[misc] # noqa
    #     return 'textDocument/documentSymbol', {
    #         'textDocument': {
    #             'uri': Path(path).as_uri()
    #         }
    #     }

    # @d_call
    # def _wkspace_symbol(query: str) -> RPC:  # type: ignore[misc] # noqa
    #     return 'workspace/symbol', {
    #         'query': query
    #     }

    # @d_call
    # def _completion(file_path: PathLike, line: int, char: int) -> RPC:
    #     return 'textDocument/completion', {
    #         'textDocument': {
    #             'uri': Path(file_path).as_uri()
    #         },
    #         'position': {
    #             'line': line,
    #             'character': char,
    #         }
    #     }

    # @d_call
    # def _diagnostic(path: PathLike) -> RPC:
    #     return 'workspace/diagnostic', {
    #         # 'textDocument': {
    #         #     'uri': Path(path).as_uri()
    #         # }
    #     }

    # @d_call
    # def _sem_token_full(path: PathLike) -> RPC:  # type: ignore[misc] # noqa
    #     return 'textDocument/semanticTokens/full', {
    #         'textDocument': {
    #             'uri': Path(path).as_uri()
    #         }
    #     }

    # @d_call
    # def debug_call(method: str, params: dict) -> RPC:  # type: ignore[misc] # noqa
    #     return method, params

    # @d_notify
    # def debug_notify(method: str, params: dict) -> RPC:  # type: ignore[misc] # noqa
    # return method, params
