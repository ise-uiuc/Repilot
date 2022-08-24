from functools import wraps
from os import PathLike
import subprocess
import json
from typing import IO, Any, Concatenate, Dict, Callable, List, Tuple, TypeVar, Type
from pathlib import Path
from typing import ParamSpec
from itertools import count

# TODO: support Content-Type
HEADER = 'Content-Length: '


def add_header(content_str: str) -> str:
    return f'Content-Length: {len(content_str)}\r\n\r\n{content_str}'


Msg = Dict[str, Any]
id_gen = count(0)


def request(method: str, params: Msg) -> Msg:
    return {
        'jsonrpc': '2.0',
        'method': method,
        'params': params,
        'id': next(id_gen)
    }


def notification(method: str, params: Msg) -> Msg:
    return {
        'jsonrpc': '2.0',
        'method': method,
        'params': params,
    }


def response(id: int, result: str | int | bool | dict | None) -> Msg:
    return {
        'jsonrpc': '2.0',
        'id': id,
        'result': result,
    }


P = ParamSpec('P')
RPC = Tuple[str, dict]
T = TypeVar('T')


def d_call(f: Callable[P, RPC]) -> Callable[Concatenate['LSPClient', P], Msg]:
    @wraps(f)
    def impl(self: 'LSPClient', *args: P.args, **kwargs: P.kwargs) -> Msg:
        method, params = f(*args, **kwargs)
        return self.call(method, params)
    return impl


def d_notify(f: Callable[P, RPC]) -> Callable[Concatenate['LSPClient', P], None]:
    @wraps(f)
    def impl(self: 'LSPClient', *args: P.args, **kwargs: P.kwargs) -> None:
        method, params = f(*args, **kwargs)
        self.notify(method, params)
    return impl


class LSPClient:
    def __init__(self, stdin: IO[bytes], stdout: IO[bytes]):
        self.stdin = stdin
        self.stdout = stdout
        # self.responses: Dict[str, Msg] = {}

    def call(self, method: str, params: Msg) -> Msg:
        message = request(method, params)
        id = message['id']
        self.send(message)
        while True:
            server_response = self.recv()
            # and server_response['method'] == 'client/registerCapability':
            if 'method' in server_response and 'id' in server_response:
                # print(server_response)
                self.send(response(server_response['id'], None))
                print(server_response)
            if 'id' in server_response and server_response['id'] == id:
                return server_response

    def notify(self, method: str, params: Msg):
        self.send(notification(method, params))
        # if method == 'textDocument/didSave':
        #     while True:
        #         self.recv()

    def send(self, message: Msg):
        content = json.dumps(message)
        content = add_header(content)
        # print(content)
        self.stdin.write(content.encode())
        self.stdin.flush()

    def recv(self) -> Msg:
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
        #     print(response)
        return json.loads(response)

    # @d_call
    # def _register(method: str) -> RPC:  # type: ignore[misc] # noqa
    #     return 'client/registerCapability', {
    #         'registrations': [{
    #             'id': '79eee87c-c409-4664-8102-e03263673f6f',
    #             'method': method,
    #         }]
    #     }

    @d_call
    def _initialize(pid: int, proj_path: PathLike) -> RPC:  # type: ignore[misc] # noqa
        path = Path(proj_path)
        return 'initialize', {
            'processId': pid,
            'rootPath': path.absolute(),
            'rootUri': path.as_uri(),
            'workspaceFolders': [{'uri': path.as_uri(), 'name': path.name}],
            'capabilities': {
                'workspace': {
                    'applyEdit': True,
                    'workspaceEdit': {
                        'documentChanges': True,
                        'resourceOperations': [
                            'create',
                            'rename',
                            'delete'
                        ],
                        'failureHandling': 'textOnlyTransactional',
                        'normalizesLineEndings': True,
                        'changeAnnotationSupport': {
                            'groupsOnLabel': True
                        }
                    },
                    'didChangeConfiguration': {
                        'dynamicRegistration': True
                    },
                    'didChangeWatchedFiles': {
                        'dynamicRegistration': True
                    },
                    'symbol': {
                        'dynamicRegistration': True,
                        'symbolKind': {
                            'valueSet': [
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
                        'tagSupport': {
                            'valueSet': [
                                1
                            ]
                        }
                    },
                    'codeLens': {
                        'refreshSupport': True
                    },
                    'executeCommand': {
                        'dynamicRegistration': True
                    },
                    'configuration': True,
                    'workspaceFolders': True,
                    'semanticTokens': {
                        'refreshSupport': True
                    },
                    'fileOperations': {
                        'dynamicRegistration': True,
                        'didCreate': True,
                        'didRename': True,
                        'didDelete': True,
                        'willCreate': True,
                        'willRename': True,
                        'willDelete': True
                    }
                },
                'textDocument': {
                    'publishDiagnostics': {
                        'relatedInformation': True,
                        'versionSupport': False,
                        'tagSupport': {
                            'valueSet': [
                                1,
                                2
                            ]
                        },
                        'codeDescriptionSupport': True,
                        'dataSupport': True
                    },
                    'synchronization': {
                        'dynamicRegistration': True,
                        'willSave': True,
                        'willSaveWaitUntil': True,
                        'didSave': True
                    },
                    'completion': {
                        'dynamicRegistration': True,
                        'contextSupport': True,
                        'completionItem': {
                            'snippetSupport': True,
                            'commitCharactersSupport': True,
                            'documentationFormat': [
                                'markdown',
                                'plaintext'
                            ],
                            'deprecatedSupport': True,
                            'preselectSupport': True,
                            'tagSupport': {
                                'valueSet': [
                                    1
                                ]
                            },
                            'insertReplaceSupport': True,
                            'resolveSupport': {
                                'properties': [
                                    'documentation',
                                    'detail',
                                    'additionalTextEdits'
                                ]
                            },
                            'insertTextModeSupport': {
                                'valueSet': [
                                    1,
                                    2
                                ]
                            },
                            'labelDetailsSupport': True
                        },
                        'insertTextMode': 2,
                        'completionItemKind': {
                            'valueSet': [
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
                    'hover': {
                        'dynamicRegistration': True,
                        'contentFormat': [
                            'markdown',
                            'plaintext'
                        ]
                    },
                    'signatureHelp': {
                        'dynamicRegistration': True,
                        'signatureInformation': {
                            'documentationFormat': [
                                'markdown',
                                'plaintext'
                            ],
                            'parameterInformation': {
                                'labelOffsetSupport': True
                            },
                            'activeParameterSupport': True
                        },
                        'contextSupport': True
                    },
                    'definition': {
                        'dynamicRegistration': True,
                        'linkSupport': True
                    },
                    'references': {
                        'dynamicRegistration': True
                    },
                    'documentHighlight': {
                        'dynamicRegistration': True
                    },
                    'documentSymbol': {
                        'dynamicRegistration': True,
                        'symbolKind': {
                            'valueSet': [
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
                        'hierarchicalDocumentSymbolSupport': True,
                        'tagSupport': {
                            'valueSet': [
                                1
                            ]
                        },
                        'labelSupport': True
                    },
                    'codeAction': {
                        'dynamicRegistration': True,
                        'isPreferredSupport': True,
                        'disabledSupport': True,
                        'dataSupport': True,
                        'resolveSupport': {
                            'properties': [
                                'edit'
                            ]
                        },
                        'codeActionLiteralSupport': {
                            'codeActionKind': {
                                'valueSet': [
                                    '',
                                    'quickfix',
                                    'refactor',
                                    'refactor.extract',
                                    'refactor.inline',
                                    'refactor.rewrite',
                                    'source',
                                    'source.organizeImports'
                                ]
                            }
                        },
                        'honorsChangeAnnotations': False
                    },
                    'codeLens': {
                        'dynamicRegistration': True
                    },
                    'formatting': {
                        'dynamicRegistration': True
                    },
                    'rangeFormatting': {
                        'dynamicRegistration': True
                    },
                    'onTypeFormatting': {
                        'dynamicRegistration': True
                    },
                    'rename': {
                        'dynamicRegistration': True,
                        'prepareSupport': True,
                        'prepareSupportDefaultBehavior': 1,
                        'honorsChangeAnnotations': True
                    },
                    'documentLink': {
                        'dynamicRegistration': True,
                        'tooltipSupport': True
                    },
                    'typeDefinition': {
                        'dynamicRegistration': True,
                        'linkSupport': True
                    },
                    'implementation': {
                        'dynamicRegistration': True,
                        'linkSupport': True
                    },
                    'colorProvider': {
                        'dynamicRegistration': True
                    },
                    'foldingRange': {
                        'dynamicRegistration': True,
                        'rangeLimit': 5000,
                        'lineFoldingOnly': True
                    },
                    'declaration': {
                        'dynamicRegistration': True,
                        'linkSupport': True
                    },
                    'selectionRange': {
                        'dynamicRegistration': True
                    },
                    'callHierarchy': {
                        'dynamicRegistration': True
                    },
                    'semanticTokens': {
                        'dynamicRegistration': True,
                        'tokenTypes': [
                            'namespace',
                            'type',
                            'class',
                            'enum',
                            'interface',
                            'struct',
                            'typeParameter',
                            'parameter',
                            'variable',
                            'property',
                            'enumMember',
                            'event',
                            'function',
                            'method',
                            'macro',
                            'keyword',
                            'modifier',
                            'comment',
                            'string',
                            'number',
                            'regexp',
                            'operator'
                        ],
                        'tokenModifiers': [
                            'declaration',
                            'definition',
                            'readonly',
                            'static',
                            'deprecated',
                            'abstract',
                            'async',
                            'modification',
                            'documentation',
                            'defaultLibrary'
                        ],
                        'formats': [
                            'relative'
                        ],
                        'requests': {
                            'range': True,
                            'full': {
                                'delta': True
                            }
                        },
                        'multilineTokenSupport': False,
                        'overlappingTokenSupport': False
                    },
                    'linkedEditingRange': {
                        'dynamicRegistration': True
                    }
                },
                'window': {
                    'showMessage': {
                        'messageActionItem': {
                            'additionalPropertiesSupport': True
                        }
                    },
                    'showDocument': {
                        'support': True
                    },
                    'workDoneProgress': True
                },
                'general': {
                    'staleRequestSupport': {
                        'cancel': True,
                        'retryOnContentModified': [
                            'textDocument/semanticTokens/full',
                            'textDocument/semanticTokens/range',
                            'textDocument/semanticTokens/full/delta'
                        ]
                    },
                    'regularExpressions': {
                        'engine': 'ECMAScript',
                        'version': 'ES2020'
                    },
                    'markdown': {
                        'parser': 'marked',
                        'version': '1.1.0'
                    }
                }
            },
        }

    @d_notify
    def _did_open(lang_id: str, file_path: PathLike) -> RPC:  # type: ignore[misc] # noqa
        with open(Path(file_path)) as f:
            text = f.read()
        return 'textDocument/didOpen', {
            'textDocument': {
                'uri': Path(file_path).as_uri(),
                'languageId': lang_id,
                'version': 0,
                'text': text,
            }
        }


    @d_notify
    def _did_save(file_path: PathLike) -> RPC:  # type: ignore[misc] # noqa
        return 'textDocument/didSave', {
            'textDocument': {
                'uri': Path(file_path).as_uri(),
            },
        }

    @d_notify
    def _initialized() -> RPC:  # type: ignore[misc] # noqa
        return 'initialized', {}

    @d_call
    def _doc_symbol(path: PathLike) -> RPC:  # type: ignore[misc] # noqa
        return 'textDocument/documentSymbol', {
            'textDocument': {
                'uri': Path(path).as_uri()
            }
        }

    @d_call
    def _wkspace_symbol(query: str) -> RPC:  # type: ignore[misc] # noqa
        return 'workspace/symbol', {
            'query': query
        }

    @d_call
    def _completion(file_path: PathLike, line: int, char: int) -> RPC:
        return 'textDocument/completion', {
            'textDocument': {
                'uri': Path(file_path).as_uri()
            },
            'position': {
                'line': line,
                'character': char,
            }
        }
    
    @d_call
    def _diagnostic(path: PathLike) -> RPC:
        return 'workspace/diagnostic', {
            # 'textDocument': {
            #     'uri': Path(path).as_uri()
            # }
        }

    @d_call
    def _sem_token_full(path: PathLike) -> RPC:  # type: ignore[misc] # noqa
        return 'textDocument/semanticTokens/full', {
            'textDocument': {
                'uri': Path(path).as_uri()
            }
        }

    @d_call
    def debug_call(method: str, params: dict) -> RPC:  # type: ignore[misc] # noqa
        return method, params

    @d_notify
    def debug_notify(method: str, params: dict) -> RPC:  # type: ignore[misc] # noqa
        return method, params


C = TypeVar('C')


def post_process(get_client: Callable[[C], LSPClient], rpc: Callable[Concatenate[LSPClient, P], Msg]) \
        -> Callable[[Callable[[Msg], T]], Callable[Concatenate[C, P], T]]:
    def decorator(f: Callable[[Msg], T]) -> Callable[Concatenate[C, P], T]:
        @wraps(f)
        def impl(self: C, *args: P.args, **kwargs: P.kwargs) -> T:
            client = get_client(self)
            return f(rpc(client, *args, **kwargs))
        return impl
    return decorator


class PipeLspAnalyzer:
    def __init__(self, server_cmd: str) -> None:
        self.process = subprocess.Popen(
            server_cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
        assert self.process.stdin is not None and self.process.stdout is not None
        self.client = LSPClient(self.process.stdin, self.process.stdout)

    def init(self, proj_path: PathLike) -> Msg:
        msg = self.client._initialize(self.process.pid, proj_path)
        self.client._initialized()
        return msg

    def _client(self) -> LSPClient:
        return self.client

    @post_process(_client, LSPClient._completion)
    def complete(completions: Msg) -> List[str]:  # type: ignore[misc] # noqa
        return [c['textEdit']['newText'] for c in completions['result']['items']]
