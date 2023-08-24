"""An INCOMPLETE, direct translation of https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/ """
from typing import Annotated, Any, Dict, Literal, TypeAlias, TypedDict

from typing_extensions import NotRequired

number: TypeAlias = int
integer: TypeAlias = number
uinteger: TypeAlias = number
decimal: TypeAlias = number
string: TypeAlias = str
boolean: TypeAlias = bool
null: TypeAlias = None
array: TypeAlias = list
object: TypeAlias = Any

# NOTE: used an experimenting feature of mypy: --enable-recursive-aliases
LSPObject: TypeAlias = Dict[str, Any]
LSPAny: TypeAlias = (
    LSPObject | list["LSPAny"] | string | integer | uinteger | decimal | boolean | null
)
LSPArray: TypeAlias = list[LSPAny]


class Message(TypedDict):
    jsonrpc: string


class RequestMessage(Message):
    id: integer | string
    method: string
    params: NotRequired[array | object]


class ResponseError(TypedDict):
    code: integer
    message: string
    data: NotRequired[string | number | boolean | array | object | null]


class ResponseMessage(Message):
    id: integer | string | null
    result: NotRequired[string | number | boolean | object | null]
    error: NotRequired[ResponseError]


class ErrorCodes:
    # Defined by JSON-RPC
    ParseError: integer = -32700
    InvalidRequest: integer = -32600
    MethodNotFound: integer = -32601
    InvalidParams: integer = -32602
    InternalError: integer = -32603

    jsonrpcReservedErrorRangeStart: integer = -32099
    serverErrorStart: integer = jsonrpcReservedErrorRangeStart

    ServerNotInitialized: integer = -32002
    UnknownErrorCode: integer = -32001

    jsonrpcReservedErrorRangeEnd = -32000
    serverErrorEnd: integer = jsonrpcReservedErrorRangeEnd

    lspReservedErrorRangeStart: integer = -32899

    RequestFailed: integer = -32803

    ServerCancelled: integer = -32802

    ContentModified: integer = -32801

    RequestCancelled: integer = -32800

    lspReservedErrorRangeEnd: integer = -32800


class NotificationMessage(Message):
    method: string
    params: NotRequired[array | object]


class CancelParams(TypedDict):
    id: integer | string


class HoverParams(TypedDict):
    textDocument: string
    position: "Position"


class HoverResult(TypedDict):
    value: string


DocumentUri: TypeAlias = string
URI: TypeAlias = string


class Position(TypedDict):
    line: int
    character: int


class Range(TypedDict):
    start: Position
    end: Position


class TextChange(TypedDict):
    text: str
    range: Range


class EntireDocumentChange(TypedDict):
    text: str


class TextDocumentItem(TypedDict):
    uri: DocumentUri
    languageId: string
    version: integer
    text: string


class TextDocumentIdentifier(TypedDict):
    uri: DocumentUri


class VersionedTextDocumentIdentifier(TextDocumentIdentifier):
    version: integer


class OptionalVersionedTextDocumentIdentifier(TextDocumentIdentifier):
    version: integer | null


TextDocumentContentChangeEvent = EntireDocumentChange | TextChange


class Registration(TypedDict):
    id: string
    method: string
    registerOptions: NotRequired[LSPAny]


class RegistrationParams(TypedDict):
    registrations: list[Registration]


ProgressToken: TypeAlias = integer | string


class WorkDoneProgressParams(TypedDict):
    workDoneToken: NotRequired[ProgressToken]


class ClientInfo(TypedDict):
    name: string
    version: NotRequired[string]


class TextDocumentClientCapabilities(dict):
    # TODO(or not todo): implement the types below and change the superclass to TypedDict
    pass
    # synchronization: NotRequired[TextDocumentSyncClientCapabilities]
    # completion: NotRequired[CompletionClientCapabilities]
    # hover: NotRequired[HoverClientCapabilities]
    # signatureHelp: NotRequired[SignatureHelpClientCapabilities]
    # declaration: NotRequired[DeclarationClientCapabilities]
    # definition: NotRequired[DefinitionClientCapabilities]
    # typeDefinition: NotRequired[TypeDefinitionClientCapabilities]
    # implementation: NotRequired[ImplementationClientCapabilities]
    # references: NotRequired[ReferenceClientCapabilities]
    # documentHighlight: NotRequired[DocumentHighlightClientCapabilities]
    # documentSymbol: NotRequired[DocumentSymbolClientCapabilities]
    # codeAction: NotRequired[CodeActionClientCapabilities]
    # codeLens: NotRequired[CodeLensClientCapabilities]
    # documentLink: NotRequired[DocumentLinkClientCapabilities]
    # colorProvider: NotRequired[DocumentColorClientCapabilities]
    # formatting: NotRequired[DocumentFormattingClientCapabilities]
    # rangeFormatting: NotRequired[DocumentRangeFormattingClientCapabilities]
    # onTypeFormatting: NotRequired[DocumentOnTypeFormattingClientCapabilities]
    # rename: NotRequired[RenameClientCapabilities]
    # publishDiagnostics: NotRequired[PublishDiagnosticsClientCapabilities]
    # foldingRange: NotRequired[FoldingRangeClientCapabilities]
    # selectionRange: NotRequired[SelectionRangeClientCapabilities]
    # linkedEditingRange: NotRequired[LinkedEditingRangeClientCapabilities]
    # callHierarchy: NotRequired[CallHierarchyClientCapabilities]
    # semanticTokens: NotRequired[SemanticTokensClientCapabilities]
    # moniker: NotRequired[MonikerClientCapabilities]
    # typeHierarchy: NotRequired[TypeHierarchyClientCapabilities]
    # inlineValue: NotRequired[InlineValueClientCapabilities]
    # inlayHint: NotRequired[InlayHintClientCapabilities]
    # diagnostic: NotRequired[DiagnosticClientCapabilities]


class ClientCapabilities(dict):
    # TODO(or not todo): implement the types below and change the superclass to TypedDict
    pass
    # workspace?: {
    # 	applyEdit?: boolean;
    # 	workspaceEdit?: WorkspaceEditClientCapabilities;
    # 	didChangeConfiguration?: DidChangeConfigurationClientCapabilities;
    # 	didChangeWatchedFiles?: DidChangeWatchedFilesClientCapabilities;
    # 	symbol?: WorkspaceSymbolClientCapabilities;
    # 	executeCommand?: ExecuteCommandClientCapabilities;
    # 	workspaceFolders?: boolean;
    # 	configuration?: boolean;
    # 	semanticTokens?: SemanticTokensWorkspaceClientCapabilities;
    # 	codeLens?: CodeLensWorkspaceClientCapabilities;
    # 	fileOperations?: {
    # 		dynamicRegistration?: boolean;
    # 		didCreate?: boolean;
    # 		willCreate?: boolean;
    # 		didRename?: boolean;
    # 		willRename?: boolean;
    # 		didDelete?: boolean;
    # 		willDelete?: boolean;
    # 	};
    # 	inlineValue?: InlineValueWorkspaceClientCapabilities;
    # 	inlayHint?: InlayHintWorkspaceClientCapabilities;
    # 	diagnostics?: DiagnosticWorkspaceClientCapabilities;
    # };
    # textDocument?: TextDocumentClientCapabilities;
    # notebookDocument?: NotebookDocumentClientCapabilities;
    # window?: {
    # 	workDoneProgress?: boolean;
    # 	showMessage?: ShowMessageRequestClientCapabilities;
    # 	showDocument?: ShowDocumentClientCapabilities;
    # };
    # general?: {
    # 	staleRequestSupport?: {
    # 		cancel: boolean;
    # 		retryOnContentModified: string[];
    # 	}
    # 	regularExpressions?: RegularExpressionsClientCapabilities;
    # 	markdown?: MarkdownClientCapabilities;
    # 	positionEncodings?: PositionEncodingKind[];
    # };
    # experimental?: LSPAny;


class WorkspaceFolder(TypedDict):
    uri: DocumentUri
    name: string


TraceValue = Literal["off", "messages", "verbose"]


class InitializeParams(WorkDoneProgressParams):
    processId: integer | null
    clientInfo: NotRequired[ClientInfo]
    locale: NotRequired[string]
    rootPath: NotRequired[string | null]
    rootUri: DocumentUri | null
    initializationOptions: NotRequired[LSPAny]
    capabilities: ClientCapabilities
    trace: NotRequired[TraceValue]
    workspaceFolders: NotRequired[list[WorkspaceFolder] | null]


class InitializedParams(TypedDict):
    pass


class DidOpenTextDocumentParams(TypedDict):
    textDocument: TextDocumentItem


class DidSaveTextDocumentParams(TypedDict):
    textDocument: TextDocumentIdentifier
    text: NotRequired[string]


class DidChangeTextDocumentParams(TypedDict):
    textDocument: VersionedTextDocumentIdentifier
    contentChanges: list[TextDocumentContentChangeEvent]


class TextDocumentPositionParams(TypedDict):
    textDocument: TextDocumentIdentifier
    position: Position


class PartialResultParams(TypedDict):
    partialResultToken: NotRequired[ProgressToken]


class CompletionTriggerKinds:
    Invoked = 1
    TriggerCharacter = 2
    TriggerForIncompleteCompletions = 3


CompletionTriggerKind = Literal[1, 2, 3]


class CompletionContext(TypedDict):
    triggerKind: CompletionTriggerKind
    triggerCharacter: NotRequired[string]


class CompletionParams(
    TextDocumentPositionParams, WorkDoneProgressParams, PartialResultParams
):
    context: NotRequired[CompletionContext]
