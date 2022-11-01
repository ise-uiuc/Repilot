from typing import cast
from javalang.parser import JavaSyntaxError, JavaToken, Parser, EndOfInput
from javalang.tokenizer import tokenize, LexerError, JavaTokenizer
from realm.lsp import MutableTextDocument, spec

class AnalysisError(Exception):
    def __init__(self, error_token: JavaToken) -> None:
        self.args = (error_token,)


class TokenizeError(Exception):
    pass


def reduce(text_document: MutableTextDocument, raise_normal_exc: bool = False, raise_unexpected_exc: bool = False):
    """Modifies the document in place by removing everything from the earliest error to the end"""
    try:
        # tokens = tokenize(self.content, False)
        tokenizer = JavaTokenizer(text_document.content, False)
        tokens = tokenizer.tokenize()
        parser = Parser(tokens)
        parser.tokens.list.extend(tokens)
        parser.parse_expression()
    except Exception as e:
        raise_exc = raise_normal_exc
        if isinstance(e, LexerError):
            text_document.change(
                [{'text': text_document.content[:tokenizer.last_position]}])
        elif isinstance(e, (JavaSyntaxError, StopIteration)):
            head = parser.tokens.look()
            if not isinstance(head, EndOfInput):
                # print(head.position)
                start_position = {
                    'line': head.position.line - 1,
                    'character': head.position.column - 1,
                }
                end_position = {
                    'line': text_document.n_lines - 1,
                    'character': text_document.n_chars[text_document.n_lines - 1]
                }
                text_document.change([cast(spec.TextChange, {
                    'range': {
                        'start': start_position,
                        'end': end_position,
                    },
                    'text': ''
                })])
        else:
            raise_exc = raise_unexpected_exc
            print('Unexpected exception', type(e), e)
        if raise_exc:
            raise e