from javalang.parser import JavaSyntaxError, JavaToken, Parser, EndOfInput
from javalang.tokenizer import tokenize, LexerError, JavaTokenizer


class AnalysisError(Exception):
    def __init__(self, error_token: JavaToken) -> None:
        self.args = (error_token,)


class TokenizeError(Exception):
    pass


class Analyzer:
    def __init__(self) -> None:
        self.content = ''

    def restrict(self, line: int, column: int):
        lines = self.content.splitlines(True)
        lines = [x[:-1] for x in lines[:line - 1]] + \
            [lines[line - 1][:column - 1]]
        self.content = '\n'.join(lines)

    def feed(self, new_content: str, raise_normal_exc: bool = False, raise_unexpected_exc: bool = False):
        self.content += new_content
        try:
            # tokens = tokenize(self.content, False)
            tokenizer = JavaTokenizer(self.content, False)
            tokens = tokenizer.tokenize()
            parser = Parser(tokens)
            parser.tokens.list.extend(tokens)
            parser.parse()
        except Exception as e:
            raise_exc = raise_normal_exc
            if isinstance(e, LexerError):
                self.content = self.content[:tokenizer.last_position]
            elif isinstance(e, (JavaSyntaxError, StopIteration)):
                head = parser.tokens.look()
                if not isinstance(head, EndOfInput):
                    # print(head.position)
                    self.restrict(head.position.line, head.position.column)
            else:
                raise_exc = raise_unexpected_exc
                print('Unexpected exception', type(e), e)
            if raise_exc:
                raise e
