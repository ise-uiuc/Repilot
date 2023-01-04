from functools import partial
from operator import add
from os import PathLike
from pathlib import Path
from typing import List, Optional, Protocol, cast
from . import spec


class MutableTextDocument(Protocol):
    """Mutable text document represented in 0-index-based lines"""

    lines: List[str]
    n_chars: List[int]
    content: str
    cursor: int

    def add(self, text: str):
        self.content = self.content[: self.cursor] + text + self.content[self.cursor :]  # type: ignore # noqa
        self.cursor += len(text)
        self.sync()

    def delete(self, length: int):
        self.content = (
            self.content[: self.cursor - length] + self.content[self.cursor :]
        )
        self.cursor -= length
        self.sync()

    def get_cursor_position(self) -> spec.Position:
        return self.get_position(self.cursor)

    # Remember to sync everytime the content changes
    def sync(self):
        # TODO?: make this more efficient (only count the numbers w/o returning str)
        self.lines = self.content.split("\n")
        self.n_chars = [len(line) for line in self.lines]
        self.check()

    def move_cursor(self, index: int):
        self.cursor = index

    def check(self):
        assert self.n_lines >= 1
        n_newline_chars = self.n_lines - 1
        assert sum(self.n_chars) + n_newline_chars == len(self.content)

    def refine_index(self, line: int, character: int) -> spec.Position:
        if line >= self.n_lines and character >= 0:
            line = self.n_lines - 1
            character = self.n_chars[line]
        return spec.Position(
            {
                "line": line,
                "character": character,
            }
        )

    def form_index(self, line: int, character: int) -> int:
        pos = self.refine_index(line, character)
        line = pos["line"]
        character = pos["character"]
        if line >= self.n_lines or character > self.n_chars[line]:
            raise IndexError(line, character)
        return sum(map(partial(add, 1), self.n_chars[:line])) + character

    def get_position(self, index: int) -> spec.Position:
        for n_line, n_char in enumerate(self.n_chars):
            # 0 ... n_char-1 (\n)
            if index <= n_char:
                return {"line": n_line, "character": n_char}
            index -= n_char + 1
        raise IndexError(index)

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    # From LSP spec: The actual content changes. The content changes describe single state
    # changes to the document. So if there are two content changes c1 (at
    # array index 0) and c2 (at array index 1) for a document in state S then
    # c1 moves the document from S to S' and c2 from S' to S''. So c1 is
    # computed on the state S and c2 is computed on the state S'.
    def change(self, text_changes: List[spec.TextDocumentContentChangeEvent]):
        for text_change in text_changes:
            if "range" in text_change:
                text_change = cast(spec.TextChange, text_change)
                start_pos = text_change["range"]["start"]
                end_pos = text_change["range"]["end"]
                start_index = self.form_index(start_pos["line"], start_pos["character"])
                end_index = self.form_index(end_pos["line"], end_pos["character"])
                self.content = (
                    self.content[:start_index]
                    + text_change["text"]
                    + self.content[end_index:]
                )
            else:
                text_change = cast(spec.EntireDocumentChange, text_change)
                self.content = text_change["text"]
        self.sync()

    def __str__(self) -> str:
        return self.content

    def __repr__(self) -> str:
        return str(self)


class TextDocument(MutableTextDocument):
    def __init__(self, content: str) -> None:
        self.content = content
        self.cursor = 0
        self.sync()


class TextFile(MutableTextDocument):
    def __init__(self, path: PathLike, content: Optional[str] = None) -> None:
        self._path = Path(path)
        self.cursor = 0
        if content is None:
            with open(path) as f:
                self.content = f.read()
        else:
            self.content = content
        self.sync()

    def write(self):
        with open(self.path, "w") as f:
            f.write(self.content)

    def copy(self) -> "TextFile":
        return TextFile(self.path, self.content)

    def repeat(self, n: int, include_self=False) -> List["TextFile"]:
        return [self.copy() for _ in range(n - 1)] + [
            self if include_self else self.copy()
        ]

    @property
    def path(self) -> Path:
        return self._path
