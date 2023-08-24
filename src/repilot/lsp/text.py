from functools import partial
from operator import add
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Protocol, cast

from repilot import utils

from . import spec


class MutableTextDocument(Protocol):
    """Mutable text document represented in 0-index-based lines"""

    content: str
    cursor: int

    def add(self, text: str):
        self.content = self.content[: self.cursor] + text + self.content[self.cursor :]  # type: ignore # noqa
        self.cursor += len(text)

    def delete(self, length: int):
        self.content = (
            self.content[: self.cursor - length] + self.content[self.cursor :]
        )
        self.cursor -= length

    def get_cursor_position(self) -> spec.Position:
        return self.get_position(self.cursor)

    @property
    def lines(self) -> list[str]:
        return self.content.split("\n")

    @property
    def n_chars(self) -> list[int]:
        return [len(line) for line in self.lines]

    # # Remember to sync everytime the content changes
    # def sync(self):
    #     # TODO?: make this more efficient (only count the numbers w/o returning str)
    #     self.lines = self.content.split("\n")
    #     self.n_chars = [len(line) for line in self.lines]
    #     self.check()

    def move_cursor(self, index: int):
        self.cursor = index

    # def check(self):
    #     assert self.n_lines >= 1
    #     n_newline_chars = self.n_lines - 1
    #     assert sum(self.n_chars) + n_newline_chars == len(self.content)

    def refine_index(self, line: int, character: int) -> spec.Position:
        n_lines = self.n_lines
        if line >= n_lines and character >= 0:
            line = n_lines - 1
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

    def modify(self, start: int, end: int, content: str):
        self.content = self.content[:start] + content + self.content[end:]

    # # From LSP spec: The actual content changes. The content changes describe single state
    # # changes to the document. So if there are two content changes c1 (at
    # # array index 0) and c2 (at array index 1) for a document in state S then
    # # c1 moves the document from S to S' and c2 from S' to S''. So c1 is
    # # computed on the state S and c2 is computed on the state S'.
    # def change(self, text_changes: list[spec.TextDocumentContentChangeEvent]):
    #     for text_change in text_changes:
    #         if "range" in text_change:
    #             text_change = cast(spec.TextChange, text_change)
    #             start_pos = text_change["range"]["start"]
    #             end_pos = text_change["range"]["end"]
    #             start_index = self.form_index(start_pos["line"], start_pos["character"])
    #             end_index = self.form_index(end_pos["line"], end_pos["character"])
    #             self.content = (
    #                 self.content[:start_index]
    #                 + text_change["text"]
    #                 + self.content[end_index:]
    #             )
    #         else:
    #             text_change = cast(spec.EntireDocumentChange, text_change)
    #             self.content = text_change["text"]

    def __str__(self) -> str:
        return self.content

    def __repr__(self) -> str:
        return str(self)


class TextDocument(MutableTextDocument):
    def __init__(self, content: str) -> None:
        self.content = content
        self.cursor = 0


class TextFile(MutableTextDocument, utils.JsonSerializable):
    def __init__(self, path: PathLike, content: str, root: Path | None) -> None:
        self._path = Path(path)
        self.cursor = 0
        self.content = content
        self.root = root

    @staticmethod
    def read(root: Path, path: Path) -> "TextFile":
        try:
            with open(root / path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(root / path, "r", encoding="latin-1") as f:
                content = f.read()
        return TextFile(path, content, root)

    def write(self):
        with open(self.path, "w") as f:
            f.write(self.content)

    def copy(self) -> "TextFile":
        text_file = TextFile(self._path, self.content, self.root)
        text_file.cursor = self.cursor
        return text_file

    def to_json(self) -> Any:
        return {
            "path": str(self._path),
            "content": self.content,
            "cursor": self.cursor,
        }

    @classmethod
    def from_json(cls, d: Any) -> "TextFile":
        text_file = TextFile(Path(d["path"]), d["content"], None)
        text_file.move_cursor(int(d["cursor"]))
        return text_file

    def repeat(self, n: int, include_self=False) -> list["TextFile"]:
        return [self.copy() for _ in range(n - 1)] + [
            self if include_self else self.copy()
        ]

    @property
    def path(self) -> Path:
        assert self.root is not None
        return self.root / self._path
