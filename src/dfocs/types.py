"""base python type and mime type contents

implementations of types should be placed in the models.py file"""

import functools
from .base import Base, field
from datetime import datetime
from pathlib import Path

from mimetypes import MimeTypes


THIS = Path(__file__)
HERE = THIS.parent
MIME_TYPES = HERE / "data" / "mime.types"
mimetypes = MimeTypes((MIME_TYPES,))


class Path(type(Path())):
    _notebook_readers = {}

    @classmethod
    def register_notebook_reader(cls, mimetype):
        def main(callable):
            cls._notebook_readers[mimetype] = callable
            return callable

        return main

    def mimetype(self):
        return mimetypes.guess_type(self)[0]

    def read_notebook(self, type=None):
        from .tools import read_notebook_raw

        return self._notebook_readers.get(type or self.mimetype(), read_notebook_raw)(
            self
        )


class NotebookType(Base):
    cells: list = field(default_factory=list)
    metadata: list = field(default_factory=dict)
    nbformat: int = 4
    nbformat_minor: int = 5


    class Cell(Base):
        source: list[str] | str = None
        metadata: dict = field(default_factory=dict)
        cell_type: str = None
        id: str = "fu"
        ct: int = -1

    class Code(Cell):
        execution_count: int | None = None
        outputs: list | None = None

    class Md(Cell):
        attachments: dict | None = None

    class Cells(Code, Md):
        pass


class FileType(Base):
    path: Path
    mimetype: str = None
    size: int = None
    created_at: datetime = None
    started_at: datetime = None
    updated_at: datetime = None
