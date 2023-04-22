"""the base builder implementations"""

from asyncio import gather
from collections import ChainMap
from enum import IntFlag, auto
from functools import lru_cache, partial, partialmethod
from inspect import getmro
from itertools import chain
from pathlib import Path
from distributed.client import Client
from pluggy import HookimplMarker, HookspecMarker, PluginManager
from dfocs.tools import get_notebooks_partition, get_cells, iter_files, get_file_info
from dfocs.types import NotebookType, FileType
from .base import Base, field
from dask import dataframe
from pandas import Index, DataFrame, Series
from toolz.curried import compose_left
from pathspec import PathSpec

BASE_SUFFIXES = [".py", ".md", ".ipynb"]


class Config(Base):
    cwd: Path = field(default_factory=Path.cwd)
    src: dict | list = field(default_factory=[Path.cwd()].copy)
    exclude: str = field(default="", metadata=dict(help="a gitigonore style scheme"))
    suffixes: list[str] = field(default_factory=BASE_SUFFIXES.copy)

    def __post_init__(self):
        from .tools import IGNORE

        self.exclude = PathSpec.from_lines(
            "gitwildmatch", self.exclude.splitlines() + IGNORE.read_text().splitlines()
        )


spec = HookspecMarker("dfocs")


class DataFrames(Base):
    """Builder collects files and their cells as dataframes.

    all files are cast to the nbformat document schema.
    a non-notebook file is a document with a single cell.

    - python files are one code cell
    - markdown files are one markdown cell
    - other files are one raw cell
    - image files are cells with a single output

    a builder can be used with multiple applications, they all cooperate with the central builder instance.
    """

    config: Config = field(default_factory=Config)
    files: dataframe.DataFrame = field(default=None, repr=False)
    cells: dataframe.DataFrame = field(default=None, repr=False)
    manager: PluginManager = field(default_factory=partial(PluginManager, "dfocs"))

    def get_src_files(self, index=None):
        src = self.config.src if index is None else index
        if isinstance(src, dict):
            for k, v in (index or self.layout).items():
                yield from self.get_src_files(v)
        elif isinstance(src, list):
            for v in src:
                yield from self.get_src_files(v)
        else:
            yield from iter_files(src, self.config.exclude, self.config.suffixes)

    @classmethod
    def get_cell_row(cls, row: Series) -> Series:
        """explode cells into valid dask objects"""
        data = NotebookType.Cells.get_series()
        data.update(row)
        return data

    @classmethod
    def get_cells(cls, df: dataframe.DataFrame) -> dataframe.DataFrame:
        """get cells from the notebook form of the dataframe."""
        return (
            df.pop("cells").explode()
            # order the cell keys for dask
            .apply(cls.get_cell_row, meta=NotebookType.Cells.get_dtype())
            # enumerate the cells positions
            .map_partitions(lambda df: df.assign(ct=range(len(df))))
        )

    def initialize(self):
        """initialize the builder's lazy dataframes"""
        self.initialize_files(self.get_src_files())
        self.initialize_cells()
        return self

    def initialize_files(self, files):
        """initialize the files frame with basic statistics"""
        index = Index(files, name="path")
        self.files = (
            DataFrame(index=index)
            .pipe(dataframe.from_pandas, len(index))
            .map_partitions(
                lambda df: df.index.to_series().apply(
                    compose_left(get_file_info, Series)
                ),
                meta=FileType.get_dtype(),
            )
        )
        return self

    def initialize_cells(self):
        """initialize the cells from the files frame"""
        self.cells = self.files.apply(
            get_notebooks_partition, axis=1, meta=NotebookType.get_dtype()
        ).pipe(self.get_cells)
        self.cells.source = self.cells.source.apply("".join, meta=("source", "O"))
        return self

    def expand(self):
        """expand columns on the cell frames"""
        # we might find markdown or python representations of the source.
        for i in self.manager.hook.expand_files.get_hookimpls():
            self.files = dataframe.concat(
                [self.files, (i.function(self.config, self.files))], axis=1
            )
        for i in self.manager.hook.expand_cells.get_hookimpls():
            self.cells = dataframe.concat(
                [self.cells, (i.function(self.config, self.files, self.cells))], axis=1
            )
        return self

    def compact(self):
        """compact expanded cells into the files frame"""
        for i in self.manager.hook.compact_cells.get_hookimpls():
            self.files = dataframe.concat(
                [self.files, (i.function(self.config, self.files, self.cells))], axis=1
            )
        return self

    def tasks(self):
        yield from chain.from_iterable(
            self.manager.hook.tasks(
                config=self.config, files=self.files, cells=self.cells
            )
        )


class DataFrames(Base):
    """Builder collects files and their cells as dataframes.

    all files are cast to the nbformat document schema.
    a non-notebook file is a document with a single cell.

    - python files are one code cell
    - markdown files are one markdown cell
    - other files are one raw cell
    - image files are cells with a single output

    a builder can be used with multiple applications, they all cooperate with the central builder instance.
    """

    config: Config = field(default_factory=Config)
    files: dataframe.DataFrame = field(default=None, repr=False)
    cells: dataframe.DataFrame = field(default=None, repr=False)


    def get_src_files(self, index=None):
        src = self.config.src if index is None else index
        if isinstance(src, dict):
            for k, v in (index or self.layout).items():
                yield from self.get_src_files(v)
        elif isinstance(src, list):
            for v in src:
                yield from self.get_src_files(v)
        else:
            yield from iter_files(src, self.config.exclude, self.config.suffixes)

    @classmethod
    def get_cell_row(cls, row: Series) -> Series:
        """explode cells into valid dask objects"""
        data = NotebookType.Cells.get_series()
        data.update(row)
        return data

    @classmethod
    def get_cells(cls, df: dataframe.DataFrame) -> dataframe.DataFrame:
        """get cells from the notebook form of the dataframe."""
        return (
            df.pop("cells").explode()
            # order the cell keys for dask
            .apply(cls.get_cell_row, meta=NotebookType.Cells.get_dtype())
            # enumerate the cells positions
            .map_partitions(lambda df: df.assign(ct=range(len(df))))
        )

    def initialize(self):
        """initialize the builder's lazy dataframes"""
        self.initialize_files(self.get_src_files())
        self.initialize_cells()
        return self

    def initialize_files(self, files):
        """initialize the files frame with basic statistics"""
        index = Index(files, name="path")
        self.files = (
            DataFrame(index=index)
            .pipe(dataframe.from_pandas, len(index))
            .map_partitions(
                lambda df: df.index.to_series().apply(
                    compose_left(get_file_info, Series)
                ),
                meta=FileType.get_dtype(),
            )
        )
        return self

    def initialize_cells(self):
        """initialize the cells from the files frame"""
        self.cells = self.files.apply(
            get_notebooks_partition, axis=1, meta=NotebookType.get_dtype()
        ).pipe(self.get_cells)
        self.cells.source = self.cells.source.apply("".join, meta=("source", "O"))
        return self

    def expand(self):
        """expand columns on the cell frames"""
        # we might find markdown or python representations of the source.

    def compact(self):
        """compact expanded cells into the files frame"""
