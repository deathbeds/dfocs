from json import loads

from pandas import DataFrame, Series
from dask import dataframe
from .types import Path, FileType, NotebookType, mimetypes
from datetime import datetime
from typing import Generator

from pathspec import PathSpec


THIS = Path(__file__)
HERE = THIS.parent
IGNORE = HERE / "data"/ "ignore"
ignore = PathSpec.from_lines("gitwildmatch", IGNORE.read_text().splitlines())


def iter_files(
    dir: Path | str, exclude: PathSpec = ignore, suffix: tuple | None = None
) -> Generator:
    dir = Path(dir)
    if dir.is_dir():
        if exclude:
            if exclude.match_file(dir):
                return

        for file in dir.iterdir():
            if exclude:
                if exclude.match_file(file):
                    continue
            yield from iter_files(file, exclude=exclude, suffix=suffix)

    elif dir.is_file():
        if suffix:
            if dir.suffix not in suffix:
                return
        yield dir


def get_file_info(path: Path) -> dict:
    """accumulate information about the file"""
    path = Path(path)
    stat = path.stat()
    m = mimetypes.guess_type((path.absolute().as_uri()))
    info = FileType.fromkeys()
    info.update(
        path=path,
        created_at=datetime.fromtimestamp(stat.st_ctime),
        started_at=datetime.fromtimestamp(stat.st_atime),
        updated_at=datetime.fromtimestamp(stat.st_mtime),
        size=stat.st_size / 2**10,
        mimetype=m and m[0],
    )
    return info


def get_files_index(files, exclude: Path = ignore, suffix: tuple | None = None) -> DataFrame:
    import pandas

    return pandas.Index(iter_files(files, exclude=exclude, suffix=suffix), name="path")


def get_files(files, exclude: Path = ignore, suffix: tuple | None = None) -> DataFrame:
    import pandas

    index = pandas.Index(iter_files(files, exclude=exclude, suffix=suffix), name="path")
    return (
        pandas.DataFrame(
            map(get_file_info, index), columns=FileType.properties(), index=index
        )
        .reset_index(drop=True)
        .set_index("path")
    )


def read_notebook_raw(path):
    data = NotebookType.fromkeys()
    data.update(cells=[dict(source=path.read_text(), cell_type="raw")])
    return data


@Path.register_notebook_reader("application/x-ipynb+json")
def read_notebook_ipynb(path):
    data = NotebookType.fromkeys()
    data.update(loads(path.read_text()))
    return data


@Path.register_notebook_reader("text/markdown")
def read_notebook_md(path):
    data = NotebookType.fromkeys()
    data.update(cells=[dict(source=path.read_text(), cell_type="markdown")])
    return data


@Path.register_notebook_reader("text/x-python")
def read_notebook_py(path):
    data = NotebookType.fromkeys()
    data.update(cells=[dict(source=path.read_text(), cell_type="code")])
    return data


def get_notebooks(df: DataFrame) -> dataframe.DataFrame:
    ddf = dataframe.from_pandas(df, len(df))
    return ddf.apply(get_notebooks_partition, axis=1, meta=NotebookType.dtype())


def get_notebooks_partition(row: Series) -> Series:
    """read a single notebook from a row partition"""
    return Series(Path(row.name).read_notebook(row.mimetype))


def get_cell_row(row: Series) -> Series:
    """explode cells into valid dask objects"""
    data = Series(NotebookType.Cells.fromkeys())
    data.update(row)
    return data


def get_cells(ddf: dataframe.DataFrame) -> dataframe.DataFrame:
    """get cells from the notebook form of the dataframe."""
    return (
        ddf.pop("cells")
        .explode()
        # order the cell keys for dask
        .apply(get_cell_row, meta=NotebookType.Cells.dtype())
        # enumerate the cells positions
        .map_partitions(lambda df: df.assign(ct=range(len(df))))
    )
