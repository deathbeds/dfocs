"""reusable pandas and jinja components for dacs"""

import asyncio
from configparser import ConfigParser
import enum
from functools import lru_cache, partial, partialmethod, singledispatch
from mimetypes import MimeTypes
from operator import attrgetter
from textwrap import dedent, indent
from typing import AsyncGenerator, AsyncIterable, AsyncIterator, Coroutine
from numpy import nan
from pandas import DataFrame, Index, MultiIndex, Series
import pandas
import anyio
import jinja2.ext
from pathlib import Path
from enum import auto, IntFlag

import slugify

HERE = Path(__file__).parent
TEMPLATES = HERE / "templates"

MIME = MimeTypes((HERE / "mime.types",))


class Kind(IntFlag):
    INDEX, SERIES, DATAFRAME = auto(), auto(), auto()


INDEX, SERIES, FRAME = Kind.INDEX, Kind.SERIES, Kind.DATAFRAME


@singledispatch
def apply(x, f, *args, **kwargs):
    """a generalized, async-aware apply method for pandas collections"""
    return apply_index(Index(x), f, *args, **kwargs)


@apply.register(Index)
def apply_index(x, f, *args, **kwargs):
    return apply_series(x.to_series(), f, *args, **kwargs)


@apply.register(Series)
def apply_series(x, f, *args, **kwargs):
    return (
        x.apply(f, args=args, **kwargs)
        .rename(getattr(f, "__name__", None))
        .pipe(_sync_or_async)
    )


@apply.register(DataFrame)
def apply_frame(x, f, *args, **kwargs):
    return x.apply(f, args=args, axis=1, **kwargs).pipe(_sync_or_async)


@apply.register(pandas.core.groupby.DataFrameGroupBy)
def apply_group(x, f, *args, **kwargs):
    return Series(dict((k, f(y, *args, **kwargs)) for k, y in x)).pipe(_sync_or_async)


async def _asyncgen(x):
    return [x async for x in x]


async def _update(s):
    from asyncio import gather

    if not len(s):
        return ()
    if isinstance(s[0], (AsyncGenerator, AsyncIterable, AsyncIterator)):
        s = s.apply(_asyncgen)
    if isinstance(s[0], Coroutine):
        y = await gather(*s)
        if isinstance(s, Series):
            s.update(Series(y, s.index))
        elif isinstance(s, DataFrame):
            s = s.combine_first(pandas.DataFrame(y, s.index))
    return s


def _sync_or_async(s):
    return _update(s) if _isasync(s) else s


def _isasync(x):
    if len(x):
        return isinstance(
            x[0], (AsyncGenerator, AsyncIterable, AsyncIterator, Coroutine)
        )
    return False


class Accessor:
    def __init__(self, object):
        self.object = object

    def __init_subclass__(cls, method=None, types=INDEX | SERIES | FRAME, name=None):
        cls.method = method

        for t in (INDEX, SERIES, FRAME):
            if t in types:
                getattr(pandas.api.extensions, f"register_{t.name.lower()}_accessor")(
                    name or cls.__name__.lower()
                )(cls)

    def apply(self, f, *args, **kwargs):
        return apply(self.object, f, *args, **kwargs)

    ACC_METHODS = {Index: "index", Series: "series", DataFrame: "dataframe"}


class Method(Accessor):
    def __init_subclass__(cls, method=None, types=INDEX | SERIES | FRAME, name=None):
        super().__init_subclass__(method=method, types=types, name=name)
        cls._register_methods(method)

    @classmethod
    def _register_methods(cls, method, properties=True, methods=True):
        for k in dir(method):
            if hasattr(cls, k):
                continue  # break when the attribute exists
            v = getattr(method, k)
            if methods and callable(v) and not isinstance(v, classmethod):
                cls._register_method(k, v)
            elif properties and isinstance(v, property):
                cls._register_property(k, v)

    @classmethod
    def _register_method(cls, k, v):
        setattr(cls, k, partialmethod(cls.apply, v))

    @classmethod
    def _register_property(cls, k, v):
        setattr(cls, k, property(lambda x: x.apply(attrgetter(k))))


async def run(command, **kwargs):
    import anyio

    result = await anyio.run_process(command, **kwargs)
    return pandas.Series([result.stdout, result.stderr], ["stdout", "stderr"])


class Shell(Accessor, types=INDEX | SERIES, name="sh"):
    async def run(self, **kwargs):
        return (await self.apply(run, **kwargs)).apply(pandas.Series)


class Bytes(Method, types=INDEX | SERIES, method=bytes):
    pass


def get_mimetype(object):
    if isinstance(object, str):
        if object.startswith("."):
            object = "xxxxx" + object
    return MIME.guess_type(object)[0]


@lru_cache
def get_markdown():
    import midgy.tangle

    md = midgy.tangle.Tangle().parser
    md.options.update(highlight=highlight)
    return md


def markdown(md, **kwargs):
    return get_markdown().render(md, **kwargs)


def markdown_parse(md, **kwargs):
    return get_markdown().render(md, **kwargs)


def highlight(code, lang="python", attrs=None):
    import pygments, html

    try:
        return pygments.highlight(
            code,
            pygments.lexers.get_lexer_by_name(lang or "python"),
            pygments.formatters.get_formatter_by_name(
                "html", debug_token_types=True, title=f"{lang} code"
            ),
        )
    except:
        return f"""<pre><code>{html.escape(code)}</code></pre>"""


HERE = Path(__file__).parent
TEMPLATES = HERE / "templates"


@lru_cache
def get_environment():
    import jinja2, html, builtins
    from html import escape
    from slugify import slugify

    env = jinja2.Environment(
        enable_async=True,
        loader=jinja2.ChoiceLoader(
            [jinja2.FileSystemLoader(TEMPLATES), jinja2.DictLoader({})]
        ),
    )
    env.globals.update(vars(builtins), markdown=get_markdown().render)
    env.filters.update(
        markdown=get_markdown().render,
        highlight=highlight,
        escape=escape,
        slug=slugify,
        dedent=dedent,
        indent=indent,
    )
    return env


class _Jinja2(Accessor, types=SERIES | FRAME | INDEX, name="template"):
    template_name = None  # need a default template

    def __init__(self, object):
        super().__init__(object)
        self.environment = get_environment()

    def render_one(self, row, template=None, **kwargs):
        if not isinstance(template, jinja2.Template):
            template = self.environment.get_template(
                template or row.get("template_name")
            )
        data = row.to_dict()
        data.update(kwargs)

        data.update(row=row)
        if self.environment.is_async:
            return template.render_async(row=row, **{**kwargs, **row.to_dict()})
        return template.render(row=row, **{**kwargs, **row.to_dict()})

    def render_template(self, template=None, **kwargs):
        object = self.object
        if isinstance(object, MultiIndex):
            object = object.to_frame()
        elif isinstance(object, Index):
            object = object.to_series()
        object = object.replace({nan: None})
        if isinstance(object, Series):
            object = object.to_frame()
        else:
            try:
                object = object.reset_index().set_index(object.index)
            except ValueError:
                pass  # dont reset on duplicate names
        return apply_frame(object, self.render_one, template=template, **kwargs)

    def render_string(self, template, **kwargs):
        return self.render_template(self.environment.from_string(template), **kwargs)


async def get_index(
    path=None, include="", exclude=None, recursive=False
) -> pandas.DataFrame:
    if not isinstance(recursive, bool):
        recursive -= 1
    return [
        path
        async for path in get_index_iter(anyio.Path(path), include, exclude, recursive)
    ]


async def get_index_iter(
    path=None, include=[".ipynb", ".py", ".md"], exclude=None, recursive=False
) -> AsyncGenerator[anyio.Path, None]:
    if isinstance(exclude, str):
        import pathspec

        exclude = pathspec.PathSpec.from_lines(
            pathspec.patterns.GitWildMatchPattern, exclude.splitlines()
        )
    if await path.is_dir():
        async for path in path.iterdir():
            if await path.is_file():
                if path.suffix in include:
                    if (not exclude) or not exclude.match_file(path):
                        yield path
            elif recursive:
                async for path in get_index_iter(path, include, exclude, recursive):
                    yield path


def get_mimetype(object):
    if isinstance(object, str):
        if object.startswith("."):
            object = "xxxxx" + object
    return MIME.guess_type(object)[0]


def loads(x, mime=None):
    try:
        f = LOADERS[mime]
    except KeyError:
        return x
    return f(x)


LOADERS = dict()


@partial(setattr, loads, "register")
def loads_register(mime, callable=None):
    if callable is None:
        return partial(loads.register, mime)
    LOADERS[mime] = callable
    return callable


@loads.register("text/toml")
def loads_toml(x):
    return __import__("tomli").loads(x)


@loads.register("application/x-ipynb+json")
@loads.register("application/json")
def loads_json(x):
    data = __import__("json").loads(x)
    for i, x in enumerate(data["cells"]):
        x.update(source="".join(x["source"]), count=i)
    return data


@loads.register("application/x-yaml")
def loads_yaml(x):
    try:
        return __import__("ruamel.yaml").yaml.safe_load(x)
    except ModuleNotFoundError:
        return __import__("yaml").safe_load(x)


@loads.register("text/ini")
def loads_ini(x):
    parser = ConfigParser()
    parser.read_string(x)
    return parser._sections


def get_cell(x, cell_type="markdown", **kwargs):
    return dict(cell_type=cell_type, source=x)


@singledispatch
def get_notebook(x, **kwargs):
    return get_cell(x, kwargs.pop("cell_type", "raw"), **kwargs)


@get_notebook.register
def get_notebook_dict(x: dict, **kwargs):
    if "cells" in x and "metadata" in x:
        return x
    return dict(metadata=x)


@get_notebook.register
def get_notebook_str(x: str, cell_type="markdown", **kwargs):
    return dict(cells=[get_cell(x, cell_type, **kwargs)])


def get_parent_glob(path, pattern):
    path = path.absolute()
    while path.parent is not path:
        x = list(path.glob(pattern))
        if x:
            return x
        path = path.parent


class Path(type(Path())):
    def load(self):
        return loads(self.read_text(), Path.mime(self))

    def notebook(self, **kwargs):
        return get_notebook(Path.load(self), **kwargs)

    def mime(self):
        return get_mimetype(self)

    iglob = get_parent_glob


async def aget_parent_glob(path, pattern):
    path = await path.absolute()
    paths = list()
    while path.parent is not path:
        async for x in path.glob(pattern):
            paths.append(x)
        if paths:
            return paths
        path = path.parent
    return []


class APath(anyio.Path):
    async def load(self):
        return loads(await self.read_text(), APath.mime(self))

    async def notebook(self, **kwargs):
        return get_notebook(await APath.load(self), **kwargs)

    def mime(self):
        return get_mimetype(self)

    iglob = aget_parent_glob


class _Path(Method, method=Path, types=INDEX | SERIES, name="path"):
    def __call__(self, *args, **kwargs):
        if isinstance(self.object, Index):
            return self.object.map(self.method)
        return self.object.apply(self.method)

    def find(self, *args, **kwargs):
        x = self.apply(get_index, *args, **kwargs)
        return x.explode().pipe(pandas.Index, name="path")

    def glob(self, *patterns, method="glob"):
        return pandas.concat(
            self.apply(getattr(self.method, method), x).apply(list) for x in patterns
        ).explode()

    rglob = partialmethod(glob, method="rglob")
    iglob = partialmethod(glob, method="iglob")

    async def notebook(self, **kwargs):
        return self.apply(self.method.notebook).apply(pandas.Series)


class _APath(Method, method=APath, types=INDEX | SERIES, name="apath"):
    def __call__(self, *args, **kwargs):
        if isinstance(self.object, Index):
            return self.object.map(self.method)
        return self.object.apply(self.method)

    async def find(self, *args, **kwargs):
        x = await self.apply(get_index, *args, **kwargs)
        return x.explode().pipe(pandas.Index, name="path")

    async def glob(self, *patterns, method="glob"):
        return pandas.concat(
            await asyncio.gather(
                *(self.apply(getattr(self.method, method), x) for x in patterns)
            )
        ).explode()

    rglob = partialmethod(glob, method="rglob")
    iglob = partialmethod(glob, method="iglob")

    async def notebook(self, **kwargs):
        return (await self.apply(self.method.notebook)).apply(pandas.Series)


@singledispatch
def get_soup(x, *args, **kwargs):
    return x


@get_soup.register(str)
def get_soup_str(x, *args, **kwargs):
    import bs4

    kwargs.setdefault("features", "html.parser")
    return bs4.BeautifulSoup(x, *args, **kwargs)


class _html(Accessor, types=SERIES | INDEX, name="html"):
    import bs4
    from html import escape

    escape = partialmethod(Method.apply, escape)
    soup = partialmethod(Method.apply, get_soup)
    select_one = partialmethod(Method.apply, bs4.BeautifulSoup.select_one)

    def select(self, *args, **kwargs):
        import bs4

        return (
            self.object.html.soup().method.apply(
                bs4.BeautifulSoup.select, *args, **kwargs
            )
        ).explode()

    def select_one(self, *args, **kwargs):
        import bs4

        return self.object.html.soup().method.apply(
            bs4.BeautifulSoup.select_one, *args, **kwargs
        )


class _git(Accessor, types=SERIES | INDEX, name="git"):
    async def authors(self):
        return (
            (
                await (
                    await self.object.rename("path").template.render_string(
                        "cd {{path.parent}} && git log --format='%an<%ae>' -- {{path.name}} | sort | uniq"
                    )
                ).sh.run()
            )
            .stdout.bytes.decode()
            .apply(str.splitlines)
            .pipe(lambda x: x[x.apply(bool)])
            .explode()
            .str.rstrip(">")
            .str.rpartition("<", expand=True)[[0, 2]]
            .rename(columns={0: "name", 2: "email"})
            .groupby(["path", "email"])
            .name.agg(list)
            .reset_index("email")
            .groupby("path")
            .apply(lambda x: x.to_dict(orient="records"))
            .rename("authors")
        )
