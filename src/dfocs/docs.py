from functools import lru_cache
from .frames import DataFrames, field, Base, Config
from .types import Path
from dask import dataframe

HTML = """<html><head></head><body>%s</body></html>"""


class HtmlMeta(Base):
    title: str = None
    headings: list = None
    description: str = None
    links: list = None


class Docs(DataFrames):
    class Config:
        target: Path = Path("site")
        layout_mapping: dict = field(default_factory=dict, repr=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_doit_tasks = self.task_html

    def expand(self):
        """expand the cells frame to include markdown and html representations"""
        self.cells["md"] = self.cells.apply(get_markdown, axis=1, meta=("md", "O"))
        self.cells["html"] = self.cells.apply(get_html_body, axis=1, meta=("html", "O"))
        self.cells = self.cells.join(self.files.mimetype)
        self.cells.persist()
        self.cells[list(HtmlMeta.fromkeys())] = self.cells.apply(get_headings, axis=1, meta=HtmlMeta.get_dtype())
        return self

    def compact(self):
        """expand the files frame to include aggregated html representations"""
        self.files[["md", "body"]] = self.cells.map_partitions(
            lambda df: pandas.DataFrame(
                [["\n\n".join(df.md), "\n\n".join(str(df.html))]],
                df.index[:1], ["md", "body"]
            ),
            meta=dict((("md", "O"), ("body", "O"))),
        )
        self.files["html"] = self.files.body.apply(HTML.__mod__, meta=("html", "O"))
        self.files[["title", "description"]] = self.cells.map_partitions(
            lambda df: df.iloc[[0]][["title", "description"]],
            meta=dict(title="O", description="O"),
        )
        return self

    def task_html(self):
        """generate the html version of documentation"""

        def write_html(row, target):
            target.parent.mkdir(exist_ok=True, parents=True)
            target.write_text(row.html.compute().iloc[0])

        for partition in self.cells.partitions:
            d = partition.divisions[0]
            p = d.relative_to("../../tonyfast")
            target = Path("site") / p.with_suffix(".html")

            yield dict(
                name=f"{p}",
                file_dep=[d],
                actions=[(write_html, (self.files.loc[d], target))],
                targets=[target],
                basename="html",
            )

    def task_md(self):
        """generate the html version of documentation"""

        def write_html(row, target):
            target.parent.mkdir(exist_ok=True, parents=True)
            target.write_text(row.html.compute().iloc[0])

        for partition in self.cells.partitions:
            d = partition.divisions[0]
            p = d.relative_to("../../tonyfast")
            target = Path("site") / p.with_suffix(".html")

            yield dict(
                name=f"{p}",
                file_dep=[d],
                actions=[(write_html, (self.files.loc[d], target))],
                targets=[target],
                basename="html",
            )

    def tasks(self):
        yield from self.task_html()


def get_headings(s):
    soup = s.html
    meta = HtmlMeta.get_series()
    meta.headings = []
    h1 = soup.select_one("h1")

    if h1:
        meta.title = h1
        meta.headings.append(h1)

    meta.headings.extend(soup.select("h2,h3,h4,h5,h6"))

    for h in meta.headings:
        if "id" not in h.attrs:
            pass

    meta.links = soup.select("a")
    meta.description = soup.select_one("h1 ~ p")

    return meta


def get_markdown_body(df):
    return df.md.pipe("\n\n\n".join)


def get_soup(s, *args, **kwargs):
    from bs4 import BeautifulSoup

    kwargs.setdefault("features", "lxml")
    return BeautifulSoup(s, *args, **kwargs)


def get_html_body(s):
    return get_soup(get_markdown_render(s.md))


def get_html_head(s):
    return """"""


@lru_cache
def get_markdown_it():
    from markdown_it import MarkdownIt

    return MarkdownIt()


def get_markdown_render(body):
    return get_markdown_it().render(body)


def get_markdown_code(s):
    return f"""```\n{s.source}\n\n```\n"""


def get_markdown_markdown(s):
    return "".join(s.source)


def get_markdown_raw(s):
    return f"""```\n{s.source}\n\n```\n"""


get_markdown_dispatch = dict(
    code=get_markdown_code, markdown=get_markdown_markdown, raw=get_markdown_raw
)


def get_markdown(s):
    return get_markdown_dispatch.get(s.cell_type)(s)


class Meta(Base):
    heading: str
    description: str
    links: list


import pandas


def get_html_info(s):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(s.body, features="lxml")
    meta = Meta.fromkeys()
    h1 = soup.select_one("h1")
    description = soup.select_one("h1 + p")
    meta.update(
        heading=h1 and h1.string or s.path,
        description=description and str(description) or "",
        links=[],
    )
    return pandas.Series(meta)
