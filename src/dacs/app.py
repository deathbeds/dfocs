from asyncio import gather
import datetime
from enum import Enum
from operator import itemgetter
from pathlib import Path
from pandas import DataFrame, Index, MultiIndex, Series, to_datetime
from pydantic import BaseModel, Field

from dacs.utils import apply, get_soup, markdown_parse


class Config(BaseModel):
    name: str | None = None
    authors: list = Field(default_factory=list)
    index: list = Field(default_factory=list)
    include: list[str] = Field(default_factory=[".py", ".md", ".ipynb"].copy)
    exclude: str | list[str] = ".git\n.nox\n*checkpoint.*\n.ipynb_checkpoints\n.*"
    plugins: list[str | dict] = Field(default_factory=list)

    @classmethod
    def from_pyproject(cls, data):
        project, dacs = data.get("project", {}), data.get("tool", {}).get("dacs", {})
        return cls(
            name=project.get("name"),
            description=project.get("description"),
            index=dacs.get("index", [""]),
            readme=project.get("readme"),
        )

    def get_index(self):
        return (
            Index(self.index)
            .apath()
            .apath.find(include=self.include, exclude=self.exclude, recursive=True)
        )


class App(BaseModel):
    config: Config
    dirs: DataFrame = Field(default_factory=DataFrame)
    files: DataFrame = Field(default_factory=DataFrame)
    cells: DataFrame = Field(default_factory=DataFrame)

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(config=config, **kwargs)

    async def get_files(self):
        return await (await self.config.get_index()).apath.notebook()

    async def load(self):
        self.files = self.files.combine_first(await self.get_files())
        self.dirs = self.dirs.combine_first(
            self.files.index.path.parent.value_counts().to_frame("files")
        )
        cells = self.files.pop("cells")
        self.cells = self.cells.combine_first(
            cells[cells.apply(bool)].explode().apply(Series)
        )
        return self

    @staticmethod
    async def template_cell(df):
        env = dict()
        return (await df.template.render_template("cell.html.j2", env=env)).sum(), env

    async def template_cells(self):
        self.files = (
            (
                await apply(
                    self.cells.groupby(self.cells.index.get_level_values("path")),
                    self.template_cell,
                )
            )
            .apply(Series, index=["main", "env"])
            .pipe(self.files.combine_first)
        )
        return self

    class Config:
        arbitrary_types_allowed = True


def get_cell_times(x):
    try:
        x = x.get("execution")
    except AttributeError:
        pass
    else:
        if x:
            try:
                return list(
                    map(
                        datetime.datetime.fromisoformat,
                        itemgetter("iopub.execute_input", "shell.execute_reply")(x),
                    )
                )
            except KeyError:
                pass
    return [None, None]


class Docs(App):
    target: Path = Path("site")

    async def load(self):
        await super().load()
        self.files = self.files.assign(target=(self.target / self.files.index))
        self.is_indexed()
        await gather(self.get_times(), self.get_authors())
        self.get_cell_times()
        return self

    def is_indexed(self):
        self.dirs = (
            self.dirs.loc[
                self.files.index[
                    self.files.index.path.stem.isin({"README", "index"})
                ].path.parent
            ]
            .assign(has_index=True)
            .pipe(self.dirs.combine_first)
            .fillna(False)
        )

    def get_cell_times(self):
        
        self.cells = self.cells.combine_first(
            self.cells["metadata"].apply(get_cell_times).apply(
                Series, index=["created_at", "updated_at"]
            )
        )
        print("cell times")
        return self

    async def get_times(self):
        self.files = (
            (
                await (
                    await MultiIndex.from_product(
                        [["", "--reverse"], self.files.index], names=["rev", "path"]
                    ).template.render_string(
                        "cd {{path.parent}} && git log --date=unix {{rev}} --format='%cd' -- {{path.name}} |  head -n1"
                    )
                ).sh.run()
            )
            .stdout.pipe(lambda x: x[x.astype(bool)])
            .bytes.decode()
            .astype(int)
            .pipe(to_datetime, unit="s")
            .unstack(0)
            .rename(columns={"--reverse": "created_at", "": "updated_at"})
        ).combine_first(self.files)
        self.files.index = self.files.index.rename("path")
        return self

    async def get_authors(self):
        self.files = (
            (
                await (
                    await self.files.index.rename("path").template.render_string(
                        "cd {{path.parent}} && git log --format='%an<%ae>' -- {{path.name}} | sort | uniq"
                    )
                ).sh.run()
            )
            .stdout.bytes.decode()
            .apply(str.splitlines)
            .pipe(lambda s: s[s.astype(bool)])
            .explode()
            .str.extract("(?P<name>.*)\<(?P<email>.*)\>")
            .apply(Series.to_dict, axis=1)
            .groupby("path")
            .agg(list)
            .to_frame("authors")
            .combine_first(self.files)
        )
        self.files.index = self.files.index.rename("path")
        return self

    def post_template_cells(self):
        self.files = (
            self.files[["main"]].dropna().apply(self.post_template_cell, axis=1)
        ).combine_first(self.files)
        self.files.index = self.files.index.rename("path")
        self.files.toc = self.files.toc.fillna("")
        return self

    @staticmethod
    def post_template_cell(x):
        from slugify import slugify

        soup = get_soup(x.main)
        for h in soup.select(".nf,.nc"):
            h.attrs.update(role="heading")
        hs = soup.select("h1,h2,h3,h4,h5,h6,[role=heading]")
        toc = """"""
        level = 1
        for h in hs:
            if "id" not in h.attrs:
                h.attrs["id"] = slugify(h.string)
                a = soup.new_tag("a", attrs=dict(href="#" + h.attrs["id"]))
                a.extend(h.children)
                h.clear()
                h.append(a)
            if h.name.startswith("h"):
                level = int(h.name[1])
                toc += "  " * level
                toc += f"""* [{h.string}](#{h.attrs["id"]})\n"""
            else:
                toc += "  " * (level + 1)
                toc += f"""* [`{h.string}`](#{h.attrs["id"]})\n"""

        for x in soup.select(".highlight"):
            x.attrs.update(tabindex=0)

        h1 = soup.select_one("h1")

        if h1:
            h1.attrs["hidden"] = None

        return Series(
            dict(
                main=soup.prettify(),
                title=str(h1.string if h1 else x.name),
                toc=toc,
                description=str(soup.select_one("h1 > p")),
            )
        )


class FontSize(str, Enum):
    xxsmall = "xx-small"
    xsmall = "x-small"
    small = "small"
    medium = "medium"
    large = "large"
    xlarge = "x-large"
    xxlarge = "xx-large"


class Settings(BaseModel):
    block_code_is_interactive: bool = Field(
        True,
        description="code cells are included in the tab order and can be copied with keyboard.",
    )
    # regions or landmarks are exposed in screen reader navigation.
    # when the landmarks are too verbose these values may be set to false
    # to reduce screen reader noise.
    markdown_cells_are_landmarks: bool = Field(
        True, description="markdown cells are landmark regions to assistive technology."
    )
    code_cells_are_landmarks: bool = Field(
        True, description="code cells are landmark regions to assistive technology."
    )
    raw_cells_are_landmarks: bool = Field(
        False, description="raw cells are landmark regions to assistive technology."
    )
    # headings as anchors create large hit areas for those with ambulatory disabilities
    headings_are_link_anchors: bool = Field(
        True,
        description="headings are transformed links to themselves",
    )
    # cells numbers provide fiducial markers for discussing content with others.
    # the are a super set of line numbers
    cell_numbers_are_visible: bool = Field(
        True,
        description="ordinal cells numbers are visible",
    )
    # code mirror uses tables for line numbers, but that adds screen reader noise
    # to table navigation. it is best to avoid artificially adding tables and lists.
    # we need a semantic solution for line numbers.
    line_numbers_are_visible: bool = Field(
        False,
        description="line numbers in block code is visible",
    )
    font_size: FontSize = FontSize.medium
    # function and classes can be included in heading navigation for assistive technology.
    # pygments is used to add role=heading to each of the declarations.
    functions_and_classes_are_headings: bool = Field(
        True, description="function and classes in code blocks are treated as headings"
    )
    functions_and_classes_are_landmarks: bool = Field(
        True, description="function and classes in code blocks are treated as headings"
    )
    functions_and_classes_are_labelled: bool = Field(
        True, description="function and classes in code blocks are treated as headings"
    )
    track_id_history: bool = Field(
        True, description="anchors with like local hrefs are tracked in the url"
    )
