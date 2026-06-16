from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputPaths:
    project_dir: Path
    output_dir: Path
    stem: str
    static_dir: Path

    index_html: Path
    chart_html: Path
    table_html: Path
    crosstable_html: Path
    csv: Path
    crosstable_data: Path

    @property
    def crosstable_href(self) -> str:
        return self.crosstable_html.name

    @property
    def static_prefix(self) -> str:
        return "static/" if self.output_dir == self.project_dir else "../static/"

    @property
    def csv_href(self) -> str:
        return f"{self.static_prefix}{self.csv.name}"


def resolve_output_paths(project_name: str, *, page_dir: Path | None = None, stem: str | None = None) -> OutputPaths:
    project_dir = Path(project_name)
    output_dir = page_dir or project_dir
    file_stem = stem or project_dir.name
    static_dir = project_dir / "static"
    return OutputPaths(
        project_dir=project_dir,
        output_dir=output_dir,
        stem=file_stem,
        static_dir=static_dir,
        index_html=output_dir / "index.html",
        chart_html=output_dir / f"{file_stem}_chart.html",
        table_html=output_dir / f"{file_stem}_table.html",
        crosstable_html=output_dir / f"{file_stem}_crosstable.html",
        csv=static_dir / f"{file_stem}.csv",
        crosstable_data=static_dir / f"{file_stem}_crosstable.data.json",
    )
