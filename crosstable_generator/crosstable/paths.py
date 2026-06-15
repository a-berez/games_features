from __future__ import annotations

from pathlib import Path

HTML_INDEX = "index.html"


def resolve_output_bundle(tournament_id: int, output: str | None) -> tuple[Path, Path]:
    """Возвращает каталог вывода и базовый путь для xlsx внутри него."""
    if output:
        bundle = Path(output)
        if bundle.suffix.lower() in {".xlsx", ".html"}:
            bundle = bundle.with_suffix("")
        output_dir = bundle
        stem = output_dir / output_dir.name
    else:
        name = str(tournament_id)
        output_dir = Path(name)
        stem = output_dir / name
    return output_dir, stem


def html_index_path(output_dir: Path) -> Path:
    return output_dir / HTML_INDEX


def data_json_path(output_dir: Path) -> Path:
    return output_dir / f"{output_dir.name}.data.json"
