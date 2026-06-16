from __future__ import annotations

import re
from pathlib import Path

from .constants import STATIC_DIR

STYLE_CSS = "style.css"
SCRIPTS_JS = "scripts.js"

_CROSSTABLE_ACCENT_REPLACEMENTS = (
    ("color: #6464a6;", "color: var(--accent-color, #71c0a1);"),
    ("background: #6464a6;", "background: var(--accent-color, #71c0a1);"),
    ("border-color: #6464a6;", "border-color: var(--accent-color, #71c0a1);"),
)


def build_style_css() -> str:
    chart_css = (STATIC_DIR / "chart_generator_style.css").read_text(encoding="utf-8")
    crosstable_css = (STATIC_DIR / "crosstable.css").read_text(encoding="utf-8")
    for old, new in _CROSSTABLE_ACCENT_REPLACEMENTS:
        crosstable_css = crosstable_css.replace(old, new)
    return f"{chart_css.rstrip()}\n\n/* === Crosstable === */\n{crosstable_css}"


def build_scripts_js() -> str:
    parts = [
        (STATIC_DIR / "d3.v4.min.js").read_text(encoding="utf-8"),
        (STATIC_DIR / "replay-table.min.js").read_text(encoding="utf-8"),
        (STATIC_DIR / "chart_generator_scripts.js").read_text(encoding="utf-8"),
        (STATIC_DIR / "crosstable.js").read_text(encoding="utf-8"),
    ]
    return "\n".join(part.rstrip() for part in parts) + "\n"


def write_bundled_static(static_dir: Path) -> None:
    static_dir.mkdir(parents=True, exist_ok=True)
    (static_dir / STYLE_CSS).write_text(build_style_css(), encoding="utf-8")
    (static_dir / SCRIPTS_JS).write_text(build_scripts_js(), encoding="utf-8")


def normalize_accent_color(raw: str | None) -> str:
    default = "#71c0a1"
    if not raw:
        return default
    color = raw.strip()
    if not color.startswith("#"):
        color = f"#{color}"
    if not re.fullmatch(r"#[0-9A-Fa-f]{3}([0-9A-Fa-f]{3})?", color):
        raise ValueError(f"Некорректный цвет: {raw!r} (ожидается #RGB или #RRGGBB)")
    return color.lower()
