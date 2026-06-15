from __future__ import annotations

import html
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Sequence

from .constants import STATIC_DIR, STATIC_FILES, TEMPLATES_DIR
from .matrix import format_position
from .models import SheetData, TeamResult

logger = logging.getLogger(__name__)


def render_template(name: str, **placeholders: str) -> str:
    text = (TEMPLATES_DIR / name).read_text(encoding="utf-8")
    for key, value in placeholders.items():
        text = text.replace(f"{{{{{key}}}}}", value)
    return text


def build_data_payload(
    teams: Sequence[TeamResult],
    sheets: Sequence[SheetData],
) -> dict[str, Any]:
    return {
        "teams": [
            {"name": team.name, "position": format_position(team.position)}
            for team in teams
        ],
        "sheets": {
            sheet.key: [
                [list(cell) if cell is not None else None for cell in row]
                for row in sheet.matrix
            ]
            for sheet in sheets
        },
    }


def build_html(
    tournament_id: int,
    tournament_name: str,
    sheets: Sequence[SheetData],
    data_href: str,
) -> str:
    tab_buttons: list[str] = []
    tab_panels: list[str] = []
    for index, sheet in enumerate(sheets):
        active = " active" if index == 0 else ""
        tab_buttons.append(
            f'<button type="button" class="tab{active}" data-tab="{html.escape(sheet.key)}">'
            f"{html.escape(sheet.label)}</button>"
        )
        tab_panels.append(
            f'<div class="tab-panel{active}" data-tab="{html.escape(sheet.key)}">'
            '<div class="matrix-mount"></div>'
            "</div>"
        )

    return render_template(
        "crosstable.html",
        page_title=html.escape(tournament_name),
        tournament_id=str(tournament_id),
        tab_buttons="".join(tab_buttons),
        tab_panels="".join(tab_panels),
        active_tab=html.escape(sheets[0].key),
        data_href=html.escape(data_href),
        css_href="crosstable.css",
        js_href="crosstable.js",
    )


def copy_static_assets(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in STATIC_FILES:
        shutil.copy2(STATIC_DIR / filename, output_dir / filename)
        logger.debug("Скопирован %s → %s", filename, output_dir)


def write_html(
    html_path: Path,
    data_path: Path,
    tournament_id: int,
    tournament_name: str,
    teams: Sequence[TeamResult],
    sheets: Sequence[SheetData],
) -> list[Path]:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    data_href = data_path.name

    data_path.write_text(
        json.dumps(build_data_payload(teams, sheets), ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    logger.info("Сохранён %s", data_path)

    html_path.write_text(
        build_html(tournament_id, tournament_name, sheets, data_href),
        encoding="utf-8",
    )
    logger.info("Сохранён %s", html_path)
    copy_static_assets(html_path.parent)
    return [html_path, data_path]
