from __future__ import annotations

import json
import logging
from pathlib import Path

from .crosstable.matrix import build_sheets, validate_mask_lengths
from .crosstable.models import TeamResult
from .crosstable.payload import build_data_payload
from .data import TeamSeries
from .html_export import build_crosstable_page, build_crosstable_tab_markup
from .paths import OutputPaths

logger = logging.getLogger(__name__)


def parse_tour_comp(raw: str) -> list[int]:
    sizes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("Некорректный формат --tour_comp")
    return sizes


def teams_to_crosstable_results(teams: list[TeamSeries]) -> list[TeamResult]:
    results: list[TeamResult] = []
    for rank, team in enumerate(teams, start=1):
        position = team.position if team.position is not None else float(rank)
        results.append(TeamResult(name=team.name, position=position, mask=team.mask))
    results.sort(key=lambda item: (item.position, item.name))
    return results


def write_crosstable(
    teams: list[TeamSeries],
    out_paths: OutputPaths,
    *,
    tournament_name: str,
    tournament_id: int | None,
    tour_sizes: list[int] | None,
    static_prefix: str,
    accent_color: str | None,
    chart_href: str,
    table_href: str | None,
) -> list[Path]:
    """Генерирует HTML-шахматку и JSON-данные в static/."""
    team_results = teams_to_crosstable_results(teams)
    expected_len = sum(tour_sizes) if tour_sizes else None
    validate_mask_lengths(team_results, expected_len)
    sheets = build_sheets(team_results, tour_sizes)
    logger.info("Генерация шахматки (%d команд, %d листов)", len(team_results), len(sheets))

    out_paths.static_dir.mkdir(parents=True, exist_ok=True)
    out_paths.crosstable_data.write_text(
        json.dumps(build_data_payload(team_results, sheets), ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    logger.info("Сохранён %s", out_paths.crosstable_data)

    tab_buttons, tab_panels, active_tab = build_crosstable_tab_markup(sheets)
    data_href = f"{static_prefix}{out_paths.crosstable_data.name}"
    page = build_crosstable_page(
        page_title=tournament_name,
        tournament_id=tournament_id,
        data_href=data_href,
        tab_buttons=tab_buttons,
        tab_panels=tab_panels,
        active_tab=active_tab,
        chart_href=chart_href,
        table_href=table_href,
        static_prefix=static_prefix,
        accent_color=accent_color,
    )
    out_paths.crosstable_html.write_text(page, encoding="utf-8")
    logger.info("Сохранён %s", out_paths.crosstable_html)
    return [out_paths.crosstable_html, out_paths.crosstable_data]
