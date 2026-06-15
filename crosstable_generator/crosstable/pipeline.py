from __future__ import annotations

import logging
from typing import Sequence

from .api import fetch_results, fetch_tournament_name
from .html_export import write_html
from .matrix import build_sheets, validate_mask_lengths
from .paths import data_json_path, html_index_path, resolve_output_bundle
from .teams import parse_teams
from .xlsx_export import build_workbook

logger = logging.getLogger(__name__)


def parse_tournament_ids(raw: str) -> list[int]:
    ids: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    if not ids:
        raise ValueError("Не указан ID турнира")
    return ids


def parse_tour_comp(raw: str) -> list[int]:
    sizes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("Некорректный формат --tour_comp")
    return sizes


def process_tournament(
    tournament_id: int,
    base_url: str,
    tour_sizes: list[int] | None,
    output: str | None,
    make_html: bool,
    make_xlsx: bool,
) -> list[str]:
    if not make_html and not make_xlsx:
        raise ValueError("Укажите --html и/или --xlsx")

    logger.info("Обработка турнира %d", tournament_id)
    entries = fetch_results(base_url, tournament_id)
    teams = parse_teams(entries)
    expected_len = sum(tour_sizes) if tour_sizes else None
    validate_mask_lengths(teams, expected_len)
    logger.debug("Длина маски: %d", len(teams[0].mask))

    sheets = build_sheets(teams, tour_sizes)
    logger.info("Листов: %d (%s)", len(sheets), ", ".join(sheet.key for sheet in sheets))

    output_dir, stem = resolve_output_bundle(tournament_id, output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Каталог вывода: %s", output_dir)
    paths: list[str] = []

    if make_xlsx:
        xlsx_path = stem.with_suffix(".xlsx")
        build_workbook(teams, sheets).save(xlsx_path)
        paths.append(str(xlsx_path))
        logger.info("Сохранён %s", xlsx_path)

    if make_html:
        tournament_name = fetch_tournament_name(base_url, tournament_id)
        html_paths = write_html(
            html_index_path(output_dir),
            data_json_path(output_dir),
            tournament_id,
            tournament_name,
            teams,
            sheets,
        )
        paths.extend(str(path) for path in html_paths)

    return paths


def run(
    tournament_ids: Sequence[int],
    base_url: str,
    tour_sizes: list[int] | None,
    output: str | None,
    make_html: bool,
    make_xlsx: bool,
) -> list[str]:
    paths: list[str] = []
    for tournament_id in tournament_ids:
        tournament_output = output if len(tournament_ids) == 1 else None
        paths.extend(
            process_tournament(
                tournament_id,
                base_url,
                tour_sizes,
                tournament_output,
                make_html,
                make_xlsx,
            )
        )
    return paths
