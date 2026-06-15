from __future__ import annotations

import argparse
import logging
from typing import Sequence

import requests

from .api import api_base_url
from .logging_config import setup_logging
from .pipeline import parse_tour_comp, parse_tournament_ids, run

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Строит шахматку попарного сравнения команд турнира КВРМ.",
    )
    parser.add_argument(
        "tournament_ids",
        help="ID турнира или несколько ID через запятую",
    )
    parser.add_argument(
        "-tc",
        "--tour_comp",
        metavar="N,N,...",
        help="Состав туров по числу вопросов, например 15,15,15,15,15,15",
    )
    parser.add_argument(
        "-u",
        "--url",
        choices=("base", "mirror"),
        default="base",
        help="Базовый API (base) или зеркало (mirror); по умолчанию base",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        help="Имя каталога для выходных файлов; по умолчанию <ID>",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Создать интерактивный HTML-файл",
    )
    parser.add_argument(
        "--xlsx",
        action="store_true",
        help="Создать XLSX-файл",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Уровень логирования (-v для INFO, -vv для DEBUG)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    tournament_ids = parse_tournament_ids(args.tournament_ids)
    tour_sizes = parse_tour_comp(args.tour_comp) if args.tour_comp else None
    base_url = api_base_url(args.url)
    logger.info("API: %s", base_url)
    if tour_sizes:
        logger.info("Туры: %s", ", ".join(str(size) for size in tour_sizes))

    if not args.html and not args.xlsx:
        logger.error("Укажите --html и/или --xlsx")
        return 2

    if args.output and len(tournament_ids) > 1:
        logger.warning(
            "--output игнорируется при нескольких ID, "
            "для каждого турнира создаётся отдельный файл",
        )

    try:
        for path in run(
            tournament_ids,
            base_url,
            tour_sizes,
            args.output,
            args.html,
            args.xlsx,
        ):
            print(path)
    except (ValueError, requests.RequestException, OSError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
