from __future__ import annotations

import argparse
import logging
from typing import Sequence

import requests

from .logging_config import setup_logging
from .pipeline import process

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Генерация HTML-файлов с графиками и таблицами взятий вопросов.",
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="ID турнира или путь к табличному файлу (xlsx/csv)",
    )
    table_group = parser.add_mutually_exclusive_group()
    table_group.add_argument(
        "-t",
        "--table",
        dest="table",
        action="store_true",
        default=True,
        help="Создавать интерактивную таблицу Replay Table (+ CSV) (по умолчанию да)",
    )
    table_group.add_argument(
        "--no-table",
        dest="table",
        action="store_false",
        help="Не создавать интерактивную таблицу",
    )
    parser.add_argument(
        "-f",
        "--flags",
        help="Список ID флагов для фильтрации команд (например: '4,5' или '4+5' или '4*7') (только API)",
    )
    parser.add_argument(
        "-m",
        "--more_files",
        default="yes",
        help="Создавать отдельные файлы для флагов (yes/no, по умолчанию yes)",
    )
    parser.add_argument(
        "-n",
        "--name",
        help="Название турнира (если не указано, берётся из API или имени файла)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Имя каталога для выходных файлов (по умолчанию ID турнира или имя файла)",
    )
    parser.add_argument(
        "--no-crosstable",
        action="store_true",
        help="Не создавать шахматку (по умолчанию создаётся)",
    )
    parser.add_argument(
        "-tc",
        "--tour_comp",
        metavar="N,N,...",
        help="Состав туров по числу вопросов для шахматки, например 15,15,15,15,15,15",
    )
    parser.add_argument(
        "--color",
        metavar="HEX",
        help="Акцентный цвет для заголовков (например #71c0a1 или 71c0a1)",
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

    more_files = str(args.more_files).lower() in {"yes", "да", "y", "д", "true", "1"}
    tour_sizes = None
    if args.tour_comp:
        from .crosstable_export import parse_tour_comp

        tour_sizes = parse_tour_comp(args.tour_comp)

    from .static_bundle import normalize_accent_color

    try:
        accent_color = normalize_accent_color(args.color)
        paths = process(
            args.source,
            flags_arg=args.flags,
            make_table=bool(args.table),
            more_files=more_files,
            name=args.name,
            output=args.output,
            make_crosstable=not args.no_crosstable,
            tour_sizes=tour_sizes,
            accent_color=accent_color,
        )
    except (ValueError, OSError, requests.RequestException) as exc:
        logger.error("%s", exc)
        return 1

    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

