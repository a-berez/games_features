from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from . import api
from .data import TeamSeries, prepare_chart_data_from_api, write_csv_for_replay_table
from .flags import describe_flag_filter, filter_teams_by_flags, parse_flags_argument
from .crosstable_export import write_crosstable
from .html_export import build_chart_page, build_index_page, build_table_page, copy_static_assets, statistics_html
from .paths import OutputPaths, resolve_output_paths
from .plotly_export import build_plotly_html
from .stats import calculate_questions_statistics

logger = logging.getLogger(__name__)


def is_tournament_id(source: str) -> bool:
    return bool(source and source.isdigit() and not Path(source).exists())


def read_table_file(file_path: str) -> tuple[pd.DataFrame, bool]:
    logger.info("Чтение файла: %s", file_path)
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    elif file_path.endswith(".csv"):
        last_exc: Exception | None = None
        for sep in [";", ",", "\t"]:
            try:
                df = pd.read_csv(file_path, sep=sep)
                if len(df.columns) > 3:
                    break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        else:
            raise ValueError(f"Не удалось прочитать CSV: {last_exc}")  # pragma: no cover
    else:
        raise ValueError("Неподдерживаемый формат файла: нужен .xlsx или .csv")

    df = df.dropna(how="all").fillna("0")
    has_tours = False
    for col_name in list(df.columns)[:5]:
        if isinstance(col_name, str) and ("тур" in col_name.lower() or "tour" in col_name.lower()):
            has_tours = True
            break
    logger.info("Файл прочитан. Строк: %d, столбцов: %d. Разбиение по турам: %s", len(df), len(df.columns), "да" if has_tours else "нет")
    return df, has_tours


def prepare_chart_data_from_table(df: pd.DataFrame, has_tours: bool) -> tuple[list[TeamSeries], int]:
    columns = list(df.columns)

    if not has_tours:
        question_columns = columns[3:]
        question_columns = [col for col in question_columns if isinstance(col, (int, float)) or (isinstance(col, str) and col.isdigit())]
        questions_count = len(question_columns)
        teams: list[TeamSeries] = []
        for _, row in df.iterrows():
            team_id = int(row.iloc[0]) if str(row.iloc[0]).isdigit() else 0
            team_name = str(row.iloc[1])
            team_town = str(row.iloc[2])
            mask_chars: list[str] = []
            takes: list[int] = []
            for col in question_columns:
                value_str = str(row[col]).strip().upper()
                if value_str in {"1", "TRUE", "+"}:
                    mask_chars.append("1")
                    takes.append(1)
                elif value_str in {"X", "Х"}:
                    mask_chars.append("X")
                    takes.append(0)
                else:
                    mask_chars.append("0")
                    takes.append(0)
            mask = "".join(mask_chars)
            teams.append(TeamSeries(id=team_id, name=team_name, town=team_town, mask=mask, takes=takes, cumulative_takes=_cumulative(takes), total_takes=sum(takes)))
        teams.sort(key=lambda t: t.total_takes, reverse=True)
        return teams, questions_count

    tour_column_idx = None
    for i, col in enumerate(columns[:5]):
        if isinstance(col, str) and ("тур" in col.lower() or "tour" in col.lower()):
            tour_column_idx = i
            break
    if tour_column_idx is None:
        return [], 0

    question_columns = columns[tour_column_idx + 1 :]
    question_columns = [col for col in question_columns if isinstance(col, (int, float)) or (isinstance(col, str) and col.isdigit())]
    max_questions_in_tour = len(question_columns)
    max_tour = int(df[columns[tour_column_idx]].max())
    questions_count = max_tour * max_questions_in_tour

    teams_data: dict[Any, dict[str, Any]] = {}
    for _, row in df.iterrows():
        team_id = row.iloc[0]
        team_name = str(row.iloc[1])
        team_town = str(row.iloc[2])
        tour = int(row[columns[tour_column_idx]])
        if team_id not in teams_data:
            teams_data[team_id] = {
                "id": int(team_id) if str(team_id).isdigit() else 0,
                "name": team_name,
                "town": team_town,
                "mask": ["0"] * questions_count,
                "takes": [0] * questions_count,
            }
        for i, col in enumerate(question_columns):
            qidx = (tour - 1) * max_questions_in_tour + i
            if qidx >= questions_count:
                continue
            value_str = str(row[col]).strip().upper()
            if value_str in {"1", "TRUE", "+"}:
                teams_data[team_id]["mask"][qidx] = "1"
                teams_data[team_id]["takes"][qidx] = 1
            elif value_str in {"X", "Х"}:
                teams_data[team_id]["mask"][qidx] = "X"
                teams_data[team_id]["takes"][qidx] = 0

    teams: list[TeamSeries] = []
    for t in teams_data.values():
        mask = "".join(t["mask"])
        takes = list(t["takes"])
        teams.append(TeamSeries(id=t["id"], name=t["name"], town=t["town"], mask=mask, takes=takes, cumulative_takes=_cumulative(takes), total_takes=sum(takes)))
    teams.sort(key=lambda t: t.total_takes, reverse=True)
    return teams, questions_count


def _cumulative(values: list[int]) -> list[int]:
    acc = 0
    out: list[int] = []
    for v in values:
        acc += int(v)
        out.append(acc)
    return out


def process(
    source: str,
    *,
    flags_arg: str | None,
    make_table: bool,
    more_files: bool,
    name: str | None,
    output: str | None,
    make_crosstable: bool = True,
    tour_sizes: list[int] | None = None,
    accent_color: str | None = None,
) -> list[Path]:
    if not source:
        raise ValueError("Не указан источник данных (ID турнира или путь к файлу)")

    is_api = is_tournament_id(source)
    tournament_id: int | None = int(source) if is_api else None

    if is_api:
        info = api.fetch_tournament_info(int(source))
        tournament_name = name or (info.get("name") or f"Турнир {source}")
        tournament_long_name = info.get("longName") or tournament_name
        teams_raw = api.fetch_tournament_results(int(source))
        flags_filters = parse_flags_argument(flags_arg)
        flags_info = api.fetch_team_flags() if flags_filters else {}

        main_teams, questions_count = prepare_chart_data_from_api(teams_raw)
        if not main_teams:
            raise ValueError(f"Не удалось подготовить данные для турнира {source}")

        flag_charts: dict[str, tuple[str, list[TeamSeries], int]] = {}
        for flag_filter in flags_filters:
            key, human_name = describe_flag_filter(flag_filter, flags_info)
            filtered_raw = filter_teams_by_flags(teams_raw, flag_filter)
            teams, qn = prepare_chart_data_from_api(filtered_raw)
            flag_charts[key] = (human_name, teams, qn)
    else:
        df, has_tours = read_table_file(source)
        tournament_name = name or Path(source).stem
        tournament_long_name = tournament_name
        main_teams, questions_count = prepare_chart_data_from_table(df, has_tours)
        if not main_teams:
            raise ValueError(f"Не удалось подготовить данные из файла {source}")
        flag_charts = {}

    output_name = output or (str(tournament_id) if tournament_id else Path(source).stem)
    paths = resolve_output_paths(output_name)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    copy_static_assets(paths.static_dir)
    written: list[Path] = []

    crosstable_href: str | None = None
    if make_crosstable:
        crosstable_paths = write_crosstable(
            main_teams,
            paths,
            tournament_name=tournament_name,
            tournament_id=tournament_id,
            tour_sizes=tour_sizes,
            static_prefix=paths.static_prefix,
            accent_color=accent_color,
            chart_href=paths.chart_html.name,
            table_href=paths.table_html.name if make_table else None,
        )
        written.extend(crosstable_paths)
        crosstable_href = paths.crosstable_href

    def write_pages_for_group(
        out_paths: OutputPaths,
        title: str,
        teams: list[TeamSeries],
        qn: int,
        *,
        chart_href: str,
        table_href: str | None,
    ) -> None:
        stats = calculate_questions_statistics(teams, qn)
        chart_html = build_plotly_html(teams, qn, title)
        page = build_chart_page(
            page_title=title,
            teams_count=len(teams),
            chart_html=chart_html,
            stats_html=statistics_html(stats),
            tournament_id=tournament_id,
            chart_href=chart_href,
            table_href=table_href,
            crosstable_href=crosstable_href,
            static_prefix=out_paths.static_prefix,
            accent_color=accent_color,
        )
        out_paths.chart_html.write_text(page, encoding="utf-8")
        written.append(out_paths.chart_html)

        if make_table:
            out_paths.static_dir.mkdir(parents=True, exist_ok=True)
            write_csv_for_replay_table(teams, qn, str(out_paths.csv))
            written.append(out_paths.csv)
            table_page = build_table_page(
                page_title=title,
                csv_href=out_paths.csv_href,
                tournament_id=tournament_id,
                chart_href=chart_href,
                crosstable_href=crosstable_href,
                static_prefix=out_paths.static_prefix,
                accent_color=accent_color,
            )
            out_paths.table_html.write_text(table_page, encoding="utf-8")
            written.append(out_paths.table_html)

    write_pages_for_group(
        paths,
        tournament_name,
        main_teams,
        questions_count,
        chart_href=paths.chart_html.name,
        table_href=paths.table_html.name if make_table else None,
    )

    index_page = build_index_page(
        page_title=tournament_long_name,
        chart_href=paths.chart_html.name,
        table_href=paths.table_html.name if make_table else None,
        crosstable_href=crosstable_href,
        tournament_id=tournament_id,
        static_prefix=paths.static_prefix,
        accent_color=accent_color,
    )
    paths.index_html.write_text(index_page, encoding="utf-8")
    written.append(paths.index_html)

    if is_api and flag_charts and more_files:
        extra_dir = paths.project_dir / "extra"
        extra_dir.mkdir(parents=True, exist_ok=True)
        for key, (human_name, teams, qn) in flag_charts.items():
            safe_key = key.replace("*", "AND").replace("+", "OR").replace("/", "_").replace("\\", "_")
            extra_stem = f"{paths.stem}_{safe_key}"
            extra_paths = resolve_output_paths(
                output_name,
                page_dir=extra_dir,
                stem=extra_stem,
            )
            write_pages_for_group(
                extra_paths,
                f"{tournament_name}. {human_name}",
                teams,
                qn,
                chart_href=extra_paths.chart_html.name,
                table_href=extra_paths.table_html.name if make_table else None,
            )

    logger.info("Готово: %s", paths.project_dir)
    return written
