from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeamSeries:
    id: int
    name: str
    town: str
    mask: str
    takes: list[int]
    cumulative_takes: list[int]
    total_takes: int
    position: float | None = None


def mask_to_takes(mask: str) -> list[int]:
    takes: list[int] = []
    for char in mask:
        if char == "1":
            takes.append(1)
        else:
            # '0', 'X', '?' и т.п. — для графика считаем невзятым.
            takes.append(0)
    return takes


def cumulative(values: list[int]) -> list[int]:
    out: list[int] = []
    acc = 0
    for v in values:
        acc += int(v)
        out.append(acc)
    return out


def prepare_chart_data_from_api(teams_data: list[dict[str, Any]]) -> tuple[list[TeamSeries], int]:
    if not teams_data:
        return [], 0

    questions_count = 0
    for team in teams_data:
        mask = team.get("mask") or ""
        if mask:
            questions_count = len(mask)
            break
    if questions_count == 0:
        logger.error("Не удалось определить количество вопросов (все маски пустые)")
        return [], 0

    teams: list[TeamSeries] = []
    for team in teams_data:
        mask = team.get("mask") or ""
        if not mask:
            logger.warning(
                "Пропускаем команду %s с пустой маской",
                (team.get("team", {}) or {}).get("name", "Неизвестная"),
            )
            continue

        team_id = int(((team.get("team", {}) or {}).get("id") or 0))
        team_name = ((team.get("current", {}) or {}).get("name")) or "Неизвестная команда"
        team_town = (((team.get("current", {}) or {}).get("town", {}) or {}).get("name")) or "Неизвестный город"

        takes = mask_to_takes(mask)
        total_takes = int(team.get("questionsTotal") or sum(takes))
        raw_position = team.get("position")
        position = float(raw_position) if raw_position is not None else None
        teams.append(
            TeamSeries(
                id=team_id,
                name=str(team_name),
                town=str(team_town),
                mask=str(mask),
                takes=takes,
                cumulative_takes=cumulative(takes),
                total_takes=total_takes,
                position=position,
            )
        )

    teams.sort(key=lambda t: t.total_takes, reverse=True)
    return teams, questions_count


def write_csv_for_replay_table(teams: list[TeamSeries], questions_count: int, output_path: str) -> None:
    if not teams or questions_count <= 0:
        raise ValueError("Нет данных для создания CSV-файла")

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ["Team"] + [str(i + 1) for i in range(questions_count)]
        csv_writer.writerow(header)
        for team in teams:
            row = [team.name]
            row.extend(team.takes)
            csv_writer.writerow(row)

