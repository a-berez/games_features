from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Union

logger = logging.getLogger(__name__)

FlagFilter = Union[int, tuple[str, list[int]]]


def parse_flags_argument(flags_arg: str | None) -> list[FlagFilter]:
    if not flags_arg:
        return []

    parts = re.split(r"[,;\s]+", flags_arg)
    result: list[FlagFilter] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "+" in part:
            flag_ids = [int(f) for f in part.split("+") if f.strip()]
            result.append(("OR", flag_ids))
        elif "*" in part:
            flag_ids = [int(f) for f in part.split("*") if f.strip()]
            result.append(("AND", flag_ids))
        else:
            try:
                result.append(int(part))
            except ValueError:
                logger.warning("Некорректный ID флага: %s, пропускаем", part)
    return result


def filter_teams_by_flags(teams: Iterable[dict[str, Any]], flag_filter: FlagFilter) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(flag_filter, int):
        flag_id = flag_filter
        for team in teams:
            team_flags = [flag.get("id") for flag in team.get("flags", []) if isinstance(flag, dict)]
            if flag_id in team_flags:
                out.append(team)
        return out

    operator, flag_ids = flag_filter
    for team in teams:
        team_flags = [flag.get("id") for flag in team.get("flags", []) if isinstance(flag, dict)]
        if operator == "OR" and any(flag_id in team_flags for flag_id in flag_ids):
            out.append(team)
        elif operator == "AND" and all(flag_id in team_flags for flag_id in flag_ids):
            out.append(team)
    return out


def describe_flag_filter(flag_filter: FlagFilter, flags_info: dict[int, dict[str, Any]]) -> tuple[str, str]:
    """Возвращает (key, human_name). key используется в имени файла."""
    if isinstance(flag_filter, int):
        flag_id = flag_filter
        name = flags_info.get(flag_id, {}).get("longName") or f"Флаг {flag_id}"
        return str(flag_id), str(name)

    operator, flag_ids = flag_filter
    op_symbol = "+" if operator == "OR" else "*"
    key = op_symbol.join(str(fid) for fid in flag_ids)
    names = [flags_info.get(fid, {}).get("longName") or f"Флаг {fid}" for fid in flag_ids]
    if operator == "OR":
        return key, f"Зачёт: {' или '.join(names)}"
    return key, f"Зачёт: {' и '.join(names)}"

