from __future__ import annotations

import logging
from typing import Iterable

from .models import TeamResult

logger = logging.getLogger(__name__)


def team_name(entry: dict) -> str:
    current = entry.get("current") or {}
    name = current.get("name")
    if name:
        return name
    team = entry.get("team") or {}
    return team.get("name") or ""


def parse_teams(entries: Iterable[dict]) -> list[TeamResult]:
    teams: list[TeamResult] = []
    skipped = 0
    total = 0
    for entry in entries:
        total += 1
        mask = entry.get("mask")
        if mask is None:
            skipped += 1
            logger.debug("Пропуск команды без маски: %s", team_name(entry))
            continue
        position = entry.get("position")
        if position is None:
            position = float("inf")
        teams.append(
            TeamResult(
                name=team_name(entry),
                position=float(position),
                mask=mask,
            )
        )
    teams.sort(key=lambda team: (team.position, team.name))
    logger.info("Команд с маской: %d из %d", len(teams), total)
    if skipped:
        logger.info("Пропущено без маски: %d", skipped)
    return teams
