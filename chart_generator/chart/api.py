from __future__ import annotations

import logging
from typing import Any

import requests

from .constants import BASE_URL

logger = logging.getLogger(__name__)


def fetch_json(url: str, *, params: dict[str, Any] | None = None, timeout: int = 10) -> Any:
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_tournament_info(tournament_id: int, base_url: str = BASE_URL) -> dict[str, Any]:
    url = f"{base_url}/tournaments/{tournament_id}"
    logger.info("Получение информации о турнире %d…", tournament_id)
    try:
        info = fetch_json(url)
    except requests.RequestException as exc:
        logger.error("Не удалось получить информацию о турнире %d: %s", tournament_id, exc)
        return {}
    return info or {}


def fetch_tournament_results(tournament_id: int, base_url: str = BASE_URL) -> list[dict[str, Any]]:
    url = f"{base_url}/tournaments/{tournament_id}/results"
    params = {
        "includeTeamMembers": 0,
        "includeMasksAndControversials": 1,
        "includeTeamFlags": 1,
        "includeRatingB": 0,
    }
    logger.info("Получение результатов турнира %d…", tournament_id)
    try:
        results = fetch_json(url, params=params, timeout=20)
    except requests.RequestException as exc:
        logger.error("Не удалось получить результаты турнира %d: %s", tournament_id, exc)
        return []
    if not isinstance(results, list):
        return []
    logger.info("Получены результаты для %d команд", len(results))
    return results


def fetch_team_flags(base_url: str = BASE_URL) -> dict[int, dict[str, Any]]:
    url = f"{base_url}/tournament_flags"
    logger.info("Получение справочника флагов команд…")
    try:
        flags_list = fetch_json(url, timeout=20)
    except requests.RequestException as exc:
        logger.error("Не удалось получить флаги команд: %s", exc)
        return {}
    if not isinstance(flags_list, list):
        return {}
    return {int(flag["id"]): flag for flag in flags_list if isinstance(flag, dict) and "id" in flag}

