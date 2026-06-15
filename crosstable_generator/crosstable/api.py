from __future__ import annotations

import logging
from pathlib import Path

import requests

from .constants import BASE_URL, MIRROR_URL

logger = logging.getLogger(__name__)


def api_base_url(choice: str) -> str:
    return MIRROR_URL if choice == "mirror" else BASE_URL


def fetch_results(base_url: str, tournament_id: int) -> list[dict]:
    url = f"{base_url}/tournaments/{tournament_id}/results"
    logger.debug("Запрос результатов: %s", url)
    response = requests.get(
        url,
        params={"includeMasksAndControversials": 1},
        timeout=30,
    )
    response.raise_for_status()
    entries = response.json()
    logger.debug("Получено записей: %d", len(entries))
    return entries


def fetch_tournament_name(base_url: str, tournament_id: int) -> str:
    url = f"{base_url}/tournaments/{tournament_id}"
    logger.debug("Запрос турнира: %s", url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    name = data.get("longName") or data.get("name") or str(tournament_id)
    logger.debug("Название турнира: %s", name)
    return name
