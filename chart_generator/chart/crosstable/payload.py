from __future__ import annotations

from typing import Any, Sequence

from .matrix import format_position
from .models import SheetData, TeamResult


def build_data_payload(
    teams: Sequence[TeamResult],
    sheets: Sequence[SheetData],
) -> dict[str, Any]:
    return {
        "teams": [
            {"name": team.name, "position": format_position(team.position)}
            for team in teams
        ],
        "sheets": {
            sheet.key: [
                [list(cell) if cell is not None else None for cell in row]
                for row in sheet.matrix
            ]
            for sheet in sheets
        },
    }
