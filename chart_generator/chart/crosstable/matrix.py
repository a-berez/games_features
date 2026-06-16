from __future__ import annotations

from typing import Sequence

from .models import SheetData, TeamResult


def is_taken(symbol: str) -> bool:
    return symbol == "1"


def is_removed(symbol: str) -> bool:
    return symbol.upper() == "X"


def compare_masks_tuple(
    mask_a: str,
    mask_b: str,
    indices: range,
) -> tuple[int, int, int, int]:
    x = y = z = w = 0
    for index in indices:
        symbol_a = mask_a[index]
        symbol_b = mask_b[index]
        if is_removed(symbol_a) or is_removed(symbol_b):
            continue
        a_taken = is_taken(symbol_a)
        b_taken = is_taken(symbol_b)
        if a_taken and not b_taken:
            x += 1
        elif not a_taken and b_taken:
            y += 1
        elif a_taken and b_taken:
            z += 1
        else:
            w += 1
    return x, y, z, w


def format_position(position: float) -> str | int | float:
    if position == float("inf"):
        return ""
    if position == int(position):
        return int(position)
    return position


def validate_mask_lengths(teams: Sequence[TeamResult], expected_len: int | None) -> None:
    if not teams:
        raise ValueError("Нет команд с маской для обработки")
    lengths = {len(team.mask) for team in teams}
    if len(lengths) != 1:
        raise ValueError(f"Маски команд разной длины: {sorted(lengths)}")
    mask_len = lengths.pop()
    if expected_len is not None and mask_len != expected_len:
        raise ValueError(
            f"Длина маски ({mask_len}) не совпадает с суммой туров ({expected_len})"
        )


def build_matrix(
    teams: Sequence[TeamResult],
    indices: range,
) -> list[list[tuple[int, int, int, int] | None]]:
    matrix: list[list[tuple[int, int, int, int] | None]] = []
    for team_a in teams:
        row: list[tuple[int, int, int, int] | None] = []
        for team_b in teams:
            if team_a is team_b:
                row.append(None)
            else:
                row.append(compare_masks_tuple(team_a.mask, team_b.mask, indices))
        matrix.append(row)
    return matrix


def tour_ranges(tour_sizes: Sequence[int]) -> list[range]:
    start = 0
    ranges: list[range] = []
    for size in tour_sizes:
        ranges.append(range(start, start + size))
        start += size
    return ranges


def build_sheets(
    teams: Sequence[TeamResult],
    tour_sizes: Sequence[int] | None,
) -> list[SheetData]:
    sheets = [
        SheetData(
            key="all",
            label="Все",
            matrix=build_matrix(teams, range(len(teams[0].mask))),
        )
    ]
    if tour_sizes:
        for tour_no, indices in enumerate(tour_ranges(tour_sizes), start=1):
            sheets.append(
                SheetData(
                    key=f"T{tour_no}",
                    label=f"Тур {tour_no}",
                    matrix=build_matrix(teams, indices),
                )
            )
    return sheets
