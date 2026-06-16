from dataclasses import dataclass


@dataclass(frozen=True)
class TeamResult:
    name: str
    position: float
    mask: str


@dataclass(frozen=True)
class SheetData:
    key: str
    label: str
    matrix: list[list[tuple[int, int, int, int] | None]]
