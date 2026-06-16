#!/usr/bin/env python

import argparse
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger("SeatingGenerator")


@dataclass
class PenaltyConfig:
    """Настраиваемые параметры расчёта штрафов."""
    number_close_diff: int = 3
    number_top_threshold: int = 5
    number_top_multiplier: float = 1.5
    diagonal_neighbor_weight: float = 1.0


@dataclass
class OptimizationConfig:
    """Параметры поиска рассадки."""
    num_initial_placements: int = 10
    num_to_optimize: int = 3
    max_iterations: int = 1000


def setup_logging() -> None:
    log_path = os.path.join(SCRIPT_DIR, "seating_generator.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def parse_arguments() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Генератор рассадки команд")
    parser.add_argument("-i", "--input", type=str, help="Путь к файлу со списком команд")
    parser.add_argument("-s", "--seating", type=str, help="Путь к файлу со схемой рассадки")
    parser.add_argument(
        "-abc",
        "--alphabetical_order",
        action="store_true",
        help="Устаревший алиас для --ignore-number",
    )
    parser.add_argument(
        "--ignore-number",
        action="store_true",
        help="Не учитывать столбец number при расчёте штрафов",
    )
    parser.add_argument(
        "-o",
        "--order",
        type=str,
        help='Порядок важности столбцов (например, "2,4,1,3")',
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Не запрашивать подтверждение порядка критериев",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Путь к выходному файлу лучшего варианта (по умолчанию seating.xlsx)",
    )
    parser.add_argument(
        "--initial-placements",
        type=int,
        default=10,
        help="Количество начальных вариантов рассадки (по умолчанию: 10)",
    )
    parser.add_argument(
        "--optimize-count",
        type=int,
        default=3,
        help="Сколько лучших начальных вариантов оптимизировать (по умолчанию: 3)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Максимум итераций оптимизации на вариант (по умолчанию: 1000)",
    )
    parser.add_argument(
        "--number-close-diff",
        type=int,
        default=3,
        help="Макс. разница номеров для штрафа за близость (по умолчанию: 3)",
    )
    parser.add_argument(
        "--diagonal-weight",
        type=float,
        default=1.0,
        help="Множитель штрафа для диагональных соседей (по умолчанию: 1.0)",
    )
    return parser.parse_args()


def load_teams_data(file_path: str) -> pd.DataFrame:
    """Загрузка данных о командах из Excel файла."""
    try:
        logger.info(f"Загрузка данных о командах из {file_path}")
        df = pd.read_excel(file_path)

        if "team" not in df.columns:
            raise ValueError("В файле отсутствует обязательный столбец 'team'")

        if len(df.columns) < 2:
            raise ValueError("В файле должен быть хотя бы один дополнительный столбец")

        seating_columns = [col for col in df.columns if col != "team"]
        if not seating_columns:
            raise ValueError("В файле должен быть хотя бы один столбец для определения рассадки")

        logger.info(f"Столбцы для определения рассадки: {seating_columns}")

        if df["team"].duplicated().any():
            logger.warning("Обнаружены дубликаты команд")
            df = df.drop_duplicates(subset=["team"])

        if df["team"].isna().any() or (df["team"].astype(str).str.strip() == "").any():
            raise ValueError("Столбец 'team' содержит пустые значения")

        return df
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных о командах: {e}")
        raise


def load_seating_plan(
    file_path: str,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], str]]:
    """
    Загрузка схемы рассадки из Excel файла.

    X — свободное место, O — недоступное место,
    любой другой текст — заранее закреплённая за командой позиция.
    """
    try:
        logger.info(f"Загрузка схемы рассадки из {file_path}")
        df = pd.read_excel(file_path, header=None)
        seating_plan = df.values
        fixed_placements: Dict[Tuple[int, int], str] = {}

        for i in range(seating_plan.shape[0]):
            for j in range(seating_plan.shape[1]):
                raw = seating_plan[i, j]
                if pd.isna(raw) or str(raw).strip() == "":
                    seating_plan[i, j] = "O"
                    continue

                value = str(raw).strip()
                upper = value.upper()
                if upper == "X":
                    seating_plan[i, j] = "X"
                elif upper == "O":
                    seating_plan[i, j] = "O"
                else:
                    seating_plan[i, j] = "X"
                    fixed_placements[(i, j)] = value

        available_seats = np.sum(seating_plan == "X")
        free_seats = available_seats - len(fixed_placements)
        logger.info(
            f"Доступно мест: {available_seats}, "
            f"из них заранее занято: {len(fixed_placements)}, свободно: {free_seats}"
        )
        return seating_plan, fixed_placements
    except Exception as e:
        logger.error(f"Ошибка при загрузке схемы рассадки: {e}")
        raise


def parse_column_order_indices(order_arg: str, num_columns: int) -> List[int]:
    """Преобразование строки порядка столбцов в список индексов с проверкой."""
    order_indices = [int(x.strip()) - 1 for x in order_arg.split(",")]

    if max(order_indices) >= num_columns or min(order_indices) < 0:
        raise ValueError(f"Недопустимые индексы в порядке столбцов: {order_arg}")

    if len(order_indices) != num_columns:
        raise ValueError("Количество индексов не соответствует количеству столбцов")

    if len(order_indices) != len(set(order_indices)):
        raise ValueError("Обнаружены дубликаты в порядке столбцов")

    return order_indices


def get_column_order(df: pd.DataFrame, order_arg: Optional[str] = None) -> List[str]:
    """Определение порядка важности столбцов."""
    columns = [col for col in df.columns if col != "team"]

    if order_arg:
        order_indices = parse_column_order_indices(order_arg, len(columns))
        return [columns[i] for i in order_indices]

    return columns


def confirm_column_order(columns: List[str]) -> List[str]:
    """Запрос подтверждения порядка важности столбцов у пользователя."""
    logger.info("Запрос подтверждения порядка важности столбцов")
    print("\nПорядок учета критериев (от наиболее важного к наименее важному):")
    for i, col in enumerate(columns, 1):
        print(f"{i}. {col}")

    base_columns = [col for col in columns]

    while True:
        try:
            confirmation = input("\nПодтвердите порядок (да/нет): ").strip().lower()
            if confirmation in ["да", "yes", "y", "д"]:
                logger.info("Порядок столбцов подтвержден")
                return columns
            if confirmation in ["нет", "no", "n", "н"]:
                new_order = input("Введите новый порядок (например, 2,4,1,3): ").strip()
                try:
                    order_indices = parse_column_order_indices(new_order, len(base_columns))
                except ValueError as exc:
                    print(f"Ошибка: {exc}")
                    continue

                new_columns = [base_columns[i] for i in order_indices]
                logger.info(f"Установлен новый порядок столбцов: {new_columns}")

                print("\nНовый порядок учета критериев:")
                for i, col in enumerate(new_columns, 1):
                    print(f"{i}. {col}")
                final_confirmation = input("\nПодтвердите новый порядок (да/нет): ").strip().lower()
                if final_confirmation in ["да", "yes", "y", "д"]:
                    logger.info("Новый порядок столбцов подтвержден")
                    return new_columns
                continue

            print("Пожалуйста, введите 'да' или 'нет'")
        except Exception as e:
            logger.error(f"Ошибка при вводе порядка столбцов: {e}")
            print(f"Ошибка: {e}. Пожалуйста, попробуйте снова.")


def get_neighbors_weighted(
    row: int,
    col: int,
    seating_plan: np.ndarray,
    penalty_config: PenaltyConfig,
) -> List[Tuple[int, int, float]]:
    """Соседние места с весом (диагональные могут иметь меньший вес)."""
    rows, cols = seating_plan.shape
    neighbors: List[Tuple[int, int, float]] = []

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue

            new_row, new_col = row + dr, col + dc
            if not (0 <= new_row < rows and 0 <= new_col < cols):
                continue
            if seating_plan[new_row, new_col] != "X":
                continue

            is_diagonal = dr != 0 and dc != 0
            weight = penalty_config.diagonal_neighbor_weight if is_diagonal else 1.0
            neighbors.append((new_row, new_col, weight))

    return neighbors


def calculate_penalty(
    team1: pd.Series,
    team2: pd.Series,
    columns: List[str],
    ignore_number: bool,
    penalty_config: PenaltyConfig,
    neighbor_weight: float = 1.0,
) -> float:
    """Расчёт штрафа за размещение двух команд рядом."""
    penalty = 0.0

    for i, column in enumerate(columns):
        weight = 1.0 / (i + 1)

        if column == "number" and not ignore_number:
            try:
                num1 = float(team1[column]) if pd.notna(team1[column]) else 0
                num2 = float(team2[column]) if pd.notna(team2[column]) else 0
                number_diff = abs(num1 - num2)
                if number_diff <= penalty_config.number_close_diff:
                    number_penalty = (penalty_config.number_close_diff + 1 - number_diff) * weight
                    if min(num1, num2) <= penalty_config.number_top_threshold:
                        number_penalty *= penalty_config.number_top_multiplier
                    penalty += number_penalty
            except (ValueError, TypeError):
                logger.warning("Не удалось преобразовать значения в столбце 'number' к числу")
        elif pd.notna(team1[column]) and pd.notna(team2[column]):
            if str(team1[column]).strip() == str(team2[column]).strip():
                penalty += weight

    return penalty * neighbor_weight


def get_available_positions(
    seating_plan: np.ndarray,
    fixed_positions: Optional[Set[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """Свободные позиции для размещения команд."""
    fixed_positions = fixed_positions or set()
    positions = []

    for r in range(seating_plan.shape[0]):
        for c in range(seating_plan.shape[1]):
            pos = (r, c)
            if seating_plan[r, c] == "X" and pos not in fixed_positions:
                positions.append(pos)

    positions.sort()
    return positions


class SeatingEvaluator:
    """Оценка рассадки с быстрым пересчётом штрафа при обмене двух команд."""

    def __init__(
        self,
        teams_df: pd.DataFrame,
        seating_plan: np.ndarray,
        columns: List[str],
        ignore_number: bool,
        penalty_config: PenaltyConfig,
        fixed_placement: Dict[str, Tuple[int, int]],
    ):
        self.seating_plan = seating_plan
        self.columns = columns
        self.ignore_number = ignore_number
        self.penalty_config = penalty_config
        self.fixed_teams = set(fixed_placement.keys())
        self.team_data = {
            row["team"]: row for _, row in teams_df.iterrows()
        }
        self.placement: Dict[str, Tuple[int, int]] = dict(fixed_placement)
        self.position_to_team: Dict[Tuple[int, int], str] = {
            pos: team for team, pos in fixed_placement.items()
        }

    def set_free_placement(self, free_placement: Dict[str, Tuple[int, int]]) -> None:
        """Установить размещение для команд, которые можно перемещать."""
        for team, pos in list(self.placement.items()):
            if team in self.fixed_teams:
                continue
            del self.placement[team]
            self.position_to_team.pop(pos, None)

        for team, pos in free_placement.items():
            if team in self.fixed_teams:
                raise ValueError(f"Команда '{team}' закреплена на схеме и не может быть перемещена")
            if pos in self.position_to_team:
                raise ValueError(f"Позиция {pos} уже занята")
            self.placement[team] = pos
            self.position_to_team[pos] = team

    def total_penalty(self) -> float:
        """Суммарный штраф; каждая пара соседей учитывается один раз."""
        total = 0.0
        processed: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

        for pos, team in self.position_to_team.items():
            row, col = pos
            team_data = self.team_data[team]
            for nrow, ncol, neighbor_weight in get_neighbors_weighted(
                row, col, self.seating_plan, self.penalty_config
            ):
                neighbor_pos = (nrow, ncol)
                if neighbor_pos not in self.position_to_team:
                    continue

                pair = (pos, neighbor_pos) if pos < neighbor_pos else (neighbor_pos, pos)
                if pair in processed:
                    continue
                processed.add(pair)

                neighbor_team = self.position_to_team[neighbor_pos]
                neighbor_data = self.team_data[neighbor_team]
                total += calculate_penalty(
                    team_data,
                    neighbor_data,
                    self.columns,
                    self.ignore_number,
                    self.penalty_config,
                    neighbor_weight,
                )

        return total

    def _affected_positions(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> Set[Tuple[int, int]]:
        affected = {pos1, pos2}
        for pos in (pos1, pos2):
            row, col = pos
            for nrow, ncol, _ in get_neighbors_weighted(
                row, col, self.seating_plan, self.penalty_config
            ):
                affected.add((nrow, ncol))
        return affected

    def _penalty_touching_positions(self, touch_positions: Set[Tuple[int, int]]) -> float:
        total = 0.0
        processed: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

        for pos in touch_positions:
            if pos not in self.position_to_team:
                continue

            row, col = pos
            team = self.position_to_team[pos]
            team_data = self.team_data[team]

            for nrow, ncol, neighbor_weight in get_neighbors_weighted(
                row, col, self.seating_plan, self.penalty_config
            ):
                neighbor_pos = (nrow, ncol)
                if neighbor_pos not in self.position_to_team:
                    continue

                pair = (pos, neighbor_pos) if pos < neighbor_pos else (neighbor_pos, pos)
                if pair in processed:
                    continue
                if pos not in touch_positions and neighbor_pos not in touch_positions:
                    continue

                processed.add(pair)
                neighbor_team = self.position_to_team[neighbor_pos]
                neighbor_data = self.team_data[neighbor_team]
                total += calculate_penalty(
                    team_data,
                    neighbor_data,
                    self.columns,
                    self.ignore_number,
                    self.penalty_config,
                    neighbor_weight,
                )

        return total

    def swap_delta(self, team1: str, team2: str) -> float:
        """Изменение штрафа при обмене двух подвижных команд местами."""
        if team1 in self.fixed_teams or team2 in self.fixed_teams:
            return 0.0

        pos1 = self.placement[team1]
        pos2 = self.placement[team2]
        affected = self._affected_positions(pos1, pos2)
        old_penalty = self._penalty_touching_positions(affected)

        self.placement[team1], self.placement[team2] = pos2, pos1
        self.position_to_team[pos1] = team2
        self.position_to_team[pos2] = team1

        new_penalty = self._penalty_touching_positions(affected)
        delta = new_penalty - old_penalty

        if delta >= 0:
            self.placement[team1], self.placement[team2] = pos1, pos2
            self.position_to_team[pos1] = team1
            self.position_to_team[pos2] = team2

        return delta


def build_fixed_placement(
    fixed_by_position: Dict[Tuple[int, int], str],
) -> Dict[str, Tuple[int, int]]:
    """Преобразование фиксированных мест в словарь команда -> позиция."""
    fixed_placement: Dict[str, Tuple[int, int]] = {}
    for pos, team in fixed_by_position.items():
        if team in fixed_placement:
            raise ValueError(f"Команда '{team}' закреплена на нескольких местах")
        fixed_placement[team] = pos
    return fixed_placement


def validate_fixed_teams(
    teams_df: pd.DataFrame,
    fixed_placement: Dict[str, Tuple[int, int]],
) -> None:
    """Проверка, что все закреплённые команды есть во входном списке."""
    known_teams = set(teams_df["team"].astype(str))
    unknown = [team for team in fixed_placement if team not in known_teams]
    if unknown:
        raise ValueError(
            "На схеме указаны команды, которых нет во входном файле: "
            + ", ".join(unknown)
        )


def generate_random_placement(
    teams: List[str],
    positions: List[Tuple[int, int]],
) -> Dict[str, Tuple[int, int]]:
    """Случайное размещение команд на свободные места."""
    if len(teams) > len(positions):
        raise ValueError("Недостаточно мест для всех команд")

    shuffled_teams = teams[:]
    random.shuffle(shuffled_teams)
    return {team: positions[i] for i, team in enumerate(shuffled_teams)}


def generate_greedy_placement(
    teams: List[str],
    positions: List[Tuple[int, int]],
    evaluator: SeatingEvaluator,
) -> Dict[str, Tuple[int, int]]:
    """Жадное размещение: каждая команда выбирает место с минимальным штрафом."""
    if len(teams) > len(positions):
        raise ValueError("Недостаточно мест для всех команд")

    remaining_positions = positions[:]
    placement: Dict[str, Tuple[int, int]] = {}
    shuffled_teams = teams[:]
    random.shuffle(shuffled_teams)

    for team in shuffled_teams:
        best_pos = None
        best_penalty = float("inf")

        for pos in remaining_positions:
            placement[team] = pos
            evaluator.set_free_placement(placement)
            penalty = evaluator.total_penalty()
            if penalty < best_penalty:
                best_penalty = penalty
                best_pos = pos
            del placement[team]

        if best_pos is None:
            raise ValueError("Не удалось разместить команду жадным алгоритмом")

        placement[team] = best_pos
        remaining_positions.remove(best_pos)

    evaluator.set_free_placement(placement)
    return placement


def optimize_seating(
    teams_df: pd.DataFrame,
    seating_plan: np.ndarray,
    columns: List[str],
    ignore_number: bool,
    penalty_config: PenaltyConfig,
    optimization_config: OptimizationConfig,
    fixed_placement: Dict[str, Tuple[int, int]],
) -> List[Tuple[Dict[str, Tuple[int, int]], float]]:
    """Оптимизация рассадки команд."""
    logger.info("Начало оптимизации рассадки")

    fixed_positions = set(fixed_placement.values())
    all_teams = teams_df["team"].tolist()
    movable_teams = [team for team in all_teams if team not in fixed_placement]
    positions = get_available_positions(seating_plan, fixed_positions)

    if len(movable_teams) > len(positions):
        raise ValueError("Недостаточно свободных мест для всех команд")

    evaluator = SeatingEvaluator(
        teams_df,
        seating_plan,
        columns,
        ignore_number,
        penalty_config,
        fixed_placement,
    )

    placements: List[Tuple[Dict[str, Tuple[int, int]], float]] = []
    num_initial = max(1, optimization_config.num_initial_placements)

    for i in range(num_initial):
        logger.info(f"Генерация начального размещения {i + 1}/{num_initial}")
        if i == 0 and movable_teams:
            free_placement = generate_greedy_placement(movable_teams, positions, evaluator)
        else:
            free_placement = generate_random_placement(movable_teams, positions)

        evaluator.set_free_placement(free_placement)
        full_placement = evaluator.placement.copy()
        penalty = evaluator.total_penalty()
        placements.append((full_placement, penalty))

    placements.sort(key=lambda item: item[1])
    num_to_optimize = min(optimization_config.num_to_optimize, len(placements))
    best_placements = placements[:num_to_optimize]
    optimized_placements: List[Tuple[Dict[str, Tuple[int, int]], float]] = []

    for idx, (placement, penalty) in enumerate(best_placements):
        logger.info(
            f"Оптимизация размещения {idx + 1}/{num_to_optimize} с начальным штрафом {penalty:.2f}"
        )
        print(f"\nОптимизация варианта {idx + 1}/{num_to_optimize} (начальный штраф: {penalty:.2f})")

        evaluator.set_free_placement(
            {team: pos for team, pos in placement.items() if team not in fixed_placement}
        )
        penalty = evaluator.total_penalty()

        improved = True
        iterations = 0
        max_iterations = optimization_config.max_iterations
        total_swaps = len(movable_teams) * (len(movable_teams) - 1) // 2

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            swap_count = 0

            print(
                f"\rИтерация {iterations}/{max_iterations} | Текущий штраф: {penalty:.2f}",
                end="",
            )

            for i, team1 in enumerate(movable_teams):
                for j in range(i + 1, len(movable_teams)):
                    team2 = movable_teams[j]
                    swap_count += 1
                    if total_swaps and swap_count % 100 == 0:
                        progress = swap_count / total_swaps * 100
                        print(
                            f"\rИтерация {iterations}/{max_iterations} | "
                            f"Прогресс: {progress:.1f}% | Штраф: {penalty:.2f}",
                            end="",
                        )

                    delta = evaluator.swap_delta(team1, team2)
                    if delta < 0:
                        penalty += delta
                        improved = True
                        logger.debug(
                            f"Улучшение: поменяли местами {team1} и {team2}, новый штраф {penalty:.2f}"
                        )

            print(
                f"\rИтерация {iterations}/{max_iterations} завершена | Текущий штраф: {penalty:.2f}"
            )

        print(
            f"\nОптимизация варианта {idx + 1} завершена после {iterations} итераций. "
            f"Итоговый штраф: {penalty:.2f}"
        )
        logger.info(
            f"Оптимизация завершена после {iterations} итераций, итоговый штраф {penalty:.2f}"
        )
        optimized_placements.append((evaluator.placement.copy(), penalty))

    optimized_placements.sort(key=lambda item: item[1])
    return optimized_placements


def save_seating_to_excel(
    placement: Dict[str, Tuple[int, int]],
    seating_plan: np.ndarray,
    filename: str,
    penalty: Optional[float] = None,
) -> None:
    """Сохранение результатов рассадки в Excel файл."""
    logger.info(f"Сохранение рассадки в файл {filename}")

    result_plan = np.full(seating_plan.shape, "", dtype=object)

    for team, (row, col) in placement.items():
        result_plan[row, col] = team

    for r in range(seating_plan.shape[0]):
        for c in range(seating_plan.shape[1]):
            if seating_plan[r, c] == "O":
                result_plan[r, c] = "НЕДОСТУПНО"
            elif seating_plan[r, c] == "X" and result_plan[r, c] == "":
                result_plan[r, c] = "СВОБОДНО"

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        pd.DataFrame(result_plan).to_excel(writer, index=False, header=False, sheet_name="Рассадка")
        if penalty is not None:
            summary = pd.DataFrame({"Параметр": ["Штраф"], "Значение": [penalty]})
            summary.to_excel(writer, index=False, sheet_name="Информация")

    logger.info(f"Рассадка успешно сохранена в {filename}")


def get_absolute_path(relative_path: str) -> str:
    """Преобразует относительный путь в абсолютный относительно местоположения скрипта."""
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(SCRIPT_DIR, relative_path)


def build_output_paths(base_output: str, count: int) -> List[str]:
    """Пути для лучшего и альтернативных вариантов."""
    directory, filename = os.path.split(base_output)
    stem, ext = os.path.splitext(filename)
    if not ext:
        ext = ".xlsx"
    if not directory:
        directory = SCRIPT_DIR

    paths = [os.path.join(directory, f"{stem}{ext}")]
    for index in range(1, count):
        paths.append(os.path.join(directory, f"{stem}_{index}{ext}"))
    return paths


def main() -> None:
    """Основная функция программы."""
    setup_logging()

    try:
        args = parse_arguments()
        ignore_number = args.ignore_number or args.alphabetical_order
        if args.alphabetical_order and not args.ignore_number:
            logger.warning(
                "Флаг --alphabetical_order устарел, используйте --ignore-number"
            )

        teams_file = args.input or input("Введите путь к файлу со списком команд: ").strip()
        seating_file = args.seating or input("Введите путь к файлу со схемой рассадки: ").strip()

        teams_file = get_absolute_path(teams_file)
        seating_file = get_absolute_path(seating_file)

        logger.info(f"Абсолютный путь к файлу команд: {teams_file}")
        logger.info(f"Абсолютный путь к файлу схемы: {seating_file}")

        teams_df = load_teams_data(teams_file)
        seating_plan, fixed_by_position = load_seating_plan(seating_file)
        fixed_placement = build_fixed_placement(fixed_by_position)
        validate_fixed_teams(teams_df, fixed_placement)

        columns = get_column_order(teams_df, args.order)
        if args.no_confirm or args.order:
            if args.order:
                logger.info("Порядок столбцов задан через --order, подтверждение пропущено")
            else:
                logger.info("Подтверждение порядка столбцов пропущено (--no-confirm)")
        else:
            columns = confirm_column_order(columns)

        fixed_positions = set(fixed_placement.values())
        available_seats = len(get_available_positions(seating_plan, fixed_positions))
        movable_teams = len(teams_df) - len(fixed_placement)

        if movable_teams > available_seats:
            logger.error(
                f"Недостаточно свободных мест: {movable_teams} подвижных команд, "
                f"{available_seats} свободных мест"
            )
            print(
                f"Ошибка: недостаточно свободных мест "
                f"({movable_teams} подвижных команд, {available_seats} свободных мест)"
            )
            return

        penalty_config = PenaltyConfig(
            number_close_diff=args.number_close_diff,
            diagonal_neighbor_weight=args.diagonal_weight,
        )
        optimization_config = OptimizationConfig(
            num_initial_placements=args.initial_placements,
            num_to_optimize=args.optimize_count,
            max_iterations=args.max_iterations,
        )

        logger.info(
            f"Начало генерации рассадки для {len(teams_df)} команд "
            f"({len(fixed_placement)} закреплены) на {available_seats} свободных местах"
        )
        print(
            f"Генерация рассадки для {len(teams_df)} команд "
            f"({len(fixed_placement)} закреплены) на {available_seats} свободных мест..."
        )
        print("Этот процесс может занять некоторое время. Пожалуйста, подождите.")

        best_placements = optimize_seating(
            teams_df,
            seating_plan,
            columns,
            ignore_number,
            penalty_config,
            optimization_config,
            fixed_placement,
        )

        if not best_placements:
            logger.error("Не удалось сгенерировать рассадку")
            print("Ошибка: не удалось сгенерировать рассадку")
            return

        output_base = get_absolute_path(args.output) if args.output else os.path.join(SCRIPT_DIR, "seating.xlsx")
        output_paths = build_output_paths(output_base, min(len(best_placements), optimization_config.num_to_optimize))

        print("\nСохранение результатов... ", end="")
        for path, (placement, penalty) in zip(output_paths, best_placements):
            save_seating_to_excel(placement, seating_plan, path, penalty)
            print(".", end="", flush=True)
        print(" готово!")

        print("\nГотово! Рассадка сохранена в файл(ы):")
        for index, (path, (_, penalty)) in enumerate(zip(output_paths, best_placements)):
            label = "лучший вариант" if index == 0 else f"альтернативный вариант {index}"
            print(f"- {path} ({label}, штраф: {penalty:.2f})")

        logger.info("Генерация рассадки успешно завершена")

    except Exception as e:
        logger.error(f"Ошибка при выполнении программы: {e}")
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()
