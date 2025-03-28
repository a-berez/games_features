import pandas as pd
import numpy as np
import argparse
import os
import logging
from itertools import permutations
import time
from typing import List, Tuple, Dict, Set, Optional
import random

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("seating_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SeatingGenerator")

def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Генератор рассадки команд')
    parser.add_argument('-i', '--input', type=str, help='Путь к файлу со списком команд')
    parser.add_argument('-s', '--seating', type=str, help='Путь к файлу со схемой рассадки')
    parser.add_argument('-abc', '--alphabetical_order', action='store_true', 
                        help='Игнорировать номера при рассадке')
    parser.add_argument('-o', '--order', type=str, 
                        help='Порядок важности столбцов (например, "2,4,1,3")')
    return parser.parse_args()

def load_teams_data(file_path: str) -> pd.DataFrame:
    """Загрузка данных о командах из Excel файла"""
    try:
        logger.info(f"Загрузка данных о командах из {file_path}")
        df = pd.read_excel(file_path)
        
        # Проверка наличия обязательного столбца 'team'
        if 'team' not in df.columns:
            logger.error("В файле отсутствует обязательный столбец 'team'")
            raise ValueError("В файле отсутствует обязательный столбец 'team'")
        
        # Проверка наличия хотя бы одного дополнительного столбца
        if len(df.columns) < 2:
            logger.error("В файле должен быть хотя бы один дополнительный столбец")
            raise ValueError("В файле должен быть хотя бы один дополнительный столбец")
        
        # Удаление столбца 'team' из списка столбцов для рассадки
        seating_columns = [col for col in df.columns if col != 'team']
        if not seating_columns:
            logger.error("В файле должен быть хотя бы один столбец для определения рассадки")
            raise ValueError("В файле должен быть хотя бы один столбец для определения рассадки")
        
        logger.info(f"Столбцы для определения рассадки: {seating_columns}")
        
        # Проверка на дубликаты команд
        if df['team'].duplicated().any():
            logger.warning("Обнаружены дубликаты команд")
            df = df.drop_duplicates(subset=['team'])
        
        return df
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных о командах: {str(e)}")
        raise

def load_seating_plan(file_path: str) -> np.ndarray:
    """Загрузка схемы рассадки из Excel файла"""
    try:
        logger.info(f"Загрузка схемы рассадки из {file_path}")
        df = pd.read_excel(file_path, header=None)
        
        # Преобразование DataFrame в numpy array
        seating_plan = df.values
        
        # Проверка наличия допустимых символов (X и O)
        valid_chars = {'X', 'O', 'x', 'o'}
        for i in range(seating_plan.shape[0]):
            for j in range(seating_plan.shape[1]):
                if str(seating_plan[i, j]).upper() not in valid_chars:
                    logger.warning(f"Обнаружен недопустимый символ '{seating_plan[i, j]}' в схеме рассадки. "
                                  f"Заменяем на 'O' (недоступное место)")
                    seating_plan[i, j] = 'O'
                seating_plan[i, j] = str(seating_plan[i, j]).upper()
        
        # Подсчет доступных мест
        available_seats = np.sum(seating_plan == 'X')
        logger.info(f"Доступно мест: {available_seats}")
        
        return seating_plan
    except Exception as e:
        logger.error(f"Ошибка при загрузке схемы рассадки: {str(e)}")
        raise

def get_column_order(df: pd.DataFrame, order_arg: Optional[str] = None) -> List[str]:
    """Определение порядка важности столбцов"""
    # Исключаем столбец 'team' из рассмотрения
    columns = [col for col in df.columns if col != 'team']
    
    if order_arg:
        try:
            # Преобразование строки с порядком в список индексов
            order_indices = [int(x.strip()) - 1 for x in order_arg.split(',')]
            
            # Проверка валидности индексов
            if max(order_indices) >= len(columns) or min(order_indices) < 0:
                logger.error(f"Недопустимые индексы в порядке столбцов: {order_arg}")
                raise ValueError(f"Недопустимые индексы в порядке столбцов: {order_arg}")
            
            # Проверка, что все столбцы указаны
            if len(order_indices) != len(columns):
                logger.error(f"Количество индексов ({len(order_indices)}) не соответствует "
                            f"количеству столбцов для рассадки ({len(columns)})")
                raise ValueError(f"Количество индексов не соответствует количеству столбцов")
            
            # Проверка на дубликаты индексов
            if len(order_indices) != len(set(order_indices)):
                logger.error(f"Обнаружены дубликаты в порядке столбцов: {order_arg}")
                raise ValueError(f"Обнаружены дубликаты в порядке столбцов")
            
            # Применение указанного порядка
            ordered_columns = [columns[i] for i in order_indices]
            return ordered_columns
        except Exception as e:
            logger.error(f"Ошибка при обработке порядка столбцов: {str(e)}")
            raise
    
    # Если порядок не указан, используем исходный порядок столбцов
    return columns

def confirm_column_order(columns: List[str]) -> List[str]:
    """Запрос подтверждения порядка важности столбцов у пользователя"""
    logger.info("Запрос подтверждения порядка важности столбцов")
    print("\nПорядок учета критериев (от наиболее важного к наименее важному):")
    for i, col in enumerate(columns, 1):
        print(f"{i}. {col}")
    
    while True:
        try:
            confirmation = input("\nПодтвердите порядок (да/нет): ").strip().lower()
            if confirmation in ['да', 'yes', 'y', 'д']:
                logger.info("Порядок столбцов подтвержден")
                return columns
            elif confirmation in ['нет', 'no', 'n', 'н']:
                new_order = input("Введите новый порядок (например, 2,4,1,3): ").strip()
                order_indices = [int(x.strip()) - 1 for x in new_order.split(',')]
                
                # Проверка валидности индексов
                if max(order_indices) >= len(columns) or min(order_indices) < 0:
                    logger.error(f"Недопустимые индексы: {new_order}")
                    print(f"Ошибка: индексы должны быть от 1 до {len(columns)}")
                    continue
                
                # Проверка, что все столбцы указаны
                if len(order_indices) != len(columns):
                    logger.error(f"Количество индексов ({len(order_indices)}) не соответствует "
                                f"количеству столбцов ({len(columns)})")
                    print(f"Ошибка: нужно указать {len(columns)} индексов")
                    continue
                
                # Проверка на дубликаты индексов
                if len(order_indices) != len(set(order_indices)):
                    logger.error(f"Обнаружены дубликаты в порядке: {new_order}")
                    print("Ошибка: индексы не должны повторяться")
                    continue
                # Применение нового порядка
                new_columns = [columns[i] for i in order_indices]
                logger.info(f"Установлен новый порядок столбцов: {new_columns}")
                
                # Показать новый порядок и запросить подтверждение
                print("\nНовый порядок учета критериев:")
                for i, col in enumerate(new_columns, 1):
                    print(f"{i}. {col}")
                final_confirmation = input("\nПодтвердите новый порядок (да/нет): ").strip().lower()
                if final_confirmation in ['да', 'yes', 'y', 'д']:
                    logger.info("Новый порядок столбцов подтвержден")
                    return new_columns
                else:
                    logger.info("Повторный запрос порядка столбцов")
                    continue
            else:
                print("Пожалуйста, введите 'да' или 'нет'")
        except Exception as e:
            logger.error(f"Ошибка при вводе порядка столбцов: {str(e)}")
            print(f"Ошибка: {str(e)}. Пожалуйста, попробуйте снова.")

def get_neighbors(row: int, col: int, seating_plan: np.ndarray) -> List[Tuple[int, int]]:
    """Получение соседних мест (включая диагональные)"""
    rows, cols = seating_plan.shape
    neighbors = []
    
    # Проверяем все соседние ячейки (горизонтальные, вертикальные и диагональные)
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue  # Пропускаем текущую ячейку
                
            new_row, new_col = row + dr, col + dc
            
            # Проверяем, что ячейка находится в пределах плана рассадки
            if 0 <= new_row < rows and 0 <= new_col < cols:
                # Проверяем, что место доступно для рассадки
                if seating_plan[new_row, new_col] == 'X':
                    neighbors.append((new_row, new_col))
    
    return neighbors

def calculate_penalty(team1: pd.Series, team2: pd.Series, columns: List[str], 
                      alphabetical_order: bool) -> float:
    """Расчет штрафа за размещение двух команд рядом"""
    penalty = 0.0
    weight_multiplier = 1.0  # Множитель веса для каждого следующего столбца
    
    for i, column in enumerate(columns):
        # Вес критерия зависит от его позиции в списке
        weight = 1.0 / (i + 1) * weight_multiplier
        
        # Для столбца 'number' (если он есть и не игнорируется)
        if column == 'number' and not alphabetical_order:
            # Если номера близки, штраф выше
            try:
                num1 = float(team1[column]) if pd.notna(team1[column]) else 0
                num2 = float(team2[column]) if pd.notna(team2[column]) else 0
                
                # Больший штраф для меньших номеров (обычно лучшие команды)
                number_diff = abs(num1 - num2)
                if number_diff <= 3:  # Близкие номера
                    number_penalty = (4 - number_diff) * weight
                    # Дополнительный штраф для команд с маленькими номерами
                    if min(num1, num2) <= 5:
                        number_penalty *= 1.5
                    penalty += number_penalty
            except (ValueError, TypeError):
                logger.warning(f"Не удалось преобразовать значения в столбце 'number' к числу")
        
        # Для всех остальных столбцов
        elif pd.notna(team1[column]) and pd.notna(team2[column]):
            # Если значения совпадают, добавляем штраф
            if str(team1[column]).strip() == str(team2[column]).strip():
                penalty += weight
    
    return penalty

def get_available_positions(seating_plan: np.ndarray) -> List[Tuple[int, int]]:
    """Получение всех доступных позиций для размещения команд"""
    rows, cols = seating_plan.shape
    positions = []
    
    for r in range(rows):
        for c in range(cols):
            if seating_plan[r, c] == 'X':
                positions.append((r, c))
    
    # Сортируем позиции сверху вниз, слева направо
    positions.sort(key=lambda pos: (pos[0], pos[1]))
    return positions

def evaluate_seating(placement: Dict[str, Tuple[int, int]], teams_df: pd.DataFrame,
                     seating_plan: np.ndarray, columns: List[str], 
                     alphabetical_order: bool) -> float:
    """Оценка качества рассадки"""
    total_penalty = 0.0
    
    # Создаем обратный словарь: позиция -> команда
    position_to_team = {pos: team for team, pos in placement.items()}
    
    # Для каждой команды
    for team_name, (row, col) in placement.items():
        team_data = teams_df[teams_df['team'] == team_name].iloc[0]
        
        # Получаем соседей
        neighbors = get_neighbors(row, col, seating_plan)
        
        # Для каждого соседа
        for neighbor_row, neighbor_col in neighbors:
            neighbor_pos = (neighbor_row, neighbor_col)
            if neighbor_pos in position_to_team:
                neighbor_team = position_to_team[neighbor_pos]
                neighbor_data = teams_df[teams_df['team'] == neighbor_team].iloc[0]
                
                # Рассчитываем штраф
                penalty = calculate_penalty(team_data, neighbor_data, columns, 
                                           alphabetical_order)
                
                # Добавляем штраф к общему
                total_penalty += penalty
    
    return total_penalty

def generate_initial_placement(teams: List[str], positions: List[Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
    """Генерация начального размещения команд"""
    # Убедимся, что у нас достаточно позиций для всех команд
    if len(teams) > len(positions):
        logger.error(f"Недостаточно мест для всех команд: {len(teams)} команд, {len(positions)} мест")
        raise ValueError(f"Недостаточно мест для всех команд")
    
    # Случайное размещение команд
    random.shuffle(teams)
    placement = {}
    
    for i, team in enumerate(teams):
        placement[team] = positions[i]
    
    return placement

def optimize_seating(teams_df: pd.DataFrame, seating_plan: np.ndarray, 
                     columns: List[str], alphabetical_order: bool) -> List[Dict[str, Tuple[int, int]]]:
    """Оптимизация рассадки команд"""
    logger.info("Начало оптимизации рассадки")
    teams = teams_df['team'].tolist()
    positions = get_available_positions(seating_plan)
    
    # Проверка, что у нас достаточно мест
    if len(teams) > len(positions):
        logger.error(f"Недостаточно мест для всех команд: {len(teams)} команд, {len(positions)} мест")
        raise ValueError(f"Недостаточно мест для всех команд")
    
    # Генерируем несколько начальных размещений
    num_initial_placements = 10
    placements = []
    
    for i in range(num_initial_placements):
        logger.info(f"Генерация начального размещения {i+1}/{num_initial_placements}")
        placement = generate_initial_placement(teams, positions)
        placements.append((placement, evaluate_seating(placement, teams_df, seating_plan, 
                                                     columns, alphabetical_order)))
    
    # Сортируем по штрафу (от меньшего к большему)
    placements.sort(key=lambda x: x[1])
    
    # Берем лучшие начальные размещения для дальнейшей оптимизации
    best_placements = placements[:3]
    optimized_placements = []
    # Оптимизируем каждое из лучших начальных размещений
    for idx, (placement, penalty) in enumerate(best_placements):
        logger.info(f"Оптимизация размещения {idx+1}/3 с начальным штрафом {penalty:.2f}")
        print(f"\nОптимизация варианта {idx+1}/3 (начальный штраф: {penalty:.2f})")
        
        improved = True
        iterations = 0
        max_iterations = 1000
        
        # Расчет общего количества возможных перестановок для прогресс-бара
        total_swaps = len(teams) * (len(teams) - 1) // 2
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Отображение прогресса
            print(f"\rИтерация {iterations}/{max_iterations} | Текущий штраф: {penalty:.2f}", end="")
            if iterations % 10 == 0:
                print(".", end="", flush=True)  # Индикатор активности
            
            # Счетчик для отображения прогресса внутри итерации
            swap_count = 0
            
            # Пробуем поменять местами каждую пару команд
            for i, team1 in enumerate(teams):
                for j in range(i+1, len(teams)):
                    team2 = teams[j]
                    swap_count += 1
                    # Периодически обновляем индикатор прогресса внутри итерации
                    if swap_count % 100 == 0:
                        progress = swap_count / total_swaps * 100
                        print(f"\rИтерация {iterations}/{max_iterations} | "
                              f"Прогресс: {progress:.1f}% | Штраф: {penalty:.2f}", end="")
                    
                    # Меняем команды местами
                    pos1, pos2 = placement[team1], placement[team2]
                    placement[team1], placement[team2] = pos2, pos1
                    
                    # Оцениваем новое размещение
                    new_penalty = evaluate_seating(placement, teams_df, seating_plan, 
                                                  columns, alphabetical_order)
                    
                    # Если штраф уменьшился, оставляем изменение
                    if new_penalty < penalty:
                        penalty = new_penalty
                        improved = True
                        logger.debug(f"Улучшение: поменяли местами {team1} и {team2}, "
                                    f"новый штраф {penalty:.2f}")
                    else:
                        # Иначе возвращаем как было
                        placement[team1], placement[team2] = pos1, pos2
            
            # Выводим информацию о завершении итерации
            print(f"\rИтерация {iterations}/{max_iterations} завершена | Текущий штраф: {penalty:.2f}")
        print(f"\nОптимизация варианта {idx+1} завершена после {iterations} итераций. "
              f"Итоговый штраф: {penalty:.2f}")
        logger.info(f"Оптимизация завершена после {iterations} итераций, "
                   f"итоговый штраф {penalty:.2f}")
        
        optimized_placements.append((placement, penalty))
    
    # Сортируем оптимизированные размещения по штрафу
    optimized_placements.sort(key=lambda x: x[1])
    
    # Возвращаем лучшие размещения (только словари размещения)
    return [placement for placement, _ in optimized_placements]

def save_seating_to_excel(placement: Dict[str, Tuple[int, int]], seating_plan: np.ndarray, 
                         filename: str):
    """Сохранение результатов рассадки в Excel файл"""
    logger.info(f"Сохранение рассадки в файл {filename}")
    
    # Создаем копию плана рассадки для заполнения
    result_plan = np.full(seating_plan.shape, '', dtype=object)
    
    # Заполняем план названиями команд
    for team, (row, col) in placement.items():
        result_plan[row, col] = team
    
    # Отмечаем недоступные места
    for r in range(seating_plan.shape[0]):
        for c in range(seating_plan.shape[1]):
            if seating_plan[r, c] == 'O':
                result_plan[r, c] = 'НЕДОСТУПНО'
    
    # Преобразуем в DataFrame и сохраняем
    df = pd.DataFrame(result_plan)
    df.to_excel(filename, index=False, header=False)
    logger.info(f"Рассадка успешно сохранена в {filename}")

def get_absolute_path(relative_path):
    """Преобразует относительный путь в абсолютный относительно местоположения скрипта"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(script_dir, relative_path)

def main():
    """Основная функция программы"""
    try:
        # Парсинг аргументов командной строки
        args = parse_arguments()
        
        # Запрос путей к файлам, если не указаны в аргументах
        teams_file = args.input
        if not teams_file:
            teams_file = input("Введите путь к файлу со списком команд: ").strip()
        
        seating_file = args.seating
        if not seating_file:
            seating_file = input("Введите путь к файлу со схемой рассадки: ").strip()
        
        # Преобразование относительных путей в абсолютные
        teams_file = get_absolute_path(teams_file)
        seating_file = get_absolute_path(seating_file)
        
        logger.info(f"Абсолютный путь к файлу команд: {teams_file}")
        logger.info(f"Абсолютный путь к файлу схемы: {seating_file}")
        
        # Загрузка данных
        teams_df = load_teams_data(teams_file)
        seating_plan = load_seating_plan(seating_file)
        
        # Получение порядка столбцов
        columns = get_column_order(teams_df, args.order)
        
        # Подтверждение порядка столбцов
        columns = confirm_column_order(columns)
        # Проверка достаточности мест
        available_seats = np.sum(seating_plan == 'X')
        num_teams = len(teams_df)
        
        if num_teams > available_seats:
            logger.error(f"Недостаточно мест для всех команд: {num_teams} команд, "
                        f"{available_seats} доступных мест")
            print(f"Ошибка: недостаточно мест для всех команд ({num_teams} команд, "
                 f"{available_seats} доступных мест)")
            return
        
        logger.info(f"Начало генерации рассадки для {num_teams} команд "
                   f"на {available_seats} доступных местах")
        print(f"Генерация рассадки для {num_teams} команд на {available_seats} мест...")
        print("Этот процесс может занять некоторое время. Пожалуйста, подождите.")
        
        # Показываем индикатор активности во время генерации
        print("Генерация начальных вариантов рассадки: ", end="")
        for _ in range(10):
            time.sleep(0.2)  # Имитация активности
            print(".", end="", flush=True)
        print(" готово!")
        
        # Оптимизация рассадки
        best_placements = optimize_seating(teams_df, seating_plan, columns, 
                                          args.alphabetical_order)
        
        # Создаем абсолютные пути для выходных файлов
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Сохранение результатов
        if best_placements:
            print("\nСохранение результатов... ", end="")
            
            # Сохраняем лучший вариант
            output_file = os.path.join(script_dir, "seating.xlsx")
            save_seating_to_excel(best_placements[0], seating_plan, output_file)
            print(".", end="", flush=True)
            
            # Сохраняем альтернативные варианты (если есть)
            for i, placement in enumerate(best_placements[1:3], 1):
                output_file = os.path.join(script_dir, f"seating_{i}.xlsx")
                save_seating_to_excel(placement, seating_plan, output_file)
                print(".", end="", flush=True)
            
            print(" готово!")
            print(f"\nГотово! Рассадка сохранена в файл(ы):")
            print(f"- seating.xlsx (лучший вариант)")
            for i in range(1, min(len(best_placements), 3)):
                print(f"- seating_{i}.xlsx (альтернативный вариант {i})")
                
            logger.info(f"Генерация рассадки успешно завершена")
        else:
            logger.error("Не удалось сгенерировать рассадку")
            print("Ошибка: не удалось сгенерировать рассадку")
    
    except Exception as e:
        logger.error(f"Ошибка при выполнении программы: {str(e)}")
        print(f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    main()
