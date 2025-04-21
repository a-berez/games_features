#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для анализа турниров.
Собирает статистику по командам и игрокам за определенный сезон.
"""

import argparse
import logging
import sys
import re
import os
import datetime
import requests
import pandas as pd
import concurrent.futures
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm

# Настройка логирования
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"games_counter_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Базовый URL API
BASE_URL = "https://api.rating.chgk.net"

# Кэши для хранения данных и уменьшения количества запросов к API
tournament_info_cache = {}  # Кэш информации о турнирах
player_tournaments_cache = {}  # Кэш турниров игроков
team_tournaments_cache = {}  # Кэш турниров команд
season_tournament_cache = {}  # Кэш принадлежности турниров к сезонам
seasons_cache = None  # Кэш списка сезонов

def parse_arguments() -> argparse.Namespace:
    """
    Парсинг аргументов командной строки.
    
    Returns:
        argparse.Namespace: Объект с аргументами командной строки
    """
    parser = argparse.ArgumentParser(description="Анализ турниров")
    parser.add_argument("-t", "--tournament", type=int, help="ID турнира для анализа")
    parser.add_argument("-s", "--season", type=str, help="ID сезона для анализа (таблицу можно посмотреть в README")
    
    return parser.parse_args()

def get_user_input() -> Tuple[int, int]:
    """
    Получение ID турнира и ID сезона от пользователя, если они не были указаны в аргументах.
    
    Returns:
        Tuple[int, int]: ID турнира и ID сезона
    """
    args = parse_arguments()
    
    # Получение ID турнира
    tournament_id = args.tournament
    if tournament_id is None:
        while True:
            try:
                tournament_id = int(input("Введите ID турнира: ").strip())
                break
            except ValueError:
                print("Пожалуйста, введите корректный ID турнира (целое число).")
    
    # Получение ID сезона
    season_id = args.season
    if season_id is None:
        while True:
            try:
                season_id = int(input("Введите ID сезона: ").strip())
                break
            except ValueError:
                print("Пожалуйста, введите корректный ID сезона (целое число).")
    
    return tournament_id, season_id

def get_seasons() -> List[Dict[str, Any]]:
    """
    Получение списка всех сезонов из API с использованием кэша.
    
    Returns:
        List[Dict[str, Any]]: Список сезонов
    """
    global seasons_cache
    
    if seasons_cache is not None:
        return seasons_cache
    
    url = f"{BASE_URL}/seasons"
    logger.info(f"Запрос списка сезонов: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        seasons = response.json()
        logger.info(f"Получено {len(seasons)} сезонов")
        seasons_cache = seasons
        return seasons
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при получении списка сезонов: {e}")
        raise


def get_tournament_results(tournament_id: int) -> List[Dict[str, Any]]:
    """
    Получение результатов турнира.
    
    Args:
        tournament_id (int): ID турнира
    
    Returns:
        List[Dict[str, Any]]: Результаты турнира
    """
    url = f"{BASE_URL}/tournaments/{tournament_id}/results?includeTeamMembers=1&includeMasksAndControversials=0&includeTeamFlags=0&includeRatingB=0"
    logger.info(f"Запрос результатов турнира {tournament_id}: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        results = response.json()
        logger.info(f"Получены результаты турнира {tournament_id}: {len(results)} команд")
        return results
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при получении результатов турнира {tournament_id}: {e}")
        raise

def get_tournament_info(tournament_id: int) -> Dict[str, Any]:
    """
    Получение информации о турнире с использованием кэша.
    
    Args:
        tournament_id (int): ID турнира
    
    Returns:
        Dict[str, Any]: Информация о турнире
    """
    if tournament_id in tournament_info_cache:
        return tournament_info_cache[tournament_id]
    
    url = f"{BASE_URL}/tournaments/{tournament_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        info = response.json()
        tournament_info_cache[tournament_id] = info
        return info
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при получении информации о турнире {tournament_id}: {e}")
        raise

def get_player_tournaments(player_id: int) -> List[Dict[str, Any]]:
    """
    Получение списка турниров игрока с использованием кэша.
    
    Args:
        player_id (int): ID игрока
    
    Returns:
        List[Dict[str, Any]]: Список турниров игрока
    """
    if player_id in player_tournaments_cache:
        return player_tournaments_cache[player_id]
    
    url = f"{BASE_URL}/players/{player_id}/tournaments"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        tournaments = response.json()
        player_tournaments_cache[player_id] = tournaments
        return tournaments
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при получении турниров игрока {player_id}: {e}")
        return []  # Возвращаем пустой список, чтобы продолжить работу с другими игроками

def get_team_tournaments(team_id: int) -> List[Dict[str, Any]]:
    """
    Получение списка турниров команды с использованием кэша.
    
    Args:
        team_id (int): ID команды
    
    Returns:
        List[Dict[str, Any]]: Список турниров команды
    """
    if team_id in team_tournaments_cache:
        return team_tournaments_cache[team_id]
    
    url = f"{BASE_URL}/teams/{team_id}/tournaments"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        tournaments = response.json()
        team_tournaments_cache[team_id] = tournaments
        return tournaments
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при получении турниров команды {team_id}: {e}")
        return []  # Возвращаем пустой список, чтобы продолжить работу с другими командами

def is_tournament_in_season(tournament_id: int, season_id: int, main_tournament_end_date: datetime.datetime) -> bool:
    """
    Проверка, относится ли турнир к указанному сезону и завершился ли до указанной даты.
    Использует кэш для уменьшения количества запросов.
    
    Args:
        tournament_id (int): ID турнира
        season_id (int): ID сезона
        main_tournament_end_date (datetime.datetime): Дата завершения основного турнира
    
    Returns:
        bool: True, если турнир относится к указанному сезону и завершился до указанной даты
    """
    # Создаем ключ для кэша
    cache_key = (tournament_id, season_id, main_tournament_end_date.isoformat())
    
    if cache_key in season_tournament_cache:
        return season_tournament_cache[cache_key]
    
    try:
        tournament_info = get_tournament_info(tournament_id)
        
        # Проверка сезона
        if "idseason" in tournament_info and tournament_info["idseason"] == season_id:
            # Проверка даты завершения
            if "dateEnd" in tournament_info:
                tournament_end_date = datetime.datetime.fromisoformat(tournament_info["dateEnd"].replace("Z", "+00:00"))
                result = tournament_end_date <= main_tournament_end_date
                season_tournament_cache[cache_key] = result
                return result
    except Exception:
        pass
    
    season_tournament_cache[cache_key] = False
    return False

def analyze_player_tournaments_batch(player_data: List[Tuple[int, int]], season_id: int, main_tournament_end_date: datetime.datetime) -> Dict[int, Tuple[int, int]]:
    """
    Анализ турниров группы игроков за указанный сезон (пакетная обработка).
    
    Args:
        player_data (List[Tuple[int, int]]): Список кортежей (ID игрока, ID команды)
        season_id (int): ID сезона
        main_tournament_end_date (datetime.datetime): Дата завершения основного турнира
    
    Returns:
        Dict[int, Tuple[int, int]]: Словарь {ID игрока: (base_games, other_games)}
    """
    results = {}
    
    for player_id, team_id in player_data:
        player_tournaments = get_player_tournaments(player_id)
        
        base_games = 0  # Турниры в составе указанной команды
        other_games = 0  # Турниры в составе других команд
        
        for tournament in player_tournaments:
            tournament_id = tournament["idtournament"]
            current_team_id = tournament["idteam"]
            
            if is_tournament_in_season(tournament_id, season_id, main_tournament_end_date):
                if current_team_id == team_id:
                    base_games += 1
                else:
                    other_games += 1
        
        results[player_id] = (base_games, other_games)
    
    return results

def analyze_team_tournaments_batch(team_ids: List[int], season_id: int, main_tournament_end_date: datetime.datetime) -> Dict[int, int]:
    """
    Анализ турниров группы команд за указанный сезон (пакетная обработка).
    
    Args:
        team_ids (List[int]): Список ID команд
        season_id (int): ID сезона
        main_tournament_end_date (datetime.datetime): Дата завершения основного турнира
    
    Returns:
        Dict[int, int]: Словарь {ID команды: количество турниров}
    """
    results = {}
    
    for team_id in team_ids:
        team_tournaments = get_team_tournaments(team_id)
        
        games_count = 0
        
        for tournament in team_tournaments:
            tournament_id = tournament["idtournament"]
            
            if is_tournament_in_season(tournament_id, season_id, main_tournament_end_date):
                games_count += 1
        
        results[team_id] = games_count
    
    return results

def process_data_in_parallel(tournament_results: List[Dict[str, Any]], season_id: int, main_tournament_end_date: datetime.datetime) -> Tuple[Dict[int, int], Dict[int, Tuple[int, int]]]:
    """
    Параллельная обработка данных о турнирах команд и игроков.
    
    Args:
        tournament_results (List[Dict[str, Any]]): Результаты турнира
        season_id (int): ID сезона
        main_tournament_end_date (datetime.datetime): Дата завершения основного турнира
    
    Returns:
        Tuple[Dict[int, int], Dict[int, Tuple[int, int]]]: Словари с результатами для команд и игроков
    """
    # Собираем ID команд и данные игроков
    team_ids = []
    player_data = []
    
    for result in tournament_results:
        team_id = result["team"]["id"]
        team_ids.append(team_id)
        
        for member in result.get("teamMembers", []):
            player_id = member["player"]["id"]
            player_data.append((player_id, team_id))
    
    print(f"Всего найдено: {len(team_ids)} команд и {len(player_data)} игроков")
    
    # Распределяем работу между потоками
    team_results = {}
    player_results = {}
    
    # Разбиваем данные на пакеты для более эффективной обработки
    team_chunks = [team_ids[i:i + 10] for i in range(0, len(team_ids), 10)]
    player_chunks = [player_data[i:i + 10] for i in range(0, len(player_data), 10)]
    
    # Обработка команд
    print(f"\nОбработка данных для {len(team_ids)} команд:")
    team_counter = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_chunk = {
            executor.submit(analyze_team_tournaments_batch, chunk, season_id, main_tournament_end_date): chunk
            for chunk in team_chunks
        }
        
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_results = future.result()
            team_results.update(chunk_results)
            team_counter += len(chunk_results)
            # Обновляем прогресс в одной строке
            print(f"\rОбработано {team_counter}/{len(team_ids)} команд ({team_counter/len(team_ids)*100:.1f}%)", end="")
    
    print()  # Переход на новую строку после завершения обработки команд
    
    # Обработка игроков
    print(f"\nОбработка данных для {len(player_data)} игроков:")
    player_counter = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_chunk = {
            executor.submit(analyze_player_tournaments_batch, chunk, season_id, main_tournament_end_date): chunk
            for chunk in player_chunks
        }
        
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_results = future.result()
            player_results.update(chunk_results)
            player_counter += len(chunk_results)
            # Обновляем прогресс в одной строке
            print(f"\rОбработано {player_counter}/{len(player_data)} игроков ({player_counter/len(player_data)*100:.1f}%)", end="")
    
    print()  # Переход на новую строку после завершения обработки игроков
    
    return team_results, player_results


def create_teams_table(tournament_results: List[Dict[str, Any]], team_results: Dict[int, int], player_results: Dict[int, Tuple[int, int]]) -> pd.DataFrame:
    """
    Создание таблицы команд с использованием предварительно обработанных данных.
    
    Args:
        tournament_results (List[Dict[str, Any]]): Результаты турнира
        team_results (Dict[int, int]): Данные о турнирах команд
        player_results (Dict[int, Tuple[int, int]]): Данные о турнирах игроков
    
    Returns:
        pd.DataFrame: Таблица команд
    """
    teams_data = []
    
    for result in tournament_results:
        team_id = result["team"]["id"]
        team_name = result["team"]["name"]
        team_town = result["team"]["town"]["name"] if "town" in result["team"] else ""
        position = result["position"]
        taken = result.get("questionsTotal", 0)
        
        # Получаем количество турниров команды
        team_games = team_results.get(team_id, 0)
        
        # Собираем данные об играх игроков команды
        player_games = []
        for member in result.get("teamMembers", []):
            player_id = member["player"]["id"]
            if player_id in player_results:
                base_games, other_games = player_results[player_id]
                player_games.append(base_games + other_games)
        
        # Расчет статистики по играм игроков
        avg_player_games = sum(player_games) / len(player_games) if player_games else 0
        min_player_games = min(player_games) if player_games else 0
        max_player_games = max(player_games) if player_games else 0
        
        teams_data.append({
            "place": position,
            "teamID": team_id,
            "teamName": team_name,
            "teamTown": team_town,
            "taken": taken,
            "teamGames": team_games,
            "avgPlayerGames": round(avg_player_games, 2),
            "minPlayerGames": min_player_games,
            "maxPlayerGames": max_player_games
        })
    
    return pd.DataFrame(teams_data)

def create_players_table(tournament_results: List[Dict[str, Any]], player_results: Dict[int, Tuple[int, int]]) -> pd.DataFrame:
    """
    Создание таблицы игроков с использованием предварительно обработанных данных.
    
    Args:
        tournament_results (List[Dict[str, Any]]): Результаты турнира
        player_results (Dict[int, Tuple[int, int]]): Данные о турнирах игроков
    
    Returns:
        pd.DataFrame: Таблица игроков
    """
    players_data = []
    
    for result in tournament_results:
        team_id = result["team"]["id"]
        team_name = result["team"]["name"]
        team_town = result["team"]["town"]["name"] if "town" in result["team"] else ""
        position = result["position"]
        taken = result.get("questionsTotal", 0)
        
        for member in result.get("teamMembers", []):
            player_id = member["player"]["id"]
            player_name = member["player"]["name"]
            player_surname = member["player"]["surname"]
            player_patronymic = member["player"].get("patronymic", "")
            
            # Получаем данные о турнирах игрока
            base_games, other_games = player_results.get(player_id, (0, 0))
            # Добавляем сумму игр
            total_games = base_games + other_games
            
            players_data.append({
                "place": position,
                "teamID": team_id,
                "teamName": team_name,
                "teamTown": team_town,
                "taken": taken,
                "playerID": player_id,
                "playerSurname": player_surname,
                "playerName": player_name,
                "playerPatronim": player_patronymic,
                "baseGames": base_games,
                "otherGames": other_games,
                "totalGames": total_games
            })
    
    return pd.DataFrame(players_data)

def save_to_excel(df: pd.DataFrame, filename: str):
    """
    Сохранение DataFrame в Excel-файл с базовой настройкой.
    Использует только стандартные библиотеки из изначального кода.
    
    Args:
        df (pd.DataFrame): DataFrame для сохранения
        filename (str): Имя файла
    """
    try:
        # Создаем безопасное имя файла
        safe_filename = "".join([c if c.isalnum() or c in "._- " else "_" for c in filename])
        if len(safe_filename) > 200:  # Ограничиваем длину имени файла
            safe_filename = safe_filename[:197] + "..."
        
        # Определяем словарь с шириной столбцов
        column_widths = {
            'place': 5,
            'teamID': 8,
            'teamName': 35,
            'teamTown': 25,
            'taken': 8,
            'playerID': 10,
            'playerSurname': 20,
            'playerName': 18,
            'playerPatronim': 17,
            'baseGames': 10,
            'otherGames': 11,
            'totalGames': 11,
            'teamGames': 10,
            'avgPlayerGames': 15,
            'minPlayerGames': 15,
            'maxPlayerGames': 15
        }
        
        # Используем контекстный менеджер для автоматического закрытия и сохранения файла
        with pd.ExcelWriter(safe_filename, engine='xlsxwriter') as writer:
            # Записываем данные
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            
            # Пытаемся настроить ширину столбцов, если доступен xlsxwriter
            try:
                worksheet = writer.sheets['Sheet1']
                for i, column in enumerate(df.columns):
                    if hasattr(worksheet, 'set_column'):  # Проверяем, доступен ли метод set_column
                        width = column_widths.get(column, 15)  # Используем значение по умолчанию 15, если столбец не найден
                        worksheet.set_column(i, i, width)
            except (AttributeError, KeyError):
                # Если не удалось настроить ширину столбцов, продолжаем без настройки
                logger.warning("Не удалось настроить ширину столбцов. Используется стандартный формат.")
        
        # При выходе из блока with файл автоматически сохраняется и закрывается
        logger.info(f"Данные сохранены в файл: {safe_filename}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении данных в файл {safe_filename}: {e}")
        raise



def main():
    """
    Основная функция скрипта.
    """
    try:
        logger.info("Запуск скрипта анализа турниров")
        
        # Получение ID турнира и сезона от пользователя
        tournament_id, season_id = get_user_input()
        
        logger.info(f"Анализ турнира {tournament_id} для сезона с ID {season_id}")
        
        # Получение информации о турнире
        tournament_info = get_tournament_info(tournament_id)
        tournament_name = tournament_info.get("name", "Unnamed_Tournament")
        
        # Получение даты завершения турнира
        if "dateEnd" not in tournament_info:
            logger.error(f"Турнир {tournament_id} не содержит даты завершения")
            raise ValueError(f"Турнир {tournament_id} не содержит даты завершения")
        
        main_tournament_end_date = datetime.datetime.fromisoformat(tournament_info["dateEnd"].replace("Z", "+00:00"))
        logger.info(f"Дата завершения турнира: {main_tournament_end_date}")
        
        # Получение результатов турнира
        tournament_results = get_tournament_results(tournament_id)
        
        if not tournament_results:
            logger.error(f"Не удалось получить результаты турнира {tournament_id}")
            raise ValueError(f"Не удалось получить результаты турнира {tournament_id}")
        
        logger.info(f"Начало анализа данных для {len(tournament_results)} команд")
        
        # Параллельная обработка данных
        team_results, player_results = process_data_in_parallel(
            tournament_results, season_id, main_tournament_end_date
        )
        
        # Создание таблиц
        logger.info("Создание таблицы команд")
        teams_df = create_teams_table(tournament_results, team_results, player_results)
        
        logger.info("Создание таблицы игроков")
        players_df = create_players_table(tournament_results, player_results)
        
        # Сохранение результатов в Excel-файлы
        teams_filename = f"{tournament_id}_teams.xlsx"
        players_filename = f"{tournament_id}_players.xlsx"
        
        save_to_excel(teams_df, teams_filename)
        save_to_excel(players_df, players_filename)
        
        logger.info("Анализ завершен успешно")
        print(f"\nРезультаты сохранены в файлы:\n- {teams_filename}\n- {players_filename}")
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении скрипта: {e}", exc_info=True)
        print(f"Произошла ошибка: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        # Установка таймаута для HTTP-запросов
        requests.adapters.DEFAULT_RETRIES = 3
        session = requests.Session()
        session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
        
        # Установка уровня логирования для библиотек
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        
        print("Скрипт анализа турниров")
        print("===========================")
        print("Скрипт собирает статистику по командам и игрокам за определенный сезон.")
        print("Результаты будут сохранены в Excel-файлы.")
        print()
        
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
        logger.info("Скрипт прерван пользователем")
        sys.exit(1)
