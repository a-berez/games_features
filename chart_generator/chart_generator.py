#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Генератор графиков для КВРМ

Скрипт генерирует HTML-файлы с интерактивными графиками взятий вопросов
и дополнительной статистикой на основе данных из API или табличных файлов.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots

# Настройка логирования
logger = logging.getLogger(__name__)

def setup_logging(dev_mode: bool = False) -> None:
    """
    Настройка системы логирования.
    
    Args:
        dev_mode: Режим разработчика с подробным логированием
    """
    level = logging.DEBUG if dev_mode else logging.INFO
    
    # Настройка форматирования логов
    log_format = '%(asctime)s - %(levelname)s - %(message)s' if dev_mode else '%(message)s'
    date_format = '%Y-%m-%d %H:%M:%S' if dev_mode else None
    
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        stream=sys.stdout
    )
    
    logger.debug("Логирование настроено в режиме разработчика") if dev_mode else None

def parse_arguments() -> argparse.Namespace:
    """
    Парсинг аргументов командной строки.
    
    Returns:
        Объект с аргументами командной строки
    """
    parser = argparse.ArgumentParser(
        description="Генерация HTML-файлов с графиками взятий вопросов"
    )
    
    parser.add_argument(
        "source", nargs="?", type=str,
        help="ID турнира или путь к табличному файлу (xlsx/csv)"
    )
    
    parser.add_argument(
        "-f", "--flags", type=str,
        help="Список ID флагов для фильтрации команд (например: '4,5' или '4+5' или '4*7')"
    )
    
    parser.add_argument(
        "-m", "--more_files", type=str, default="yes",
        help="Создавать отдельные файлы для флагов (yes/no, по умолчанию yes)"
    )
    
    parser.add_argument(
        "-n", "--name", type=str,
        help="Название турнира"
    )
    
    parser.add_argument(
        "-o", "--output", type=str,
        help="Имя выходного файла (без расширения)"
    )
    
    parser.add_argument(
        "-dev", action="store_true",
        help="Режим разработчика с подробным логированием"
    )
    
    return parser.parse_args()
def is_tournament_id(source: str) -> bool:
    """
    Проверяет, является ли источник данных ID турнира.
    
    Args:
        source: Строка с ID турнира или путем к файлу
        
    Returns:
        True, если источник похож на ID турнира, иначе False
    """
    # Если это число и не существует файла с таким именем
    if source and source.isdigit() and not os.path.exists(source):
        return True
    return False

def get_tournament_info(tournament_id: str) -> Dict:
    """
    Получает информацию о турнире по его ID через API.
    
    Args:
        tournament_id: ID турнира
        
    Returns:
        Словарь с информацией о турнире
    """
    url = f"https://api.rating.chgk.net/tournaments/{tournament_id}"
    logger.info(f"Получение информации о турнире {tournament_id}...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        tournament_info = response.json()
        logger.debug(f"Получена информация о турнире: {tournament_info}")
        return tournament_info
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при получении информации о турнире: {e}")
        return {}

def get_tournament_results(tournament_id: str) -> List[Dict]:
    """
    Получает результаты турнира по его ID через API.
    
    Args:
        tournament_id: ID турнира
        
    Returns:
        Список словарей с результатами команд
    """
    url = f"https://api.rating.chgk.net/tournaments/{tournament_id}/results"
    params = {
        "includeTeamMembers": 0,
        "includeMasksAndControversials": 1,
        "includeTeamFlags": 1,
        "includeRatingB": 0
    }
    
    logger.info(f"Получение результатов турнира {tournament_id}...")
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()
        logger.info(f"Получены результаты для {len(results)} команд")
        logger.debug(f"Результаты турнира: {results[:2]}...")  # Логируем только первые 2 команды
        return results
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при получении результатов турнира: {e}")
        return []

def get_tournament_flags() -> Dict[int, Dict]:
    """
    Получает информацию о всех возможных флагах команд через API.
    
    Returns:
        Словарь с информацией о флагах, где ключ - ID флага
    """
    url = "https://api.rating.chgk.net/tournament_flags"
    logger.info("Получение информации о флагах команд...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        flags_list = response.json()
        
        # Преобразуем список в словарь для удобного доступа по ID
        flags_dict = {flag["id"]: flag for flag in flags_list}
        
        logger.info(f"Получена информация о {len(flags_dict)} флагах")
        return flags_dict
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при получении информации о флагах: {e}")
        return {}
def parse_flags_argument(flags_arg: str) -> List[Union[int, Tuple[int, ...], List[int]]]:
    """
    Парсит аргумент с флагами команд.
    
    Args:
        flags_arg: Строка с флагами (например: '4,5' или '4+5' или '4*7')
        
    Returns:
        Список флагов или их комбинаций для фильтрации команд
    """
    if not flags_arg:
        return []
    
    # Разделяем строку по запятым, точкам с запятой или пробелам
    parts = re.split(r'[,;\s]+', flags_arg)
    result = []
    
    for part in parts:
        if '+' in part:  # Логическое ИЛИ
            flag_ids = [int(f) for f in part.split('+')]
            result.append(('OR', flag_ids))
            logger.debug(f"Добавлен фильтр по флагам (ИЛИ): {flag_ids}")
        elif '*' in part:  # Логическое И
            flag_ids = [int(f) for f in part.split('*')]
            result.append(('AND', flag_ids))
            logger.debug(f"Добавлен фильтр по флагам (И): {flag_ids}")
        else:
            try:
                flag_id = int(part)
                result.append(flag_id)
                logger.debug(f"Добавлен фильтр по флагу: {flag_id}")
            except ValueError:
                logger.warning(f"Некорректный ID флага: {part}, пропускаем")
    
    return result

def filter_teams_by_flags(teams: List[Dict], flag_filter: Union[int, Tuple[str, List[int]]]) -> List[Dict]:
    """
    Фильтрует команды по флагам.
    
    Args:
        teams: Список команд
        flag_filter: ID флага или кортеж с оператором ('OR'/'AND') и списком ID флагов
        
    Returns:
        Отфильтрованный список команд
    """
    filtered_teams = []
    
    if isinstance(flag_filter, int):
        # Простая фильтрация по одному флагу
        flag_id = flag_filter
        for team in teams:
            team_flags = [flag["id"] for flag in team.get("flags", [])]
            if flag_id in team_flags:
                filtered_teams.append(team)
    else:
        # Сложная фильтрация по нескольким флагам с логическим оператором
        operator, flag_ids = flag_filter
        
        for team in teams:
            team_flags = [flag["id"] for flag in team.get("flags", [])]
            
            if operator == 'OR' and any(flag_id in team_flags for flag_id in flag_ids):
                filtered_teams.append(team)
            elif operator == 'AND' and all(flag_id in team_flags for flag_id in flag_ids):
                filtered_teams.append(team)
    
    return filtered_teams
def process_api_data(tournament_id: str, flags_filters: List) -> Dict:
    """
    Обрабатывает данные, полученные через API.
    
    Args:
        tournament_id: ID турнира
        flags_filters: Список фильтров по флагам
        
    Returns:
        Словарь с данными для построения графиков
    """
    # Получаем информацию о турнире
    tournament_info = get_tournament_info(tournament_id)
    tournament_name = tournament_info.get('longName') or tournament_info.get('name', f'Турнир {tournament_id}')
    
    # Получаем результаты турнира
    teams_data = get_tournament_results(tournament_id)
    
    if not teams_data:
        logger.error(f"Не удалось получить данные о турнире {tournament_id}")
        return {}
    
    # Получаем информацию о флагах, если нужно
    flags_info = get_tournament_flags() if flags_filters else {}
    
    # Готовим данные для основного графика (все команды)
    main_data = prepare_chart_data_from_api(teams_data)
    
    # Готовим данные для дополнительных графиков по флагам
    flag_charts = {}
    for flag_filter in flags_filters:
        filtered_teams = filter_teams_by_flags(teams_data, flag_filter)
        
        if isinstance(flag_filter, int):
            # Простой флаг
            flag_id = flag_filter
            flag_name = flags_info.get(flag_id, {}).get('longName', f'Флаг {flag_id}')
            chart_key = str(flag_id)
        else:
            # Комбинированный флаг
            operator, flag_ids = flag_filter
            op_symbol = '+' if operator == 'OR' else '*'
            
            # Получаем названия флагов и убеждаемся, что они не None
            flag_names = []
            for f_id in flag_ids:
                flag_info = flags_info.get(f_id, {})
                flag_long_name = flag_info.get('longName')
                if flag_long_name:
                    flag_names.append(flag_long_name)
                else:
                    flag_names.append(f'Флаг {f_id}')
            
            # Формируем название с учетом оператора
            if operator == 'OR':
                flag_name = f"Зачёт: {' или '.join(flag_names)}"
            else:  # AND
                flag_name = f"Зачёт: {' и '.join(flag_names)}"
                
            chart_key = f"{op_symbol.join(map(str, flag_ids))}"
        
        logger.info(f"Подготовка данных для графика по флагу: {flag_name} ({len(filtered_teams)} команд)")
        flag_charts[chart_key] = {
            'name': flag_name,
            'data': prepare_chart_data_from_api(filtered_teams)
        }
    
    return {
        'tournament_id': tournament_id,
        'tournament_name': tournament_name,
        'main_data': main_data,
        'flag_charts': flag_charts
    }
def prepare_chart_data_from_api(teams_data: List[Dict]) -> Dict:
    """
    Подготавливает данные для графика из API-данных.
    
    Args:
        teams_data: Список словарей с данными команд
        
    Returns:
        Словарь с подготовленными данными для графика
    """
    if not teams_data:
        return {
            'teams': [],
            'questions_count': 0,
            'statistics': {
                'tombs': [],
                'anti_tombs': [],
                'hardest': [],
                'easiest': [],
                'removed': []
            }
        }
    
    # Ищем первую команду с непустой маской для определения количества вопросов
    questions_count = 0
    for team in teams_data:
        mask = team.get('mask')
        if mask:
            questions_count = len(mask)
            break
    
    if questions_count == 0:
        logger.error("Не удалось определить количество вопросов (все маски пустые)")
        return {
            'teams': [],
            'questions_count': 0,
            'statistics': {
                'tombs': [],
                'anti_tombs': [],
                'hardest': [],
                'easiest': [],
                'removed': []
            }
        }
    
    # Подготавливаем данные о командах
    teams = []
    for team in teams_data:
        mask = team.get('mask', '')
        
        # Пропускаем команды с пустой маской
        if not mask:
            logger.warning(f"Пропускаем команду {team.get('team', {}).get('name', 'Неизвестная')} с пустой маской")
            continue
            
        team_id = team.get('team', {}).get('id', 0)
        team_name = team.get('current', {}).get('name', 'Неизвестная команда')
        team_town = team.get('current', {}).get('town', {}).get('name', 'Неизвестный город')
        
        # Преобразуем маску в список взятий (1) и невзятий (0)
        takes = []
        for i, char in enumerate(mask, 1):
            if char == '1':
                takes.append(1)
            elif char == '0':
                takes.append(0)
            else:  # 'X' или другие символы - снятый вопрос
                takes.append(0)  # Для графика считаем как невзятый
        
        # Накопительная сумма взятий для графика
        cumulative_takes = [sum(takes[:i+1]) for i in range(len(takes))]
        
        teams.append({
            'id': team_id,
            'name': team_name,
            'town': team_town,
            'mask': mask,
            'takes': takes,
            'cumulative_takes': cumulative_takes,
            'total_takes': team.get('questionsTotal', sum(takes))
        })
    
    # Проверяем, остались ли команды после фильтрации
    if not teams:
        logger.warning("После обработки не осталось команд с валидными данными")
        return {
            'teams': [],
            'questions_count': questions_count,
            'statistics': {
                'tombs': [],
                'anti_tombs': [],
                'hardest': [],
                'easiest': [],
                'removed': []
            }
        }
    
    # Сортируем команды по количеству взятых вопросов (по убыванию)
    teams.sort(key=lambda x: x['total_takes'], reverse=True)
    
    # Собираем статистику по вопросам
    statistics = calculate_questions_statistics(teams, questions_count)
    
    return {
        'teams': teams,
        'questions_count': questions_count,
        'statistics': statistics
    }
def calculate_questions_statistics(teams: List[Dict], questions_count: int) -> Dict:
    """
    Рассчитывает статистику по вопросам (гробы, антигробы и т.д.).
    
    Args:
        teams: Список словарей с данными команд
        questions_count: Общее количество вопросов в турнире
        
    Returns:
        Словарь со статистикой по вопросам
    """
    # Подсчитываем количество взятий каждого вопроса
    takes_by_question = [0] * questions_count
    removed_questions = set()
    
    for team in teams:
        mask = team['mask']
        for i, char in enumerate(mask):
            if char == '1':
                takes_by_question[i] += 1
            elif char == 'X':
                removed_questions.add(i + 1)  # Номера вопросов начинаются с 1
    
    teams_count = len(teams)
    
    # Находим гробы (вопросы, которые не взяла ни одна команда)
    tombs = [i + 1 for i, takes in enumerate(takes_by_question) if takes == 0 and (i + 1) not in removed_questions]
    
    # Находим антигробы (вопросы, которые взяли все команды)
    anti_tombs = [i + 1 for i, takes in enumerate(takes_by_question) if takes == teams_count and (i + 1) not in removed_questions]
    
    # Находим самые сложные вопросы (исключая гробы)
    takes_with_index = [(takes, i + 1) for i, takes in enumerate(takes_by_question) if takes > 0 and (i + 1) not in removed_questions]
    takes_with_index.sort()  # Сортируем по возрастанию количества взятий
    
    hardest = []
    for takes, question_num in takes_with_index[:5]:  # Берем 5 самых сложных
        # Собираем список команд, которые взяли этот вопрос
        teams_took = [team['name'] for team in teams if team['mask'][question_num - 1] == '1']
        hardest.append({
            'question_num': question_num,
            'takes_count': takes,
            'teams_took': teams_took
        })
    
    # Находим самые легкие вопросы (исключая антигробы)
    takes_with_index = [(takes, i + 1) for i, takes in enumerate(takes_by_question) if takes < teams_count and (i + 1) not in removed_questions]
    takes_with_index.sort(reverse=True)  # Сортируем по убыванию количества взятий
    
    easiest = []
    for takes, question_num in takes_with_index[:5]:  # Берем 5 самых легких
        easiest.append({
            'question_num': question_num,
            'takes_count': takes
        })
    
    return {
        'tombs': tombs,
        'anti_tombs': anti_tombs,
        'hardest': hardest,
        'easiest': easiest,
        'removed': sorted(list(removed_questions))
    }
def read_table_file(file_path: str) -> Tuple[pd.DataFrame, bool]:
    """
    Читает табличный файл (xlsx или csv) и определяет его формат.
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        Кортеж из DataFrame и флага, указывающего на наличие разбиения по турам
    """
    logger.info(f"Чтение файла: {file_path}")
    
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            # Пробуем разные разделители
            for sep in [';', ',', '\t']:
                try:
                    df = pd.read_csv(file_path, sep=sep)
                    if len(df.columns) > 3:  # Успешно прочитали с нужным разделителем
                        break
                except:
                    continue
        else:
            logger.error(f"Неподдерживаемый формат файла: {file_path}")
            return pd.DataFrame(), False
        
        # Очищаем от пустых строк
        df = df.dropna(how='all')
        df = df.fillna("0")
        
        # Проверяем, есть ли разбиение по турам
        has_tours = False
        for col_name in df.columns[:5]:  # Проверяем первые 5 столбцов
            if isinstance(col_name, str) and ('тур' in col_name.lower() or 'tour' in col_name.lower()):
                has_tours = True
                break
        
        logger.info(f"Файл успешно прочитан. Строк: {len(df)}, столбцов: {len(df.columns)}")
        logger.info(f"Формат с разбиением по турам: {'Да' if has_tours else 'Нет'}")
        
        return df, has_tours
    except Exception as e:
        logger.error(f"Ошибка при чтении файла: {e}")
        return pd.DataFrame(), False
def process_table_data(file_path: str) -> Dict:
    """
    Обрабатывает данные из табличного файла.
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        Словарь с данными для построения графиков
    """
    # Читаем файл и определяем его формат
    df, has_tours = read_table_file(file_path)
    
    if df.empty:
        logger.error("Не удалось прочитать данные из файла")
        return {}
    
    # Получаем название турнира из имени файла
    file_name = os.path.basename(file_path)
    tournament_name = os.path.splitext(file_name)[0]
    
    # Обрабатываем данные в зависимости от формата
    if has_tours:
        chart_data = prepare_chart_data_from_table_with_tours(df)
    else:
        chart_data = prepare_chart_data_from_table_without_tours(df)
    
    return {
        'tournament_id': None,
        'tournament_name': tournament_name,
        'main_data': chart_data,
        'flag_charts': {}  # Для табличных данных флаги не поддерживаются
    }
def prepare_chart_data_from_table_without_tours(df: pd.DataFrame) -> Dict:
    """
    Подготавливает данные для графика из таблицы без разбиения по турам.
    
    Args:
        df: DataFrame с данными
        
    Returns:
        Словарь с подготовленными данными для графика
    """
    # Определяем столбцы с данными о командах и вопросах
    columns = list(df.columns)
    
    # Первые три столбца - ID команды, название и город
    question_columns = columns[3:]
    
    # Фильтруем только числовые столбцы (номера вопросов)
    question_columns = [col for col in question_columns if isinstance(col, (int, float)) or 
                        (isinstance(col, str) and col.isdigit())]
    
    questions_count = len(question_columns)
    logger.info(f"Обнаружено {questions_count} вопросов")
    
    # Подготавливаем данные о командах
    teams = []
    for _, row in df.iterrows():
        team_id = row.iloc[0]
        team_name = row.iloc[1]
        team_town = row.iloc[2]
        # Собираем маску и данные о взятиях
        mask = ""
        takes = []
        
        for col in question_columns:
            value = row[col]
            # Преобразуем значение в строку для проверки
            value_str = str(value).strip().upper()
            
            if value_str == '1' or value_str == 'TRUE' or value_str == '+':
                mask += '1'
                takes.append(1)
            elif value_str == '0' or value_str == 'FALSE' or value_str == '-':
                mask += '0'
                takes.append(0)
            elif value_str == 'X' or value_str == 'Х':  # Латинская и кириллическая X
                mask += 'X'
                takes.append(0)  # Для графика считаем как невзятый
            else:
                # Если значение не распознано, считаем вопрос невзятым
                mask += '0'
                takes.append(0)
        
        # Накопительная сумма взятий для графика
        cumulative_takes = [sum(takes[:i+1]) for i in range(len(takes))]
        total_takes = sum(takes)
        
        teams.append({
            'id': team_id,
            'name': team_name,
            'town': team_town,
            'mask': mask,
            'takes': takes,
            'cumulative_takes': cumulative_takes,
            'total_takes': total_takes
        })
    
    # Сортируем команды по количеству взятых вопросов (по убыванию)
    teams.sort(key=lambda x: x['total_takes'], reverse=True)
    
    # Собираем статистику по вопросам
    statistics = calculate_questions_statistics(teams, questions_count)
    
    return {
        'teams': teams,
        'questions_count': questions_count,
        'statistics': statistics
    }
def prepare_chart_data_from_table_with_tours(df: pd.DataFrame) -> Dict:
    """
    Подготавливает данные для графика из таблицы с разбиением по турам.
    
    Args:
        df: DataFrame с данными
        
    Returns:
        Словарь с подготовленными данными для графика
    """
    # Определяем столбцы с данными о командах, туре и вопросах
    columns = list(df.columns)
    
    # Находим индекс столбца с туром
    tour_column_idx = None
    for i, col in enumerate(columns[:5]):
        if isinstance(col, str) and ('тур' in col.lower() or 'tour' in col.lower()):
            tour_column_idx = i
            break
    
    if tour_column_idx is None:
        logger.error("Не найден столбец с информацией о туре")
        return {
            'teams': [],
            'questions_count': 0,
            'statistics': {
                'tombs': [],
                'anti_tombs': [],
                'hardest': [],
                'easiest': [],
                'removed': []
            }
        }
    
    # Первые три столбца - ID команды, название и город
    # Затем идет столбец с туром
    question_columns = columns[tour_column_idx + 1:]
    
    # Фильтруем только числовые столбцы или строковые с цифрами (номера вопросов)
    question_columns = [col for col in question_columns if isinstance(col, (int, float)) or 
                        (isinstance(col, str) and col.isdigit())]
    
    # Определяем максимальное количество вопросов в туре
    max_questions_in_tour = len(question_columns)
    
    # Определяем максимальный номер тура
    max_tour = df[columns[tour_column_idx]].max()
    
    # Общее количество вопросов в турнире
    questions_count = max_tour * max_questions_in_tour
    logger.info(f"Обнаружено {questions_count} вопросов ({max_tour} туров по {max_questions_in_tour} вопросов)")
    
    # Группируем данные по командам
    teams_data = {}
    
    for _, row in df.iterrows():
        team_id = row.iloc[0]
        team_name = row.iloc[1]
        team_town = row.iloc[2]
        tour = int(row[columns[tour_column_idx]])
        
        # Инициализируем данные команды, если еще не сделали это
        if team_id not in teams_data:
            teams_data[team_id] = {
                'id': team_id,
                'name': team_name,
                'town': team_town,
                'mask': ['0'] * questions_count,  # Инициализируем маску нулями
                'takes': [0] * questions_count    # Инициализируем взятия нулями
            }
        # Обрабатываем вопросы текущего тура
        for i, col in enumerate(question_columns):
            # Вычисляем абсолютный номер вопроса в турнире
            question_idx = (tour - 1) * max_questions_in_tour + i
            
            if question_idx >= questions_count:
                continue  # Пропускаем, если индекс выходит за пределы
            
            value = row[col]
            # Преобразуем значение в строку для проверки
            value_str = str(value).strip().upper()
            
            if value_str == '1' or value_str == 'TRUE' or value_str == '+':
                teams_data[team_id]['mask'][question_idx] = '1'
                teams_data[team_id]['takes'][question_idx] = 1
            elif value_str == '0' or value_str == 'FALSE' or value_str == '-':
                teams_data[team_id]['mask'][question_idx] = '0'
                teams_data[team_id]['takes'][question_idx] = 0
            elif value_str == 'X' or value_str == 'Х':  # Латинская и кириллическая X
                teams_data[team_id]['mask'][question_idx] = 'X'
                teams_data[team_id]['takes'][question_idx] = 0  # Для графика считаем как невзятый
    
    # Преобразуем данные в список команд
    teams = []
    for team_data in teams_data.values():
        # Преобразуем маску из списка в строку
        team_data['mask'] = ''.join(team_data['mask'])
        
        # Накопительная сумма взятий для графика
        cumulative_takes = [sum(team_data['takes'][:i+1]) for i in range(len(team_data['takes']))]
        total_takes = sum(team_data['takes'])

        teams.append({
            'id': team_data['id'],
            'name': team_data['name'],
            'town': team_data['town'],
            'mask': team_data['mask'],
            'takes': team_data['takes'],
            'cumulative_takes': cumulative_takes,
            'total_takes': total_takes
        })
    
    # Сортируем команды по количеству взятых вопросов (по убыванию)
    teams.sort(key=lambda x: x['total_takes'], reverse=True)
    
    # Собираем статистику по вопросам
    statistics = calculate_questions_statistics(teams, questions_count)
    
    return {
        'teams': teams,
        'questions_count': questions_count,
        'statistics': statistics
    }
def create_chart_html(data: Dict, output_path: str, more_files: bool) -> None:
    """
    Создает HTML-файл с интерактивным графиком и статистикой.
    
    Args:
        data: Данные для построения графика
        output_path: Путь для сохранения HTML-файла
        more_files: Создавать ли отдельные файлы для графиков по флагам
    """
    tournament_name = data['tournament_name']
    main_data = data['main_data']
    flag_charts = data.get('flag_charts', {})
    
    if not main_data:
        logger.error("Нет данных для построения графика")
        return
    
    # Создаем директорию для сохранения файлов, если она не существует
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем основной график
    logger.info(f"Создание основного графика для турнира: {tournament_name}")
    main_html = create_single_chart_html(main_data, tournament_name, data.get('tournament_id'))
    
    # Если есть графики по флагам и нужно создавать отдельные файлы
    if flag_charts and more_files:
        # Создаем директорию для дополнительных файлов
        extra_dir = os.path.join(output_dir, 'extra')
        os.makedirs(extra_dir, exist_ok=True)
        
        # Добавляем ссылки на дополнительные графики в основной HTML
        flag_links_html = '<div class="flag-links"><h3>Дополнительные графики по флагам:</h3><ul>'
        
        for flag_key, flag_data in flag_charts.items():
            flag_name = flag_data['name']
            
            # Создаем безопасное имя файла, заменяя недопустимые символы
            safe_flag_key = flag_key.replace('*', 'AND').replace('+', 'OR').replace('/', '_').replace('\\', '_')
            flag_file_name = f"{os.path.basename(output_path).split('.')[0]}_{safe_flag_key}.html"
            flag_output_path = os.path.join(extra_dir, flag_file_name)
            
            # Создаем HTML для графика по флагу
            logger.info(f"Создание графика для флага: {flag_name}")
            flag_html = create_single_chart_html(flag_data['data'], f"{tournament_name}. {flag_name}", data.get('tournament_id'))
            
            # Сохраняем HTML-файл
            with open(flag_output_path, 'w', encoding='utf-8') as f:
                f.write(flag_html)
            
            # Добавляем ссылку в основной HTML
            relative_path = os.path.relpath(flag_output_path, output_dir)
            flag_links_html += f'<li><a href="{relative_path}" target="_blank">{flag_name}</a></li>'
        
        flag_links_html += '</ul></div>'
        
        # Вставляем ссылки перед закрывающим тегом body
        main_html = main_html.replace('</body>', f'{flag_links_html}</body>')
    # Если есть графики по флагам, но не нужно создавать отдельные файлы
    elif flag_charts:
        # Добавляем все графики в один HTML-файл
        for flag_key, flag_data in flag_charts.items():
            flag_name = flag_data['name']
            logger.info(f"Добавление графика для флага: {flag_name} в основной файл")
            
            # Создаем HTML для графика по флагу (только график, без обертки)
            flag_chart_html = create_chart_figure(flag_data['data'], f"{tournament_name}. {flag_name}")
            
            # Добавляем якорь и заголовок
            flag_section_html = f'''
            <div id="flag_{flag_key}" class="flag-section">
                <h2>{flag_name}</h2>
                {flag_chart_html}
            </div>
            '''
            
            # Вставляем перед закрывающим тегом body
            main_html = main_html.replace('</body>', f'{flag_section_html}</body>')
        # Добавляем навигацию по флагам в начало страницы
        flag_nav_html = '<div class="flag-nav"><h3>Навигация по флагам:</h3><ul>'
        for flag_key, flag_data in flag_charts.items():
            flag_name = flag_data['name']
            flag_nav_html += f'<li><a href="#flag_{flag_key}">{flag_name}</a></li>'
        flag_nav_html += '</ul></div>'
        
        # Вставляем навигацию после основного графика
        main_html = main_html.replace('</div><!-- main-chart -->', '</div><!-- main-chart -->' + flag_nav_html)
    
    # Сохраняем основной HTML-файл
    logger.info(f"Сохранение HTML-файла: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(main_html)
    
    logger.info(f"HTML-файл успешно создан: {output_path}")
def create_single_chart_html(chart_data: Dict, title: str, tournament_id: Optional[str] = None) -> str:
    """
    Создает HTML-код для одного графика с обрамляющей страницей.
    
    Args:
        chart_data: Данные для построения графика
        title: Заголовок графика
        tournament_id: ID турнира (если есть)
        
    Returns:
        HTML-код страницы с графиком
    """
    # Создаем график
    chart_figure_html = create_chart_figure(chart_data, title)
    
    # Создаем HTML-код для статистики
    statistics_html = create_statistics_html(chart_data['statistics'])
    
    # Формируем полный HTML-код страницы с адаптивными стилями
    html = f'''
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: 'Noto Sans', sans-serif;
                margin: 0;
                padding: 10px;
                background-color: #f5f5f5;
                font-size: 16px;
            }}
            .container {{
                max-width: 100%;
                margin: 0 auto;
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1, h2, h3 {{
                color: #333;
                word-wrap: break-word;
            }}
            h1 {{
                font-size: 1.8em;
            }}
            h2 {{
                font-size: 1.5em;
            }}
            h3 {{
                font-size: 1.2em;
            }}
            .chart-container {{
                margin-top: 20px;
                margin-bottom: 20px;
                width: 100%;
                height: auto;
                min-height: 400px;
            }}
            .statistics {{
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #eee;
            }}
            .statistics-section {{
                margin-bottom: 20px;
            }}
            .flag-links, .flag-nav {{
                margin-top: 30px;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 8px;
            }}
            .flag-section {{
                margin-top: 50px;
                padding-top: 20px;
                border-top: 2px solid #ddd;
            }}
            .question-list {{
                list-style-type: none;
                padding-left: 0;
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }}
            .question-list li {{
                background-color: #f0f0f0;
                padding: 5px 10px;
                border-radius: 4px;
            }}
            .hardest-questions .question-item {{
                margin-bottom: 10px;
            }}
            .teams-took {{
                font-size: 0.9em;
                color: #666;
                margin-top: 5px;
            }}
            .footer {{
                margin-top: 30px;
                text-align: center;
                font-size: 0.8em;
                color: #999;
            }}
            
            /* Медиа-запросы для адаптивности */
            @media (min-width: 768px) {{
                .container {{
                    max-width: 90%;
                    padding: 20px;
                }}
                body {{
                    padding: 20px;
                }}
            }}
            
            @media (min-width: 1200px) {{
                .container {{
                    max-width: 1200px;
                }}
            }}
            
            /* Стили для мобильных устройств */
            @media (max-width: 767px) {{
                h1 {{
                    font-size: 1.5em;
                }}
                h2 {{
                    font-size: 1.3em;
                }}
                h3 {{
                    font-size: 1.1em;
                }}
                .chart-container {{
                    min-height: 300px;
                }}
                .question-list {{
                    gap: 5px;
                }}
                .question-list li {{
                    padding: 3px 6px;
                    font-size: 0.9em;
                }}
            }}
        </style>
        <link href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;700&display=swap" rel="stylesheet">
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            
            <div class="main-chart">
                <h2>График взятий вопросов</h2>
                <div class="chart-container">
                    {chart_figure_html}
                </div>
            </div><!-- main-chart -->
            
            <div class="statistics">
                <h2>Статистика турнира</h2>
                {statistics_html}
            </div>
            
            <div class="footer">
                <p>Сгенерировано с помощью <a href="https://github.com/a-berez/games_features/tree/main/chart_generator" target="_blank">chart_generator</a></p>
                {f'<p>ID турнира: <a href="https://rating.chgk.info/tournament/{tournament_id}" target="_blank">{tournament_id}</a></p>' if tournament_id else ''}
            </div>
        </div>
    </body>
    </html>
    '''
    return html
def create_chart_figure(chart_data: Dict, title: str) -> str:
    """
    Создает HTML-код с интерактивным графиком Plotly.
    
    Args:
        chart_data: Данные для построения графика
        title: Заголовок графика
        
    Returns:
        HTML-код с графиком
    """
    teams = chart_data['teams']
    questions_count = chart_data['questions_count']
    
    if not teams:
        return "<p>Нет данных для построения графика</p>"
    
    # Создаем фигуру Plotly
    fig = go.Figure()
    
    # Добавляем линии для каждой команды
    for i, team in enumerate(teams):
        team_name = team['name']
        town = team['town']
        total_takes = team['total_takes']
        
        # Создаем метку для легенды - укорачиваем для мобильных устройств
        legend_name = f"{team_name} ({town}) - {total_takes}"
        
        # Добавляем линию графика
        fig.add_trace(go.Scatter(
            x=list(range(1, questions_count + 1)),  # Номера вопросов от 1 до questions_count
            y=team['cumulative_takes'],
            mode='lines',
            name=legend_name,
            hovertemplate='Вопрос %{x}<br>Всего взято: %{y}<extra></extra>'
        ))
    
    # Настраиваем макет графика с адаптивными размерами
    fig.update_layout(
        title=f"{title}: график взятий вопросов",
        xaxis_title="Номер вопроса",
        yaxis_title="Количество взятых вопросов",
        hovermode="closest",
        # Располагаем легенду внизу для лучшего отображения на мобильных устройствах
        legend=dict(
            orientation="h",      # Горизонтальная ориентация для мобильных
            yanchor="top",
            y=-0.2,               # Располагаем под графиком
            xanchor="left",
            x=0,
            font=dict(size=10),
            itemwidth=30,
            tracegroupgap=5       # Уменьшаем расстояние между группами
        ),
        margin=dict(l=50, r=50, t=80, b=150),  # Увеличиваем нижний отступ для легенды
        autosize=True,
        # Добавляем аннотацию для отображения общего количества вопросов
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"Всего вопросов: {questions_count}",
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1,
                borderpad=4,
                align="left"
            )
        ]
    )
    
    # Настраиваем оси
    fig.update_xaxes(
        tickmode='linear',
        dtick=5,  # Шаг меток на оси X
        range=[0.5, questions_count + 0.5]  # Немного расширяем диапазон для лучшего отображения
    )
    
    fig.update_yaxes(
        range=[0, questions_count],  # Строго ограничиваем верхнюю границу количеством вопросов
        tickmode='linear',
        dtick=max(1, questions_count // 10),  # Адаптивный шаг для меток
        # Добавляем явную метку для максимального значения
        tickvals=list(range(0, questions_count + 1, max(1, questions_count // 10))) + [questions_count],
        ticktext=None  # Автоматические метки
    )
    
    # Добавляем настройку для адаптивного отображения
    config = {
        'responsive': True,
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'displaylogo': False,
        'scrollZoom': True  # Добавляем возможность масштабирования прокруткой
    }
    # Добавляем JavaScript для адаптивного расположения легенды в зависимости от размера экрана
    post_script = """
    <script>
    (function() {
        var graphDiv = document.querySelector('.plotly-graph-div');
        
        function updateLegendPosition() {
            if (window.innerWidth < 768) {
                // Для мобильных устройств - легенда внизу
                Plotly.relayout(graphDiv, {
                    'legend.orientation': 'h',
                    'legend.y': -0.2,
                    'legend.x': 0,
                    'margin.b': 150
                });
            } else {
                // Для десктопов - легенда справа
                Plotly.relayout(graphDiv, {
                    'legend.orientation': 'v',
                    'legend.y': 1,
                    'legend.x': 1.02,
                    'margin.r': 160,
                    'margin.b': 50
                });
            }
        }
        
        // Обновляем при загрузке и изменении размера окна
        updateLegendPosition();
        window.addEventListener('resize', updateLegendPosition);
    })();
    </script>
    """
    
    # Конвертируем график в HTML с дополнительными настройками
    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config=config)
    
    # Добавляем JavaScript для адаптивной легенды
    chart_html += post_script
    
    return chart_html
def create_statistics_html(statistics: Dict) -> str:
    """
    Создает HTML-код с статистикой турнира.
    
    Args:
        statistics: Данные статистики
        
    Returns:
        HTML-код с статистикой
    """
    tombs = statistics.get('tombs', [])
    anti_tombs = statistics.get('anti_tombs', [])
    hardest = statistics.get('hardest', [])
    easiest = statistics.get('easiest', [])
    removed = statistics.get('removed', [])
    
    html = '<div class="statistics-container">'
    
    # Гробы
    html += '<div class="statistics-section tombs">'
    html += '<h3>Гробы (вопросы, которые не взяла ни одна команда)</h3>'
    if tombs:
        html += '<ul class="question-list">'
        for question in tombs:
            html += f'<li>Вопрос {question}</li>'
        html += '</ul>'
    else:
        html += '<p>Нет гробов</p>'
    html += '</div>'
    
    # Антигробы
    html += '<div class="statistics-section anti-tombs">'
    html += '<h3>Антигробы (вопросы, которые взяли все команды)</h3>'
    if anti_tombs:
        html += '<ul class="question-list">'
        for question in anti_tombs:
            html += f'<li>Вопрос {question}</li>'
        html += '</ul>'
    else:
        html += '<p>Нет антигробов</p>'
    html += '</div>'
    # Самые сложные вопросы
    html += '<div class="statistics-section hardest-questions">'
    html += '<h3>Самые сложные вопросы</h3>'
    if hardest:
        html += '<ul>'
        for question in hardest:
            html += f'<li class="question-item">Вопрос {question["question_num"]} — взяли {question["takes_count"]} команд'
            
            # Добавляем список команд, которые взяли вопрос
            teams_took = question.get('teams_took', [])
            if teams_took and len(teams_took) <= 10:
                html += f'<div class="teams-took">Взяли: {", ".join(teams_took)}</div>'
            elif teams_took and len(teams_took) > 10:
                html += f'<div class="teams-took">Взяли {len(teams_took)} команд</div>'
            
            html += '</li>'
        html += '</ul>'

    else:
        html += '<p>Нет данных</p>'
    html += '</div>'
    
    # Самые легкие вопросы
    html += '<div class="statistics-section easiest-questions">'
    html += '<h3>Самые легкие вопросы</h3>'
    if easiest:
        html += '<ul>'
        for question in easiest:
            html += f'<li>Вопрос {question["question_num"]} — взяли {question["takes_count"]} команд</li>'
        html += '</ul>'
    else:
        html += '<p>Нет данных</p>'
    html += '</div>'
    
    # Снятые вопросы
    html += '<div class="statistics-section removed-questions">'
    html += '<h3>Снятые вопросы</h3>'
    if removed:
        html += '<ul class="question-list">'
        for question in removed:
            html += f'<li>Вопрос {question}</li>'
        html += '</ul>'
    else:
        html += '<p>Нет снятых вопросов</p>'
    html += '</div>'
    
    html += '</div>'  # Закрываем statistics-container
    
    return html
def prompt_for_tournament_name(default_name: str) -> str:
    """
    Запрашивает у пользователя название турнира.
    
    Args:
        default_name: Название по умолчанию
        
    Returns:
        Введенное пользователем название или значение по умолчанию
    """
    print(f"\nВведите название турнира (или нажмите Enter для использования '{default_name}'): ", end="")
    user_input = input().strip()
    return user_input if user_input else default_name
def main():
    """
    Основная функция скрипта.
    """
    # Парсим аргументы командной строки
    args = parse_arguments()
    
    # Настраиваем логирование
    setup_logging(args.dev)
    
    logger.info("Запуск chart_generator")
    logger.debug(f"Аргументы командной строки: {args}")
    
    # Определяем источник данных
    source = args.source
    
    if not source:
        logger.error("Не указан источник данных (ID турнира или путь к файлу)")
        return 1
    
    # Проверяем, является ли источник ID турнира или путем к файлу
    is_api_source = is_tournament_id(source)
    
    # Обрабатываем флаги
    flags_filters = []
    if args.flags:
        if not is_api_source:
            logger.warning("Фильтрация по флагам поддерживается только при использовании API. Аргумент будет проигнорирован.")
        else:
            flags_filters = parse_flags_argument(args.flags)
    # Определяем, нужно ли создавать отдельные файлы для флагов
    more_files = args.more_files.lower() in ['yes', 'да', 'y', 'д', 'true', '1']
    
    # Получаем данные для графика
    if is_api_source:
        # Получаем данные из API
        tournament_id = source
        data = process_api_data(tournament_id, flags_filters)
        
        if not data:
            logger.error(f"Не удалось получить данные для турнира {tournament_id}")
            return 1
        
        # Определяем название турнира
        if args.name:
            data['tournament_name'] = args.name
        elif not data.get('tournament_name'):
            # Запрашиваем название у пользователя
            data['tournament_name'] = prompt_for_tournament_name(f"Турнир {tournament_id}")
    else:
        # Проверяем существование файла
        if not os.path.exists(source):
            logger.error(f"Файл не найден: {source}")
            return 1
        
        # Получаем данные из файла
        data = process_table_data(source)
        
        if not data:
            logger.error(f"Не удалось обработать данные из файла {source}")
            return 1
        
        # Определяем название турнира
        if args.name:
            data['tournament_name'] = args.name
        else:
            # Запрашиваем название у пользователя
            default_name = os.path.basename(source).split('.')[0]
            data['tournament_name'] = prompt_for_tournament_name(default_name)
    
    # Определяем имя выходного файла
    if args.output:
        output_name = args.output
    else:
        if is_api_source:
            output_name = tournament_id
        else:
            output_name = os.path.basename(source).split('.')[0]
    
    # Создаем директорию charts, если ее нет
    charts_dir = Path('charts')
    charts_dir.mkdir(exist_ok=True)
    
    # Создаем поддиректорию для текущего турнира/файла
    output_dir = charts_dir / output_name
    output_dir.mkdir(exist_ok=True)
    
    # Полный путь к выходному файлу
    output_path = output_dir / f"{output_name}.html"
    
    # Создаем HTML-файл с графиком
    create_chart_html(data, str(output_path), more_files)
    
    logger.info(f"Работа завершена. Результат сохранен в {output_path}")
    return 0
if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nПрервано пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Произошла непредвиденная ошибка: {e}")
        if logging.getLogger().level == logging.DEBUG:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)

