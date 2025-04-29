#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Chart Generator - скрипт для создания HTML-графиков взятий вопросов

Скрипт создает интерактивные HTML-графики и таблицы, показывающие
динамику взятий вопросов командами в турнирах КВРМ.
Данные могут быть получены из API или из табличных файлов.
"""

import os
import sys
import re
import argparse
import logging
import requests
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import shutil
import csv

# Настройка логгера
logger = logging.getLogger(__name__)

def setup_logging(dev_mode: bool) -> None:
    """
    Настройка логирования.
    
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
        description="Генерация HTML-файлов с графиками и таблицами взятий вопросов"
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
        "-t", "--table", action="store_true",
        help="Создавать дополнительные HTML-файлы с таблицами Replay Table"
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
def parse_flags_argument(flags_arg: str) -> List[Union[int, Tuple[str, List[int]]]]:
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
        Словарь с данными для построения графиков и таблиц
    """
    # Получаем информацию о турнире
    tournament_info = get_tournament_info(tournament_id)
    
    # Получаем название турнира (короткое и длинное)
    tournament_name = tournament_info.get('name', f'Турнир {tournament_id}')
    tournament_long_name = tournament_info.get('longName', tournament_name)
    
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
        'tournament_long_name': tournament_long_name,
        'main_data': main_data,
        'flag_charts': flag_charts
    }

def prepare_chart_data_from_api(teams_data: List[Dict]) -> Dict:
    """
    Подготавливает данные для графика из API-данных.
    
    Args:
        teams_data: Список словарей с данными команд
        
    Returns:
        Словарь с подготовленными данными для графика и таблицы
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
                takes.append(0)  # Для графика и таблицы считаем как невзятый
        
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
        Словарь с данными для построения графиков и таблиц
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
        'tournament_long_name': tournament_name,  # Для файлов длинное и короткое имя совпадают
        'main_data': chart_data,
        'flag_charts': {}  # Для табличных данных флаги не поддерживаются
    }

def prepare_chart_data_from_table_without_tours(df: pd.DataFrame) -> Dict:
    """
    Подготавливает данные для графика из таблицы без разбиения по турам.
    
    Args:
        df: DataFrame с данными
        
    Returns:
        Словарь с подготовленными данными для графика и таблицы
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
                takes.append(0)  # Для графика и таблицы считаем как невзятый
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
        Словарь с подготовленными данными для графика и таблицы
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
                teams_data[team_id]['takes'][question_idx] = 0  # Для графика и таблицы считаем как невзятый
    
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

def create_csv_file(chart_data: Dict, output_path: str) -> None:
    """
    Создает CSV-файл для таблицы Replay Table.
    
    Args:
        chart_data: Данные для построения графика и таблицы
        output_path: Путь для сохранения CSV-файла
    """
    teams = chart_data.get('teams', [])
    questions_count = chart_data.get('questions_count', 0)
    
    if not teams or questions_count == 0:
        logger.error("Нет данных для создания CSV-файла")
        return
    
    logger.info(f"Создание CSV-файла для таблицы: {output_path}")
    try:
        # Открываем файл для записи
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Создаем writer с разделителем-запятой
            csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
            # Записываем заголовок: Название команды и номера вопросов
            header = ['Team'] + [str(i+1) for i in range(questions_count)]
            csv_writer.writerow(header)
            
            # Записываем данные каждой команды
            for team in teams:
                # Первый столбец - название команды
                row = [team['name']]
                
                # Добавляем данные о взятиях (1 - взят, 0 - не взят)
                row.extend(team['takes'])
                
                csv_writer.writerow(row)
        
        logger.info(f"CSV-файл успешно создан: {output_path}")
    except Exception as e:
        logger.error(f"Ошибка при создании CSV-файла: {e}")

def create_table_html(data: Dict, output_path: str, csv_path: str, chart_path: str, 
                     tournament_id: Optional[str] = None, is_extra_file: bool = False,
                     main_chart_path: Optional[str] = None) -> None:
    """
    Создает HTML-файл с таблицей Replay Table.
    
    Args:
        data: Данные для построения таблицы
        output_path: Путь для сохранения HTML-файла
        csv_path: Путь к CSV-файлу с данными
        chart_path: Путь к HTML-файлу с графиком для ссылки
        tournament_id: ID турнира (если есть)
        is_extra_file: Является ли файл дополнительным (находится в поддиректории)
        main_chart_path: Путь к основному графику (для ссылки "Назад")
    """
    # Получаем относительный путь к CSV-файлу
    base_dir = os.path.dirname(output_path)
    relative_csv_path = os.path.relpath(csv_path, base_dir)
    relative_chart_path = os.path.relpath(chart_path, base_dir)
    
    # Получаем название турнира
    tournament_name = data.get('tournament_name', 'Турнир')
    tournament_long_name = data.get('tournament_long_name', tournament_name)
    
    # Определяем относительный путь к директории work
    work_path = get_relative_work_path(output_path, is_extra_file)
    
    # Добавляем ссылку на основной график, если это дополнительная страница
    back_link_html = ''
    if is_extra_file and main_chart_path:
        # Относительный путь к основному графику
        relative_main_chart_path = os.path.relpath(main_chart_path, base_dir)
        back_link_html = f'<p><a href="{relative_main_chart_path}" class="back-link">← Вернуться к основному графику</a></p>'

    # Подготавливаем HTML-код для таблицы
    html = f'''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{tournament_name}</title>
    
    <!-- Подключаем шрифт Noto Sans с Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;700&display=swap" rel="stylesheet">

    <!-- Подключаем D3.js -->
    <script type="text/javascript" src="https://d3js.org/d3.v4.min.js"></script>
    
    <!-- Подключаем Replay Table -->
    <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/replay-table/dist/replay-table.min.js"></script>
    
    <!-- Подключаем стили -->
    <link rel="stylesheet" type="text/css" href="{work_path}/chart_generator_style.css">
</head>
<body>
    <h1>{tournament_name}</h1>
    
    <div class="tournament-info">
        {back_link_html}
        <p>Следите за изменением положения команд после каждого вопроса</p>
        <p>Интерактивный график можно посмотреть <a href="{relative_chart_path}">тут</a></p>
    </div>
    
    <div class="content">
        <div class="replayTable" id="chgk-table" 
             data-source="{relative_csv_path}" 
             data-preset="chgk">
        </div>
    </div>
    
    <footer>
        <p>Сгенерировано с помощью <a href="https://github.com/a-berez/games_features/tree/main/chart_generator" target="_blank">chart_generator</a></p>
        {f'<p>ID турнира: <a href="https://rating.chgk.info/tournament/{tournament_id}" target="_blank">{tournament_id}</a></p>' if tournament_id else ''}
        <p>Создано с помощью библиотеки <a href="https://github.com/antoniokov/replay-table">Replay Table</a> by <a href="https://github.com/antoniokov/">antoniokov</a></p>
        
        
    </footer>

    <!-- Подключаем скрипты -->
    <script type="text/javascript" src="{work_path}/chart_generator_scripts.js"></script>
</body>
</html>'''
    # Создаем директорию для сохранения файла, если она не существует
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем HTML-файл
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"HTML-файл с таблицей успешно создан: {output_path}")
    except Exception as e:
        logger.error(f"Ошибка при создании HTML-файла с таблицей: {e}")

def create_chart_html(data: Dict, output_path: str, table_path: Optional[str] = None, more_files: bool = True) -> None:
    """
    Создает HTML-файл с интерактивным графиком и статистикой.
    
    Args:
        data: Данные для построения графика
        output_path: Путь для сохранения HTML-файла
        table_path: Путь к HTML-файлу с таблицей для ссылки (если есть)
        more_files: Создавать ли отдельные файлы для графиков по флагам
    """
    tournament_name = data['tournament_name']
    tournament_long_name = data.get('tournament_long_name', tournament_name)
    main_data = data['main_data']
    flag_charts = data.get('flag_charts', {})
    
    if not main_data:
        logger.error("Нет данных для построения графика")
        return
    
    # Создаем директорию для сохранения файлов, если она не существует
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем относительный путь к файлу с таблицей, если он есть
    relative_table_path = None
    if table_path:
        relative_table_path = os.path.relpath(table_path, output_dir)
    
    # Создаем основной график
    logger.info(f"Создание основного графика для турнира: {tournament_name}")
    main_html = create_single_chart_html(
        main_data, 
        tournament_name, 
        data.get('tournament_id'), 
        relative_table_path,
        output_path,  # Передаем путь к выходному файлу
        False,  # Не является дополнительным файлом
        None    # Нет основного графика (это и есть основной)
    )
    
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
            
            # Формируем имена файлов для графика и таблицы
            base_name = os.path.basename(output_path).split('_chart.html')[0]
            flag_chart_file_name = f"{base_name}_{safe_flag_key}_chart.html"
            flag_chart_path = os.path.join(extra_dir, flag_chart_file_name)
            
            # Пути к соответствующим таблицам, если они есть
            flag_table_path = None
            if table_path:
                flag_table_file_name = f"{base_name}_{safe_flag_key}_table.html"
                flag_table_path = os.path.join(extra_dir, flag_table_file_name)
                
                # Путь к CSV-файлу для таблицы
                flag_csv_file_name = f"{base_name}_{safe_flag_key}.csv"
                flag_csv_path = os.path.join(extra_dir, flag_csv_file_name)
                
                # Создаем CSV-файл для таблицы
                create_csv_file(flag_data['data'], flag_csv_path)
                
                # Создаем HTML для таблицы с названием флага в заголовке
                create_table_html(
                    {
                        'tournament_name': f"{tournament_name}. {flag_name}",
                        'tournament_long_name': f"{tournament_long_name}. {flag_name}"
                    },
                    flag_table_path,
                    flag_csv_path,
                    flag_chart_path,
                    data.get('tournament_id'),
                    True  # Является дополнительным файлом
                )
            
            # Относительный путь к таблице для ссылки в графике
            relative_flag_table_path = os.path.relpath(flag_table_path, os.path.dirname(flag_chart_path)) if flag_table_path else None
            
            # Относительный путь к основному графику для ссылки "Назад"
            relative_main_chart_path = os.path.relpath(output_path, os.path.dirname(flag_chart_path))
            
            # Создаем HTML для графика по флагу
            logger.info(f"Создание графика для флага: {flag_name}")
            flag_html = create_single_chart_html(
                flag_data['data'],
                f"{tournament_name}. {flag_name}",
                data.get('tournament_id'),
                relative_flag_table_path,
                flag_chart_path,  # Передаем путь к выходному файлу
                True,  # Является дополнительным файлом
                relative_main_chart_path  # Путь к основному графику для ссылки "Назад"
            )
            
            # Сохраняем HTML-файл с графиком
            with open(flag_chart_path, 'w', encoding='utf-8') as f:
                f.write(flag_html)
            
            # Добавляем ссылку в основной HTML
            relative_chart_path = os.path.relpath(flag_chart_path, output_dir)
            flag_links_html += f'<li><a href="{relative_chart_path}" target="_blank">{flag_name}</a></li>'
            
            # Если есть таблица, добавляем ссылку на нее
            if flag_table_path:
                relative_table_path = os.path.relpath(flag_table_path, output_dir)
                flag_links_html += f' (<a href="{relative_table_path}" target="_blank">таблица</a>)'
        
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
    logger.info(f"Сохранение HTML-файла с графиком: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(main_html)
    
    logger.info(f"HTML-файл с графиком успешно создан: {output_path}")

def get_relative_work_path(output_path: str, is_extra_file: bool = False) -> str:
    """
    Определяет относительный путь к директории work в зависимости от уровня вложенности файла.
    
    Args:
        output_path: Путь к выходному HTML-файлу
        is_extra_file: Является ли файл дополнительным (находится в поддиректории)
        
    Returns:
        Относительный путь к директории work
    """
    # Проверка на None
    if output_path is None:
        return "../work"  # Возвращаем путь по умолчанию
        
    # Определяем количество уровней вложенности
    # charts/{output_name}/... - это базовый уровень (2 уровня)
    # charts/{output_name}/extra/... - дополнительный уровень (3 уровня)
    
    # Получаем количество компонентов пути
    path_parts = Path(output_path).parts
    
    # Базовый уровень - 2 (charts/output_name)
    # Если файл в extra, то уровень вложенности на 1 больше
    if is_extra_file or len(path_parts) > 3:
        return "../../work"
    else:
        return "../work"

def create_single_chart_html(chart_data: Dict, title: str, tournament_id: Optional[str] = None, 
                            table_path: Optional[str] = None, output_path: Optional[str] = None, 
                            is_extra_file: bool = False, main_chart_path: Optional[str] = None) -> str:
    """
    Создает HTML-код для одного графика с обрамляющей страницей.
    
    Args:
        chart_data: Данные для построения графика
        title: Заголовок графика
        tournament_id: ID турнира (если есть)
        table_path: Путь к HTML-файлу с таблицей для ссылки (если есть)
        output_path: Путь к выходному HTML-файлу
        is_extra_file: Является ли файл дополнительным (находится в поддиректории)
        main_chart_path: Путь к основному графику (для ссылки "Назад")
        
    Returns:
        HTML-код страницы с графиком
    """
    # Создаем график
    chart_figure_html = create_chart_figure(chart_data, title)
    
    # Создаем HTML-код для статистики
    statistics_html = create_statistics_html(chart_data['statistics'])
    
    # Добавляем ссылку на таблицу, если она есть
    table_link_html = f'<p>Интерактивную таблицу можно посмотреть <a href="{table_path}" target="_blank">тут</a></p>' if table_path else ''
    
    # Добавляем ссылку на основной график, если это дополнительная страница
    back_link_html = ''
    if is_extra_file and main_chart_path:
        # Если это дополнительная страница, добавляем ссылку "Назад"
        back_link_html = f'<p><a href="{main_chart_path}" class="back-link">← Вернуться к основному графику</a></p>'
    
    # Определяем относительный путь к директории work
    work_path = get_relative_work_path(output_path, is_extra_file)
    
    # Формируем полный HTML-код страницы
    html = f'''
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <link href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" type="text/css" href="{work_path}/chart_generator_style.css">
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            
            <div class="tournament-info">
                {back_link_html}
                {table_link_html}
            </div>
            
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
            data['tournament_long_name'] = args.name
        elif not data.get('tournament_name'):
            # Запрашиваем название у пользователя
            data['tournament_name'] = prompt_for_tournament_name(f"Турнир {tournament_id}")
            data['tournament_long_name'] = data['tournament_name']
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
            data['tournament_long_name'] = args.name
        else:
            # Запрашиваем название у пользователя
            default_name = os.path.basename(source).split('.')[0]
            data['tournament_name'] = prompt_for_tournament_name(default_name)
            data['tournament_long_name'] = data['tournament_name']
    
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
    
    # Полные пути к выходным файлам
    chart_path = output_dir / f"{output_name}_chart.html"
    
    # Если нужно создавать таблицу
    if args.table:
        # Путь к CSV-файлу для таблицы
        csv_path = output_dir / f"{output_name}.csv"
        # Путь к HTML-файлу с таблицей
        table_path = output_dir / f"{output_name}_table.html"
        
        # Создаем CSV-файл для таблицы
        create_csv_file(data['main_data'], csv_path)
        
        # Создаем HTML-файл с таблицей
        create_table_html(data, table_path, csv_path, chart_path, data.get('tournament_id'), False)
        
        # Создаем HTML-файл с графиком, включая ссылку на таблицу
        create_chart_html(data, chart_path, table_path, more_files)
        
        logger.info(f"Созданы файлы: {chart_path} и {table_path}")
    else:
        # Создаем только HTML-файл с графиком
        create_chart_html(data, chart_path, None, more_files)
        
        logger.info(f"Создан файл: {chart_path}")
    
    logger.info(f"Работа завершена. Результаты сохранены в директории {output_dir}")
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
