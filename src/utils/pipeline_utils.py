# src/utils/pipeline_utils.py

from typing import Dict, List

import pandas as pd

from src.backend.normalization import Time2Vec
from src.core.logger import logger


def generate_possible_date(
    df: pd.DataFrame,
    time_column: str,
    col_target: str,
    forecast_horizon_time: str,
) -> List[str]:
    """
    Генерирует возможные даты прогнозирования на основе входного временного ряда.
    
    Description:
    - Принимает DataFrame, временную колонку, целевую колонку и горизонт прогнозирования.
    - Возвращает список возможных дат в интервале между последней известной датой и горизонтом прогнозирования.
    
    Parameters:
    - **df (pd.DataFrame)**: Входной DataFrame.
    - **time_column (str)**: Название временной колонки.
    - **col_target (str)**: Название целевой колонки.
    - **forecast_horizon_time (str)**: Горизонт прогнозирования (дата или временная метка).
    
    Returns:
    - **List[str]**: Список возможных дат прогнозирования.
    
    Example Request:
    ```python
    df = pd.DataFrame([
        {"time": "2023-01-01 00:00:00", "value": 10},
        {"time": "2023-01-01 00:15:00", "value": 20}
    ])
    generate_possible_date(df, "time", "value", "2023-01-01 01:00:00")
    ```
    
    Example Response:
    ```python
    ["2023-01-01 00:30:00", "2023-01-01 00:45:00", "2023-01-01 01:00:00"]
    ```
    
    Raises:
    - **ValueError**: Если входной DataFrame пустой или временная колонка отсутствует.
    """
    try:
        # Проверка на пустой DataFrame
        if df.empty:
            raise ValueError("Входной DataFrame не может быть пустым.")

        # Проверка наличия временной колонки
        if time_column not in df.columns:
            raise ValueError(f"Колонка '{time_column}' отсутствует во входном DataFrame.")

        # Преобразование временной колонки в datetime
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
        
        # Преобразование горизонта прогнозирования в Timestamp
        try:
            forecast_horizon_time = pd.to_datetime(forecast_horizon_time)
        except ValueError as e:
            raise ValueError(f"Неверный формат даты горизонта прогнозирования: {forecast_horizon_time}") from e

        # Определение последней известной даты
        last_known_date = df[time_column].max()
        if pd.isna(last_known_date):
            raise ValueError("Не удалось определить последнюю известную дату.")

        # Генерация возможных дат
        possible_dates = []
        current_date = last_known_date
        while current_date < forecast_horizon_time:
            current_date += pd.Timedelta(minutes=15)  # Шаг в 15 минут
            possible_dates.append(current_date.strftime('%Y-%m-%d %H:%M:%S'))

        return possible_dates

    except Exception as e:
        logger.error(f"Ошибка в generate_possible_date: {e}")
        raise


def prepare_data_for_pipeline(
    df: pd.DataFrame,
    time_column: str,
    col_target: str,
    norm_values: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Подготавливает данные для использования в пайплайне.
    
    Description:
    - Выполняет нормализацию данных, добавление новых фичей и разбиение на обучающую/тестовую выборки.
    
    Parameters:
    - **df (pd.DataFrame)**: Входной DataFrame.
    - **time_column (str)**: Название временной колонки.
    - **col_target (str)**: Название целевой колонки.
    - **norm_values (bool)**: Флаг для нормализации значений (по умолчанию True).
    
    Returns:
    - **Dict[str, pd.DataFrame]**: Словарь с подготовленными данными:
        - "df_train": Обучающая выборка.
        - "df_test": Тестовая выборка.
    
    Example Request:
    ```python
    df = pd.DataFrame([
        {"time": "2023-01-01 00:00:00", "value": 10},
        {"time": "2023-01-01 00:15:00", "value": 20}
    ])
    prepare_data_for_pipeline(df, "time", "value")
    ```
    
    Raises:
    - **ValueError**: Если входной DataFrame пустой или временная колонка отсутствует.
    """
    try:
        # Проверка на пустой DataFrame
        if df.empty:
            raise ValueError("Входной DataFrame не может быть пустым.")

        # Проверка наличия временной колонки
        if time_column not in df.columns:
            raise ValueError(f"Колонка '{time_column}' отсутствует во входном DataFrame.")

        # Преобразование временной колонки в datetime
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')

        # Добавление новых фичей
        t2v = Time2Vec(col_time=time_column, col_target=col_target)
        df_processed = t2v.vectorization(df)[0]

        # Нормализация данных
        if norm_values:
            min_val = df_processed[col_target].min()
            max_val = df_processed[col_target].max()
            df_processed[col_target] = (df_processed[col_target] - min_val) / (max_val - min_val)

        # Разделение на обучающую и тестовую выборки
        train_size = int(len(df_processed) * 0.8)
        df_train = df_processed[:train_size]
        df_test = df_processed[train_size:]

        return {
            "df_train": df_train,
            "df_test": df_test,
        }

    except Exception as e:
        logger.error(f"Ошибка в prepare_data_for_pipeline: {e}")
        raise