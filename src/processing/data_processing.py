#src/processing/data_processing.py
import pandas as pd
import numpy as np
from typing import List, Tuple

def split_sequence(sequence: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Разделяет одномерную последовательность на образцы для обучения.
    
    :param sequence: Входная последовательность.
    :param n_steps: Количество шагов для lookback.
    :return: Кортеж из массивов входных данных (X) и целевых значений (y).
    """
    X, y = [], []
    for i in range(len(sequence) - n_steps):
        seq_x, seq_y = sequence[i:i + n_steps, :], sequence[i + n_steps, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def create_x_input(df_train: pd.DataFrame, n_steps: int) -> np.ndarray:
    """
    Создает входной массив для прогнозирования из обучающего DataFrame.
    
    :param df_train: Обучающие данные.
    :param n_steps: Количество шагов для lookback.
    :return: Входной массив для прогнозирования.
    """
    return df_train.iloc[-n_steps:].values

def calculate_time_interval(df: pd.DataFrame, time_column: str) -> int:
    """
    Вычисляет средний временной интервал в секундах между записями.
    
    :param df: DataFrame с временными метками.
    :param time_column: Название колонки с временными метками.
    :return: Средний временной интервал в секундах.
    """
    df[time_column] = pd.to_datetime(df[time_column])
    time_interval = df[time_column].diff().dt.total_seconds().mean()
    return int(time_interval)

def clean_column(val: str) -> float:
    """
    Очищает значение колонки от ненужных символов и преобразует в число.
    
    :param val: Исходное значение.
    :return: Очищенное числовое значение или None, если преобразование невозможно.
    """
    if isinstance(val, str):
        val = val.replace('%', '').replace('M', '').replace(',', '.')
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def preprocess_data(df: pd.DataFrame, cols_to_convert: List[str], time_column: str) -> pd.DataFrame:
    """
    Предобработка данных: очистка числовых колонок и форматирование временных меток.
    
    :param df: Исходный DataFrame.
    :param cols_to_convert: Список колонок для очистки и преобразования.
    :param time_column: Название колонки с временными метками.
    :return: Обработанный DataFrame.
    """
    # Очистка числовых колонок
    df[cols_to_convert] = df[cols_to_convert].applymap(clean_column)
    
    # Форматирование временных меток
    df[time_column] = pd.to_datetime(df[time_column], format='%d.%m.%Y', errors='coerce')
    df[time_column] = df[time_column].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df
