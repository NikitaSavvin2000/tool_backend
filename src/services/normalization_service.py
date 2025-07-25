# src/services/normalization_service.py

from typing import Any, Dict

import pandas as pd

from src.backend.normalization import Time2Vec
from src.core.logger import logger


def run_normalization(
    df: pd.DataFrame, col_time: str, col_target: str
) -> Dict[str, Any]:
    """
    Выполняет нормализацию временного ряда.
    
    :param df: Исходный DataFrame.
    :param col_time: Название временной колонки.
    :param col_target: Название целевой колонки для нормализации.
    :return: Словарь с результатами нормализации:
        - df_all_data_norm (dict): Нормализованный DataFrame.
        - min_val (float): Минимальное значение целевой колонки.
        - max_val (float): Максимальное значение целевой колонки.
    """
    try:
        # Инициализация Time2Vec
        t2v = Time2Vec(col_time=col_time, col_target=col_target)

        # Выполнение нормализации
        df_all_data_norm, min_val, max_val = t2v.vectorization(df)

        return {
            "df_all_data_norm": df_all_data_norm.to_dict(orient="records"),
            "min_val": min_val,
            "max_val": max_val,
        }

    except Exception as e:
        logger.error(f"Ошибка в run_normalization: {e}")
        raise


def run_reverse_normalization(
    df: pd.DataFrame, col_time: str, col_target: str, min_val: float, max_val: float
) -> Dict[str, Any]:
    """
    Выполняет денормализацию временного ряда.
    
    :param df: Нормализованный DataFrame.
    :param col_time: Название временной колонки.
    :param col_target: Название целевой колонки для денормализации.
    :param min_val: Минимальное значение целевой колонки до нормализации.
    :param max_val: Максимальное значение целевой колонки до нормализации.
    :return: Словарь с результатами денормализации:
        - df_all_data_reverse_norm (dict): Денормализованный DataFrame.
    """
    try:
        # Инициализация Time2Vec
        t2v = Time2Vec(col_time=col_time, col_target=col_target)

        # Выполнение денормализации
        df_reversed = t2v.reverse_vectorization(df, min_val=min_val, max_val=max_val)

        return {
            "df_all_data_reverse_norm": df_reversed.to_dict(orient="records"),
        }

    except Exception as e:
        logger.error(f"Ошибка в run_reverse_normalization: {e}")
        raise