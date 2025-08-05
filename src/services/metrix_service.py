# src/services/metrix_service.py

from typing import Any, Dict

import pandas as pd

from src.core.logger import logger
from src.utils.metrics import calculate_metrics


def run_metrix_all(
    col_time: str,
    col_target: str,
    df_evaluation: pd.DataFrame,
    df_comparative: pd.DataFrame
) -> Dict[str, Any]:
    """
    Выполняет расчёт метрик между двумя DataFrame'ами.
    
    :param col_time: Название временной колонки.
    :param col_target: Название целевой колонки.
    :param df_evaluation: Основной DataFrame.
    :param df_comparative: Сравнительный DataFrame.
    :return: Словарь с результатами расчёта метрик.
    """
    try:
        # Объединение данных по временной колонке
        merged_df = pd.merge(
            df_evaluation[[col_time, col_target]],
            df_comparative[[col_time, col_target]],
            on=col_time,
            suffixes=("_true", "_pred")
        )

        # Извлечение истинных и предсказанных значений
        y_true = merged_df[f"{col_target}_true"].values
        y_pred = merged_df[f"{col_target}_pred"].values

        # Расчёт метрик
        metrics = calculate_metrics(y_true, y_pred)

        return {
            "metrics": metrics,
            "df_metrics": merged_df.to_dict(orient="records"),
        }

    except Exception as e:
        logger.error(f"Ошибка в run_metrix_all: {e}")
        raise