# src/backend/metrix.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Вычисляет метрики качества прогноза.
    
    :param y_true: Массив истинных значений.
    :param y_pred: Массив предсказанных значений.
    :return: Словарь с метриками качества.
    """
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)

    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "MAPE": round(mape, 3),
    }


def metrix_all(
    col_time: str,
    col_target: str,
    df_evaluation: pd.DataFrame,
    df_comparative: pd.DataFrame
) -> tuple:
    """
    Выполняет расчёт метрик между двумя DataFrame'ами.
    
    :param col_time: Название временной колонки.
    :param col_target: Название целевой колонки.
    :param df_evaluation: Основной DataFrame.
    :param df_comparative: Сравнительный DataFrame.
    :return: Кортеж с метриками и DataFrame с детализацией.
    """
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

    return metrics, merged_df