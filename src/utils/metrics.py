# src/utils/metrics.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисляет среднюю абсолютную процентную ошибку (MAPE).
    
    :param y_true: Массив истинных значений.
    :param y_pred: Массив предсказанных значений.
    :return: Значение MAPE.
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисляет симметричную среднюю абсолютную процентную ошибку (sMAPE).
    
    :param y_true: Массив истинных значений.
    :param y_pred: Массив предсказанных значений.
    :return: Значение sMAPE.
    """
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


def normalized_root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисляет нормализованную корневую среднеквадратичную ошибку (NRMSE).
    
    :param y_true: Массив истинных значений.
    :param y_pred: Массив предсказанных значений.
    :return: Значение NRMSE.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (np.max(y_true) - np.min(y_true))


def mean_absolute_range_normalized_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисляет среднюю абсолютную ошибку, нормализованную по диапазону (MARNE).
    
    :param y_true: Массив истинных значений.
    :param y_pred: Массив предсказанных значений.
    :return: Значение MARNE.
    """
    return np.mean(np.abs(y_true - y_pred) / (np.max(y_true) - np.min(y_true)))


def weighted_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисляет взвешенную среднюю абсолютную процентную ошибку (WMAPE).
    
    :param y_true: Массив истинных значений.
    :param y_pred: Массив предсказанных значений.
    :return: Значение WMAPE.
    """
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Вычисляет метрики качества прогноза.
    
    :param y_true: Массив истинных значений.
    :param y_pred: Массив предсказанных значений.
    :return: Словарь с метриками качества.
    """
    # Проверка на совпадение длин массивов
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

    # Вычисление метрик
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    nrmse = normalized_root_mean_squared_error(y_true, y_pred)
    marne = mean_absolute_range_normalized_error(y_true, y_pred)
    wmape = weighted_mean_absolute_percentage_error(y_true, y_pred)

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3),
        "MAPE": round(mape, 3),
        "sMAPE": round(smape, 3),
        "NRMSE": round(nrmse, 3),
        "MARNE": round(marne, 3),
        "WMAPE": round(wmape, 3),
    }