#src/utils/metrics.py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Вычисляет метрики качества прогноза.
    
    :param y_true: Массив истинных значений.
    :param y_pred: Массив предсказанных значений.
    :return: Кортеж из RMSE, R2, MAE, MAPE, WMAPE.
    """
    # Среднее значение истинных данных
    y_true_mean = np.mean(y_true)
    
    print(f"Shape of y_true: {y_true.shape}")
    print(f"Shape of y_pred: {y_pred.shape}")
    assert len(y_true) == len(y_pred), f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Коэффициент детерминации (R2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true_mean) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.nan
    
    # Weighted Mean Absolute Percentage Error (WMAPE)
    wmape = (np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))) * 100 if np.sum(np.abs(y_true)) != 0 else np.nan
    
    return rmse, r2, mae, mape, wmape