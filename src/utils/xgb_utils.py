# src/utils/xgb_utils.py
import numpy as np
from typing import Any

def make_predictions_xgb(model, x_input: np.ndarray) -> np.ndarray:
    """
    Генерирует прогнозы для XGBoost.

    Args:
        model: Обученная модель XGBoost (XGBRegressor).
        x_input (np.ndarray): Входные данные для модели. 
                              Ожидается форма (lag, n_features) или (1, lag, n_features).

    Returns:
        np.ndarray: Предсказанные значения. Форма (n_predictions,) или скаляр.
    """
    # Проверим и преобразуем x_input в нужный формат для XGBoost
    if x_input.ndim == 3:
        # Если форма (1, lag, n_features) или (n_samples, lag, n_features)
        # Преобразуем в (n_samples, lag * n_features)
        n_samples, lag, n_features = x_input.shape
        x_flat = x_input.reshape(n_samples, lag * n_features)
    elif x_input.ndim == 2:
        # Если форма (lag, n_features)
        # Преобразуем в (1, lag * n_features)
        lag, n_features = x_input.shape
        x_flat = x_input.reshape(1, lag * n_features)
    else:
        # Если форма (lag * n_features,) - уже плоский
        x_flat = x_input.reshape(1, -1)

    # Предсказание
    predictions = model.predict(x_flat)
    
    return predictions

# Дополнительная универсальная функция (по желанию)
def predict_single_step_xgb(model, x_input_single: np.ndarray) -> float:
    """
    Предсказывает одно значение на основе одного входного окна.

    Args:
        model: Обученная модель XGBoost.
        x_input_single (np.ndarray): Входное окно, форма (lag, n_features).

    Returns:
        float: Предсказанное значение.
    """
    prediction = make_predictions_xgb(model, x_input_single)
    # XGBoost.predict возвращает массив, берем первый элемент
    return float(prediction[0])
