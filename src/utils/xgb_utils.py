# src/utils/xgb_utils.py

import numpy as np


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

    if x_input.ndim == 3:
        n_samples, lag, n_features = x_input.shape
        x_flat = x_input.reshape(n_samples, lag * n_features)
    elif x_input.ndim == 2:
        lag, n_features = x_input.shape
        x_flat = x_input.reshape(1, lag * n_features)
    else:
        x_flat = x_input.reshape(1, -1)

    predictions = model.predict(x_flat)
    
    return predictions

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
    return float(prediction[0])
