#src/models/xgboost_model.py
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from typing import List, Tuple
from src.processing.data_processing import split_sequence, create_x_input
from src.utils.metrics import calculate_metrics

def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_params: dict
) -> XGBRegressor:
    """
    Обучает модель XGBoost на предоставленных данных.
    
    :param X_train: Массив признаков для обучения.
    :param y_train: Массив целевых значений для обучения.
    :param model_params: Параметры модели XGBoost.
    :return: Обученная модель XGBoost.
    """
    xgb_model = XGBRegressor(**model_params)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def forecast_XGBoost_sistem(
    col_target, time_column, df_all_data_norm, last_known_index, lag,
    model_architecture_params, col_for_train
):
    # Преобразование временной колонки
    df_all_data_norm[time_column] = pd.to_datetime(df_all_data_norm[time_column], errors='coerce')
    df_all_data_norm = df_all_data_norm.sort_values(by=time_column).reset_index(drop=True)
    
    # Убедиться, что целевая колонка включена в обучающие признаки
    col_for_train = [col_target] + col_for_train
    df_all_data_norm = df_all_data_norm[col_for_train].copy()
    
    # Преобразование целевой колонки в числовой формат
    df_all_data_norm[col_target] = df_all_data_norm[col_target].replace('None', None).astype(float)
    
    # Разделение данных на обучающую и тестовую выборки
    df_train = df_all_data_norm.iloc[:last_known_index].dropna(subset=[col_target])
    df_test = df_all_data_norm.iloc[last_known_index:].copy()
    df_test[col_target] = np.nan
    df_real_predict = df_test.copy()
    
    # Подготовка данных для XGBoost
    values = df_train[col_for_train].values
    x_input = create_x_input(df_train, lag)
    X, y = split_sequence(values, lag)
    
    # Проверка формы данных
    print(f"Shape of X before reshape: {X.shape}")
    print(f"Shape of y: {y.shape}")
    
    # Исправление формы X
    X = X.reshape(X.shape[0], -1)
    print(f"Shape of X after reshape: {X.shape}")
    
    # Преобразование в числовой тип
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    
    # Исключение строк с NaN
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    if np.isnan(X).sum() > 0 or np.isnan(y).sum() > 0:
        raise ValueError("X or y contains NaN values.")
    
    n_features = values.shape[1]
    
    # Обучение модели XGBoost
    if isinstance(model_architecture_params, list):
        model_params = model_architecture_params[0]  # Берем первый элемент списка
    else:
        model_params = model_architecture_params
    
    xgb_model = XGBRegressor(**model_params)
    X_reshaped = X.reshape(X.shape[0], -1)
    xgb_model.fit(X_reshaped, y)
    
    # Генерация прогнозов
    x_input = x_input.reshape((1, lag, n_features))
    predict_values = _make_xgboost_predictions(x_input, df_test.values, n_features, xgb_model, lag)
    df_real_predict[col_target] = np.array(predict_values).flatten()
    
    return df_train, df_real_predict

def _make_xgboost_predictions(
    x_input: np.ndarray,
    df_test: np.ndarray,
    n_features: int,
    model: XGBRegressor,
    lag: int
) -> List[float]:
    """
    Генерирует прогнозы для будущего горизонта с использованием обученной модели XGBoost.
    
    :param x_input: Исходные входные данные.
    :param df_test: Массив тестовых данных.
    :param n_features: Количество признаков.
    :param model: Обученная модель XGBoost.
    :param lag: Количество временных шагов.
    :return: Список прогнозируемых значений.
    """
    predict_values = []
    for _ in range(len(df_test)):
        # Предсказание следующего значения
        y_predict = model.predict(x_input.reshape(1, -1))[0]
        predict_values.append(y_predict)
        
        # Обновление входных данных
        x_input = np.delete(x_input, 0, axis=1)
        future_lag = df_test[0]
        df_test = np.delete(df_test, 0, axis=0)
        future_lag[0] = y_predict
        x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)
        x_input = x_input.reshape((1, lag, n_features))
    
    return predict_values
