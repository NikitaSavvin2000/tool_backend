# src/services/hyperparameter_tuning.py
from typing import Any, Dict

from sklearn.metrics import mean_absolute_percentage_error

from src.core.constants.constants import (
    COLSAMPLE_BYTREE_MAX,
    COLSAMPLE_BYTREE_MIN,
    LEARNING_RATE_MAX,
    LEARNING_RATE_MIN,
    MAX_DEPTH_MAX,
    MAX_DEPTH_MIN,
    MIN_CHILD_WEIGHT_MAX,
    MIN_CHILD_WEIGHT_MIN,
    N_ESTIMATORS_MAX,
    N_ESTIMATORS_MIN,
    SUBSAMPLE_MAX,
    SUBSAMPLE_MIN,
)
from src.core.logger import logger
from src.utils.metrics import calculate_metrics

# src/services/hyperparameter_tuning.py
import pandas as pd
import optuna
from typing import Dict, List
from xgboost import XGBRegressor
from src.processing.data_processing import split_sequence, create_x_input
from src.models.xgboost_model import _make_xgboost_predictions
from src.normalization.time2vec import Time2Vec
from src.utils.metrics import calculate_metrics
import numpy as np


def objective(
    trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """
    Целевая функция для подбора гиперпараметров с помощью Optuna.
    
    :param trial: Объект Trial из Optuna.
    :param X_train: Массив признаков обучающей выборки.
    :param y_train: Массив целевых значений обучающей выборки.
    :param X_val: Массив признаков валидационной выборки.
    :param y_val: Массив целевых значений валидационной выборки.
    :return: Значение метрики (MAPE) для оценки модели.
    """
    params = {
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "n_estimators": trial.suggest_int("n_estimators", N_ESTIMATORS_MIN, N_ESTIMATORS_MAX),
        "learning_rate": trial.suggest_loguniform("learning_rate", LEARNING_RATE_MIN, LEARNING_RATE_MAX),
        "max_depth": trial.suggest_int("max_depth", MAX_DEPTH_MIN, MAX_DEPTH_MAX),
        "subsample": trial.suggest_uniform("subsample", SUBSAMPLE_MIN, SUBSAMPLE_MAX),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", COLSAMPLE_BYTREE_MIN, COLSAMPLE_BYTREE_MAX),
        "min_child_weight": trial.suggest_int("min_child_weight", MIN_CHILD_WEIGHT_MIN, MIN_CHILD_WEIGHT_MAX),
        "random_state": 42,
    }

    # Обучение модели
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Предсказание на валидационной выборке
    y_pred = model.predict(X_val)

    # Вычисление MAPE
    mape = mean_absolute_percentage_error(y_val, y_pred)
    return mape


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> Dict[str, Any]:
    """
    Подбор оптимальных гиперпараметров для модели XGBoost с использованием Optuna.
    
    :param X_train: Массив признаков обучающей выборки.
    :param y_train: Массив целевых значений обучающей выборки.
    :param X_val: Массив признаков валидационной выборки.
    :param y_val: Массив целевых значений валидационной выборки.
    :param n_trials: Количество испытаний для Optuna.
    :return: Словарь с лучшими гиперпараметрами.
    """
    try:
        # Создание исследования Optuna
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
        )

        # Лучшие параметры
        best_params = study.best_params
        best_params["objective"] = "reg:squarederror"
        best_params["booster"] = "gbtree"
        best_params["random_state"] = 42

        logger.info(f"Лучшие параметры: {best_params}")
        return best_params

    except Exception as e:
        logger.error(f"Ошибка при подборе гиперпараметров: {e}")
        raise


def evaluate_model(
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_params: Dict[str, Any],
) -> Dict[str, float]:
    """
    Оценка модели с использованием лучших гиперпараметров.
    
    :param X_test: Массив признаков тестовой выборки.
    :param y_test: Массив целевых значений тестовой выборки.
    :param best_params: Лучшие гиперпараметры модели.
    :return: Словарь с метриками качества модели.
    """
    try:
        # Обучение модели с лучшими параметрами
        model = XGBRegressor(**best_params)
        model.fit(X_test, y_test)

        # Предсказание
        y_pred = model.predict(X_test)

        # Вычисление метрик
        rmse, r2, mae, mape, wmape = calculate_metrics(y_true=y_test, y_pred=y_pred)

        metrics = {
            "RMSE": rmse,
            "R2": r2,
            "MAE": mae,
            "MAPE": mape,
            "WMAPE": wmape,
        }
        logger.info(f"Метрики модели: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Ошибка при оценке модели: {e}")
        raise


def params_selection_xgboots(
        df_init: pd.DataFrame,
        time_column: str,
        col_target: str,
        cols: List[str],
        lag: int
) -> Dict:
    """
    Выбирает оптимальные гиперпараметры для модели XGBoost с фиксированным лагом.

    :param df_init: Исходный DataFrame, содержащий временные ряды.
    :param time_column: Название временной колонки.
    :param col_target: Название целевой переменной.
    :param cols: Список признаков для обучения.
    :param lag: Фиксированное значение лага для прогнозирования временных рядов.
    :return: Словарь, содержащий лучшие параметры XGBoost.
    """
    # Сохранение оригинального формата временной колонки
    original_column = df_init[time_column].copy()
    df_init = df_init.copy()
    df_init[time_column] = pd.to_datetime(df_init[time_column], errors='coerce')
    df_init = df_init.sort_values(by=time_column).reset_index(drop=True)
    df_init[time_column] = original_column[df_init.index]

    # Определение размера тестовой выборки
    optimal_evaluation_points = int(min(len(df_init) * MIN_TEST_FRACTION, OPTIMAL_EVALUATION_POINTS))

    # Создание пустого DataFrame для тестовых данных
    df_all_data = pd.concat([df_init, df_init[-optimal_evaluation_points:].copy()], ignore_index=True).sort_values(by=time_column).reset_index(drop=True)
    last_known_index = len(df_all_data) - optimal_evaluation_points

    # Векторизация данных
    t2v = Time2Vec(col_time=time_column, col_target=col_target)
    df_all_data_norm, min_val, max_val = t2v.vectorization(df_all_data)
    df_all_data_norm = df_all_data_norm.sort_values(by=time_column).reset_index(drop=True)

    # Подготовка обучающих данных для подбора гиперпараметров
    df_train = df_all_data_norm.iloc[:last_known_index].copy()
    df_train = df_train[[col_target] + cols]
    values = df_train.values
    X, y = split_sequence(values, lag)
    X_train = X.reshape(X.shape[0], -1)


    # Определение функции для подбора гиперпараметров
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "booster": "gbtree",
            "n_estimators": trial.suggest_int("n_estimators", N_ESTIMATORS_MIN, N_ESTIMATORS_MAX),
            "learning_rate": trial.suggest_float("learning_rate", LEARNING_RATE_MIN, LEARNING_RATE_MAX, log=True),
            "max_depth": trial.suggest_int("max_depth", MAX_DEPTH_MIN, MAX_DEPTH_MAX),
            "subsample": trial.suggest_float("subsample", SUBSAMPLE_MIN, SUBSAMPLE_MAX),
            "colsample_bytree": trial.suggest_float("colsample_bytree", COLSAMPLE_BYTREE_MIN, COLSAMPLE_BYTREE_MAX),
            "min_child_weight": trial.suggest_int("min_child_weight", MIN_CHILD_WEIGHT_MIN, MIN_CHILD_WEIGHT_MAX),
        }

        # Обучение модели
        xgb_model = XGBRegressor(**params)
        xgb_model.fit(X_train, y)
        x_input = create_x_input(df_train, lag)
        print(f'x_input = {x_input}')

        print(f'cols = {cols}')

        n_features = values.shape[1]

        print(f'n_features = {n_features}')
        print(f'lag = {lag}')

        x_input = x_input.reshape((1, lag, n_features))



        # Прогнозирование
        df_test = df_all_data_norm.iloc[last_known_index:].copy()
        df_real_predict = df_test.copy()
        df_test = df_test[[col_target] + cols]

        predict_values = _make_xgboost_predictions(x_input, df_test.values, len(cols) + 1, xgb_model, lag)
        df_real_predict[col_target] = np.array(predict_values).flatten()
        # Обратная нормализация прогнозов
        df_real_predict = t2v.light_reverse_vectorization(df_real_predict, min_val, max_val)
        y_true = df_all_data.iloc[last_known_index:][col_target].values
        y_pred = df_real_predict[col_target].values

        # Расчет метрики MAPE
        _, _, _, mape, _ = calculate_metrics(y_true=y_true, y_pred=y_pred)
        return mape

    # Запуск оптимизации
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    print('Завершение подбора параметров')
    print(f'Лучшие параметры: {best_params}')

    # Вернуть только лучшие параметры
    return {"best_params": best_params}
