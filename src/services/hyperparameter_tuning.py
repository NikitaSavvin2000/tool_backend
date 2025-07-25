# src/services/hyperparameter_tuning.py

import pandas as pd
import numpy as np
from typing import Dict, Any
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from src.utils.metrics import calculate_metrics
from src.configuration.constants import (
    N_ESTIMATORS_MIN,
    N_ESTIMATORS_MAX,
    LEARNING_RATE_MIN,
    LEARNING_RATE_MAX,
    MAX_DEPTH_MIN,
    MAX_DEPTH_MAX,
    SUBSAMPLE_MIN,
    SUBSAMPLE_MAX,
    COLSAMPLE_BYTREE_MIN,
    COLSAMPLE_BYTREE_MAX,
    MIN_CHILD_WEIGHT_MIN,
    MIN_CHILD_WEIGHT_MAX,
)
from src.config import logger


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