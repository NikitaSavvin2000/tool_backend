#src/services/hyperparameter_tuning.py
import pandas as pd
import numpy as np
from xgboost import DMatrix, cv
import optuna
from typing import Dict, List
from src.processing.data_processing import split_sequence
from src.normalization.time2vec import Time2Vec
from src.models.xgboost_model import forecast_XGBoost_sistem
from src.utils.metrics import calculate_metrics

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
    :return: Словарь, содержащий фиксированный лаг, MAPE и лучшие параметры XGBoost.
    """
    # Сохранение оригинального формата временной колонки
    original_column = df_init[time_column].copy()
    df_init = df_init.copy()
    df_init[time_column] = pd.to_datetime(df_init[time_column], errors='coerce')
    df_init = df_init.sort_values(by=time_column).reset_index(drop=True)
    df_init[time_column] = original_column[df_init.index]
    
    # Определение размера тестовой выборки
    optimal_evaluation_points = 300
    if len(df_init) < optimal_evaluation_points / 0.1:
        optimal_evaluation_points = int(len(df_init) * 0.1)
    df_evaluation = df_init[-optimal_evaluation_points:]
    df = df_init[:-optimal_evaluation_points]
    
    # Подготовка полного набора данных с пустыми целевыми значениями для периода оценки
    df_empty = df_evaluation.copy()
    df_empty[col_target] = None
    df_all_data = pd.concat([df, df_empty], ignore_index=True).sort_values(by=time_column).reset_index(drop=True)
    last_known_index = len(df_all_data) - optimal_evaluation_points
    
    # Векторизация данных (предполагается, что Time2Vec определен в другом месте)
    t2v = Time2Vec(col_time=time_column, col_target=col_target)
    df_all_data_norm, min_val, max_val = t2v.vectorization(df_all_data)
    df_all_data_norm = df_all_data_norm.sort_values(by=time_column).reset_index(drop=True)
    
    # Подготовка обучающих данных для подбора гиперпараметров
    df_train = df_all_data_norm.iloc[:last_known_index].copy()
    values = df_train[[col_target] + cols].values
    X, y = split_sequence(values, lag)  # Предполагается, что split_sequence определен
    X_train = X.reshape(X.shape[0], -1)
    
    # Определение функции для подбора гиперпараметров
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "booster": "gbtree",
            "n_estimators": trial.suggest_int("n_estimators", 800, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42
        }
        dtrain = DMatrix(X_train, label=y)
        cv_results = cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=params["n_estimators"],
            nfold=3,
            metrics="mape",
            early_stopping_rounds=10,
            seed=42
        )
        return cv_results["test-mape-mean"].iloc[-1]
    
    # Запуск подбора гиперпараметров
    print('=' * 100)
    print('Начало подбора оптимальных параметров')
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    best_params = study.best_params
    best_params["objective"] = "reg:squarederror"
    best_params["booster"] = "gbtree"
    best_params["random_state"] = 42
    print('Завершение подбора параметров')
    print(f'Лучшие параметры: {best_params}')
    
    # Прогнозирование с использованием лучших параметров
    df_true_all, df_pred_vector = forecast_XGBoost_sistem(
        col_target=col_target,
        time_column=time_column,
        df_all_data_norm=df_all_data_norm,
        last_known_index=last_known_index,
        lag=lag,
        model_architecture_params=[best_params],  # Передача лучших параметров
        col_for_train=cols
    )
    
    # Обратная векторизация для прогнозов
    df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)
    df_real_predict[col_target] = df_real_predict[col_target].astype('float64')
    
    # Установка последнего значения (в соответствии с исходной логикой)
    last_value = df_init[col_target].iloc[0]
    df_real_predict.at[df_real_predict.index[-1], col_target] = last_value
    df_real_predict[time_column] = df_evaluation[time_column]
    
    # Расчет MAPE
    df_evaluation[time_column] = pd.to_datetime(df_evaluation[time_column], errors='coerce')
    df_evaluation = df_evaluation.sort_values(by=time_column).reset_index(drop=True)
    y_true = df_evaluation[col_target].reset_index(drop=True)
    y_pred = df_real_predict[col_target].reset_index(drop=True)
    _, _, _, mape, _ = calculate_metrics(y_true=y_true, y_pred=y_pred)
    
    print(f"Финальные результаты: {best_params}")
    return {"best_params": best_params, "mape": mape}