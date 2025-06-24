#src/services/feature_selection.py
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from src.normalization.time2vec import Time2Vec
from src.processing.data_processing import calculate_time_interval
from src.models.xgboost_model import forecast_XGBoost_sistem
from src.utils.metrics import calculate_metrics

def col_selection_xgboots(
        df_init: pd.DataFrame,
        time_column: str,
        col_target: str,
        lag: int = 1,
) -> Dict[str, List[str]]:
    """
    Выполняет выбор оптимальных признаков для прогнозирования с использованием XGBoost.
    
    :param df_init: Исходный DataFrame.
    :param time_column: Название колонки с временными метками.
    :param col_target: Название целевой переменной.
    :param lag: Количество временных шагов для лаговых признаков.
    :return: Словарь с выбранными признаками и значением MAPE.
    """
    # Сохранение оригинального формата временной колонки
    original_column = df_init[time_column].copy()
    df_init[time_column] = pd.to_datetime(df_init[time_column], errors='coerce')
    df_init = df_init.sort_values(by=time_column).reset_index(drop=True)
    df_init[time_column] = original_column[df_init.index]
    
    # Определение размера тестовой выборки
    last_value = df_init[col_target].iloc[0]
    optimal_evaluation_points = 300
    if len(df_init) < optimal_evaluation_points / 0.1:
        optimal_evaluation_points = int(len(df_init) * 0.1)
    df_evaluation = df_init[-optimal_evaluation_points:]
    df = df_init[:-optimal_evaluation_points]
    df_empty = df_evaluation.copy()
    df_empty[col_target] = None
    
    # Объединение данных для нормализации
    df_all_data = pd.concat([df, df_empty], ignore_index=True).sort_values(by=time_column).reset_index(drop=True)
    last_known_index = len(df_all_data) - optimal_evaluation_points
    
    # Нормализация данных
    t2v = Time2Vec(col_time=time_column, col_target=col_target)
    df_all_data_norm, min_val, max_val = t2v.vectorization(df_all_data)
    df_all_data_norm = df_all_data_norm.sort_values(by=time_column).reset_index(drop=True)
    
    # Список всех возможных признаков
    all_possible_cols = [
        "year", "month", "day", "week", "day_of_week", "hour", "minute", "second",
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
        "week_sin", "week_cos", "month_sin", "month_cos",
        "part_of_day", "is_night", "is_weekend", "day_of_year",
        "is_working_hours", "season", "season_sin", "season_cos",
        "quarter", "quarter_sin", "quarter_cos", "moon_phase",
    ]
    
    # Поиск лучших признаков
    col_for_train = []
    best_mape = float('inf')
    df_evaluation[time_column] = pd.to_datetime(df_evaluation[time_column], errors='coerce')
    df_evaluation = df_evaluation.sort_values(by=time_column).reset_index(drop=True)
    
    for col in tqdm(all_possible_cols):
        current_cols = col_for_train + [col]
        df_true_all, df_pred_vector = forecast_XGBoost_sistem(
            col_target=col_target,
            time_column=time_column,
            df_all_data_norm=df_all_data_norm,
            last_known_index=last_known_index,
            lag=lag,
            model_architecture_params=[{"objective": "reg:squarederror"}],
            col_for_train=current_cols
        )
        df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)
        df_real_predict[col_target] = df_real_predict[col_target].astype('float64')
        df_real_predict.at[df_real_predict.index[-1], col_target] = last_value
        df_real_predict[time_column] = df_evaluation[time_column]
        
        y_true = df_evaluation[col_target].reset_index(drop=True)
        y_pred = df_real_predict[col_target].reset_index(drop=True)
        _, _, _, mape, _ = calculate_metrics(y_true=y_true, y_pred=y_pred)
        
        print(f"CURRENT MAPE = {mape} | BEST MAPE = {best_mape}")
        if mape < best_mape:
            best_mape = mape
            col_for_train.append(col)
    
    return {"col_for_train": col_for_train, "best_mape": best_mape}

def lag_selection_xgboots(
        df_init: pd.DataFrame,
        time_column: str,
        col_target: str,
        cols: List[str]
) -> Dict[str, int]:
    """
    Выполняет подбор оптимального значения лага для прогнозирования.
    
    :param df_init: Исходный DataFrame.
    :param time_column: Название колонки с временными метками.
    :param col_target: Название целевой переменной.
    :param cols: Список признаков для обучения.
    :return: Словарь с оптимальным значением лага и значением MAPE.
    """
    # Сохранение оригинального формата временной колонки
    original_column = df_init[time_column].copy()
    df_init[time_column] = pd.to_datetime(df_init[time_column], errors='coerce')
    df_init = df_init.sort_values(by=time_column).reset_index(drop=True)
    df_init[time_column] = original_column[df_init.index]
    
    # Определение размера тестовой выборки
    last_value = df_init[col_target].iloc[0]
    optimal_evaluation_points = 300
    if len(df_init) < optimal_evaluation_points / 0.1:
        optimal_evaluation_points = int(len(df_init) * 0.1)
    df_evaluation = df_init[-optimal_evaluation_points:]
    df = df_init[:-optimal_evaluation_points]
    df_empty = df_evaluation.copy()
    df_empty[col_target] = None
    
    # Объединение данных для нормализации
    df_all_data = pd.concat([df, df_empty], ignore_index=True).sort_values(by=time_column).reset_index(drop=True)
    last_known_index = len(df_all_data) - optimal_evaluation_points
    
    # Нормализация данных
    t2v = Time2Vec(col_time=time_column, col_target=col_target)
    df_all_data_norm, min_val, max_val = t2v.vectorization(df_all_data)
    df_all_data_norm = df_all_data_norm.sort_values(by=time_column).reset_index(drop=True)
    
    # Поиск оптимального значения лага
    best_lag = None
    best_mape = float('inf')
    df_evaluation[time_column] = pd.to_datetime(df_evaluation[time_column], errors='coerce')
    df_evaluation = df_evaluation.sort_values(by=time_column).reset_index(drop=True)
    
    for lag in tqdm(range(1, 22)):
        df_true_all, df_pred_vector = forecast_XGBoost_sistem(
            col_target=col_target,
            time_column=time_column,
            df_all_data_norm=df_all_data_norm,
            last_known_index=last_known_index,
            lag=lag,
            model_architecture_params=[{"objective": "reg:squarederror"}],
            col_for_train=cols
        )
        df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)
        df_real_predict[col_target] = df_real_predict[col_target].astype('float64')
        df_real_predict.at[df_real_predict.index[-1], col_target] = last_value
        df_real_predict[time_column] = df_evaluation[time_column]
        
        y_true = df_evaluation[col_target].reset_index(drop=True)
        y_pred = df_real_predict[col_target].reset_index(drop=True)
        _, _, _, mape, _ = calculate_metrics(y_true=y_true, y_pred=y_pred)
        
        print(f"CURRENT MAPE = {mape} | BEST LAG = {best_lag}")
        if mape < best_mape:
            best_mape = mape
            best_lag = lag
    
    return {"best_lag": best_lag, "best_mape": best_mape}