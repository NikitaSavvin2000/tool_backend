#src/services/feature_selection
import pandas as pd
import numpy as np
import yaml

from tqdm import tqdm
from typing import List, Dict
from pathlib import Path
from src.normalization.time2vec import Time2Vec
from src.processing.data_processing import calculate_time_interval
from src.models.xgboost_model import forecast_XGBoost_sistem
from src.utils.metrics import calculate_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # Корень проекта
CONFIG_DIR = PROJECT_ROOT / "src" / "configuration"

MAX_SEARCH_LAG = 21  # Максимальный лаг для поиска

def load_possible_cols():
    config_path = CONFIG_DIR / "possible_cols.yaml"
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config.get('all_possible_cols', [])
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

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
    all_possible_cols = load_possible_cols()

    original_column = df_init[time_column].copy()
    df_init[time_column] = pd.to_datetime(df_init[time_column], errors='coerce')
    df_init = df_init.sort_values(by=time_column).reset_index(drop=True)
    df_init[time_column] = original_column[df_init.index]

    last_value = df_init[col_target].iloc[0]
    optimal_evaluation_points = 300
    if len(df_init) < optimal_evaluation_points / 0.1:
        optimal_evaluation_points = int(len(df_init) * 0.1)
    df_evaluation = df_init[-optimal_evaluation_points:]
    df = df_init[:-optimal_evaluation_points]
    df_empty = df_evaluation.copy()
    df_empty[col_target] = None

    df_all_data = pd.concat([df, df_empty], ignore_index=True).sort_values(by=time_column).reset_index(drop=True)
    last_known_index = len(df_all_data) - optimal_evaluation_points

    t2v = Time2Vec(col_time=time_column, col_target=col_target)
    df_all_data_norm, min_val, max_val = t2v.vectorization(df_all_data)
    df_all_data_norm = df_all_data_norm.sort_values(by=time_column).reset_index(drop=True)

    col_for_train = []
    best_mape = float('inf')

    df_evaluation.loc[:, time_column] = pd.to_datetime(df_evaluation[time_column], errors='coerce')
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
    """
    if len(df_init) < 2:
        raise ValueError("Для подбора лага требуется минимум 2 строки во входных данных.")

    df_init.loc[:, time_column] = pd.to_datetime(df_init[time_column], errors="coerce")
    df_init = df_init.sort_values(by=time_column).reset_index(drop=True)

    optimal_evaluation_points = min(300, len(df_init) // 2)
    if len(df_init) <= optimal_evaluation_points:
        raise ValueError("Недостаточно данных для разделения на обучающую и тестовую выборки.")

    df_evaluation = df_init[-optimal_evaluation_points:].copy()
    df = df_init[:-optimal_evaluation_points].copy()

    df_empty = df_evaluation.copy()
    df_empty[col_target] = None

    df_all_data = pd.concat([df, df_empty.dropna(how="all")], ignore_index=True)
    df_all_data = df_all_data.sort_values(by=time_column).reset_index(drop=True)

    t2v = Time2Vec(col_time=time_column, col_target=col_target)
    df_all_data_norm, min_val, max_val = t2v.vectorization(df_all_data)

    best_lag = None
    best_mape = float('inf')

    df_evaluation.loc[:, time_column] = pd.to_datetime(df_evaluation[time_column], errors="coerce")
    df_evaluation = df_evaluation.sort_values(by=time_column).reset_index(drop=True)

    max_lag = min(len(df) - 1, MAX_SEARCH_LAG)
    for lag in tqdm(range(1, max_lag + 1)):
        df_true_all, df_pred_vector = forecast_XGBoost_sistem(
            col_target=col_target,
            time_column=time_column,
            df_all_data_norm=df_all_data_norm,
            last_known_index=len(df_all_data) - optimal_evaluation_points,
            lag=lag,
            model_architecture_params=[{"objective": "reg:squarederror"}],
            col_for_train=cols
        )

        if df_pred_vector.size == 0:
            raise ValueError("Forecast returned empty predictions. Check the input data and model configuration.")

        df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)
        df_real_predict[col_target] = df_real_predict[col_target].astype('float64')
        df_real_predict.at[df_real_predict.index[-1], col_target] = df_init[col_target].iloc[0]
        df_real_predict[time_column] = df_evaluation[time_column]

        y_true = df_evaluation[col_target].reset_index(drop=True)
        y_pred = df_real_predict[col_target].reset_index(drop=True)

        _, _, _, mape, _ = calculate_metrics(y_true=y_true, y_pred=y_pred)
        print(f"CURRENT MAPE = {mape} | BEST LAG = {best_lag}")

        if mape < best_mape:
            best_mape = mape
            best_lag = lag

    return {"best_lag": best_lag, "best_mape": best_mape}