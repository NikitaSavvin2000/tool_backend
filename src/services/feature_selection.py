# src/services/feature_selection.py
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from src.models.xgboost_model import forecast_XGBoost_sistem
from src.normalization.time2vec import Time2Vec
from src.utils.metrics import calculate_metrics
from src.utils.possible_cols import load_possible_cols

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # Корень проекта
CONFIG_DIR = PROJECT_ROOT / "src" / "configuration"


model_architecture_params = [{
    "objective": "reg:squarederror",
    "device": "cuda",
    "tree_method": "hist",
    "n_estimators": 1000,
    "learning_rate": 0.1,
    "max_depth": 15,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 5,
    "booster": "gbtree"
}]


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

    # for col in tqdm(all_possible_cols):
    best_errors = {}
    for col in tqdm(all_possible_cols, bar_format='{l_bar}{n_fmt}/{total_fmt} ({percentage:3.0f}%)'):
        current_cols = col_for_train + [col]
        df_true_all, df_pred_vector = forecast_XGBoost_sistem(
            col_target=col_target,
            time_column=time_column,
            df_all_data_norm=df_all_data_norm,
            last_known_index=last_known_index,
            lag=lag,
            model_architecture_params=model_architecture_params,
            col_for_train=current_cols
        )
        df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)
        df_real_predict[col_target] = df_real_predict[col_target].astype('float64')
        df_real_predict.at[df_real_predict.index[-1], col_target] = last_value
        df_real_predict[time_column] = df_evaluation[time_column]

        y_true = df_evaluation[col_target].reset_index(drop=True)
        y_pred = df_real_predict[col_target].reset_index(drop=True)
        rmse, r2, mae, mape, wmape = calculate_metrics(y_true=y_true, y_pred=y_pred)

        print(f">>> BEST MAPE = {round(best_mape, 3)} BEST COLS = {col_for_train} | CURRENT MAPE = {round(mape, 3)} | CUR COLS =  {current_cols}")
        logger.info(f">>> BEST MAPE = {round(best_mape, 3)} BEST COLS = {col_for_train} | CURRENT MAPE = {round(mape, 3)} | CUR COLS =  {current_cols}")

        if mape < best_mape:
            best_mape = mape
            col_for_train.append(col)
            best_errors["RMSE"] = rmse
            best_errors["R2"] = r2
            best_errors["MAE"] = mae
            best_errors["MAPE"] = mape
            best_errors["WMAPE"] = wmape



    return {"col_for_train": col_for_train, "errors": best_errors}


def n_estimators_selection_xgboots(
        df_init: pd.DataFrame,
        time_column: str,
        col_target: str,
        col_for_train: list,
        model_params: dict,
        lag: int = 1,
        min_n_estimators: int = 300,
        max_n_estimators: int = 3000,
        start_step: int = 5,
        max_step: int = 100,
) -> Dict[str, float]:

    df_init.loc[:, time_column] = pd.to_datetime(df_init[time_column], errors="coerce")
    df_init = df_init.sort_values(by=time_column).reset_index(drop=True)

    optimal_evaluation_points = min(300, len(df_init) // 2)
    df_evaluation = df_init[-optimal_evaluation_points:].copy()
    df = df_init[:-optimal_evaluation_points].copy()

    df_empty = df_evaluation.copy()
    df_empty[col_target] = None

    df_all_data = pd.concat([df, df_empty.dropna(how="all")], ignore_index=True)
    df_all_data = df_all_data.sort_values(by=time_column).reset_index(drop=True)

    t2v = Time2Vec(col_time=time_column, col_target=col_target)
    df_all_data_norm, min_val, max_val = t2v.vectorization(df_all_data)

    df_evaluation.loc[:, time_column] = pd.to_datetime(df_evaluation[time_column], errors="coerce")
    df_evaluation = df_evaluation.sort_values(by=time_column).reset_index(drop=True)

    n_estimators = min_n_estimators
    step = start_step
    best_mape = float('inf')
    best_n_estimators = n_estimators

    while n_estimators <= max_n_estimators:
        params = model_params.copy()
        params["n_estimators"] = n_estimators

        df_true_all, df_pred_vector = forecast_XGBoost_sistem(
            col_target=col_target,
            time_column=time_column,
            df_all_data_norm=df_all_data_norm,
            last_known_index=len(df_all_data) - optimal_evaluation_points,
            lag=lag,
            model_architecture_params=[params],
            col_for_train=col_for_train
        )

        df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)
        y_true = df_evaluation[col_target].reset_index(drop=True)
        y_pred = df_real_predict[col_target].reset_index(drop=True)

        _, _, _, mape, _ = calculate_metrics(y_true=y_true, y_pred=y_pred)

        print(f"n_estimators={n_estimators} | MAPE={mape} | Best MAPE={best_mape} with n_estimators={best_n_estimators}")

        if mape < best_mape:
            best_mape = mape
            best_n_estimators = n_estimators
            step = start_step  # при улучшении сбрасываем шаг в маленький
        else:
            step = min(step * 2, max_step)  # при ухудшении увеличиваем шаг

        n_estimators += step

    return {"n_estimators": best_n_estimators, "best_mape": best_mape}

def learning_rate_selection_xgboots(
        df_init: pd.DataFrame,
        time_column: str,
        col_target: str,
        col_for_train: list,
        model_params: dict,
        lag: int = 1,
        min_lr: float = 0.001,
        max_lr: float = 0.5,
        start_step: float = 0.005,
        max_step: float = 0.05,
) -> Dict[str, float]:

    df_init.loc[:, time_column] = pd.to_datetime(df_init[time_column], errors="coerce")
    df_init = df_init.sort_values(by=time_column).reset_index(drop=True)

    optimal_evaluation_points = min(300, len(df_init) // 2)
    df_evaluation = df_init[-optimal_evaluation_points:].copy()
    df = df_init[:-optimal_evaluation_points].copy()

    df_empty = df_evaluation.copy()
    df_empty[col_target] = None

    df_all_data = pd.concat([df, df_empty.dropna(how="all")], ignore_index=True)
    df_all_data = df_all_data.sort_values(by=time_column).reset_index(drop=True)

    t2v = Time2Vec(col_time=time_column, col_target=col_target)
    df_all_data_norm, min_val, max_val = t2v.vectorization(df_all_data)

    df_evaluation.loc[:, time_column] = pd.to_datetime(df_evaluation[time_column], errors="coerce")
    df_evaluation = df_evaluation.sort_values(by=time_column).reset_index(drop=True)

    lr = min_lr
    step = start_step
    best_mape = float('inf')
    best_lr = lr

    while lr <= max_lr:
        params = model_params.copy()
        params["learning_rate"] = lr

        df_true_all, df_pred_vector = forecast_XGBoost_sistem(
            col_target=col_target,
            time_column=time_column,
            df_all_data_norm=df_all_data_norm,
            last_known_index=len(df_all_data) - optimal_evaluation_points,
            lag=lag,
            model_architecture_params=[params],
            col_for_train=col_for_train
        )

        df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)
        y_true = df_evaluation[col_target].reset_index(drop=True)
        y_pred = df_real_predict[col_target].reset_index(drop=True)

        _, _, _, mape, _ = calculate_metrics(y_true=y_true, y_pred=y_pred)

        print(f"learning_rate={lr:.5f} | MAPE={mape:.5f} | Best MAPE={best_mape:.5f} with learning_rate={best_lr:.5f}")

        if mape < best_mape:
            best_mape = mape
            best_lr = lr
            step = start_step
        else:
            step = min(step * 2, max_step)

        lr += step

    return {"learning_rate": best_lr, "best_mape": best_mape}



def max_depth_selection_xgboots(
        df_init: pd.DataFrame,
        time_column: str,
        col_target: str,
        col_for_train: list,
        model_params: dict,
        lag: int = 1,
        min_depth: int = 5,
        max_depth: int = 30,
        step: int = 1,
) -> Dict[str, float]:

    df_init.loc[:, time_column] = pd.to_datetime(df_init[time_column], errors="coerce")
    df_init = df_init.sort_values(by=time_column).reset_index(drop=True)

    optimal_evaluation_points = min(300, len(df_init) // 2)
    df_evaluation = df_init[-optimal_evaluation_points:].copy()
    df = df_init[:-optimal_evaluation_points].copy()

    df_empty = df_evaluation.copy()
    df_empty[col_target] = None

    df_all_data = pd.concat([df, df_empty.dropna(how="all")], ignore_index=True)
    df_all_data = df_all_data.sort_values(by=time_column).reset_index(drop=True)

    t2v = Time2Vec(col_time=time_column, col_target=col_target)
    df_all_data_norm, min_val, max_val = t2v.vectorization(df_all_data)

    df_evaluation.loc[:, time_column] = pd.to_datetime(df_evaluation[time_column], errors="coerce")
    df_evaluation = df_evaluation.sort_values(by=time_column).reset_index(drop=True)

    best_mape = float('inf')
    best_depth = min_depth

    d = min_depth
    while d <= max_depth:
        params = model_params.copy()
        params["max_depth"] = int(d)

        df_true_all, df_pred_vector = forecast_XGBoost_sistem(
            col_target=col_target,
            time_column=time_column,
            df_all_data_norm=df_all_data_norm,
            last_known_index=len(df_all_data) - optimal_evaluation_points,
            lag=lag,
            model_architecture_params=[params],
            col_for_train=col_for_train
        )

        df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)
        y_true = df_evaluation[col_target].reset_index(drop=True)
        y_pred = df_real_predict[col_target].reset_index(drop=True)

        _, _, _, mape, _ = calculate_metrics(y_true=y_true, y_pred=y_pred)

        print(f"max_depth={d} | MAPE={mape:.5f} | Best MAPE={best_mape:.5f} with max_depth={best_depth}")

        if mape < best_mape:
            best_mape = mape
            best_depth = d

        d += step

    return {"max_depth": best_depth, "best_mape": best_mape}

#
#  def lag_selection_xgboots(
#         df_init: pd.DataFrame,
#         time_column: str,
#         col_target: str,
#         cols: List[str]
# ) -> Dict[str, int]:
#     if len(df_init) < 2:
#         raise ValueError("Для подбора лага требуется минимум 2 строки во входных данных.")
#
#     df_init[time_column] = pd.to_datetime(df_init[time_column], errors="coerce")
#     df_init = df_init.sort_values(by=time_column).reset_index(drop=True)
#
#     optimal_evaluation_points = min(300, len(df_init) // 2)
#     if len(df_init) <= optimal_evaluation_points:
#         raise ValueError("Недостаточно данных для разделения на обучающую и тестовую выборки.")
#
#     df_evaluation = df_init[-optimal_evaluation_points:].copy()
#     df = df_init[:-optimal_evaluation_points].copy()
#
#     df_empty = df_evaluation.copy()
#     df_empty[col_target] = None
#
#     df_all_data = pd.concat([df, df_empty.dropna(how="all")], ignore_index=True)
#     df_all_data = df_all_data.sort_values(by=time_column).reset_index(drop=True)
#
#     print(f'[INFO] Time2Vec is working')
#
#     t2v = Time2Vec(col_time=time_column, col_target=col_target)
#     df_all_data_norm, min_val, max_val = t2v.vectorization(df_all_data)
#
#     df_evaluation[time_column] = pd.to_datetime(df_evaluation[time_column], errors="coerce")
#     df_evaluation = df_evaluation.sort_values(by=time_column).reset_index(drop=True)
#
#     max_lag = min(len(df) - 1, MAX_SEARCH_LAG)
#
#     best_lag = None
#     best_mape = float('inf')
#
#     async def evaluate_lag(lag: int):
#         df_true_all, df_pred_vector = await asyncio.to_thread(
#             forecast_XGBoost_sistem,
#             col_target,
#             time_column,
#             df_all_data_norm,
#             len(df_all_data) - optimal_evaluation_points,
#             lag,
#             model_architecture_params,
#             cols
#         )
#
#         if df_pred_vector.size == 0:
#             raise ValueError("Forecast returned empty predictions.")
#
#         df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)
#         df_real_predict[col_target] = df_real_predict[col_target].astype('float64')
#         df_real_predict.at[df_real_predict.index[-1], col_target] = df_init[col_target].iloc[0]
#         df_real_predict[time_column] = df_evaluation[time_column]
#
#         y_true = df_evaluation[col_target].reset_index(drop=True)
#         y_pred = df_real_predict[col_target].reset_index(drop=True)
#
#         _, _, _, mape, _ = await asyncio.to_thread(calculate_metrics, y_true, y_pred)
#
#         print(f"CURRENT MAPE = {mape} | CURRENT LAG = {lag}")
#         logger.info(f"CURRENT MAPE = {mape} | CURRENT LAG = {lag}")
#
#         return lag, mape
#
#     print(f'[INFO] tasks is working')
#
#     tasks = [evaluate_lag(lag) for lag in range(1, max_lag + 1)]
#     results = await asyncio.gather(*tasks)
#
#     for lag, mape in results:
#         if mape < best_mape:
#             best_mape = mape
#             best_lag = lag
#
#     return {"best_lag": best_lag, "best_mape": best_mape}

def lag_selection_xgboots(
        df_init: pd.DataFrame,
        time_column: str,
        col_target: str,
        cols: List[str],
        lag_search_depth: int
) -> Dict[str, int]:
    if len(df_init) < 2:
        raise ValueError("Для подбора лага требуется минимум 2 строки во входных данных.")

    df_init[time_column] = pd.to_datetime(df_init[time_column], errors="coerce")
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

    print('[INFO] Time2Vec is working')

    t2v = Time2Vec(col_time=time_column, col_target=col_target)
    df_all_data_norm, min_val, max_val = t2v.vectorization(df_all_data)

    df_evaluation[time_column] = pd.to_datetime(df_evaluation[time_column], errors="coerce")
    df_evaluation = df_evaluation.sort_values(by=time_column).reset_index(drop=True)

    max_lag = min(len(df) - 1, lag_search_depth)

    best_lag = None
    best_mape = float('inf')

    print('[INFO] lag evaluation is working')

    for lag in tqdm(range(1, max_lag + 1), bar_format='{l_bar}{n_fmt}/{total_fmt} ({percentage:3.0f}%)'):
        df_true_all, df_pred_vector = forecast_XGBoost_sistem(
            col_target,
            time_column,
            df_all_data_norm,
            len(df_all_data) - optimal_evaluation_points,
            lag,
            model_architecture_params,
            cols
        )

        if df_pred_vector.size == 0:
            raise ValueError("Forecast returned empty predictions.")

        df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)
        df_real_predict[col_target] = df_real_predict[col_target].astype('float64')
        df_real_predict.at[df_real_predict.index[-1], col_target] = df_init[col_target].iloc[0]
        df_real_predict[time_column] = df_evaluation[time_column]

        y_true = df_evaluation[col_target].reset_index(drop=True)
        y_pred = df_real_predict[col_target].reset_index(drop=True)

        metrix = calculate_metrics(y_true, y_pred)
        mape = metrix["MAPE"]

        print(f">>> CURRENT MAPE = {round(mape, 3)}  CURRENT LAG = {lag} | BEST MAPE = {round(best_mape, 3)} BEST LAG = {best_lag}")
        logger.info(f">>> CURRENT MAPE = {round(mape, 3)}  CURRENT LAG = {lag} | BEST MAPE = {round(best_mape, 3)} BEST LAG = {best_lag}")

        if mape < best_mape:
            best_mape = mape
            best_lag = lag

    return {"best_lag": best_lag, "best_mape": best_mape}
