import numpy as np
import pandas as pd

from src.backend.xgb import forecast_XGBoost_user
from src.backend.normalization import Time2Vec
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm


MODEL_ARCHITECTURE_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 500,
    "learning_rate": 0.1,
    "max_depth": 15,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 5,
    "booster": "gbtree"
}

def calculate_metrics(y_true, y_pred):
    y_true_mean = y_true.mean()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true_mean) ** 2)

    r2 = 1 - (ss_res / ss_tot)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

    return rmse, r2, mae, mape, wmape


def calculate_time_interval(df: pd.DataFrame, time_column: str) -> int:
    """
    Вычисляет средний временной интервал в минутах между записями.

    :param df: DataFrame с временными метками
    :param time_column: Название колонки с временными метками
    :return: Средний временной интервал в минутах
    """
    df[time_column] = pd.to_datetime(df[time_column])
    time_interval = df[time_column].diff().dt.total_seconds().mean()
    return round(time_interval)



def col_selection_xgboots(
        df_init: pd.DataFrame,
        time_column: str,
        col_target: str,
        lag: int = 1,
) -> dict:
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

    all_possible_cols = [
        "year", "month", "day", "week", "day_of_week", "hour", "minute", "second",
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
        "week_sin", "week_cos", "month_sin", "month_cos",
        "part_of_day", "is_night", "is_weekend", "day_of_year",
        "is_working_hours", "season", "season_sin", "season_cos",
        "quarter", "quarter_sin", "quarter_cos", "moon_phase",
    ]

    col_for_train = []
    best_mape = float('inf')

    df_evaluation[time_column] = pd.to_datetime(df_evaluation[time_column], errors='coerce')
    df_evaluation = df_evaluation.sort_values(by=time_column).reset_index(drop=True)


    for col in tqdm(all_possible_cols):
        current_cols = col_for_train + [col]
        df_true_all, df_pred_vector = forecast_XGBoost_user(
            col_target=col_target,
            time_column=time_column,
            df_all_data_norm=df_all_data_norm,
            last_known_index=last_known_index,
            lag=lag,
            model_architecture_params=[MODEL_ARCHITECTURE_PARAMS],
            col_for_train=current_cols
        )

        df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)
        df_real_predict.iloc[-1, df_real_predict.columns.get_loc(col_target)] = last_value

        df_real_predict[time_column] = df_evaluation[time_column]
        print(df_real_predict)

        print(df_evaluation)

        y_true = df_evaluation[col_target].reset_index(drop=True)
        y_pred = df_real_predict[col_target].reset_index(drop=True)

        _, _, _, mape, _ = calculate_metrics(y_true=y_true, y_pred=y_pred)

        print(f"CURRENT MAPE = {mape} | BEST MAPE = {best_mape}")

        if mape < best_mape:
            best_mape = mape
            col_for_train.append(col)

    return {"col_for_train": col_for_train, "best_mape": best_mape}




# "Morocco Zone 1": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSgwB47qVFZcr1Aq--UWxZ6fDi9CGLZm-1i8QoMgfdaHUbV8EqSli3ayPxYYxD8kqfYYHD41uuNxbjZ/pub?gid=1952392108&single=true&output=csv",
    # "Morocco Zone 2": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQT1DfqAB5Yec8MIQ_E5A8w-SXNcRmTwbXsv2W-ZT1ZcXN_G83BHlb6QBgnWkO-MpH3oVgfLoE0SnLx/pub?gid=1952392108&single=true&output=csv",
    # "Morocco Zone 3": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQSHw5k7n3_RM6ksGbvdQJsa1i9-zF-18CFLCFnXFkCxQwqLcQ4Wu2_8EF2H1lF02ih2NLL9BDecFzQ/pub?gid=1952392108&single=true&output=csv",

df = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vSgwB47qVFZcr1Aq--UWxZ6fDi9CGLZm-1i8QoMgfdaHUbV8EqSli3ayPxYYxD8kqfYYHD41uuNxbjZ/pub?gid=1952392108&single=true&output=csv")
time_column = 'Datetime'
col_target = 'consumption'

dict = col_selection_xgboots(
    df_init=df,
    time_column=time_column,
    col_target=col_target,
)

print(dict)
