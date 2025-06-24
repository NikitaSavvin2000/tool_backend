# src/xgboost_selection_of_parameters/main.py

import pandas as pd
from src.normalization.time2vec import Time2Vec
from src.processing.data_processing import calculate_time_interval
from src.services.feature_selection import col_selection_xgboots, lag_selection_xgboots
from src.models.xgboost_model import forecast_XGBoost_sistem
from src.utils.metrics import calculate_metrics
from src.utils.date_utils import standardize_datetime


def user_predict_XGBoost(
    df: pd.DataFrame,
    time_column: str,
    col_target: str,
    forecast_horizon_time: str,
) -> dict:
    """
    Генерирует прогноз временного ряда с использованием XGBoost.
    
    :param df: Исходный DataFrame с временным рядом.
    :param time_column: Название колонки с временными метками.
    :param col_target: Название целевой переменной.
    :param forecast_horizon_time: Временная граница прогнозирования.
    :return: Словарь с прогнозными данными.
    """
    # Стандартизация границ прогнозирования
    forecast_horizon_time = standardize_datetime(forecast_horizon_time)

    # Выбор оптимальных признаков
    data_cols = col_selection_xgboots(
        df_init=df,
        time_column=time_column,
        col_target=col_target,
    )
    col_for_train = data_cols["col_for_train"]

    # Подбор оптимального значения лага
    data_lag = lag_selection_xgboots(
        df_init=df,
        time_column=time_column,
        col_target=col_target,
        cols=col_for_train,
    )
    lag = data_lag["best_lag"]  # Определяем переменную lag

    # Сохранение оригинального формата временной колонки
    original_format = df[time_column].copy()
    df.loc[:, time_column] = pd.to_datetime(df[time_column], errors="coerce")
    df = df.sort_values(by=time_column, ascending=True).reset_index(drop=True)
    df[time_column] = original_format

    # Определение временного интервала
    last_value = df[[time_column, col_target]].iloc[-1]
    last_known_data = df.iloc[-1][time_column]
    time_point_interval = abs(calculate_time_interval(df, time_column))

    # Создание будущих временных меток
    last_time = df[time_column].iloc[-1]
    date_range = pd.date_range(
        start=last_time,
        end=forecast_horizon_time,
        freq=f"{int(time_point_interval)}s",
    )
    date_range = date_range[1:]
    df_future = pd.DataFrame({time_column: date_range, col_target: [None] * len(date_range)})
    df = df.dropna(axis=1, how="all")
    df_future = df_future.dropna(axis=1, how="all")
    df_all_data = pd.concat([df, df_future], ignore_index=True)
    df_all_data = df_all_data.sort_values(by=time_column, ascending=True).reset_index(drop=True)
    last_known_index = len(df_all_data) - len(date_range)

    # Нормализация данных
    t2v = Time2Vec(col_time=time_column, col_target=col_target)
    df_all_data_norm, min_val, max_val = t2v.vectorization(df_all_data)

    # Прогнозирование
    df_true_all, df_pred_vector = forecast_XGBoost_sistem(
        col_target=col_target,
        time_column=time_column,
        df_all_data_norm=df_all_data_norm,
        last_known_index=last_known_index,
        lag=lag,  # Используем определенную переменную lag
        model_architecture_params={
            "objective": "reg:squarederror",
            "n_estimators": 500,
            "learning_rate": 0.1,
            "max_depth": 15,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 5,
            "booster": "gbtree",
        },
        col_for_train=col_for_train,
    )

    # Обратная нормализация прогнозов
    df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)
    df_real_predict[time_column] = date_range
    df_real_predict = df_real_predict.reset_index(drop=True)

    # Добавление последней известной записи
    new_row = pd.DataFrame(
        {
            time_column: [pd.to_datetime(last_value[time_column])],
            col_target: [float(last_value[col_target])],
        }
    )
    df_real_predict = pd.concat([new_row, df_real_predict]).reset_index(drop=True)

    # Стандартизация временных меток
    df[time_column] = df[time_column].apply(lambda x: standardize_datetime(str(x)))
    df_real_predict[time_column] = df_real_predict[time_column].apply(lambda x: standardize_datetime(str(x)))

    # Подготовка результатов
    last_real_data = df.to_dict(orient="records")
    predictions = df_real_predict.to_dict(orient="records")

    return {
        "map_data": {
            "data": {
                "last_real_data": last_real_data,
                "predictions": predictions,
            },
            "last_know_data": last_known_data,
            "title": f"Реальный прогноз {col_target}",
            "legend": {
                "last_know_data_line": {
                    "text": {
                        "en": "Last known date",
                        "ru": "Последняя известная дата",
                    },
                    "color": "#A9A9A9",
                },
                "real_data_line": {
                    "text": {
                        "en": "Real data",
                        "ru": "Реальные данные",
                    },
                    "color": "#0000FF",
                },
                "predict_data_line": {
                    "text": {
                        "en": "Current forecast",
                        "ru": "Актуальный прогноз",
                    },
                    "color": "#FF0000",
                },
            },
        }
    }