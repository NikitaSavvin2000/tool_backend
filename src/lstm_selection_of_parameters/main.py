import os

import pandas as pd
from src.normalization.time2vec import Time2Vec
from src.core.utils.possible_forecast_date import calculate_time_interval
from src.lstm_selection_of_parameters.feature_selection import (get_lstm_lag, get_points_per_call,
                                                                col_selection_lstm, forecast_LSTM_user)
from src.core.utils.possible_cols import load_possible_cols

home_path = os.getcwd()


# Валидация входных данных
def validate_input_data(df: pd.DataFrame) -> None:
    """
    Проверяет входные данные на соответствие минимальным требованиям.

    :param df: Исходный DataFrame с временными метками и данными.
    :raises ValueError: Если количество строк меньше 2.
    """
    if len(df) < 2:
        raise ValueError("Входной временной ряд должен содержать минимум 2 наблюдения.")


model_architecture_params= {
    "architecture": [{"layer": 1, "type": "Bi-LSTM", "neurons": 6},
                     {"layer": 2, "type": "Bi-LSTM", "neurons": 4},
                     {"layer": 3, "type": "Bi-LSTM", "neurons": 2}],
    "dropout_count": 0.1,
    "activation": "relu",
    "optimizer": "adam"
}

async def user_predict_LSTM(
        df: pd.DataFrame,
        time_column: str,
        col_target: str,
        forecast_horizon_time: str,
) -> dict:
    """
    Генерирует прогноз временного ряда с использованием XGBoost.

    :param df: Исходный DataFrame с временным рядом
    :param time_column: Название колонки с временными метками
    :param col_target: Название целевой переменной
    :param forecast_horizon_time: Временная граница прогнозирования
    :param lag: Количество временных лагов
    :param forecast_type: Тип прогноза
    :param norm_values: Флаг нормализации значений
    :return: Словарь с прогнозными данными
    """
    debag = False
    res = get_lstm_lag(df_init=df, time_column=time_column, col_target=col_target, debag=debag)
    lag = res["best_lag"]
    # lag = 1
    res = get_points_per_call(df_init=df, time_column=time_column, col_target=col_target, lag=lag, debag=debag)

    points_per_call = res["best_points_per_call"]

    # res = col_selection_lstm(df_init=df,
    #                                    col_target=col_target,
    #                                    time_column=time_column,
    #                                    lag=lag,
    #                                    points_per_call=points_per_call,
    #                                    debag=debag
    #                                    )
    #
    # col_for_train = res["col_for_train"]
    col_for_train = load_possible_cols()
    errors = {"mape": res["best_mape"]}

    original_format = df[time_column].copy()
    df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
    df = df.sort_values(by=time_column, ascending=True).reset_index(drop=True)
    df[time_column] = original_format

    last_value = df[[time_column, col_target]].iloc[-1]
    last_known_data = df.iloc[-1][time_column]
    time_point_interval = abs(calculate_time_interval(df, time_column))
    last_time = df[time_column].iloc[-1]


    date_range = pd.date_range(
        start=last_time,
        end=forecast_horizon_time,
        freq=f'{int(time_point_interval)}s'
    )

    date_range = date_range[1:]

    df_future = pd.DataFrame({time_column: date_range, col_target: [None] * len(date_range)})

    df_all_data = pd.concat([df, df_future], ignore_index=True)


    df_all_data = df_all_data.sort_values(by=time_column, ascending=True).reset_index(drop=True)

    print(f"df_all_data-"*12)
    print(f"df_all_data = {df_all_data}")
    last_known_index = len(df_all_data) - len(date_range)

    t2v = Time2Vec(col_time=time_column, col_target=col_target)
    df_all_data_norm, min_val, max_val = t2v.vectorization(df_all_data)

    print(f"="*120)
    print(f"points_per_call = {points_per_call}")

    df_true_all, df_pred_vector = forecast_LSTM_user(
        col_target=col_target,
        time_column=time_column,
        df_all_data_norm=df_all_data_norm,
        last_known_index=last_known_index,
        lag=lag,
        model_architecture_params_user=model_architecture_params,
        col_for_train=col_for_train,
        points_per_call=points_per_call,
        epochs=3

    )

    df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)

    df_real_predict[time_column] = date_range

    df_real_predict = df_real_predict.reset_index(drop=True)

    print(df_real_predict)

    new_row = pd.DataFrame({
        time_column: [pd.to_datetime(last_value[time_column])],
        col_target: [float(last_value[col_target])]
    })

    df_real_predict = pd.concat([new_row, df_real_predict]).reset_index(drop=True)

    df[time_column] = df[time_column].dt.strftime("%Y-%m-%d %H:%M:%S")

    df_real_predict[time_column] = df_real_predict[time_column].dt.strftime("%Y-%m-%d %H:%M:%S")
    last_real_data = df.to_dict(orient="records")
    predictions = df_real_predict.to_dict(orient="records")

    return {
        "map_data": {
            "data": {
                "last_real_data": last_real_data,
                "predictions": predictions,
            },
            "errors": errors,
            "last_know_data": last_known_data,
            "title": f"Реальный прогноз {col_target}",
            "legend": {
                "last_know_data_line": {
                    "text": {
                        "en": "Last known date",
                        "ru": "Последняя известная дата"
                    },
                    "color": "#A9A9A9"
                },
                "real_data_line": {
                    "text": {
                        "en": "Real data",
                        "ru": "Реальные данные"
                    },
                    "color": "#0000FF"
                },
                "predict_data_line": {
                    "text": {
                        "en": "Current forecast",
                        "ru": "Актуальный прогноз"
                    },
                    "color": "#FF0000"
                },
            },
        }
    }

