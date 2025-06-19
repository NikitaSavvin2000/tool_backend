import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tqdm import tqdm

from src.backend.normalization import Time2Vec
from src.config import logger


home_path = os.getcwd()


def calculate_metrics(y_true, y_pred):
    y_true_mean = y_true.mean()
    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true_mean) ** 2)

        r2 = 1 - (ss_res / ss_tot)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    except Exception as e:
        print('='*150)
        print(e)
        print('='*150)
        print(y_pred)
        print('='*150)


    return rmse, r2, mae, mape, wmape


def split_sequence(sequence, n_steps, horizon):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        out_end_ix = end_ix + horizon
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix:out_end_ix, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def create_x_input(df_train, n_steps):
    df_input = df_train.iloc[len(df_train) - n_steps:]
    x_input = df_input.values
    return x_input


def make_predictions(x_input, x_future, points_per_call, model):
    predict_values = []
    x_future_len = len(x_future)
    remaining_horizon = x_future_len

    while remaining_horizon > 0:
        current_points_to_predict = min(remaining_horizon, points_per_call)
        x_input_tensor = tf.convert_to_tensor(x_input.reshape((1, x_input.shape[1], x_input.shape[2])), dtype=tf.float32)
        y_predict = model.predict(x_input_tensor, verbose=0)

        if len(y_predict.shape) == 2 and y_predict.shape[0] == 1:
            y_predict = y_predict[0]

        y_predict = y_predict[:current_points_to_predict]
        predict_values.extend(y_predict)

        for i in range(current_points_to_predict):
            cur_val = y_predict[i]
            x_input = np.delete(x_input, (0), axis=1)
            future_lag = x_future[0]
            x_future = np.delete(x_future, 0, axis=0)
            future_lag[0] = cur_val
            x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)

        remaining_horizon -= current_points_to_predict

    return predict_values


def calculate_time_interval(df: pd.DataFrame, time_column: str) -> int:
    """
    Вычисляет средний временной интервал в секундах между записями.

    :param df: DataFrame с временными метками
    :param time_column: Название колонки с временными метками
    :return: Средний временной интервал в минутах
    """
    df[time_column] = pd.to_datetime(df[time_column])
    time_interval = df[time_column].diff().dt.total_seconds().mean()
    return time_interval


def forecast_LSTM_user(
        col_target, time_column, df_all_data_norm, last_known_index, lag,
        model_architecture_params, col_for_train, points_per_call
):
    """
    Perform forecasting using XGBoost for regression.

    Parameters:
        col_target (str): Target column name.
        df_all_data_norm (pd.DataFrame): Normalized data.
        last_known_index (int): Last known data index.
        lag (int): Number of time steps for lagged features.
        model_architecture_params (dict): Parameters for XGBoost model.
        col_for_train (list): List of feature columns for training.

    Returns:
        tuple: (df_train, df_real_predict) DataFrames with true and predicted values.
    """

    df_all_data_norm[time_column] = pd.to_datetime(df_all_data_norm[time_column], errors='coerce')
    df_all_data_norm = df_all_data_norm.sort_values(by=time_column).reset_index(drop=True)

    col_for_train = [col_target] + col_for_train
    df_all_data_norm = df_all_data_norm[col_for_train].copy()

    df_all_data_norm[col_target] = df_all_data_norm[col_target].replace('None', None).astype(float)

    df_train = df_all_data_norm.iloc[:last_known_index]
    print('=============================================  forecast_XGBoost_user df_train ======================================')

    print(df_train)
    df_test = df_all_data_norm.iloc[last_known_index:].copy()
    print('=============================================  forecast_XGBoost_user df_test ======================================')
    print(df_test)

    df_test[col_target] = np.nan
    df_real_predict = df_test.copy()

    values = df_train[col_for_train].values
    x_input = create_x_input(df_train, lag)

    X, y = split_sequence(sequence=values, n_steps=lag, horizon=points_per_call)
    n_features = values.shape[1]

    architecture = model_architecture_params["architecture"]
    dropout_count = model_architecture_params["dropout_count"]
    activation = model_architecture_params["activation"]
    optimizer = model_architecture_params["optimizer"]


    model = Sequential()

    if len(architecture) == 3:
        model.add(Bidirectional(
            LSTM(int(architecture[0]['neurons']), activation=activation, return_sequences=True),
            input_shape=(lag, n_features)))
        model.add(Dropout(dropout_count))
        model.add(Bidirectional(
            LSTM(int(architecture[1]['neurons']), activation=activation, return_sequences=True)))
        model.add(Dropout(dropout_count))
        model.add(Bidirectional(LSTM(int(architecture[2]['neurons']), activation=activation)))
        model.add(Dropout(dropout_count))
        model.add(Dense(points_per_call))

    elif len(architecture) == 2:
        model.add(Bidirectional(
            LSTM(int(architecture[0]['neurons']), activation=activation, return_sequences=True),
            input_shape=(lag, n_features)))
        model.add(Dropout(dropout_count))
        model.add(Bidirectional(
            LSTM(int(architecture[1]['neurons']), activation=activation)))
        model.add(Dropout(dropout_count))
        model.add(Dense(points_per_call))

    elif len(architecture) == 1:
        model.add(Bidirectional(LSTM(int(architecture[0]['neurons']), activation=activation)))
        model.add(Dropout(dropout_count))
        model.add(Dense(points_per_call))

    else:
        model.add(Bidirectional(LSTM(32, activation='relu')))
        model.add(Dropout(0.01))
        model.add(Dense(points_per_call))

    model.compile(optimizer=optimizer, loss='mse')

    epochs = 2

    history = model.fit(X, y, epochs=epochs, verbose=1,)
    x_input = x_input.reshape((1, lag, n_features))

    predict_values = make_predictions(x_input=x_input, x_future=df_test.values, points_per_call=points_per_call, model=model)


    df_real_predict[col_target] = np.array(predict_values).flatten()
    for name, df in {'df_train': df_train, 'df_real_predict': df_real_predict}.items():
        none_indices = df[df.isnull().any(axis=1)].index.tolist()
        if none_indices:
            logger.error(f"DataFrame '{name}' contains None values at rows: {none_indices}")

    return df_train, df_real_predict


def forecast_LSTM_sistem(
        col_target, time_column, df_all_data_norm, last_known_index, lag,
        model_architecture_params, col_for_train
):
    """
    Perform forecasting using XGBoost for regression.

    Parameters:
        col_target (str): Target column name.
        df_all_data_norm (pd.DataFrame): Normalized data.
        last_known_index (int): Last known data index.
        lag (int): Number of time steps for lagged features.
        model_architecture_params (dict): Parameters for XGBoost model.
        col_for_train (list): List of feature columns for training.

    Returns:
        tuple: (df_train, df_real_predict) DataFrames with true and predicted values.
    """

    df_all_data_norm[time_column] = pd.to_datetime(df_all_data_norm[time_column], errors='coerce')
    df_all_data_norm = df_all_data_norm.sort_values(by=time_column).reset_index(drop=True)

    col_for_train = [col_target] + col_for_train
    df_all_data_norm = df_all_data_norm[col_for_train].copy()

    # Convert target column to float, handling 'None' strings
    df_all_data_norm[col_target] = df_all_data_norm[col_target].replace('None', None).astype(float)

    # Split data into training and prediction sets
    df_train = df_all_data_norm.iloc[:last_known_index]

    df_test = df_all_data_norm.iloc[last_known_index:].copy()

    df_test[col_target] = np.nan
    df_real_predict = df_test.copy()

    # Prepare data for XGBoost
    values = df_train[col_for_train].values
    x_input = create_x_input(df_train, lag)
    # X, y = split_sequence(values, lag)
    points_per_call = 1
    X, y = split_sequence(sequence=values, n_steps=lag, horizon=points_per_call)
    n_features = values.shape[1]


    architecture = model_architecture_params["architecture"]
    dropout_count = model_architecture_params["dropout_count"]
    activation = model_architecture_params["activation"]
    optimizer = model_architecture_params["optimizer"]

    model = Sequential()

    model.add(Bidirectional(LSTM(int(architecture[0]['neurons']), activation=activation)))
    model.add(Dropout(dropout_count))
    model.add(Dense(points_per_call))

    model.compile(optimizer=optimizer, loss='mse')

    epochs = 3

    history = model.fit(X, y, epochs=epochs, verbose=0)

    # Make predictions
    x_input = x_input.reshape((1, lag, n_features))

    # predict_values = make_predictions(x_input, df_test.values, n_features, model, lag)
    predict_values = make_predictions(x_input=x_input, x_future=df_test.values, points_per_call=points_per_call, model=model)


    df_real_predict[col_target] = np.array(predict_values).flatten()

    # Log any remaining None values
    for name, df in {'df_train': df_train, 'df_real_predict': df_real_predict}.items():
        none_indices = df[df.isnull().any(axis=1)].index.tolist()
        if none_indices:
            logger.error(f"DataFrame '{name}' contains None values at rows: {none_indices}")

    return df_train, df_real_predict

"""########################################## Блок подбора параметров #############################################"""

def get_lstm_lag(df_init, time_column, col_target):

    col_for_train = ['year', 'month', 'day', 'week', 'day_of_week',
                     'hour', 'minute', 'second', 'hour_sin', 'hour_cos',
                     'day_of_week_sin', 'day_of_week_cos', 'week_sin', 'week_cos',
                     'month_sin', 'month_cos', 'part_of_day', 'is_night', 'is_weekend', 'day_of_year']


    model_architecture_params= {
        "architecture": [{"layer": 3, "type": "Bi-LSTM", "neurons": 10}],
        "dropout_count": 0.01,
        "activation": "relu",
        "optimizer": "adam"
    }

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

    # cols = [
    #     "year", "month", "day", "week", "day_of_week", "hour", "minute", "second",
    #     "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    #     "week_sin", "week_cos", "month_sin", "month_cos",
    #     "part_of_day", "is_night", "is_weekend", "day_of_year",
    #     "is_working_hours", "season", "season_sin", "season_cos",
    #     "quarter", "quarter_sin", "quarter_cos", "moon_phase",
    # ]
    best_lag = None
    # cols = [
    #     "year", "month"
    # ]

    lag_list = range(1, 22)
    # lag_list = range(1, 2)

    best_mape = float('inf')

    df_evaluation[time_column] = pd.to_datetime(df_evaluation[time_column], errors='coerce')
    df_evaluation = df_evaluation.sort_values(by=time_column).reset_index(drop=True)

    for lag in tqdm(lag_list):
        df_true_all, df_pred_vector = forecast_LSTM_sistem(
            col_target=col_target,
            time_column=time_column,
            df_all_data_norm=df_all_data_norm,
            last_known_index=last_known_index,
            lag=lag,
            model_architecture_params=model_architecture_params,
            col_for_train=col_for_train
        )

        df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)
        df_real_predict[col_target] = df_real_predict[col_target].astype('float64')

        df_real_predict.at[df_real_predict.index[-1], col_target] = last_value


        df_real_predict[time_column] = df_evaluation[time_column]

        y_true = df_evaluation[col_target].reset_index(drop=True)
        y_pred = df_real_predict[col_target].reset_index(drop=True)

        _, _, _, mape, _ = calculate_metrics(y_true=y_true, y_pred=y_pred)

        print(f"CURRENT MAPE = {mape} | BEST lag = {best_lag}")

        if mape < best_mape:
            best_mape = mape
            best_lag = lag

    result = {"best_lag": best_lag, "best_mape": best_mape}
    print(result)

    return result


def get_model_architecture_params_lstm():
    pass

def get_points_per_call():
    pass

def get_col_for_train():
    pass

"""##################################################################################################################"""



def user_predict_LSTM(
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


    # res = get_lstm_lag(df_init=df, time_column=time_column, col_target=col_target)
    # lag = res["best_lag"]

    # model_architecture_params = get_model_architecture_params_lstm()

    # points_per_call = get_points_per_call()

    # col_for_train = get_col_for_train()


    """"Эти параметры должны подбираться автоматически, для достижения наулучшего результата"""

    lag = 4

    model_architecture_params= {
        "architecture": [{"layer": 1, "type": "Bi-LSTM", "neurons": 30},
                         {"layer": 2, "type": "Bi-LSTM", "neurons": 20},
                         {"layer": 3, "type": "Bi-LSTM", "neurons": 10}],
        "dropout_count": 0.01,
        "activation": "relu",
        "optimizer": "adam"
    }

    points_per_call = 4

    col_for_train = ['year', 'month', 'day', 'week', 'day_of_week',
                     'hour', 'minute', 'second', 'hour_sin', 'hour_cos',
                     'day_of_week_sin', 'day_of_week_cos', 'week_sin', 'week_cos',
                     'month_sin', 'month_cos', 'part_of_day', 'is_night', 'is_weekend', 'day_of_year']



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


    last_known_index = len(df_all_data) - len(date_range)

    t2v = Time2Vec(col_time=time_column, col_target=col_target)
    df_all_data_norm, min_val, max_val = t2v.vectorization(df_all_data)


    df_true_all, df_pred_vector = forecast_LSTM_user(
        col_target=col_target,
        time_column=time_column,
        df_all_data_norm=df_all_data_norm,
        last_known_index=last_known_index,
        lag=lag,
        model_architecture_params=model_architecture_params,
        col_for_train=col_for_train,
        points_per_call=points_per_call
    )

    df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)

    df_real_predict[time_column] = date_range

    df_real_predict = df_real_predict.reset_index(drop=True)

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
