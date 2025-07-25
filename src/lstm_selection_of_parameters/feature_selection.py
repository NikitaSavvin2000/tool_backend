# src/lstm_selection_of_parameters/feature_selection.py
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tqdm import tqdm

from src.backend.normalization import Time2Vec
from src.config import logger
from src.utils.metrics import calculate_metrics
from src.utils.possible_cols import load_possible_cols


home_path = os.getcwd()

model_architecture_params= {
    "architecture": [{"layer": 1, "type": "Bi-LSTM", "neurons": 6},
                     {"layer": 2, "type": "Bi-LSTM", "neurons": 4},
                     {"layer": 3, "type": "Bi-LSTM", "neurons": 2}],
    "dropout_count": 0.1,
    "activation": "relu",
    "optimizer": "adam"
}

model_architecture_params_user = {
    "architecture": [{"layer": 1, "type": "Bi-LSTM", "neurons": 6},
                     {"layer": 2, "type": "Bi-LSTM", "neurons": 4},
                     {"layer": 3, "type": "Bi-LSTM", "neurons": 2}],
    "dropout_count": 0.1,
    "activation": "relu",
    "optimizer": "adam"
}


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


def forecast_LSTM_user(
        col_target, time_column, df_all_data_norm, last_known_index, lag,
        model_architecture_params_user, col_for_train, points_per_call, epochs=5
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

    df_test = df_all_data_norm.iloc[last_known_index:].copy()

    df_test[col_target] = np.nan
    df_real_predict = df_test.copy()

    values = df_train[col_for_train].values
    x_input = create_x_input(df_train, lag)


    print('='*100)
    print(df_train)
    print(f'lag = {lag}')
    print(f'points_per_call = {points_per_call}')
    print(df_train[df_train.isna().any(axis=1)])



    X, y = split_sequence(sequence=values, n_steps=lag, horizon=points_per_call)
    print(type(X), X.shape, X.dtype)
    print(type(y), y.shape, y.dtype)
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)
    n_features = values.shape[1]

    architecture = model_architecture_params_user["architecture"]
    dropout_count = model_architecture_params_user["dropout_count"]
    activation = model_architecture_params_user["activation"]
    optimizer = model_architecture_params_user["optimizer"]


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
        model_architecture_params, col_for_train, points_per_call=1
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
    X, y = split_sequence(sequence=values, n_steps=lag, horizon=points_per_call)
    n_features = values.shape[1]


    architecture = model_architecture_params["architecture"]
    dropout_count = model_architecture_params["dropout_count"]
    activation = model_architecture_params["activation"]
    optimizer = model_architecture_params["optimizer"]

    model = Sequential()

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

    model.compile(optimizer=optimizer, loss='mse')

    epochs = 2

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

def get_lstm_lag(df_init, time_column, col_target, debag=False):

    col_for_train = load_possible_cols()



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

    best_lag = None

    if debag:
        max_range = 2
    else:
        max_range = 10

    lag_list = range(1, max_range)

    best_mape = float('inf')

    df_evaluation[time_column] = pd.to_datetime(df_evaluation[time_column], errors='coerce')
    df_evaluation = df_evaluation.sort_values(by=time_column).reset_index(drop=True)

    for lag in tqdm(lag_list, bar_format='{l_bar}{n_fmt}/{total_fmt} ({percentage:3.0f}%)'):
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

        print(f">>> CURRENT MAPE = {round(mape, 3)} CURRENT lag = {lag} | BEST lag = {best_lag} BEST MAPE = {round(best_mape, 3)}%")
        logger.info(f">>> CURRENT MAPE = {round(mape, 3)} CURRENT lag = {lag} | BEST lag = {best_lag} BEST MAPE = {round(best_mape, 3)}%")


        if mape < best_mape:
            best_mape = mape
            best_lag = lag

    result = {"best_lag": best_lag, "best_mape": best_mape}
    print(result)

    return result


def get_model_architecture_params_lstm():
    pass

def col_selection_lstm(df_init, col_target, time_column, lag, points_per_call, debag=False):

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

    if debag:
        all_possible_cols = all_possible_cols[:1]
    else:
        all_possible_cols = all_possible_cols

    best_errors = {}
    for col in tqdm(all_possible_cols, bar_format='{l_bar}{n_fmt}/{total_fmt} ({percentage:3.0f}%)'):
        current_cols = col_for_train + [col]

        df_true_all, df_pred_vector = forecast_LSTM_sistem(
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

def get_points_per_call(df_init, time_column, col_target, lag, debag=False):

    # col_for_train = ['year', 'month', 'day', 'week', 'day_of_week',
    #                  'hour', 'minute', 'second', 'hour_sin', 'hour_cos',
    #                  'day_of_week_sin', 'day_of_week_cos', 'week_sin', 'week_cos',
    #                  'month_sin', 'month_cos', 'part_of_day', 'is_night', 'is_weekend', 'day_of_year']

    col_for_train = load_possible_cols()


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

    best_points_per_call = None

    if debag:
        max_range = 2
    else:
        max_range = 10

    points_per_call_list = range(1, max_range)


    best_mape = float('inf')

    df_evaluation[time_column] = pd.to_datetime(df_evaluation[time_column], errors='coerce')
    df_evaluation = df_evaluation.sort_values(by=time_column).reset_index(drop=True)


    for points_per_call in tqdm(points_per_call_list, bar_format='{l_bar}{n_fmt}/{total_fmt} ({percentage:3.0f}%)'):

        df_true_all, df_pred_vector = forecast_LSTM_sistem(
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
        df_real_predict[col_target] = df_real_predict[col_target].astype('float64')

        df_real_predict.at[df_real_predict.index[-1], col_target] = last_value

        df_real_predict[time_column] = df_evaluation[time_column]
        y_true = df_evaluation[col_target].reset_index(drop=True)
        y_pred = df_real_predict[col_target].reset_index(drop=True)

        _, _, _, mape, _ = calculate_metrics(y_true=y_true, y_pred=y_pred)

        print(f">>> CURRENT MAPE = {round(mape, 3)} CURRENT points_per_call = {points_per_call} | BEST points_per_call = {best_points_per_call} BEST MAPE = {round(best_mape, 3)}")
        logger.info(f">>> CURRENT MAPE = {round(mape, 3)} CURRENT points_per_call = {points_per_call} | BEST points_per_call = {best_points_per_call} BEST MAPE = {round(best_mape, 3)}")

        if mape < best_mape:
            best_mape = mape
            best_points_per_call = points_per_call

    result = {"best_points_per_call": best_points_per_call, "best_mape": best_mape}
    print(result)

    return result
