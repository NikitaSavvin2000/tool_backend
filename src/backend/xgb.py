import numpy as np
import pandas as pd
import tensorflow as tf

from xgboost import XGBRegressor
from src.config import logger
import yaml
import os
home_path = os.getcwd()

def split_sequence(sequence, n_steps):
    """
    Split a univariate sequence into samples for supervised learning.

    Parameters:
        sequence (np.ndarray): Input sequence.
        n_steps (int): Number of steps to look back.

    Returns:
        tuple: Arrays of input samples (X) and targets (y).
    """
    X, y = [], []
    for i in range(len(sequence) - n_steps):
        seq_x, seq_y = sequence[i:i + n_steps, :], sequence[i + n_steps, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def create_x_input(df_train, n_steps):
    """
    Create the input array for predictions from the training DataFrame.

    Parameters:
        df_train (pd.DataFrame): Training data.
        n_steps (int): Number of steps to look back.

    Returns:
        np.ndarray: Input array for predictions.
    """
    return df_train.iloc[-n_steps:].values


def make_predictions(x_input, x_future, n_features, model, lag):
    """
    Generate predictions for a future horizon using an iterative approach.

    Parameters:
        x_input (np.ndarray): Initial input data.
        x_future (np.ndarray): Future data.
        n_features (int): Number of features in the data.
        model (tf.keras.Model): Trained prediction model.
        lag (int): Number of time steps used for predictions.

    Returns:
        list: Predicted values.
    """
    predict_values = []
    for _ in range(len(x_future)):
        x_input_tensor = tf.convert_to_tensor(x_input.reshape((1, -1)), dtype=tf.float32)
        y_predict = model.predict(x_input_tensor)
        predict_values.append(y_predict)

        x_input = np.delete(x_input, 0, axis=1)
        future_lag = x_future[0]
        x_future = np.delete(x_future, 0, axis=0)
        future_lag[0] = y_predict
        x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)
        x_input = x_input.reshape((1, lag, n_features))

    return predict_values


def forecast_XGBoost(
        col_target, df_all_data_norm, evaluation_index, last_known_index, lag,
        model_architecture_params, forecast_type, norm_values
):
    """
    Perform forecasting using XGBoost for regression.

    Parameters:
        col_target (str): Target column name.
        df_all_data_norm (pd.DataFrame): Normalized data.
        evaluation_index (int): Start index for evaluation.
        last_known_index (int): Last known data index.
        lag (int): Number of time steps for prediction.
        model_architecture_params (dict): Parameters for XGBoost model.
        forecast_type (str): Type of forecast ('predictions' or other).
        norm_values (bool): Whether to normalize input values.

    Returns:
        dict: DataFrames containing evaluation, true values, and predictions.
    """
    if norm_values:
        possible_cols = [col_target,
            "year", "month", "day", "week", "day_of_week", "hour", "minute", "second",
            "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
            "week_sin", "week_cos", "month_sin", "month_cos",
            "part_of_day", "is_night", "is_weekend", "day_of_year",
            "is_working_hours", "season", "season_sin", "season_cos",
            "quarter", "quarter_sin", "quarter_cos", "moon_phase",
            "time_trend", "fourier_time"
        ]

        file_path = f'{home_path}/src/backend/col_for_train.yaml'

        with open(file_path, 'r', encoding='utf-8') as f:
            col_for_train_init = yaml.safe_load(f)
            col_for_train_init = col_for_train_init['col_for_train']

        col_for_train_init.insert(0, col_target)

        col_for_train = [
            col for col in col_for_train_init if len(df_all_data_norm[col].unique()) > 1
        ]

    else:
        possible_cols = [
            col_target, 'year', 'month', 'week', 'day', 'day_of_week',
            'hour', 'minute', 'second'
        ]
        all_columns = df_all_data_norm.columns.tolist()
        col_time = [col for col in all_columns if col != 'col_target'][0]
        df_all_data_norm[col_time] = pd.to_datetime(df_all_data_norm[col_time])

        df_all_data_norm['year'] = df_all_data_norm[col_time].dt.year
        df_all_data_norm['month'] = df_all_data_norm[col_time].dt.month
        df_all_data_norm['day'] = df_all_data_norm[col_time].dt.day
        df_all_data_norm['week'] = df_all_data_norm[col_time].dt.isocalendar().week
        df_all_data_norm['day_of_week'] = df_all_data_norm[col_time].dt.dayofweek
        df_all_data_norm['hour'] = df_all_data_norm[col_time].dt.hour
        df_all_data_norm['minute'] = df_all_data_norm[col_time].dt.minute
        df_all_data_norm['second'] = df_all_data_norm[col_time].dt.second

        col_for_train = [
            col_target, 'year', 'month', 'week', 'day', 'day_of_week', 'hour', 'minute'
        ]


    model_architecture_params = model_architecture_params[0]

    df_all_data_norm = df_all_data_norm[possible_cols]
    df_true_all_col = df_all_data_norm.iloc[evaluation_index: last_known_index]
    df_true_all_col_skip = df_all_data_norm.iloc[last_known_index:]
    df_all_data_norm[col_target] = df_all_data_norm[col_target].replace('None', None)
    df_all_data_norm[col_target] = df_all_data_norm[col_target].astype(float)


    all_columns = df_all_data_norm.columns
    diff_cols = all_columns.difference(col_for_train)
    columns = col_for_train
    train_index = evaluation_index

    df_true_all_col = df_true_all_col.iloc[:last_known_index + 1]

    df = df_all_data_norm[col_for_train]
    df_train = df.iloc[:train_index]
    df_test = df.iloc[train_index + 1: last_known_index + 1]
    df_test.loc[:, col_target] = np.nan

    df_evaluetion = df_test.copy()

    values = df_train[columns].values
    x_input = create_x_input(df_train, lag)

    x_future = df_test.values
    X, y = split_sequence(values, lag)

    n_features = values.shape[1]

    xgb_model = XGBRegressor(**model_architecture_params)

    if forecast_type != 'predictions':
        X_reshaped = X.reshape(X.shape[0], -1)
        X_reshaped = np.array(X_reshaped, dtype=float)
        y = np.array(y, dtype=float)

        xgb_model.fit(X_reshaped, y)

        try:
            x_input = x_input.reshape((1, lag, n_features))
        except Exception as e:
            logger.error(e)

        predict_values = make_predictions(x_input, x_future, n_features, xgb_model, lag)

        predict_values = np.array(predict_values).flatten()

        df_evaluetion[col_target] = predict_values

        if len(diff_cols) > 0:
            for col in diff_cols:
                df_evaluetion[col] = df_true_all_col[col]

        df_evaluetion[col_target] = predict_values
    else:
        df_evaluetion = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c']
        })
        df_evaluetion['minute'] = 0
        df_evaluetion['second'] = 0

    df_all_data_norm = df_all_data_norm[col_for_train]

    train_index = last_known_index

    if forecast_type != 'predictions':
        df_train = df_all_data_norm[evaluation_index: last_known_index + 1]
    else:
        df_train = df_all_data_norm[:last_known_index + 1]

    df_test = df_all_data_norm.iloc[train_index + 1:]
    df_test.loc[:, col_target] = np.nan
    df_real_predict = df_test.copy()
    values = df_train[columns].values
    x_input = create_x_input(df_train, lag)
    x_future = df_test.values
    X, y = split_sequence(values, lag)
    n_features = values.shape[1]

    X_reshaped = X.reshape(X.shape[0], -1)
    xgb_model.fit(X_reshaped, y)

    x_input = x_input.reshape((1, lag, n_features))

    predict_values = make_predictions(x_input, x_future, n_features, xgb_model, lag)

    predict_values = np.array(predict_values).flatten()

    df_real_predict[col_target] = predict_values
    if len(diff_cols) > 0:
        for col in diff_cols:
            df_real_predict[col] = df_true_all_col_skip[col]

    df_real_predict[col_target] = predict_values

    for df in [df_evaluetion, df_true_all_col, df_real_predict]:
        df['second'] = df['second'].fillna(0)
        df['minute'] = df['minute'].fillna(method='ffill')
        df['second'] = df['second'].fillna(method='ffill')

        for col in ['year', 'hour', 'hour_sin', 'hour_cos']:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')

    loss_list = [1]

    df_evaluetion.fillna(method='ffill', inplace=True)

    dataframes = {
        'df_evaluation': df_evaluetion,
        'df_true_all_col': df_true_all_col,
        'df_real_predict': df_real_predict
    }
    for name, df in dataframes.items():
        none_indices = df[df.isnull().any(axis=1)].index.tolist()
        if none_indices:
            logger.error(f"В DataFrame '{name}' есть None на строках: {none_indices}")

    response_code, response_message = 200, 'The training was successful'

    return df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_message
