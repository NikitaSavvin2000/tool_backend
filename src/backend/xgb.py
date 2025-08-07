# src/backend/xgb.py
import os

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRegressor

from src.core.logger import logger
from src.utils.xgb_utils import make_predictions_xgb
from utils.lstm_utils import create_x_input, make_predictions, split_sequence

home_path = os.getcwd()

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


    xgb_model = XGBRegressor(**model_architecture_params)


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

    print("x_input shape:", x_input.shape)
    print("n_features used for training:", n_features)


    if forecast_type != 'predictions':
        X_reshaped = X.reshape(X.shape[0], -1)
        X_reshaped = np.array(X_reshaped, dtype=float)
        y = np.array(y, dtype=float)

        xgb_model.fit(X_reshaped, y)

        try:
            x_input = x_input.reshape((1, lag, n_features))
        except Exception as e:
            logger.error(e)

        try:
            y_pred_array = make_predictions_xgb(xgb_model, x_input)
            # y_pred_array будет массивом, берем первый элемент
            predict_values = [float(y_pred_array[0])]
        except Exception as e:
            logger.error(f"Ошибка при предсказании с XGBoost: {e}")
            raise

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


def forecast_XGBoost_user(
        col_target, time_column, df_all_data_norm, last_known_index, lag,
        model_architecture_params, col_for_train
):
    """
    Perform forecasting using XGBoost for regression.

    Parameters:
        col_target (str): Target column name.
        time_column (str): Name of the time column.
        df_all_data_norm (pd.DataFrame): Normalized data (should include time_column and col_for_train).
        last_known_index (int): Last known data index.
        lag (int): Number of time steps for lagged features.
        model_architecture_params (dict): Parameters for XGBoost model.
        col_for_train (list): List of feature columns for training (might include col_target for autoregression).

    Returns:
        tuple: (df_train, df_real_predict) DataFrames with true and predicted values.
    """

    df_all_data_norm[time_column] = pd.to_datetime(df_all_data_norm[time_column], errors='coerce')
    df_all_data_norm = df_all_data_norm.sort_values(by=time_column).reset_index(drop=True)


    col_for_train = list(dict.fromkeys(col_for_train))

    if col_target not in col_for_train:
        col_for_train = [col_target] + col_for_train
    else:
        pass
        
    print("col_for_train после обработки:", col_for_train)
    print("n_features:", len(col_for_train))

    df_working = df_all_data_norm[[time_column] + col_for_train].copy()

    df_working[col_target] = df_working[col_target].replace('None', np.nan).astype(float)

    df_train = df_working.iloc[:last_known_index].copy()
    df_test = df_working.iloc[last_known_index:].copy()
    df_test.loc[:, col_target] = np.nan 
    df_real_predict = df_test.copy()

    nan_locations = df_train.isna()
    if nan_locations.any().any():
        print("NaN values found in df_train:")
        nan_rows = df_train[nan_locations.any(axis=1)]
        print(f"Rows with NaN:\n{nan_rows}")
        nan_columns = nan_locations.sum()
        print(f"NaN counts per column:\n{nan_columns[nan_columns > 0]}")
    else:
        print("No NaN values found in df_train.")

    values = df_train[col_for_train].values 

    x_input = create_x_input(df_train[col_for_train], lag) 

    X, y = split_sequence(values, lag)

    n_features = len(col_for_train)

    xgb_model = XGBRegressor(**model_architecture_params)
    X_reshaped = X.reshape(X.shape[0], -1)

    y_target = y[:, 0] 
    xgb_model.fit(X_reshaped, y_target)

    print("x_input shape:", x_input.shape)
    print("n_features used for training:", n_features)


    try:
        y_pred_array = make_predictions_xgb(xgb_model, x_input)
        predict_value_scalar = float(y_pred_array[0])
        predict_values = [predict_value_scalar] 
    except Exception as pred_error:
        logger.error(f"Ошибка при генерации прогноза с XGBoost: {pred_error}")
        raise pred_error

    df_real_predict.loc[df_real_predict.index[0], col_target] = predict_value_scalar 

    for name, df in {'df_train': df_train, 'df_real_predict': df_real_predict}.items():
        none_indices = df[df.isnull().any(axis=1)].index.tolist()
        if none_indices:
            print(f"WARNING: DataFrame '{name}' contains None/NaN values at rows: {none_indices}")

    return df_train, df_real_predict