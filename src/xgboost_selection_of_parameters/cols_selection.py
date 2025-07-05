#src/xgboost_selection_of_parameters/cols_selection
import os

import pandas as pd
import psycopg2
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from xgboost import XGBRegressor

import plotly.graph_objects as go

from src.backend.normalization import Time2Vec
from src.config import logger
from src.utils.possible_forecast_date import calculate_time_interval



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


def forecast_XGBoost_sistem(
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

    print(df_all_data_norm)

    df_all_data_norm[time_column] = pd.to_datetime(df_all_data_norm[time_column], errors='coerce')
    df_all_data_norm = df_all_data_norm.sort_values(by=time_column).reset_index(drop=True)

    # Ensure col_target is included in training columns
    col_for_train = [col_target] + col_for_train
    df_all_data_norm = df_all_data_norm[col_for_train].copy()

    # Convert target column to float, handling 'None' strings
    df_all_data_norm[col_target] = df_all_data_norm[col_target].replace('None', None).astype(float)

    # Split data into training and prediction sets
    df_train = df_all_data_norm.iloc[:last_known_index]
    df_test = df_all_data_norm.iloc[last_known_index:].copy()
    df_test[col_target] = np.nan
    df_real_predict = df_test.copy()

    print(df_train)
    # Prepare data for XGBoost
    values = df_train[col_for_train].values
    x_input = create_x_input(df_train, lag)
    X, y = split_sequence(values, lag)
    n_features = values.shape[1]

    # Train XGBoost model
    xgb_model = XGBRegressor(**model_architecture_params[0])

    X_reshaped = X.reshape(X.shape[0], -1)
    xgb_model.fit(X_reshaped, y)

    # Make predictions
    x_input = x_input.reshape((1, lag, n_features))
    predict_values = make_predictions(x_input, df_test.values, n_features, xgb_model, lag)
    df_real_predict[col_target] = np.array(predict_values).flatten()

    # Log any remaining None values
    for name, df in {'df_train': df_train, 'df_real_predict': df_real_predict}.items():
        none_indices = df[df.isnull().any(axis=1)].index.tolist()
        if none_indices:
            logger.error(f"DataFrame '{name}' contains None values at rows: {none_indices}")

    return df_train, df_real_predict

def forecast_XGBoost_sistem_params(
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

    # Ensure col_target is included in training columns
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
    X, y = split_sequence(values, lag)
    n_features = values.shape[1]

    # Train XGBoost model
    xgb_model = XGBRegressor(**model_architecture_params[0])

    X_reshaped = X.reshape(X.shape[0], -1)
    xgb_model.fit(X_reshaped, y)

    # Make predictions
    x_input = x_input.reshape((1, lag, n_features))
    predict_values = make_predictions(x_input, df_test.values, n_features, xgb_model, lag)
    df_real_predict[col_target] = np.array(predict_values).flatten()

    # Log any remaining None values
    for name, df in {'df_train': df_train, 'df_real_predict': df_real_predict}.items():
        none_indices = df[df.isnull().any(axis=1)].index.tolist()
        if none_indices:
            logger.error(f"DataFrame '{name}' contains None values at rows: {none_indices}")

    return df_train, df_real_predict


def forecast_XGBoost_user(
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
    print('=============================================  forecast_XGBoost_user df_train ======================================')

    print(df_train)
    df_test = df_all_data_norm.iloc[last_known_index:].copy()
    print('=============================================  forecast_XGBoost_user df_test ======================================')
    print(df_test)

    df_test[col_target] = np.nan
    df_real_predict = df_test.copy()

    # Prepare data for XGBoost
    values = df_train[col_for_train].values
    x_input = create_x_input(df_train, lag)
    X, y = split_sequence(values, lag)
    n_features = values.shape[1]

    # Train XGBoost model
    xgb_model = XGBRegressor(**model_architecture_params)

    X_reshaped = X.reshape(X.shape[0], -1)
    xgb_model.fit(X_reshaped, y)

    # Make predictions
    x_input = x_input.reshape((1, lag, n_features))

    predict_values = make_predictions(x_input, df_test.values, n_features, xgb_model, lag)

    print(f" predict_values = {predict_values}")

    df_real_predict[col_target] = np.array(predict_values).flatten()

    # Log any remaining None values
    for name, df in {'df_train': df_train, 'df_real_predict': df_real_predict}.items():
        none_indices = df[df.isnull().any(axis=1)].index.tolist()
        if none_indices:
            logger.error(f"DataFrame '{name}' contains None values at rows: {none_indices}")

    return df_train, df_real_predict


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
    #
    # all_possible_cols = [
    #     "year", "month"
    # ]

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
            model_architecture_params=[MODEL_ARCHITECTURE_PARAMS],
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
        cols: list
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
        df_true_all, df_pred_vector = forecast_XGBoost_sistem(
            col_target=col_target,
            time_column=time_column,
            df_all_data_norm=df_all_data_norm,
            last_known_index=last_known_index,
            lag=lag,
            model_architecture_params=[MODEL_ARCHITECTURE_PARAMS],
            col_for_train=cols
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



import pandas as pd
import numpy as np
from xgboost import DMatrix, cv
import optuna

def params_selection_xgboots(
        df_init: pd.DataFrame,
        time_column: str,
        col_target: str,
        cols: list,
        lag: int
) -> dict:
    """
    Select optimal XGBoost hyperparameters for time-series forecasting with a fixed lag.

    Parameters:
    -----------
    df_init : pd.DataFrame
        Input DataFrame containing time-series data.
    time_column : str
        Name of the time column.
    col_target : str
        Name of the target column to predict.
    cols : list
        List of feature columns for training.
    lag : int
        Fixed lag value for time-series forecasting.

    Returns:
    --------
    dict : Dictionary containing the fixed lag, MAPE, and best XGBoost parameters.
    """
    # Preserve original time column
    original_column = df_init[time_column].copy()
    df_init = df_init.copy()
    df_init[time_column] = pd.to_datetime(df_init[time_column], errors='coerce')
    df_init = df_init.sort_values(by=time_column).reset_index(drop=True)
    df_init[time_column] = original_column[df_init.index]

    # Define evaluation set size
    optimal_evaluation_points = 300
    if len(df_init) < optimal_evaluation_points / 0.1:
        optimal_evaluation_points = int(len(df_init) * 0.1)

    df_evaluation = df_init[-optimal_evaluation_points:]
    df = df_init[:-optimal_evaluation_points]

    # Prepare full dataset with empty target for evaluation period
    df_empty = df_evaluation.copy()
    df_empty[col_target] = None
    df_all_data = pd.concat([df, df_empty], ignore_index=True).sort_values(by=time_column).reset_index(drop=True)

    last_known_index = len(df_all_data) - optimal_evaluation_points

    # Vectorize data (assuming Time2Vec is defined elsewhere)
    t2v = Time2Vec(col_time=time_column, col_target=col_target)
    df_all_data_norm, min_val, max_val = t2v.vectorization(df_all_data)
    df_all_data_norm = df_all_data_norm.sort_values(by=time_column).reset_index(drop=True)

    # Prepare training data for hyperparameter tuning
    df_train = df_all_data_norm.iloc[:last_known_index].copy()
    values = df_train[[col_target] + cols].values
    X, y = split_sequence(values, lag)  # Assumes split_sequence is defined
    X_train = X.reshape(X.shape[0], -1)

    # Define hyperparameter tuning function
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "booster": "gbtree",
            "n_estimators": trial.suggest_int("n_estimators", 800, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42
        }
        dtrain = DMatrix(X_train, label=y)
        cv_results = cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=params["n_estimators"],
            nfold=3,
            metrics="mape",
            early_stopping_rounds=10,
            seed=42
        )
        return cv_results["test-mape-mean"].iloc[-1]

    # Run hyperparameter tuning
    print('=' * 100)
    print('Начали подбор оптимальных параметров')
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    best_params = study.best_params
    best_params["objective"] = "reg:squarederror"
    best_params["booster"] = "gbtree"
    best_params["random_state"] = 42
    print('Закончили подбор параметров')
    print(f'Лучшие параметры: {best_params}')

    # Forecast with best parameters (assumes forecast_XGBoost_sistem accepts params)
    df_true_all, df_pred_vector = forecast_XGBoost_sistem(
        col_target=col_target,
        time_column=time_column,
        df_all_data_norm=df_all_data_norm,
        last_known_index=last_known_index,
        lag=lag,
        model_architecture_params=[best_params],  # Pass best parameters
        col_for_train=cols
    )

    # Reverse vectorization for predictions
    df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)
    df_real_predict[col_target] = df_real_predict[col_target].astype('float64')

    # Set last value (as per original logic)
    last_value = df_init[col_target].iloc[0]
    df_real_predict.at[df_real_predict.index[-1], col_target] = last_value

    df_real_predict[time_column] = df_evaluation[time_column]

    # Calculate MAPE
    df_evaluation[time_column] = pd.to_datetime(df_evaluation[time_column], errors='coerce')
    df_evaluation = df_evaluation.sort_values(by=time_column).reset_index(drop=True)
    y_true = df_evaluation[col_target].reset_index(drop=True)
    y_pred = df_real_predict[col_target].reset_index(drop=True)
    _, _, _, mape, _ = calculate_metrics(y_true=y_true, y_pred=y_pred)

    print(f"\nFinal results: {best_params}")

    return best_params


def user_predict_XGBoost(
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

    data_cols = col_selection_xgboots(
        df_init=df,
        time_column=time_column,
        col_target=col_target,
    )

    col_for_train = data_cols["col_for_train"]


    data_lag = lag_selection_xgboots(
        df_init=df,
        time_column=time_column,
        col_target=col_target,
        cols=col_for_train,
    )

    lag = data_lag["best_lag"]


    # MODEL_ARCHITECTURE_PARAMS = params_selection_xgboots(
    #     df_init=df,
    #     time_column=time_column,
    #     col_target=col_target,
    #     cols=col_for_train,
    #     lag=lag
    # )


    # col_for_train = [
    #     "year", "month"
    # ]
    # lag = 1


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

    df_true_all, df_pred_vector = forecast_XGBoost_user(
        col_target=col_target,
        time_column=time_column,
        df_all_data_norm=df_all_data_norm,
        last_known_index=last_known_index,
        lag=lag,
        model_architecture_params=MODEL_ARCHITECTURE_PARAMS,
        col_for_train=col_for_train
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




# "Morocco Zone 1": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSgwB47qVFZcr1Aq--UWxZ6fDi9CGLZm-1i8QoMgfdaHUbV8EqSli3ayPxYYxD8kqfYYHD41uuNxbjZ/pub?gid=1952392108&single=true&output=csv",
#     "Morocco Zone 2": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQT1DfqAB5Yec8MIQ_E5A8w-SXNcRmTwbXsv2W-ZT1ZcXN_G83BHlb6QBgnWkO-MpH3oVgfLoE0SnLx/pub?gid=1952392108&single=true&output=csv",
#     "Morocco Zone 3": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQSHw5k7n3_RM6ksGbvdQJsa1i9-zF-18CFLCFnXFkCxQwqLcQ4Wu2_8EF2H1lF02ih2NLL9BDecFzQ/pub?gid=1952392108&single=true&output=csv",
#
df_data = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQJrlRwIHeCwf3DUiu_WkG_bwgKcyOmXKv8aKN5GSjTbqCvae9OiTSkHoaMpMfOstTvRvGnj6-3gtRk/pub?gid=1448488998&single=true&output=csv")

def fetch_data_from_db():
    table_name = 'load_consumption'
    measurement = 'load_consumption'

    DB_PARAMS = {
        "dbname": "mydb",
        "user": "myuser",
        "password": "mypassword",
        "host": "77.37.136.11",
        "port": 8083
    }


    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    select_query = f"""
    SELECT * FROM {table_name} ORDER BY datetime;
    """

    cur.execute(select_query)
    rows = cur.fetchall()

    df_result = pd.DataFrame(rows, columns=["datetime", measurement])
    df_result["datetime"] = df_result["datetime"].dt.tz_localize(None)

    cur.close()
    conn.close()
    return df_result


# df_data = fetch_data_from_db()

def clean_column(val):
    if isinstance(val, str):
        val = val.replace('%', '').replace('M', '').replace(',', '.')
    try:
        return float(val)
    except:
        return None


time_column = 'День'
col_target = 'Сумма заказов минус комиссия WB, руб.'
df_data = df_data[[time_column, col_target]]
df_data = df_data.dropna()
df_data[time_column] = pd.to_datetime(df_data[time_column], format="%m/%d/%Y")
df_data[time_column] = df_data[time_column].dt.strftime("%Y-%m-%d %H:%M:%S")
df_data[col_target] = df_data[col_target].str.replace(",", ".").astype(float)

print(df_data)

forecast_horizon_time = '2025-06-16 00:00:00'

df_to_predict = df_data.iloc[:-90]
df_test = df_data.iloc[-90:]

predict_dict = user_predict_XGBoost(
    df=df_to_predict,
    time_column=time_column,
    col_target=col_target,
    forecast_horizon_time=forecast_horizon_time
)
#
predictions= predict_dict["map_data"]["data"]["predictions"]
df_predictions = pd.DataFrame(predictions)
df_predictions = df_predictions.iloc[1:]

print('=============================== ИТОГ - df_test')
print(df_test)
print('=============================== ИТОГ - df_predictions')

print(df_predictions)

y_true = df_test[col_target].reset_index(drop=True)
y_pred = df_predictions[col_target].reset_index(drop=True)

rmse, r2, mae, mape, wmape = calculate_metrics(y_true=y_true, y_pred=y_pred)


print('=============== METRIX =================')
print(f'MAPE = {mape}')
print(f'R2 = {r2}')
print(f'RMSE = {rmse}')
print(f'MAE = {mae}')
print(f'WMAPE = {wmape}')
print('========================================')

fig = go.Figure()

fig.add_trace(go.Scatter(
    y=y_true,
    mode='lines',
    name='Реальные данные',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    y=y_pred,
    mode='lines',
    name='Прогноз',
    line=dict(color='orange')
))

fig.update_layout(
    title='Сравнение прогноза и реальных значений',
    xaxis_title='Индекс',
    yaxis_title='Значение',
    legend=dict(x=0, y=1)
)

fig.show()
