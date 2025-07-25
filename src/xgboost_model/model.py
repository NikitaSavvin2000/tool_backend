# src/xgboost_model/model.py
import tensorflow as tf

from src.config import logger
import os

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

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

    nan_locations = df_train.isna()
    if nan_locations.any().any():
        print("NaN values found in df_train:")
        # Print rows with NaN values
        nan_rows = df_train[nan_locations.any(axis=1)]
        print(f"Rows with NaN:\n{nan_rows}")
        # Print which columns have NaN and their counts
        nan_columns = nan_locations.sum()
        print(f"NaN counts per column:\n{nan_columns[nan_columns > 0]}")
    else:
        print("No NaN values found in df_train.")

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