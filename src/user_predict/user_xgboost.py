import os
from src.backend.normalization import Time2Vec
from src.xgboost_selection_of_parameters.feature_selection import col_selection_xgboots
import numpy as np
import pandas as pd
from src.backend.xgb import forecast_XGBoost_user
from src.utils.possible_forecast_date import calculate_time_interval


home_path = os.getcwd()

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


def forecast_XGBoost(
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


def all_available_forecast(
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
    lag = 4

    data = col_selection_xgboots(
        df_init=df,
        time_column=time_column,
        col_target=col_target,
    )
    col_for_train = data["col_for_train"]


    df = df.sort_values(by=time_column, ascending=False).reset_index(drop=True)

    last_value = df[col_target].iloc[0]
    last_known_data = df.iloc[0][time_column]
    time_point_interval = abs(calculate_time_interval(df, time_column))
    last_time = df[time_column].iloc[0]

    date_range = pd.date_range(start=last_time, end=forecast_horizon_time, freq=f'{int(time_point_interval)}T')
    date_range = date_range[1:]

    df_future = pd.DataFrame({time_column: date_range, col_target: [None] * len(date_range)})
    df_all_data = pd.concat([df, df_future], ignore_index=True)

    df_all_data = df_all_data.sort_values(by=time_column, ascending=True).reset_index(drop=True)

    last_known_index = len(df_all_data) - len(date_range) - 1

    t2v = Time2Vec(col_time=time_column, col_target=col_target)
    df_all_data_norm, min_val, max_val = t2v.vectorization(df_all_data)

    df_true_all, df_pred_vector = forecast_XGBoost_user(
        col_target=col_target,
        df_all_data_norm=df_all_data_norm,
        last_known_index=last_known_index,
        lag=lag,
        model_architecture_params=[MODEL_ARCHITECTURE_PARAMS],
        col_for_train=col_for_train
    )

    df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)
    df_real_predict.iloc[-1, df_real_predict.columns.get_loc(col_target)] = last_value

    df_real_predict[time_column] = df_future[time_column]

    df_real_predict.iloc[-1, df_real_predict.columns.get_loc(col_target)] = last_value
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


df = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vSgwB47qVFZcr1Aq--UWxZ6fDi9CGLZm-1i8QoMgfdaHUbV8EqSli3ayPxYYxD8kqfYYHD41uuNxbjZ/pub?gid=1952392108&single=true&output=csv")
time_column = 'Datetime'
col_target = 'consumption'
forecast_horizon_time = '2018-01-05 23:45:00'

dict = all_available_forecast(
    df=df,
    time_column=time_column,
    col_target=col_target,
    forecast_horizon_time=forecast_horizon_time
)

print(dict)
