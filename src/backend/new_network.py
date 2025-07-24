import numpy as np
import pandas as pd
import tensorflow as tf

from src.core.logger import logger
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, MaxPooling1D, Conv1D
import os
import yaml

home_path = os.getcwd()


class SaveBestWeights(Callback):
    def __init__(self):
        super(SaveBestWeights, self).__init__()
        self.best_weights = None
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        if current_loss is None:
            return
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_weights = self.model.get_weights()


class TerminateOnNaNCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')

        if loss is None or np.isnan(loss):
            logger.info(f'\nОбучение остановлено на эпохе {epoch + 1} из-за NaN/None значений в loss.')
            self.model.stop_training = True
            df_evaluetion, df_true_all_col, loss_list, df_real_predict = None, None, None, None
            response_code = 100
            response_massage = 'The training was interrupted due to overfitting. Try to simplify the model'
            return df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_massage


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


def make_predictions(x_input, x_future, n_features, model, lag):
    predict_values = []
    x_future_len = len(x_future)
    for i in range(x_future_len):
        try:
            x_input_tensor = tf.convert_to_tensor(x_input.reshape((1, lag, n_features)), dtype=tf.float32)
        except Exception as e:
            logger.error(e)
        y_predict = model.predict(x_input_tensor, verbose=1)
        predict_values.append(y_predict)
        x_input = np.delete(x_input, (0), axis=1)
        future_lag = x_future[0]
        x_future = np.delete(x_future, 0, axis=0)
        future_lag[0] = y_predict
        x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)
    return predict_values


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


def forecast_neural_networks(
        col_target,
        df_all_data_norm,
        evaluation_index,
        last_know_index,
        model_architecture_params,
        type,
        norm_values
):
    model_architecture_params = model_architecture_params[0]
    layers = model_architecture_params["layers"]
    epochs = model_architecture_params["epochs"]
    lag = model_architecture_params["lag"]
    optimizer_last = model_architecture_params["optimizer"]
    points_per_call = model_architecture_params["points_per_call"]
    final_l2_regularizer = model_architecture_params["final_l2_regularizer"]
    last_activation = model_architecture_params["activation"]

    possible_cols = [
        col_target, 'year', 'month', 'day', 'week', 'day_of_week',
        'hour', 'minute', 'second', 'hour_sin', 'hour_cos',
        'day_of_week_sin', 'day_of_week_cos', 'week_sin', 'week_cos',
        'month_sin', 'month_cos', 'part_of_day', 'is_night', 'is_weekend', 'day_of_year'
    ]

    file_path = f'{home_path}/src/backend/col_for_train_lstm.yaml'

    with open(file_path, 'r', encoding='utf-8') as f:
        col_for_train_init = yaml.safe_load(f)
        col_for_train_init = col_for_train_init['col_for_train']

    col_for_train_init.insert(0, col_target)

    col_for_train = [
        col for col in col_for_train_init if len(df_all_data_norm[col].unique()) > 1
    ]


    if not norm_values:
        all_columns = df_all_data_norm.columns.tolist()
        col_time = [col for col in all_columns if col != 'col_target'][0]
        df_all_data_norm[col_time] = pd.to_datetime(df_all_data_norm[col_time])

        df_all_data_norm['year'] = df_all_data_norm[col_time].dt.year
        df_all_data_norm['month'] = df_all_data_norm[col_time].dt.month
        df_all_data_norm['day'] = df_all_data_norm[col_time].dt.day
        df_all_data_norm['week'] = df_all_data_norm[col_time].dt.isocalendar().week
        df_all_data_norm['day_of_week'] = df_all_data_norm[col_time].dt.dayofweek  # 0 - понедельник, 6 - воскресенье
        df_all_data_norm['hour'] = df_all_data_norm[col_time].dt.hour
        df_all_data_norm['minute'] = df_all_data_norm[col_time].dt.minute
        df_all_data_norm['second'] = df_all_data_norm[col_time].dt.second


    df_all_data_norm = df_all_data_norm[possible_cols]
    df_true_all_col = df_all_data_norm.iloc[evaluation_index: last_know_index]
    df_true_all_col_skip = df_all_data_norm.iloc[last_know_index:]
    df_all_data_norm[col_target] = df_all_data_norm[col_target].replace('None', None)
    df_all_data_norm[col_target] = df_all_data_norm[col_target].astype(float)

    all_columns = df_all_data_norm.columns

    diff_cols = all_columns.difference(col_for_train)

    columns = col_for_train

    train_index = evaluation_index

    df_true_all_col = df_true_all_col.iloc[:last_know_index + 1]

    df = df_all_data_norm[col_for_train]
    df_train = df.iloc[:train_index]

    df_test = df.iloc[train_index : last_know_index]
    df_test.loc[:, col_target] = np.nan

    df_evaluetion = df_test.copy()
    values = df_train[columns].values
    x_input = create_x_input(df_train, lag)

    x_future = df_test.values
    X, y = split_sequence(values, lag, points_per_call)


    n_features = values.shape[1]


    layerList = list(layers.keys())
    last_layer = layerList[-1]
    first_layer = layerList[0]


    model = Sequential()
    tf.keras.utils.set_random_seed(91)
    tf.config.experimental.enable_op_determinism()
    for layer, layer_params in layers.items():
        if len(layers) == 1:
            pass
        model_type = layer_params["model_type"]
        neurons = int(layer_params["neurons"])
        recurrent_dropout = float(layer_params["recurrent_dropout"])
        activation = layer_params["activation"]
        l2_regularizers = float(layer_params["l2_regularizers"])

        if last_layer == layer:
            return_sequences = False
        else:
            return_sequences = True

        if layer == first_layer:
            input_shape = (lag, n_features)
        else:
            input_shape = None

        if model_type == 'LSTM':
            model.add(LSTM(
                neurons,
                activation=activation,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=regularizers.l2(l2_regularizers),
                return_sequences=return_sequences,
                input_shape=input_shape
                )
            )

        elif model_type == 'Bi-LSTM':
            model.add(Bidirectional(LSTM(
                neurons,
                activation=activation,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=regularizers.l2(l2_regularizers),
                return_sequences=return_sequences,

            ), input_shape=input_shape
            ))

        elif model_type == 'CNN':
            model.add(Conv1D(
                filters=neurons,
                kernel_size=1,
                activation=activation,
                input_shape=input_shape
                )
            )
            model.add(MaxPooling1D(pool_size=1))

    model.add(Dense(points_per_call, activation=last_activation, kernel_regularizer=regularizers.l2(final_l2_regularizer)))

    model.compile(optimizer=optimizer_last, loss='mean_squared_error', metrics=['mae'])



    if type != 'predictions':
        history = model.fit(X, y, epochs=epochs, verbose=1,)
        try:
            x_input = x_input.reshape((1, lag, n_features))
        except Exception as e:
            logger.error(e)


        x_input = x_input.reshape((1, lag, n_features))

        # predict_values = make_predictions(x_input, x_future, n_features, model, lag)

        predict_values = make_predictions(x_input, x_future, points_per_call, model)
        predict_values = np.array(predict_values).flatten()
        logger.info(f'predict_values = {predict_values}')

        df_evaluetion[col_target] = predict_values
        if len(diff_cols) > 0:
            for col in diff_cols:
                df_evaluetion[col] = df_true_all_col[col]

            df_evaluetion[col_target] = predict_values
            loss_list = [1]
        else:
            df_evaluetion = pd.DataFrame({
                'column1': [1, 2, 3],
                'column2': ['a', 'b', 'c']
            })
            df_evaluetion['minute'] = 0
            df_evaluetion['second'] = 0
    else:

        df_evaluetion = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c']
        })
        df_evaluetion['minute'] = 0
        df_evaluetion['second'] = 0


    # part 2 fine tuning -------------------------------------

    df_all_data_norm = df_all_data_norm[col_for_train]


    train_index = last_know_index

    if type != 'predictions':
        df_train = df_all_data_norm[evaluation_index :last_know_index+1]
    else:
        df_train = df_all_data_norm[:last_know_index+1]

    df_test = df_all_data_norm.iloc[train_index + 1:]

    df_test.loc[:, col_target] = np.nan
    df_real_predict = df_test.copy()
    values = df_train[columns].values
    x_input = create_x_input(df_train, lag)
    x_future = df_test.values

    try:
        X, y = split_sequence(values, lag, points_per_call)
    except Exception as e:
        logger.error(e)

    n_features = values.shape[1]


    try:
        n_features = values.shape[1]
    except Exception as e:
        logger.error(e)

    model.compile(optimizer=optimizer_last, loss='mse')

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    model.fit(X, y, epochs=epochs, verbose=1)

    x_input = x_input.reshape((1, lag, n_features))
    x_input = x_input.reshape((1, lag, n_features))

    predict_values = make_predictions(x_input, x_future, points_per_call, model)


    predict_values = np.array(predict_values).flatten()

    logger.info(f'predict_values = {predict_values}')

    df_real_predict[col_target] = predict_values
    if len(diff_cols) > 0:
        for col in diff_cols:
            df_real_predict[col] = df_true_all_col_skip[col]

    df_real_predict[col_target] = predict_values

    if "second" in df_evaluetion.columns:
        df_evaluetion['second'] = df_evaluetion['second'].fillna(0)

    if "second" in df_true_all_col.columns:
        df_true_all_col['second'] = df_true_all_col['second'].fillna(0)

    if "second" in df_real_predict.columns:
        df_real_predict['second'] = df_real_predict['second'].fillna(0)

    if "minute" in df_evaluetion.columns:
        df_evaluetion['minute'] = df_evaluetion['minute'].fillna(method='ffill')

    if "second" in df_evaluetion.columns:
        df_evaluetion['second'] = df_evaluetion['second'].fillna(method='ffill')

    if "year" in df_evaluetion.columns:
        df_evaluetion['year'] = df_evaluetion['year'].fillna(method='ffill')

    if "hour" in df_evaluetion.columns:
        df_evaluetion['hour'] = df_evaluetion['hour'].fillna(method='ffill')
    if "hour_sin" in df_evaluetion.columns:
        df_evaluetion['hour_sin'] = df_evaluetion['hour_sin'].fillna(method='ffill')
    if "hour_cos" in df_evaluetion.columns:
        df_evaluetion['hour_cos'] = df_evaluetion['hour_cos'].fillna(method='ffill')


    df_true_all_col['minute'] = df_true_all_col['minute'].fillna(method='ffill')
    df_true_all_col['second'] = df_true_all_col['second'].fillna(method='ffill')

    if "hour" in df_true_all_col.columns:
        df_true_all_col['hour'] = df_true_all_col['hour'].fillna(method='ffill')
    if "hour_sin" in df_true_all_col.columns:
        df_true_all_col['hour_sin'] = df_true_all_col['hour_sin'].fillna(method='ffill')
    if "hour_cos" in df_true_all_col.columns:
        df_true_all_col['hour_cos'] = df_true_all_col['hour_cos'].fillna(method='ffill')

    df_real_predict['minute'] = df_real_predict['minute'].fillna(method='ffill')
    df_real_predict['second'] = df_real_predict['second'].fillna(method='ffill')
    if "hour" in df_real_predict.columns:
        df_real_predict['hour'] = df_real_predict['hour'].fillna(method='ffill')
    if "hour_sin" in df_real_predict.columns:
        df_real_predict['hour_sin'] = df_real_predict['hour_sin'].fillna(method='ffill')
    if "hour_cos" in df_real_predict.columns:
        df_real_predict['hour_cos'] = df_real_predict['hour_cos'].fillna(method='ffill')

    loss_list = [1]

    dataframes = {
        'df_evaluation': df_evaluetion,
        'df_true_all_col': df_true_all_col,
        'df_real_predict': df_real_predict
    }
    for name, df in dataframes.items():
        none_indices = df[df.isnull().any(axis=1)].index.tolist()
        if none_indices:
            logger.error(f"В DataFrame '{name}' есть None на строках: {none_indices}")

    response_code, response_massage = 200, 'The training was successful'

    return df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_massage
