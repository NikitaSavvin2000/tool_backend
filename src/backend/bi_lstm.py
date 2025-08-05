# src/backend/bi_lstm.py
import numpy as np
import pandas as pd
import tensorflow as tf
from src.core.logger import logger
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Sequential


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

        # Если loss является NaN или None, остановим обучение
        if loss is None or np.isnan(loss):
            self.model.stop_training = True
            df_evaluetion, df_true_all_col, loss_list, df_real_predict = None, None, None, None
            response_code = 100
            response_massage = 'The training was interrupted due to overfitting. Try to simplify the model'
            return df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_massage


def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix, 0]
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
            logger.error(f"Error converting x_input to tensor: {e}")
        y_predict = model.predict(x_input_tensor, verbose=1)
        predict_values.append(y_predict)
        x_input = np.delete(x_input, (0), axis=1)
        future_lag = x_future[0]
        x_future = np.delete(x_future, 0, axis=0)
        future_lag[0] = y_predict
        x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)
    return predict_values


def forecast(
        col_target,
        df_all_data_norm,
        evaluation_index,
        last_know_index,
        epochs,
        lag,
        activation,
        optimizer,
        dropout_count,
        model_architecture_params,
):
    model_architecture_params = model_architecture_params[0]

    possible_cols = [col_target, 'year', 'week', 'day_of_week', 'hour', 'minute', 'second', 'hour_sin', 'hour_cos',
                     'day_of_week_sin', 'day_of_week_cos', 'week_sin', 'week_cos',]

    all_col = df_all_data_norm.columns.tolist()  # Убедитесь, что all_col - это список

    # Используем пересечение списков
    available_cols = [col for col in possible_cols if col in all_col]

    possible_cols = available_cols

    df_all_data_norm = df_all_data_norm[possible_cols]
    df_true_all_col = df_all_data_norm.iloc[evaluation_index: last_know_index]
    df_true_all_col_skip = df_all_data_norm.iloc[last_know_index:]
    df_all_data_norm[col_target] = df_all_data_norm[col_target].replace('None', None)
    df_all_data_norm[col_target] = df_all_data_norm[col_target].astype(float)

    all_columns = df_all_data_norm.columns

    col_for_train = [col for col in df_all_data_norm.columns if len(df_all_data_norm[col].unique()) > 1]

    diff_cols = all_columns.difference(col_for_train)

    columns = col_for_train

    train_index = evaluation_index

    df_true_all_col = df_true_all_col.iloc[:last_know_index + 1]

    df = df_all_data_norm[col_for_train]
    df_train = df.iloc[:train_index]

    df_test = df.iloc[train_index + 1: last_know_index + 1]
    df_test.loc[:, col_target] = np.nan

    df_evaluetion = df_test.copy()
    values = df_train[columns].values
    x_input = create_x_input(df_train, lag)

    x_future = df_test.values
    X, y = split_sequence(values, lag)

    n_features = values.shape[1]

    model = Sequential()
    if len(model_architecture_params) == 3:
        model.add(Bidirectional(
            LSTM(int(model_architecture_params[0]['neurons']), activation=activation, return_sequences=True),
            input_shape=(lag, n_features)))
        model.add(Dropout(dropout_count))
        model.add(Bidirectional(
            LSTM(int(model_architecture_params[1]['neurons']), activation=activation, return_sequences=True)))
        model.add(Dropout(dropout_count))
        model.add(Bidirectional(LSTM(int(model_architecture_params[2]['neurons']), activation=activation)))
        model.add(Dropout(dropout_count))
        model.add(Dense(1))

    elif len(model_architecture_params) == 2:
        model.add(Bidirectional(
            LSTM(int(model_architecture_params[0]['neurons']), activation=activation, return_sequences=True),
            input_shape=(lag, n_features)))
        model.add(Dropout(dropout_count))
        model.add(Bidirectional(
            LSTM(int(model_architecture_params[1]['neurons']), activation=activation)))
        model.add(Dropout(dropout_count))
        model.add(Dense(1))

    elif len(model_architecture_params) == 1:
        model.add(Bidirectional(LSTM(int(model_architecture_params[0]['neurons']), activation=activation)))
        model.add(Dropout(dropout_count))
        model.add(Dense(1))

    else:
        model.add(Bidirectional(LSTM(32, activation='relu')))
        model.add(Dropout(0.01))
        model.add(Dense(1))

    model.compile(optimizer=optimizer, loss='mse')

    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.001)
    save_best_weights_callback = SaveBestWeights()


    if type != 'predictions':
        try:
            x_input = x_input.reshape((1, lag, n_features))
        except Exception as e:
            logger.error(f"Error reshaping x_input: {e}")


        x_input = x_input.reshape((1, lag, n_features))

        predict_values = make_predictions(x_input, x_future, n_features, model, lag)


        predict_values = np.array(predict_values).flatten()
        logger.debug(f'predict_values = {predict_values}')

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
    X, y = split_sequence(values, lag)

    n_features = values.shape[1]


    model.compile(optimizer=optimizer, loss='mse')

    model.fit(X, y, epochs=epochs, verbose=1, callbacks=[early_stopping, reduce_lr, save_best_weights_callback, TerminateOnNaNCallback()])
    model.set_weights(save_best_weights_callback.best_weights)

    x_input = x_input.reshape((1, lag, n_features))
    x_input = x_input.reshape((1, lag, n_features))


    predict_values = make_predictions(x_input, x_future, n_features, model, lag)

    predict_values = np.array(predict_values).flatten()
    logger.debug(f'predict_values = {predict_values}')

    df_real_predict[col_target] = predict_values
    if len(diff_cols) > 0:
        for col in diff_cols:
            df_real_predict[col] = df_true_all_col_skip[col]

    df_real_predict[col_target] = predict_values

    df_evaluetion['second'] = df_evaluetion['second'].fillna(0)
    df_true_all_col['second'] = df_true_all_col['second'].fillna(0)
    df_real_predict['second'] = df_real_predict['second'].fillna(0)

    df_evaluetion['minute'] = df_evaluetion['minute'].fillna(method='ffill')
    df_evaluetion['second'] = df_evaluetion['second'].fillna(method='ffill')

    if "year" in df_evaluetion.columns:
        df_evaluetion['year'] = df_evaluetion['year'].fillna(method='ffill')

    if "hour" in df_evaluetion.columns:
        df_evaluetion['hour'] = df_evaluetion['hour'].fillna(method='ffill')
        df_evaluetion['hour_sin'] = df_evaluetion['hour_sin'].fillna(method='ffill')
        df_evaluetion['hour_cos'] = df_evaluetion['hour_cos'].fillna(method='ffill')


    df_true_all_col['minute'] = df_true_all_col['minute'].fillna(method='ffill')
    df_true_all_col['second'] = df_true_all_col['second'].fillna(method='ffill')
    if "hour" in df_true_all_col.columns:
        df_true_all_col['hour'] = df_true_all_col['hour'].fillna(method='ffill')
        df_true_all_col['hour_sin'] = df_true_all_col['hour_sin'].fillna(method='ffill')
        df_true_all_col['hour_cos'] = df_true_all_col['hour_cos'].fillna(method='ffill')

    df_real_predict['minute'] = df_real_predict['minute'].fillna(method='ffill')
    df_real_predict['second'] = df_real_predict['second'].fillna(method='ffill')
    if "hour" in df_real_predict.columns:
        df_real_predict['hour'] = df_real_predict['hour'].fillna(method='ffill')
        df_real_predict['hour_sin'] = df_real_predict['hour_sin'].fillna(method='ffill')
        df_real_predict['hour_cos'] = df_real_predict['hour_cos'].fillna(method='ffill')

    loss_list = [1]

    dataframes = {
        'df_evaluation': df_evaluetion,
        'df_true_all_col': df_true_all_col,
        'df_real_predict': df_real_predict
    }
    # Проверка на наличие None
    for name, df in dataframes.items():
        none_indices = df[df.isnull().any(axis=1)].index.tolist()
        if none_indices:
            logger.warning(f"DataFrame '{name}' contains None/NaN values at rows: {none_indices}")
            df_evaluetion = pd.DataFrame()
            df_true_all_col = pd.DataFrame()
            df_real_predict = pd.DataFrame()
            loss_list = [1]
            response_code = 100
            response_massage = 'The training was interrupted due to overfitting. Try to simplify the model'
            return df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_massage


    response_code, response_massage = 200, 'The training was successful'
    return df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_massage

