import json

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, Dropout, LSTM, Dense, Attention
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import numpy as np



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


def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def create_x_input(df_train, n_steps):
    df_input = df_train.iloc[len(df_train)-n_steps:]
    x_input = df_input.values
    return x_input

def make_predictions(x_input, x_future, n_features, model, lag):
    predict_values = []
    x_future_len = len(x_future)
    for i in range(x_future_len):
        x_input_tensor = tf.convert_to_tensor(x_input.reshape((1, lag, n_features)), dtype=tf.float32)
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
        time_points_horizon,
        epochs,
        lag,
        activation,
        optimizer,
        dropout_count,
        model_architecture_params,
):

    time_points_horizon += 1
    all_columns = df_all_data_norm.columns

    col_for_train = [col for col in df_all_data_norm.columns if len(df_all_data_norm[col].unique()) > 1]


    diff_cols = all_columns.difference(col_for_train)

    columns = col_for_train

    df = df_all_data_norm
    train_index = len(df) - time_points_horizon
    df_train_all_col = df.iloc[:train_index]
    df_test_all_col = df.iloc[train_index + 1:]
    df_true_all_col = df_test_all_col.copy()

    df = df_all_data_norm[col_for_train]
    df_train = df.iloc[:train_index]
    df_test = df.iloc[train_index + 1:]
    df_true = df_test.copy()
    df_test.loc[:, col_target] = np.nan
    df_forecast = df_test.copy()
    values = df_train[columns].values
    x_input = create_x_input(df_train, lag)
    x_future = df_test.values
    X, y = split_sequence(values, lag)


    n_features = values.shape[1]
    inputs = Input(shape=(lag, n_features))


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
    history = model.fit(X, y, epochs=epochs, verbose=1, callbacks=[early_stopping, reduce_lr, save_best_weights_callback])
    model.set_weights(save_best_weights_callback.best_weights)

    x_input = x_input.reshape((1, lag, n_features))

    predict_values = make_predictions(x_input, x_future, n_features, model, lag)

    predict_values = np.array(predict_values).flatten()

    df_forecast[col_target] = predict_values
    if len(diff_cols) > 0:
        for col in diff_cols:
            df_forecast[col] = df_true_all_col[col]


    df_forecast[col_target] = predict_values
    loss_list = history.history['loss']


    return df_forecast, df_true_all_col, loss_list