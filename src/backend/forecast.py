from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd


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
            print(f'\nОбучение остановлено на эпохе {epoch + 1} из-за NaN/None значений в loss.')
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
            print('--------------------ERROR---------------------------')
            print(e)
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
    possible_cols = [col_target, 'year', 'week', 'day_of_week', 'hour', 'minute', 'second', 'hour_sin', 'hour_cos',
                     'day_of_week_sin', 'day_of_week_cos', 'week_sin', 'week_cos',]

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
    df_train = df.iloc[:train_index + 1]

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
    print(f'----------------------------model_architecture_params-------------------------------')
    print(f'{model_architecture_params}')
    print(f'------------------------------------------------------------------------------------')


    history = model.fit(X, y, epochs=epochs, verbose=1,
                        callbacks=[early_stopping, reduce_lr, save_best_weights_callback, TerminateOnNaNCallback()])

    model.set_weights(save_best_weights_callback.best_weights)


    try:
        x_input = x_input.reshape((1, lag, n_features))
    except Exception as e:
        print('--------------------ERROR---------------------------')
        print(e)

    predict_values = make_predictions(x_input, x_future, n_features, model, lag)

    predict_values = np.array(predict_values).flatten()

    print(f'predict_values = {predict_values}')




    df_evaluetion[col_target] = predict_values
    if len(diff_cols) > 0:
        for col in diff_cols:
            df_evaluetion[col] = df_true_all_col[col]

    df_evaluetion[col_target] = predict_values
    loss_list = history.history['loss']

    # part 2 fine tuning -------------------------------------

    df_all_data_norm = df_all_data_norm[col_for_train]

    df_all_data_norm = df_all_data_norm[col_for_train]

    train_index = last_know_index
    print(f'last_know_index = {last_know_index}')

    df_train = df_all_data_norm[evaluation_index + 1:train_index]

    df_test = df_all_data_norm.iloc[train_index + 1:]

    df_test.loc[:, col_target] = np.nan
    df_real_predict = df_test.copy()
    values = df_train[columns].values
    x_input = create_x_input(df_train, lag)
    x_future = df_test.values
    X, y = split_sequence(values, lag)

    n_features = values.shape[1]

    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=optimizer, loss='mse')

    model.fit(X, y, epochs=epochs, verbose=1, callbacks=[early_stopping, reduce_lr, save_best_weights_callback, TerminateOnNaNCallback()])
    model.set_weights(save_best_weights_callback.best_weights)

    x_input = x_input.reshape((1, lag, n_features))

    print(f'x_input {x_input}')
    print(f'x_future {x_future}')

    predict_values = make_predictions(x_input, x_future, n_features, model, lag)

    predict_values = np.array(predict_values).flatten()

    print(f'predict_values = {predict_values}')

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

    df_true_all_col['minute'] = df_true_all_col['minute'].fillna(method='ffill')
    df_true_all_col['second'] = df_true_all_col['second'].fillna(method='ffill')

    df_real_predict['minute'] = df_real_predict['minute'].fillna(method='ffill')
    df_real_predict['second'] = df_real_predict['second'].fillna(method='ffill')

    #
    # df_evaluetion.to_csv('/Users/nikitasavvin/Desktop/Учеба/tool_backend/experiments/df_evaluetion.csv')
    # df_true_all_col.to_csv('/Users/nikitasavvin/Desktop/Учеба/tool_backend/experiments/df_true_all_col.csv')
    # df_real_predict.to_csv('/Users/nikitasavvin/Desktop/Учеба/tool_backend/experiments/df_real_predict.csv')

    response_code, response_massage = 200, 'The training was successful'
    return df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_massage
