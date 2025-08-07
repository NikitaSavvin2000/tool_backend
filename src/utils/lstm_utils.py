# utils/lstm_utils.py

import numpy as np
import tensorflow as tf


def split_sequence(sequence, n_steps):
    """
    Разделяет последовательность на входные и выходные данные.

    Args:
        sequence (np.ndarray): Исходная последовательность.
        n_steps (int): Количество шагов для lookback.

    Returns:
        tuple: Массивы X и y.
    """
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def create_x_input(df_train, n_steps):
    """
    Создает входной массив для прогнозирования из обучающего DataFrame.

    Args:
        df_train (pd.DataFrame): Обучающие данные.
        n_steps (int): Количество шагов для lookback.

    Returns:
        np.ndarray: Входной массив для прогнозирования.
    """
    df_input = df_train.iloc[len(df_train) - n_steps:]
    x_input = df_input.values
    return x_input


def make_predictions(x_input, x_future, n_features, model, lag):
    """
    Генерирует прогнозы для будущего горизонта с использованием итеративного подхода.

    Args:
        x_input (np.ndarray): Входные данные для модели.
        x_future (np.ndarray): Будущие данные.
        n_features (int): Количество признаков.
        model: Обученная модель.
        lag (int): Размер лага.

    Returns:
        list: Список предсказанных значений.
    """
    predict_values = []
    x_future_len = len(x_future)
    for i in range(x_future_len):
        try:
            x_input_tensor = tf.convert_to_tensor(x_input.reshape((1, lag, n_features)), dtype=tf.float32)
        except Exception as e:
            print('-ERROR-')
            print(e)
        try:
            y_predict = model.predict(x_input_tensor, verbose=1)
        except TypeError:
            y_predict = model.predict(x_input_tensor)
        # ---------------------------------------------------
        predict_values.append(y_predict)
        x_input = np.delete(x_input, (0), axis=1)
        future_lag = x_future[0]
        x_future = np.delete(x_future, 0, axis=0)
        future_lag[0] = y_predict
        x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)
    return predict_values
