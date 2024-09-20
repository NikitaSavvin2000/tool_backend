import pandas as pd
import yaml
import os
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, Dropout, LSTM, Dense, Attention
# from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Dropout, Attention

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from ui.backend.normalization import TimeNormalization
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


st.session_state.setdefault('dropout_count', 0.01)
st.session_state.setdefault('optimizer', 'adam')
st.session_state.setdefault('activation', 'relu')
st.session_state.setdefault('target', 2)
st.session_state.setdefault('epochs', 1)
st.session_state.setdefault('', 'Set learning model params')




lstm0_units = 15
lstm1_units = 20
lstm2_units = 30

lag = st.session_state.target

activation = st.session_state.activation
optimizer = st.session_state.optimizer
dense_units = 1
dropout_count = st.session_state.dropout_count
epochs = st.session_state.epochs

# batch_size = params['batch_size']

def test_run():

    df_all_data = st.session_state.df_for_train
    df_all_data = df_all_data[[st.session_state.time_col, st.session_state.target_value_col,]]
    df_all_data[st.session_state.time_col] = pd.to_datetime(df_all_data[st.session_state.time_col]).apply(lambda x: x.replace(second=0))
    df_all_data = df_all_data.sort_values(by=st.session_state.time_col)
    df_all_data[st.session_state.time_col] = df_all_data[st.session_state.time_col].dt.strftime('%Y-%m-%d %H:%M:%S')


    df_all_data = df_all_data.rename(columns={st.session_state.target_value_col: st.session_state.target_value_col})

    df_all_data = df_all_data[[st.session_state.target_value_col, st.session_state.time_col]]


    tn = TimeNormalization()
    #
    df_all_data_norm = tn.df_normalize_with_meta(df_all_data)



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

    def make_predictions(x_input, x_future):
        predict_values = []
        x_future_len = len(x_future)
        for i in range(x_future_len):
            x_input_tensor = tf.convert_to_tensor(x_input.reshape((1, lag, n_features)), dtype=tf.float32)
            y_predict = model.predict(x_input_tensor, verbose=1)
            print(y_predict)
            predict_values.append(y_predict)
            x_input = np.delete(x_input, (0), axis=1)
            future_lag = x_future[0]
            x_future = np.delete(x_future, 0, axis=0)
            future_lag[0] = y_predict
            x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)
        return predict_values



    df = df_all_data_norm
    train_index = len(df) - 288
    df_train_all_col = df.loc[:train_index]
    df_test_all_col = df.loc[train_index+1:]
    df_true_all_col = df_test_all_col.copy()

    df = df_all_data_norm
    df_train = df.loc[:train_index]
    df_test = df.loc[train_index+1:]
    df_true = df_test.copy()

    df_test[st.session_state.target_value_col] = None
    df_forecast = df_test.copy()
    values = df_train.values
    x_input = create_x_input(df_train, lag)
    x_future = df_test.values
    X, y = split_sequence(values, lag)

    n_features = values.shape[1]
    inputs = Input(shape=(lag, n_features))

    # Define the model
    model = Sequential()
    model.add(Bidirectional(LSTM(lstm0_units, activation=activation, return_sequences=True), input_shape=(lag, n_features)))
    model.add(Dropout(dropout_count))
    model.add(Bidirectional(LSTM(lstm1_units, activation=activation, return_sequences=True)))
    model.add(Dropout(dropout_count))
    model.add(Bidirectional(LSTM(lstm2_units, activation=activation)))
    model.add(Dropout(dropout_count))
    # model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=optimizer, loss='mse')

    # Define callbacks
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.001)
    save_best_weights_callback = SaveBestWeights()

    # Train the model
    history = model.fit(X, y, epochs=epochs, verbose=1, callbacks=[early_stopping, reduce_lr, save_best_weights_callback])
    model.set_weights(save_best_weights_callback.best_weights)

    x_input = x_input.reshape((1, lag, n_features))

    predict_values = make_predictions(x_input, x_future)

    predict_values = np.array(predict_values).flatten()


    df_forecast[st.session_state.target_value_col] = predict_values


    df_comparative = tn.df_denormalize_with_meta(df_forecast)
    df_true = tn.df_denormalize_with_meta(df_true_all_col)



    fig_p_l = make_subplots(rows=1, cols=1, subplot_titles=['P_l_real vs P_l_predict'])

    fig_p_l.add_trace(go.Scatter(x=df_true[st.session_state.time_col], y=df_true[st.session_state.target_value_col], mode='lines', name='P_l_real', line=dict(color='blue')), row=1, col=1)
    fig_p_l.add_trace(go.Scatter(x=df_comparative[st.session_state.time_col], y=df_comparative[st.session_state.target_value_col], mode='lines', name='P_l_predict', line=dict(color='orange')), row=1, col=1)
    template = "presentation"

    st.plotly_chart(fig_p_l, use_container_width=True)




    def calculate_metrics(y_true, y_pred):

        y_true_mean = y_true.mean()
            # Расчет RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Расчет R^2
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true_mean) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # Расчет MAE
        mae = mean_absolute_error(y_true, y_pred)

        # Расчет MAPE
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # Расчет WMAPE
        wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

        return rmse, r2, mae, mape, wmape

    y_true=df_true[st.session_state.target_value_col]
    y_pred=df_comparative[st.session_state.target_value_col]

    rmse, r2, mae, mape, wmape = calculate_metrics(y_true=y_true, y_pred=y_pred)
    print(f'RMSE = {rmse}')
    print(f'R-squared = {r2}')
    print(f'MAE = {mae}')
    print(f'MAPE = {mape}')
    print(f'WMAPE = {wmape}')