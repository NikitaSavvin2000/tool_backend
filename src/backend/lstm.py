from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import regularizers
import yaml
import os


home_path = os.getcwd()


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


def forecast_LSTM(
        col_target,
        df_all_data_norm,
        evaluation_index,
        last_know_index,
        lag,
        model_architecture_params,
        type,
        norm_values
):

    possible_cols = [
        col_target, 'year', 'month', 'day', 'week', 'day_of_week',
        'hour', 'minute', 'second', 'hour_sin', 'hour_cos',
        'day_of_week_sin', 'day_of_week_cos', 'week_sin', 'week_cos',
        'month_sin', 'month_cos', 'part_of_day', 'is_night', 'is_weekend', 'day_of_year'
    ]

    if norm_values:
        print('is norm_values')
        file_path = f'{home_path}/src/backend/col_for_train_lstm.yaml'

        with open(file_path, 'r', encoding='utf-8') as f:
            col_for_train_init = yaml.safe_load(f)
            col_for_train_init = col_for_train_init['col_for_train']

        col_for_train_init.insert(0, col_target)

        col_for_train = [
            col for col in col_for_train_init if len(df_all_data_norm[col].unique()) > 1
        ]
    else:
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

    file_path = f'{home_path}/src/backend/col_for_train_lstm.yaml'

    with open(file_path, 'r', encoding='utf-8') as f:
        col_for_train_init = yaml.safe_load(f)
        col_for_train_init = col_for_train_init['col_for_train']

    col_for_train_init.insert(0, col_target)

    col_for_train = [
        col for col in col_for_train_init if len(df_all_data_norm[col].unique()) > 1
    ]


    print(f'col_for_train = {col_for_train}')

    diff_cols = all_columns.difference(col_for_train)

    columns = col_for_train

    train_index = evaluation_index

    df_true_all_col = df_true_all_col.iloc[:last_know_index + 1]

    df = df_all_data_norm[col_for_train]
    df_train = df.iloc[:train_index]
    print('is work0')

    df_test = df.iloc[train_index + 1: last_know_index + 1]
    df_test.loc[:, col_target] = np.nan

    df_evaluetion = df_test.copy()
    values = df_train[columns].values
    x_input = create_x_input(df_train, lag)

    x_future = df_test.values
    points_per_call = 16
    X, y = split_sequence(values, lag, points_per_call)


    n_features = values.shape[1]

    print('is work1')

    activation = "relu"
    optimizer = "adam"
    epochs = 3


    model = Sequential()
    tf.keras.utils.set_random_seed(91)

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

    else:
        epochs = 3
        lstm0_units, lstm1_units, lstm2_units = 30, 20, 10
        activation = 'relu'
        recurrent_dropout_rate = 0.2
        regularizers_l2 = 0.2
        points_per_call = 16
        model.add(LSTM(lstm0_units, activation='softplus', return_sequences=True ,recurrent_dropout=recurrent_dropout_rate, input_shape=(lag, n_features)))
        model.add(LSTM(lstm1_units, activation=activation, return_sequences=True,recurrent_dropout=recurrent_dropout_rate))
        model.add(LSTM(lstm2_units, activation=activation,recurrent_dropout=recurrent_dropout_rate))
        model.add(Dense(points_per_call, activation='linear', kernel_regularizer=regularizers.l2(regularizers_l2)))

        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    model.compile(optimizer=optimizer, loss='mse')



    print('is work2')

    if type != 'predictions':
        history = model.fit(X, y, epochs=epochs, verbose=1,)
        try:
            x_input = x_input.reshape((1, lag, n_features))
        except Exception as e:
            print('--------------------ERROR---------------------------')
            print(e)


        x_input = x_input.reshape((1, lag, n_features))

        predict_values = make_predictions(x_input, x_future, points_per_call, model)


        predict_values = np.array(predict_values).flatten()
        print(f'predict_values = {predict_values}')

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


    print(f'train_index = {train_index}')
    print(f'last_know_index = {last_know_index}')
    print(f'evaluation_index = {evaluation_index}')


    print('-----------------------df_train--------------------')

    print(df_train)

    df_test = df_all_data_norm.iloc[train_index + 1:]
    print('-----------------------df_test--------------------')

    print(df_test)

    print('-----------------------df_all_data_norm--------------------')

    print(df_all_data_norm)


    df_test.loc[:, col_target] = np.nan
    df_real_predict = df_test.copy()
    values = df_train[columns].values
    x_input = create_x_input(df_train, lag)
    x_future = df_test.values

    try:
        X, y = split_sequence(values, lag, points_per_call)
    except Exception as e:
        print('-----------ERROR- ------------')
        print(e)

    n_features = values.shape[1]


    try:
        n_features = values.shape[1]
    except Exception as e:
        print('-----------ERROR- ------------')
        print(e)




    model.compile(optimizer=optimizer, loss='mse')



    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    model.fit(X, y, epochs=epochs, verbose=1)


    # model.set_weights(save_best_weights_callback.best_weights)

    x_input = x_input.reshape((1, lag, n_features))
    x_input = x_input.reshape((1, lag, n_features))


    # predict_values = make_predictions(x_input, x_future, n_features, model, lag)
    predict_values = make_predictions(x_input, x_future, points_per_call, model)

    predict_values = np.array(predict_values).flatten()

    print(f'predict_values = {predict_values}')

    df_real_predict[col_target] = predict_values
    if len(diff_cols) > 0:
        for col in diff_cols:
            df_real_predict[col] = df_true_all_col_skip[col]

    df_real_predict[col_target] = predict_values
    print('is work')

    df_evaluetion['second'] = df_evaluetion['second'].fillna(0)
    df_true_all_col['second'] = df_true_all_col['second'].fillna(0)
    df_real_predict['second'] = df_real_predict['second'].fillna(0)

    df_evaluetion['minute'] = df_evaluetion['minute'].fillna(method='ffill')
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

    print('----------------------------ПРОВЕРКА----------------------------------------------')

    dataframes = {
        'df_evaluation': df_evaluetion,
        'df_true_all_col': df_true_all_col,
        'df_real_predict': df_real_predict
    }
    # Проверка на наличие None
    for name, df in dataframes.items():
        none_indices = df[df.isnull().any(axis=1)].index.tolist()
        if none_indices:
            print(f"В DataFrame '{name}' есть None на строках: {none_indices}")


    response_code, response_massage = 200, 'The training was successful'
    return df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_massage




