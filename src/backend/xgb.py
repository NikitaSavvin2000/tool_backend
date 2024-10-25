import tensorflow as tf
import numpy as np
from xgboost import XGBRegressor
import pandas as pd



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
            x_input_tensor = tf.convert_to_tensor(x_input.reshape((1, -1)), dtype=tf.float32)  # (1, 48)
        except Exception as e:
            print('--------------------ERROR---------------------------')
            print(e)

        y_predict = model.predict(x_input_tensor)

        predict_values.append(y_predict)

        # Обновление x_input для следующей итерации
        x_input = np.delete(x_input, (0), axis=1)
        future_lag = x_future[0]
        x_future = np.delete(x_future, 0, axis=0)
        future_lag[0] = y_predict
        x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)
        x_input = x_input.reshape((1, lag, n_features))


    return predict_values


def forecast_XGBoost(
        col_target,
        df_all_data_norm,
        evaluation_index,
        last_know_index,
        lag,
        model_architecture_params,
        type
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

    model_architecture_params = model_architecture_params[0]

    xgb_model = XGBRegressor(
        objective=model_architecture_params['objective'],  # Регрессия
        n_estimators=model_architecture_params['n_estimators'],   # Количество деревьев
        learning_rate=model_architecture_params['learning_rate'],    # Скорость обучения
        max_depth=model_architecture_params['max_depth'],    # Глубина дерева
        subsample=model_architecture_params['subsample'],    # Доля выборки для построения каждого дерева
        colsample_bytree=model_architecture_params['colsample_bytree'],   # Доля признаков для каждого дерева
    )


    X_reshaped = X.reshape(X.shape[0], -1)
    X_reshaped = np.array(X_reshaped, dtype=float)
    y = np.array(y, dtype=float)




    if type != 'predictions':
        xgb_model.fit(X_reshaped, y)
        try:
            x_input = x_input.reshape((1, lag, n_features))
        except Exception as e:
            print('--------------------ERROR---------------------------')
            print(e)



        predict_values = make_predictions(x_input, x_future, n_features, xgb_model, lag)

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


    # part 2 fine tuning -------------------------------------

    df_all_data_norm = df_all_data_norm[col_for_train]

    df_all_data_norm = df_all_data_norm[col_for_train]

    train_index = last_know_index

    df_train = df_all_data_norm[evaluation_index + 1:train_index]

    df_test = df_all_data_norm.iloc[train_index + 1:]

    df_test.loc[:, col_target] = np.nan
    df_real_predict = df_test.copy()
    values = df_train[columns].values
    x_input = create_x_input(df_train, lag)
    x_future = df_test.values
    X, y = split_sequence(values, lag)

    n_features = values.shape[1]

    X_reshaped = X.reshape(X.shape[0], -1)

    xgb_model.fit(X_reshaped, y)

    x_input = x_input.reshape((1, lag, n_features))


    predict_values = make_predictions(x_input, x_future, n_features, xgb_model, lag)

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

    loss_list = [1]
    # df_evaluetion.to_csv('/Users/nikitasavvin/Desktop/Учеба/tool_backend/experiments/df_evaluetion.csv')
    # df_true_all_col.to_csv('/Users/nikitasavvin/Desktop/Учеба/tool_backend/experiments/df_true_all_col.csv')
    # df_real_predict.to_csv('/Users/nikitasavvin/Desktop/Учеба/tool_backend/experiments/df_real_predict.csv')

    response_code, response_massage = 200, 'The training was successful'
    return df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_massage
