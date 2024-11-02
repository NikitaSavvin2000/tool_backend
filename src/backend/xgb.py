import tensorflow as tf
import numpy as np
from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import GridSearchCV


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
    model_architecture_params = model_architecture_params[0]

    print(df_all_data_norm)
    # best_params = forecast_XGBoost_search(
    #     col_target,
    #     df_all_data_norm,
    #     lag,
    #     last_know_index
    # )
    # model_architecture_params = model_architecture_params[0]
    #
    # model_architecture_params['colsample_bytree'] = float(best_params['colsample_bytree'])
    # model_architecture_params['max_depth'] = int(best_params['max_depth'])
    # model_architecture_params['n_estimators'] = int(best_params['n_estimators'])
    # model_architecture_params['subsample'] = float(best_params['subsample'])
    # # model_architecture_params['colsample_bytree'] = float(best_params['colsample_bytree'])



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

    print(f'all_columns = {all_columns}')


    print(f'col_for_train = {col_for_train}')

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
    X, y = split_sequence(values, lag)

    n_features = values.shape[1]

    print('is work1')

    xgb_model = XGBRegressor(
        objective=model_architecture_params['objective'],  # Регрессия
        n_estimators=model_architecture_params['n_estimators'],   # Количество деревьев
        learning_rate=model_architecture_params['learning_rate'],    # Скорость обучения
        max_depth=model_architecture_params['max_depth'],    # Глубина дерева
        subsample=model_architecture_params['subsample'],    # Доля выборки для построения каждого дерева
        colsample_bytree=model_architecture_params['colsample_bytree'],   # Доля признаков для каждого дерева
        reg_alpha=model_architecture_params.get('reg_alpha', 0),  # L1-регуляризация (по умолчанию 0)
        reg_lambda=model_architecture_params.get('reg_lambda', 1)
    )

    print('is work2')

    if type != 'predictions':
        X_reshaped = X.reshape(X.shape[0], -1)
        X_reshaped = np.array(X_reshaped, dtype=float)
        y = np.array(y, dtype=float)
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
    #
    # df_evaluetion.to_csv('/Users/nikitasavvin/Desktop/Учеба/tool_backend/experiments/df_evaluetion.csv')
    # df_true_all_col.to_csv('/Users/nikitasavvin/Desktop/Учеба/tool_backend/experiments/df_true_all_col.csv')
    # df_real_predict.to_csv('/Users/nikitasavvin/Desktop/Учеба/tool_backend/experiments/df_real_predict.csv')

    response_code, response_massage = 200, 'The training was successful'
    return df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_massage




def forecast_XGBoost_search(
        col_target,
        df_all_data_norm,
        lag,
        last_know_index,
):

    possible_cols = [col_target, 'year', 'week', 'day_of_week', 'hour', 'minute', 'second', 'hour_sin', 'hour_cos',
                     'day_of_week_sin', 'day_of_week_cos', 'week_sin', 'week_cos',]

    df_all_data_norm = df_all_data_norm[possible_cols]
    df_all_data_norm[col_target] = df_all_data_norm[col_target].replace('None', None)
    df_all_data_norm[col_target] = df_all_data_norm[col_target].astype(float)


    col_for_train = [col for col in df_all_data_norm.columns if len(df_all_data_norm[col].unique()) > 1]


    columns = col_for_train


    df = df_all_data_norm[col_for_train]
    df_train = df[:last_know_index-1]


    values = df_train[columns].values

    X, y = split_sequence(values, lag)

    # Параметры для Grid Search
    param_grid = {
        'n_estimators': [200, 500, 1000],  # Можно добавить больше значений
        'learning_rate': [0.05],
        'max_depth': [10, 15],
        'subsample': [0.6, 0.8,],
        'colsample_bytree': [0.6, 0.8,]
    }

    # Создаем модель XGBoost
    xgb_model = XGBRegressor(objective='reg:squarederror')  # Или другой подходящий objective

    # Преобразуем данные
    X_reshaped = X.reshape(X.shape[0], -1)
    X_reshaped = np.array(X_reshaped, dtype=float)
    y = np.array(y, dtype=float)

    # Настраиваем Grid Search
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                               scoring='neg_mean_squared_error',  # Или другой метрика
                               cv=2,  # Количество кросс-валидаций
                               verbose=2,  # Уровень подробности
                               n_jobs=-1)  # Использовать все доступные ядра

    # Запускаем Grid Search
    grid_search.fit(X_reshaped, y)

    # Получаем лучшие параметры и результаты
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Лучшие параметры:", best_params)
    print("Лучший результат (MSE):", best_score)

    return best_params
