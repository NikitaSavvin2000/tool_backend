# src/xgboost_selection_of_parameters/main.py
import pandas as pd
from src.normalization.time2vec import Time2Vec
from src.utils.possible_forecast_date import calculate_time_interval
from src.services.feature_selection import col_selection_xgboots, lag_selection_xgboots, n_estimators_selection_xgboots, learning_rate_selection_xgboots, max_depth_selection_xgboots
from src.services.hyperparameter_tuning import params_selection_xgboots
from src.models.xgboost_model import forecast_XGBoost_sistem
from src.utils.date_utils import standardize_datetime
from src.utils.possible_cols import load_possible_cols


# Валидация входных данных
def validate_input_data(df: pd.DataFrame) -> None:
    """
    Проверяет входные данные на соответствие минимальным требованиям.
    
    :param df: Исходный DataFrame с временными метками и данными.
    :raises ValueError: Если количество строк меньше 2.
    """
    if len(df) < 2:
        raise ValueError("Входной временной ряд должен содержать минимум 2 наблюдения.")


def user_predict_XGBoost(
    df: pd.DataFrame,
    time_column: str,
    col_target: str,
    forecast_horizon_time: str,
    lag_search_depth: int = 1
) -> dict:
    """
    Генерирует прогноз временного ряда с использованием XGBoost.

    :param df: Исходный DataFrame с временным рядом.
    :param time_column: Название колонки с временными метками.
    :param col_target: Название целевой переменной.
    :param forecast_horizon_time: Временная граница прогнозирования.
    :return: Словарь с прогнозными данными.
    """
    # Валидация входных данных
    validate_input_data(df)

    # Стандартизация границ прогнозирования
    forecast_horizon_time = standardize_datetime(forecast_horizon_time)
    forecast_horizon_time = pd.to_datetime(forecast_horizon_time)

    # Выбор оптимальных признаков

    # Подбор оптимального значения лага
    # default_cols = ['year', 'week', 'day_of_week', 'hour', 'minute', 'second', 'hour_sin', 'hour_cos',
    #                 'day_of_week_sin', 'day_of_week_cos', 'week_sin', 'week_cos',]
    default_cols = load_possible_cols()

    print('[INFO] >>>> lag_selection_xgboots is working')
    if lag_search_depth == 1 or lag_search_depth == 0 or lag_search_depth > 21 or lag_search_depth < 0:
        lag = 1
        errors = {"mape": "unknown"}
    else:
        data_lag = lag_selection_xgboots(
            df_init=df,
            time_column=time_column,
            col_target=col_target,
            cols=default_cols,
            lag_search_depth=lag_search_depth,
        )
        lag = data_lag["best_lag"]
        errors = {"mape": data_lag["best_mape"]}

    # data_cols = col_selection_xgboots(
    #     df_init=df,
    #     time_column=time_column,
    #     col_target=col_target,
    #     lag=lag
    # )
    # col_for_train = data_cols["col_for_train"]
    # errors = data_cols["errors"]
    col_for_train = default_cols


    # Автоматический подбор параметров
    # best_params = params_selection_xgboots(
    #     df_init=df,
    #     time_column=time_column,
    #     col_target=col_target,
    #     cols=col_for_train,
    #     lag=lag
    # )
    best_params = {}
    # model_architecture_params=[{"objective": "reg:squarederror"}],

    # model_params = {
    #     "objective": "reg:squarederror",
    #     "tree_method": "hist",
    #     "device": "cuda",
    #     "learning_rate": 0.1,
    #     "max_depth": 15,
    #     "subsample": 0.9,
    #     "colsample_bytree": 0.9,
    #     "min_child_weight": 5,
    #     "booster": "gbtree",
    #     "random_state": 42,
    #     "lambda": 1,
    #     "alpha": 0,
    #     "max_delta_step": 1,
    #     "max_bin": 1024,
    #     "num_parallel_tree": 3
    # }
    model_params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",           # Быстрый алгоритм для CPU
        "device": "cpu",                 # Явно указываем CPU
        "learning_rate": 0.1,
        "max_depth": 15,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 5,
        "booster": "gbtree",
        "random_state": 42,
        "lambda": 1,
        "alpha": 0,
        "max_delta_step": 1,
        "max_bin": 1024,
        "num_parallel_tree": 3
    }




    # TODO: Блок ниже к доработке, по какой-то причине прогоз становится хуже

    # n_estimators = n_estimators_selection_xgboots(
    #     df_init=df,
    #     time_column=time_column,
    #     col_target=col_target,
    #     col_for_train=col_for_train,
    #     model_params=model_params,
    #     lag=lag
    # )["n_estimators"]
    #
    # model_params["n_estimators"] = n_estimators
    #
    # learning_rate = learning_rate_selection_xgboots(
    #     df_init=df,
    #     time_column=time_column,
    #     col_target=col_target,
    #     col_for_train=col_for_train,
    #     model_params=model_params,
    #     lag=lag
    # )["learning_rate"]
    #
    # model_params["learning_rate"] = learning_rate
    #
    #
    # max_depth = max_depth_selection_xgboots(
    #     df_init=df,
    #     time_column=time_column,
    #     col_target=col_target,
    #     col_for_train=col_for_train,
    #     model_params=model_params,
    #     lag=lag
    # )["max_depth"]
    #
    # model_params["max_depth"] = max_depth
    #
    best_params["best_params"] = model_params

    original_format = df[time_column].copy()
    df.loc[:, time_column] = pd.to_datetime(df[time_column], errors="coerce")
    df = df.sort_values(by=time_column, ascending=True).reset_index(drop=True)
    df[time_column] = original_format

    # Определение временного интервала
    last_value = df[[time_column, col_target]].iloc[-1]
    last_known_data = df.iloc[-1][time_column]
    time_point_interval = abs(calculate_time_interval(df, time_column))

    last_time = df[time_column].iloc[-1]

    date_range = pd.date_range(
        start=last_time,
        end=forecast_horizon_time,
        freq=f"{int(time_point_interval)}s",
    )
    date_range = date_range[1:]
    df_future = pd.DataFrame({time_column: date_range, col_target: [None] * len(date_range)})


    df_all_data = pd.concat([df, df_future], ignore_index=True).sort_values(by=time_column, ascending=True).reset_index(drop=True)
    last_known_index = len(df_all_data) - len(date_range)


    if forecast_horizon_time <= last_time:
        raise ValueError(
            f"Время горизонта ({forecast_horizon_time}) должно быть позже последней известной даты ({last_time})"
        )

    if df_all_data.empty:
        raise ValueError("Итоговый DataFrame пуст — невозможно построить прогноз.")

    if df_all_data[time_column].isna().any():
        raise ValueError("Обнаружены некорректные временные метки (NaT) после обработки.")

    if len(date_range) == 0:
        raise ValueError(
            f"Невозможно построить прогноз: горизонта ('{forecast_horizon_time}') недостаточно после последней известной даты ('{last_time}')."
        )

    t2v = Time2Vec(col_time=time_column, col_target=col_target)
    df_all_data_norm, min_val, max_val = t2v.vectorization(df_all_data)
    print(df_all_data)

    n_future_points = df_all_data_norm.shape[0] - last_known_index
    if n_future_points <= lag:
        raise ValueError(
            f"Недостаточно данных для прогноза: доступно {n_future_points} точек, но требуется как минимум {lag + 1}."
        )

    df_true_all, df_pred_vector = forecast_XGBoost_sistem(
        col_target=col_target,
        time_column=time_column,
        df_all_data_norm=df_all_data_norm,
        last_known_index=last_known_index,
        lag=lag,
        model_architecture_params=best_params["best_params"],
        col_for_train=col_for_train,
    )

    if df_pred_vector.empty or df_pred_vector.shape[0] == 0:
        raise ValueError("Модель не вернула предсказания. Возможно, недостаточно данных после применения лага.")

    # Обратная нормализация прогнозов
    df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)
    print("df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)")
    print(df_all_data)

    df_real_predict[time_column] = date_range
    df_real_predict = df_real_predict.reset_index(drop=True)

    # Добавление последней известной записи
    new_row = pd.DataFrame({
        time_column: [pd.to_datetime(last_value[time_column])],
        col_target: [float(last_value[col_target])],
    })
    df_real_predict = pd.concat([new_row, df_real_predict]).reset_index(drop=True)

    # Стандартизация временных меток
    df[time_column] = df[time_column].apply(lambda x: standardize_datetime(str(x)))

    # Подготовка результатов
    predictions = df_real_predict.to_dict(orient="records")
    return {
        "map_data": {
            "data": {
                "predictions": predictions,
            },
            "errors": errors,
            "last_know_data": last_known_data,
            "title": f"Реальный прогноз {col_target}",
            "legend": {
                "last_know_data_line": {
                    "text": {
                        "en": "Last known date",
                        "ru": "Последняя известная дата",
                    },
                    "color": "#A9A9A9",
                },
                "real_data_line": {
                    "text": {
                        "en": "Real data",
                        "ru": "Реальные данные",
                    },
                    "color": "#0000FF",
                },
            },
        },
    }