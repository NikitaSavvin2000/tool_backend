import pandas as pd
from src.normalization.time2vec import Time2Vec
from src.utils.possible_forecast_date import calculate_time_interval
from src.services.feature_selection import lag_selection_xgboots
from src.utils.date_utils import standardize_datetime
from src.utils.possible_cols import load_possible_cols
from xgboost import XGBRegressor
from src.utils.metrics import calculate_metrics
from typing import List, Tuple, Dict
import math
import numpy as np

model_architecture_params = [{
    "objective": "reg:squarederror",
    "device": "cuda",
    "tree_method": "hist",
    "n_estimators": 1000,
    "learning_rate": 0.1,
    "max_depth": 15,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 5,
    "booster": "gbtree"
}]


# Валидация входных данных
def validate_input_data(df: pd.DataFrame) -> None:
    """
    Проверяет входные данные на соответствие минимальным требованиям.

    :param df: Исходный DataFrame с временными метками и данными.
    :raises ValueError: Если количество строк меньше 2.
    """
    if len(df) < 2:
        raise ValueError("Входной временной ряд должен содержать минимум 2 наблюдения.")


def split_sequence(sequence: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Разделяет одномерную последовательность на образцы для обучения.

    :param sequence: Входная последовательность.
    :param n_steps: Количество шагов для lookback.
    :return: Кортеж из массивов входных данных (X) и целевых значений (y).
    """
    X, y = [], []
    for i in range(len(sequence) - n_steps):
        seq_x, seq_y = sequence[i:i + n_steps, :], sequence[i + n_steps, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def create_x_input(df_train: pd.DataFrame, n_steps: int) -> np.ndarray:
    """
    Создает входной массив для прогнозирования из обучающего DataFrame.

    :param df_train: Обучающие данные.
    :param n_steps: Количество шагов для lookback.
    :return: Входной массив для прогнозирования.
    """
    return df_train.iloc[-n_steps:].values


def clean_column(val: str) -> float:
    """
    Очищает значение колонки от ненужных символов и преобразует в число.

    :param val: Исходное значение.
    :return: Очищенное числовое значение или None, если преобразование невозможно.
    """
    if isinstance(val, str):
        val = val.replace('%', '').replace('M', '').replace(',', '.')
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def preprocess_data(df: pd.DataFrame, cols_to_convert: List[str], time_column: str) -> pd.DataFrame:
    """
    Предобработка данных: очистка числовых колонок и форматирование временных меток.

    :param df: Исходный DataFrame.
    :param cols_to_convert: Список колонок для очистки и преобразования.
    :param time_column: Название колонки с временными метками.
    :return: Обработанный DataFrame.
    """
    # Очистка числовых колонок
    df[cols_to_convert] = df[cols_to_convert].applymap(clean_column)

    # Форматирование временных меток
    df[time_column] = pd.to_datetime(df[time_column], format='%d.%m.%Y', errors='coerce')
    df[time_column] = df[time_column].dt.strftime('%Y-%m-%d %H:%M:%S')

    return df


def forecast_XGBoost_sistem(
        col_target, time_column, df_all_data_norm, last_known_index, lag,
        model_architecture_params, col_for_train
):

    # Преобразование временной колонки
    df_all_data_norm[time_column] = pd.to_datetime(df_all_data_norm[time_column], errors='coerce')
    df_all_data_norm = df_all_data_norm.sort_values(by=time_column).reset_index(drop=True)

    # Убедиться, что целевая колонка включена в обучающие признаки
    col_for_train = [col_target] + col_for_train
    df_all_data_norm = df_all_data_norm[col_for_train].copy()

    # Преобразование целевой колонки в числовой формат
    df_all_data_norm[col_target] = df_all_data_norm[col_target].replace('None', None).astype(float)

    # Разделение данных на обучающую и тестовую выборки
    df_train = df_all_data_norm.iloc[:last_known_index].dropna(subset=[col_target])
    df_test = df_all_data_norm.iloc[last_known_index:].copy()
    df_test[col_target] = np.nan
    df_real_predict = df_test.copy()

    # Подготовка данных для XGBoost
    values = df_train[col_for_train].values
    x_input = create_x_input(df_train, lag)
    X, y = split_sequence(values, lag)

    X = X.reshape(X.shape[0], -1)

    # Преобразование в числовой тип
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # Исключение строк с NaN
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]

    if np.isnan(X).sum() > 0 or np.isnan(y).sum() > 0:
        raise ValueError("X or y contains NaN values.")

    n_features = values.shape[1]

    # Обучение модели XGBoost
    if isinstance(model_architecture_params, list):
        model_params = model_architecture_params[0]  # Берем первый элемент списка
    else:
        model_params = model_architecture_params

    xgb_model = XGBRegressor(**model_params)
    X_reshaped = X.reshape(X.shape[0], -1)
    xgb_model.fit(X_reshaped, y)

    # Генерация прогнозов
    x_input = x_input.reshape((1, lag, n_features))
    predict_values = _make_xgboost_predictions(x_input, df_test.values, n_features, xgb_model, lag)
    df_real_predict[col_target] = np.array(predict_values).flatten()

    return df_train, df_real_predict


def lag_selection_xgboots(
        df_init: pd.DataFrame,
        time_column: str,
        col_target: str,
        cols: List[str],
        lag_search_depth: int
) -> Dict[str, int]:
    if len(df_init) < 2:
        raise ValueError("Для подбора лага требуется минимум 2 строки во входных данных.")

    df_init[time_column] = pd.to_datetime(df_init[time_column], errors="coerce")
    df_init = df_init.sort_values(by=time_column).reset_index(drop=True)

    optimal_evaluation_points = min(300, len(df_init) // 2)
    if len(df_init) <= optimal_evaluation_points:
        raise ValueError("Недостаточно данных для разделения на обучающую и тестовую выборки.")

    df_evaluation = df_init[-optimal_evaluation_points:].copy()
    df = df_init[:-optimal_evaluation_points].copy()

    df_empty = df_evaluation.copy()
    df_empty[col_target] = None

    df_all_data = pd.concat([df, df_empty.dropna(how="all")], ignore_index=True)
    df_all_data = df_all_data.sort_values(by=time_column).reset_index(drop=True)

    print('[INFO] Time2Vec is working')

    t2v = Time2Vec(col_time=time_column, col_target=col_target)
    df_all_data_norm, min_val, max_val = t2v.vectorization(df_all_data)

    df_evaluation[time_column] = pd.to_datetime(df_evaluation[time_column], errors="coerce")
    df_evaluation = df_evaluation.sort_values(by=time_column).reset_index(drop=True)

    max_lag = min(len(df) - 1, lag_search_depth)

    best_lag = None
    best_mape = float('inf')

    print('[INFO] lag evaluation is working')

    for lag in tqdm(range(1, max_lag + 1), bar_format='{l_bar}{n_fmt}/{total_fmt} ({percentage:3.0f}%)'):
        df_true_all, df_pred_vector = forecast_XGBoost_sistem(
            col_target,
            time_column,
            df_all_data_norm,
            len(df_all_data) - optimal_evaluation_points,
            lag,
            model_architecture_params,
            cols
        )

        if df_pred_vector.size == 0:
            raise ValueError("Forecast returned empty predictions.")

        df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)
        df_real_predict[col_target] = df_real_predict[col_target].astype('float64')
        df_real_predict.at[df_real_predict.index[-1], col_target] = df_init[col_target].iloc[0]
        df_real_predict[time_column] = df_evaluation[time_column]

        y_true = df_evaluation[col_target].reset_index(drop=True)
        y_pred = df_real_predict[col_target].reset_index(drop=True)

        metrix = calculate_metrics(y_true, y_pred)
        mape = metrix["MAPE"]

        print(f">>> CURRENT MAPE = {round(mape, 3)}  CURRENT LAG = {lag} | BEST MAPE = {round(best_mape, 3)} BEST LAG = {best_lag}")
        logger.info(f">>> CURRENT MAPE = {round(mape, 3)}  CURRENT LAG = {lag} | BEST MAPE = {round(best_mape, 3)} BEST LAG = {best_lag}")

        if mape < best_mape:
            best_mape = mape
            best_lag = lag

    return {"best_lag": best_lag, "best_mape": best_mape}


def _make_xgboost_predictions(
        x_input: np.ndarray,
        df_test: np.ndarray,
        n_features: int,
        model: XGBRegressor,
        lag: int
) -> List[float]:
    """
    Генерирует прогнозы для будущего горизонта с использованием обученной модели XGBoost.

    :param x_input: Исходные входные данные.
    :param df_test: Массив тестовых данных.
    :param n_features: Количество признаков.
    :param model: Обученная модель XGBoost.
    :param lag: Количество временных шагов.
    :return: Список прогнозируемых значений.
    """
    predict_values = []
    for _ in range(len(df_test)):
        # Предсказание следующего значения
        y_predict = model.predict(x_input.reshape(1, -1))[0]
        predict_values.append(y_predict)

        # Обновление входных данных
        x_input = np.delete(x_input, 0, axis=1)
        future_lag = df_test[0]
        df_test = np.delete(df_test, 0, axis=0)
        future_lag[0] = y_predict
        x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)
        x_input = x_input.reshape((1, lag, n_features))

    return predict_values


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

    default_cols = load_possible_cols()

    print('[INFO] >>>> lag_selection_xgboots is working')

    if lag_search_depth is None:
        lag_search_depth = 1

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

    col_for_train = default_cols

    best_params = {}

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

    best_params["best_params"] = model_params

    df[time_column] = pd.to_datetime(df[time_column])  # если ещё не datetime
    df = df.sort_values(by=time_column, ascending=False).reset_index(drop=True)


    time_point_interval = abs(calculate_time_interval(df, time_column))

    def _smart_round(x):
        if x == 0:
            return 0
        magnitude = 10 ** int(math.log10(abs(x)) - 1)
        return round(x / magnitude) * magnitude

    last_time = df[time_column].max()
    last_value = df[df[time_column] == last_time][[time_column, col_target]].iloc[0]
    time_point_interval = _smart_round(time_point_interval)

    print(f"time_point_interval = {time_point_interval}")

    date_range = pd.date_range(
        start=last_time,
        end=forecast_horizon_time,
        freq=f"{int(time_point_interval)}s",
    )
    print(f"date_range = {date_range}")
    date_range = date_range[1:]
    df_future = pd.DataFrame({time_column: date_range, col_target: [None] * len(date_range)})
    df_future[time_column] = pd.to_datetime(df_future[time_column])

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
    cols_to_show = [time_column, col_target]
    df_real_predict = df_real_predict[cols_to_show]
    predictions = df_real_predict.to_dict(orient="records")

    return {
        "map_data": {
            "data": {
                "predictions": predictions,
            },
            "errors": errors,
            "last_know_data": last_time,
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