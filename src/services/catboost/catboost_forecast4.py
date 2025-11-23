# colab_template
import pandas as pd
import numpy as np
from src.normalization.time2vec import Time2Vec
from src.utils.possible_forecast_date import calculate_time_interval
from src.utils.date_utils import standardize_datetime
from src.utils.possible_cols import load_possible_cols
from src.utils.metrics import calculate_metrics
from typing import List, Tuple, Dict
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

model_architecture_params = {
    "iterations": 100,
    "learning_rate": 0.1,
    "depth": 5,
    "random_seed": 42,
    "verbose": 0,
    "l2_leaf_reg": 5  
}


def validate_input_data(df: pd.DataFrame) -> None:
    """
    Проверяет входные данные на соответствие минимальным требованиям.
    """
    if len(df) < 2:
        raise ValueError("Входной временной ряд должен содержать минимум 2 наблюдения.")


def split_sequence(sequence: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Разделяет одномерную последовательность на образцы для обучения.
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
    """
    return df_train.iloc[-n_steps:].values


def clean_column(val: str) -> float:
    """
    Очищает значение колонки от ненужных символов и преобразует в число.
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
    """
    df[cols_to_convert] = df[cols_to_convert].applymap(clean_column)
    df[time_column] = pd.to_datetime(df[time_column], format='%d.%m.%Y', errors='coerce')
    df[time_column] = df[time_column].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df


def forecast_N_method(
        col_target, time_column, df_all_data_norm, last_known_index, lag,
        model_architecture_params, col_for_train
):
    """
    Стабильная версия: без diff, но с регуляризацией и сглаживанием.
    """
    from catboost import CatBoostRegressor

    # Работаем только с целевой переменной (univariate)
    df_all_data_norm = df_all_data_norm[[col_target]].copy()

    df_train = df_all_data_norm.iloc[:last_known_index].copy()
    df_future = df_all_data_norm.iloc[last_known_index:].copy()

    df_train[col_target] = pd.to_numeric(df_train[col_target], errors='coerce')
    df_future[col_target] = np.nan

    predictions = []
    current_df = df_train.copy()

    # Параметры сглаживания
    alpha = 0.85  # 0.8–0.9

    for i in range(len(df_future)):
        start_idx = max(0, len(current_df) - 1000)
        window = current_df.iloc[start_idx:].copy()
        values = window[col_target].values

        if len(values) <= lag:
            pred = values[-1] if len(values) > 0 else 0.0
        else:
            # Генерация лагов
            X_train, y_train = [], []
            for j in range(lag, len(values)):
                X_train.append(values[j - lag:j])
                y_train.append(values[j])
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            # Обучение модели с регуляризацией
            model = CatBoostRegressor(
                iterations=model_architecture_params.get("iterations", 100),
                learning_rate=model_architecture_params.get("learning_rate", 0.1),
                depth=model_architecture_params.get("depth", 6),
                l2_leaf_reg=model_architecture_params.get("l2_leaf_reg", 3),
                random_seed=model_architecture_params.get("random_seed", 42),
                verbose=model_architecture_params.get("verbose", 0)
            )
            model.fit(X_train, y_train)

            last_input = values[-lag:].reshape(1, -1)
            pred_raw = model.predict(last_input)[0]

            # Сглаживание относительно последнего значения (реального или прогнозного)
            last_val = values[-1]
            pred = alpha * pred_raw + (1 - alpha) * last_val

        predictions.append(pred)

        # Добавляем прогноз в историю
        new_row = df_future.iloc[[i]].copy()
        new_row[col_target] = pred
        current_df = pd.concat([current_df, new_row], ignore_index=True)

    df_real_predict = df_future.copy()
    df_real_predict[col_target] = predictions

    return df_train, df_real_predict


def lag_selection_N(
        df_init: pd.DataFrame,
        time_column: str,
        col_target: str,
        cols: List[str],
        lag_search_depth: int
) -> Dict[str, int]:
    """
    Подбор оптимального лага для метода N.
    """
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
        df_true_all, df_pred_vector = forecast_N_method(
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


def user_predict_N(
        df: pd.DataFrame,
        time_column: str,
        col_target: str,
        forecast_horizon_time: str,
        lag_search_depth: int = 1
) -> dict:
    """
    Генерирует прогноз временного ряда с использованием метода N.
    """
    # Валидация входных данных
    validate_input_data(df)

    # Стандартизация границ прогнозирования
    forecast_horizon_time = standardize_datetime(forecast_horizon_time)
    forecast_horizon_time = pd.to_datetime(forecast_horizon_time)

    default_cols = load_possible_cols()

    print('[INFO] >>>> lag_selection_N is working')

    if lag_search_depth is None:
        lag_search_depth = 1

    if lag_search_depth == 1 or lag_search_depth == 0 or lag_search_depth > 21 or lag_search_depth < 0:
        lag = 1
        errors = {"mape": "unknown"}
    else:
        data_lag = lag_selection_N(
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

    # Параметры модели N
    model_params = {
        # TODO: Заменить на реальные параметры метода N
        "param1": "value1",
        "param2": "value2"
    }

    best_params["best_params"] = model_params

    df.loc[:, time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(by=time_column, ascending=False).reset_index(drop=True)

    time_point_interval = abs(calculate_time_interval(df, time_column))
    last_time = df[time_column].max()
    last_value = df[df[time_column] == last_time][[time_column, col_target]].iloc[0]

    date_range = pd.date_range(
        start=last_time,
        end=forecast_horizon_time,
        freq=f"{int(time_point_interval)}s",
    )
    date_range = date_range[1:]
    df_future = pd.DataFrame({time_column: date_range, col_target: [None] * len(date_range)})
    df_future[time_column] = pd.to_datetime(df_future[time_column])

    df_all_data = pd.concat([df.dropna(how='all'), df_future.dropna(how='all')], ignore_index=True)
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

    df_true_all, df_pred_vector = forecast_N_method(
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