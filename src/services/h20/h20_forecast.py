import sys
import os
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
import h2o
from h2o.estimators import H2OGradientBoostingEstimator
from h2o import H2OFrame
from src.utils.metrics import calculate_metrics
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

model_architecture_params = [{
        "ntrees": 1000,
        "learn_rate": 0.1,
        "max_depth": 15,
        "sample_rate": 0.9,
        "col_sample_rate": 0.9,
        "min_rows": 5,          
        "seed": 42,             
        "stopping_rounds": 10,
        "stopping_metric": "RMSE",
        "stopping_tolerance": 1e-4,
        "score_tree_interval": 1
    }]

def ensure_h2o():
    try:
        _ = h2o.connection()          
        _ = h2o.api("GET /3/Cloud")   
        return                        
    except Exception:
        pass

    try:
        h2o.init(
            ip="localhost",
            port=54321,
            start_h2o=False,          
            strict_version_check=False
        )
        _ = h2o.api("GET /3/Cloud")   
        h2o.no_progress()
        logger.info("✅ Подключился к уже запущенному кластеру H2O")
        return
    except Exception:
        pass

    try:
        h2o.shutdown(prompt=False)
    except Exception:
        pass

    h2o.init(
        ip="localhost",
        port=54321,
        nthreads=-1,
        strict_version_check=False
    )
    h2o.no_progress()
    logger.info("✅ H2O инициализировано с нуля")

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


def forecast_h20_sistem(
        col_target, time_column, df_all_data_norm, last_known_index, lag,
        model_architecture_params, col_for_train
):

    ensure_h2o()

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

    # Подготовка данных для h20
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
    
    n_features = values.shape[1]
    
    colnames_features = [f"lag{t}_f{j}" for t in range(lag) for j in range(n_features)]
    
    
    if np.isnan(X).sum() > 0 or np.isnan(y).sum() > 0:
        raise ValueError("X or y contains NaN values.")

    # Обучение модели h20
    if isinstance(model_architecture_params, list):
        model_params = model_architecture_params[0]  # Берем первый элемент списка
    else:
        model_params = model_architecture_params

    h20_model = H2OGradientBoostingEstimator(**model_params)
    
    train_data = np.column_stack([X, y])
    colnames = colnames_features + [col_target]
    
    train_h2o = h2o.H2OFrame(train_data, column_names=colnames)
    
    train_h2o = train_h2o.asnumeric()

    h20_model.train(x=colnames_features, y=col_target, training_frame=train_h2o)

    # Генерация прогнозов
    x_input = x_input.reshape((1, lag, n_features))
    predict_values = _make_h20_predictions(x_input=x_input, df_test = df_test.values, n_features=n_features, lag=lag, h20_model=h20_model, colnames_features=colnames_features)
    df_real_predict[col_target] = np.array(predict_values).flatten()

    return df_train, df_real_predict


def lag_selection_h20(
        df_init: pd.DataFrame,
        time_column: str,
        col_target: str,
        cols: List[str],
        lag_search_depth: int
) -> Dict[str, int]:
    """
    Оптимизированный подбор лага с использованием бинарного поиска
    и упрощенных параметров модели.
    """
    
    if len(df_init) < 2:
        raise ValueError("Для подбора лага требуется минимум 2 строки во входных данных.")

    df_init[time_column] = pd.to_datetime(df_init[time_column], errors="coerce")
    df_init = df_init.sort_values(by=time_column).reset_index(drop=True)

    # Уменьшаем размер evaluation выборки для ускорения
    optimal_evaluation_points = min(100, len(df_init) // 3)
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

    # Упрощенные параметры модели для быстрого подбора лага
    quick_model_params = [{
        "ntrees": 100,
        "learn_rate": 0.1,
        "max_depth": 10,
        "sample_rate": 0.9,
        "col_sample_rate": 0.9,
        "min_rows": 5,
        "histogram_type": "AUTO",
        "stopping_rounds": 5,
        "stopping_metric": "RMSE",
        "stopping_tolerance": 1e-3,
        "score_tree_interval": 5
    }]

    def evaluate_lag(lag: int) -> float:
        """Вспомогательная функция для оценки одного лага"""
        try:
            df_true_all, df_pred_vector = forecast_h20_sistem(
                col_target,
                time_column,
                df_all_data_norm,
                len(df_all_data) - optimal_evaluation_points,
                lag,
                quick_model_params,
                cols
            )

            if df_pred_vector.size == 0:
                return float('inf')

            df_real_predict = t2v.light_reverse_vectorization(df_pred_vector, min_val, max_val)
            df_real_predict[col_target] = df_real_predict[col_target].astype('float64')
            df_real_predict.at[df_real_predict.index[-1], col_target] = df_init[col_target].iloc[0]
            df_real_predict[time_column] = df_evaluation[time_column]

            y_true = df_evaluation[col_target].reset_index(drop=True)
            y_pred = df_real_predict[col_target].reset_index(drop=True)

            metrix = calculate_metrics(y_true, y_pred)
            return metrix["MAPE"]
        except Exception as e:
            logger.warning(f"Ошибка при оценке лага {lag}: {e}")
            return float('inf')

    print('[INFO] lag evaluation is working (optimized binary search)')

    # Словарь для хранения протестированных лагов
    tested_lags = {}

    # Стратегия 1: Если диапазон маленький (<=5), проверяем все
    if max_lag <= 5:
        for lag in tqdm(range(1, max_lag + 1), desc="Testing all lags"):
            mape = evaluate_lag(lag)
            tested_lags[lag] = mape
            print(f">>> LAG = {lag} | MAPE = {round(mape, 3)}")
            logger.info(f">>> LAG = {lag} | MAPE = {round(mape, 3)}")
    
    # Стратегия 2: Для большего диапазона используем умный поиск
    else:
        # Шаг 1: Проверяем ключевые точки (начало, треть, две трети, конец)
        key_lags = [1, max_lag // 4, max_lag // 2, max_lag * 3 // 4, max_lag]
        key_lags = sorted(set(key_lags))  # Убираем дубликаты
        
        print(f"[INFO] Step 1: Testing key lags: {key_lags}")
        for lag in tqdm(key_lags, desc="Testing key lags"):
            mape = evaluate_lag(lag)
            tested_lags[lag] = mape
            print(f">>> LAG = {lag} | MAPE = {round(mape, 3)}")
            logger.info(f">>> LAG = {lag} | MAPE = {round(mape, 3)}")
        
        # Шаг 2: Находим лучший из ключевых лагов
        best_key_lag = min(tested_lags, key=tested_lags.get)
        best_key_mape = tested_lags[best_key_lag]
        
        print(f"[INFO] Best key lag: {best_key_lag} with MAPE = {round(best_key_mape, 3)}")
        
        # Шаг 3: Уточняем поиск в окрестности лучшего лага (±3 шага)
        window_size = 3
        search_range = range(
            max(1, best_key_lag - window_size),
            min(max_lag + 1, best_key_lag + window_size + 1)
        )
        
        lags_to_test = [lag for lag in search_range if lag not in tested_lags]
        
        if lags_to_test:
            print(f"[INFO] Step 2: Refining search around lag {best_key_lag}: {lags_to_test}")
            for lag in tqdm(lags_to_test, desc="Refining search"):
                mape = evaluate_lag(lag)
                tested_lags[lag] = mape
                print(f">>> LAG = {lag} | MAPE = {round(mape, 3)}")
                logger.info(f">>> LAG = {lag} | MAPE = {round(mape, 3)}")

    # Находим лучший лаг из всех протестированных
    best_lag = min(tested_lags, key=tested_lags.get)
    best_mape = tested_lags[best_lag]

    print(f"\n[RESULT] Best lag found: {best_lag} with MAPE = {round(best_mape, 3)}")
    print(f"[RESULT] Total lags tested: {len(tested_lags)} out of {max_lag}")
    logger.info(f"Best lag: {best_lag}, MAPE: {best_mape}, Tested: {len(tested_lags)}/{max_lag}")

    return {"best_lag": best_lag, "best_mape": best_mape}


def _make_h20_predictions(
        x_input: np.ndarray,
        df_test: np.ndarray,
        n_features: int,
        lag: int,
        h20_model,
        colnames_features: list
) -> List[float]:
    """
    Генерирует прогнозы для будущего горизонта с использованием обученной модели h20.

    :param x_input: Исходные входные данные.
    :param df_test: Массив тестовых данных.
    :param n_features: Количество признаков.
    :param model: Обученная модель h20.
    :param lag: Количество временных шагов.
    :return: Список прогнозируемых значений.
    """
    
    predict_values = []
    for _ in range(len(df_test)):
        # Предсказание следующего значения
        flat = x_input.reshape(1, lag * n_features)
        flat = x_input.reshape(1, lag * n_features)
        hf = H2OFrame(flat, column_names=colnames_features)
        for c in hf.col_names:
            hf[c] = hf[c].asnumeric()
        pred_hf = h20_model.predict(hf)
        y_predict = float(pred_hf.as_data_frame(use_pandas=True, use_multi_thread=True).iloc[0, 0])
        predict_values.append(y_predict)

        # Обновление входных данных
        x_input = np.delete(x_input, 0, axis=1)
        future_lag = df_test[0]
        df_test = np.delete(df_test, 0, axis=0)
        future_lag[0] = y_predict
        x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)
        x_input = x_input.reshape((1, lag, n_features))

    return predict_values


def user_predict_h20(
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

    print('[INFO] >>>> lag_selection_h20 is working')

    if lag_search_depth is None:
        lag_search_depth = 1

    if lag_search_depth == 1 or lag_search_depth == 0 or lag_search_depth > 21 or lag_search_depth < 0:
        lag = 1
        errors = {"mape": "unknown"}
    else:
        data_lag = lag_selection_h20(
            df_init=df,
            time_column=time_column,
            col_target=col_target,
            cols=default_cols,
            lag_search_depth=lag_search_depth,
        )
        lag = data_lag["best_lag"]
        errors = {"mape": data_lag["best_mape"]}

    col_for_train = default_cols

    model_params = {
            "ntrees": 1000,
            "learn_rate": 0.1,
            "max_depth": 15,
            "sample_rate": 0.9,
            "col_sample_rate": 0.9,
            "min_rows": 5,          
            "seed": 42,             
            "stopping_rounds": 10,
            "stopping_metric": "RMSE",
            "stopping_tolerance": 1e-4,
            "score_tree_interval": 1
        }

    best_params = {}

    best_params["best_params"] = model_params

    df[time_column] = pd.to_datetime(df[time_column])  # если ещё не datetime
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

    df_true_all, df_pred_vector = forecast_h20_sistem(
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
    
    if h2o.connection() is not None:
        h2o.cluster().shutdown(prompt=False)

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

