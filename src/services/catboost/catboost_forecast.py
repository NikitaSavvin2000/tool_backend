# src/services/catboost/catboost_forecast.py
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from src.utils.possible_forecast_date import calculate_time_interval
from src.utils.date_utils import standardize_datetime
from src.utils.metrics import calculate_metrics
from typing import List, Tuple, Dict
from tqdm import tqdm
import logging
from statsmodels.tsa.seasonal import seasonal_decompose

logger = logging.getLogger(__name__)


def validate_input_data(df: pd.DataFrame) -> None:
    if len(df) < 2:
        raise ValueError("Входной временной ряд должен содержать минимум 2 наблюдения.")


def clean_numeric_column(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        val = val.replace('%', '').replace('M', '').replace(',', '.')
        if val in ('Ночная зона', 'Дневная зона'):
            return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def create_features(df: pd.DataFrame, time_column: str, col_target: str, lag_features: List[int]) -> pd.DataFrame:
    df_features = df.copy()
    df_features[col_target] = df_features[col_target].apply(clean_numeric_column)

    dt = pd.to_datetime(df_features[time_column])
    df_features['hour'] = dt.dt.hour
    df_features['day_of_week'] = dt.dt.dayofweek
    df_features['month'] = dt.dt.month
    df_features['quarter'] = dt.dt.quarter

    # Циклические признаки
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)

    # Лаги уровня
    for lag in lag_features:
        df_features[f'lag_{lag}'] = df_features[col_target].shift(lag)

    # Разности
    df_features['diff_1'] = df_features[col_target].diff(1)
    df_features['abs_diff_1'] = df_features['diff_1'].abs()
    df_features['diff_2'] = df_features[col_target].diff(2)

    # Rolling stats
    for w in [3, 5, 6, 7, 12, 24]:
        df_features[f'rolling_mean_{w}'] = df_features[col_target].rolling(window=w, min_periods=1).mean()
        df_features[f'rolling_std_{w}'] = df_features[col_target].rolling(window=w, min_periods=1).std().fillna(0)
        if w >= 5:
            df_features[f'rolling_min_{w}'] = df_features[col_target].rolling(window=w, min_periods=1).min()
            df_features[f'rolling_max_{w}'] = df_features[col_target].rolling(window=w, min_periods=1).max()
        if w == 7:
            df_features[f'rolling_median_{w}'] = df_features[col_target].rolling(window=w, min_periods=1).median()

    # Отношения и нормализации
    df_features['lag1_div_mean3'] = df_features['lag_1'] / (df_features['rolling_mean_3'] + 1e-9)
    df_features['lag1_minus_mean3'] = df_features['lag_1'] - df_features['rolling_mean_3']


    df_features['ewm_0.9'] = df_features[col_target].ewm(alpha=0.9).mean()
    df_features['ewm_0.5'] = df_features[col_target].ewm(alpha=0.5).mean()

    return df_features


def prepare_data_for_catboost(df: pd.DataFrame, time_column: str, col_target: str, lag_features: List[int]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df_features = create_features(df, time_column, col_target, lag_features)
    feature_columns = [
        c for c in df_features.columns
        if c not in [time_column, col_target] and df_features[c].dtype in ['int64', 'float64', 'int32', 'float32']
    ]
    df_clean = df_features.dropna(subset=feature_columns + [col_target])
    if df_clean.empty:
        raise ValueError("После очистки данных не осталось строк для обучения")
    X = df_clean[feature_columns]
    y = df_clean[col_target]
    return X, y, feature_columns


def forecast_catboost_method(
    df_train: pd.DataFrame,
    time_column: str,
    col_target: str,
    lag_features: List[int],
    forecast_steps: int,
    catboost_params: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Прогнозирование с использованием CatBoost модели.
    Лаги формируются из фактов, а при их отсутствии — из предсказаний.
    После прогноза применяется масштабирование по дисперсии обучающей выборки.
    """
    X_train, y_train, feature_columns = prepare_data_for_catboost(df_train, time_column, col_target, lag_features)

    df_train_sorted = df_train.sort_values(by=time_column)
    series = df_train_sorted[col_target].dropna()

    try:
        decomposition = seasonal_decompose(series, model="additive", period=24)
        trend_train = decomposition.trend.bfill().ffill().values
    except Exception as e:
        logger.warning(f"Не удалось decomposировать тренд: {e}. Используем rolling mean.")
        trend_train = series.rolling(window=24, center=True).mean().bfill().ffill().values

    trend_forecast = np.full(forecast_steps, trend_train[-1])

    if X_train.empty:
        raise ValueError("Нет признаков для обучения модели")

    if X_train.isna().any().any() or y_train.isna().any():
        raise ValueError("Данные содержат NaN значения после очистки")


    model = CatBoostRegressor(
        iterations=catboost_params.get('iterations', 100),
        learning_rate=catboost_params.get('learning_rate', 0.1),
        depth=catboost_params.get('depth', 6),
        random_seed=catboost_params.get('random_seed', 42),
        verbose=catboost_params.get('verbose', False),
        loss_function=catboost_params.get('loss_function', 'RMSE')
    )
    model.fit(X_train, y_train)

    df_extended = df_train.copy()
    df_extended["__pred"] = np.nan  

    predictions = []

    for step in range(forecast_steps):
        try:
            current_features = create_features(df_extended.rename(columns={col_target: "__target_used"}),
                                               time_column, "__target_used", lag_features)

            for lag in lag_features:
                lag_col = f"lag_{lag}"
                if lag_col in current_features:
                    fact_vals = df_extended[col_target].shift(lag)
                    pred_vals = df_extended["__pred"].shift(lag)
                    current_features[lag_col] = fact_vals.combine_first(pred_vals)

            available_features = [col for col in feature_columns if col in current_features.columns]
            last_row = current_features.iloc[[-1]][available_features]

            if last_row.isna().any().any():
                last_row = last_row.fillna(X_train.mean())

            prediction = model.predict(last_row)[0]
            predictions.append(prediction)

            next_time = df_extended[time_column].iloc[-1] + pd.Timedelta(
                seconds=calculate_time_interval(df_extended, time_column)
            )

            new_row = pd.DataFrame({
                time_column: [next_time],
                col_target: [np.nan],  
                "__pred": [prediction] 
            })
            df_extended = pd.concat([df_extended, new_row], ignore_index=True)

        except Exception as e:
            logger.error(f"Ошибка при предсказании шага {step}: {str(e)}")
            predictions.append(np.nan)

    pred_array = np.array(predictions)

    # Определяем порог выброса — например, 3 стандартных отклонения от среднего
    mean_pred = np.nanmean(pred_array)
    std_pred = np.nanstd(pred_array)
    threshold_low = mean_pred - 3 * std_pred
    threshold_high = mean_pred + 3 * std_pred

    # Заменяем выбросы на медиану последних 5 точек (или среднее)
    for i in range(len(pred_array)):
        if pred_array[i] < threshold_low or pred_array[i] > threshold_high:
            # Берём медиану предыдущих 5 значений (если есть)
            window = pred_array[max(0, i-5):i]
            if len(window) > 0:
                pred_array[i] = np.nanmedian(window)
            else:
                pred_array[i] = mean_pred  # если нет окна — заменяем на среднее

    predictions = pred_array.tolist()

    # МАСШТАБИРОВАНИЕ ПО ДИСПЕРСИИ
    train_std = y_train.std()

    if train_std == 0:
        logger.warning("Стандартное отклонение обучающей выборки = 0. Прогноз не масштабируется.")
    else:
        pred_array = np.array(predictions)
        pred_mean = np.nanmean(pred_array)
        pred_std = np.nanstd(pred_array)

        if pred_std == 0:
            logger.warning("Стандартное отклонение прогноза = 0. Масштабирование невозможно.")
        else:
            scaled_predictions = (pred_array - pred_mean) * (train_std / pred_std) + pred_mean
            predictions = scaled_predictions.tolist()

    # === ОГРАНИЧЕНИЕ ПО MIN/MAX ОБУЧАЮЩЕЙ ВЫБОРКИ ===
    # train_min = y_train.min()
    # train_max = y_train.max()

    # pred_array = np.array(predictions)
    # pred_array = np.clip(pred_array, train_min, train_max)

    # clipped_count = np.sum((np.array(predictions) < train_min) | (np.array(predictions) > train_max))
    # if clipped_count > 0:
    #     logger.warning(f"Обрезано {clipped_count} значений прогноза до диапазона [{train_min:.1f}, {train_max:.1f}]")

    # predictions = pred_array.tolist()

    # === СГЛАЖИВАНИЕ МЕДИАНОЙ (окно 7) ===
    df_temp = pd.DataFrame({col_target: predictions})
    df_temp[col_target] = (
        df_temp[col_target]
        .rolling(window=7, center=True)
        .median()
        .bfill()
        .ffill()
    )

    predictions = df_temp[col_target].tolist()

    # === ТРЕНД-КОРРЕКЦИЯ ===
    pred_array = np.array(predictions)
    pred_rolling_mean = pd.Series(pred_array).rolling(window=7, center=True).mean().bfill().ffill().values


    alpha = 0.5  # сглаживаем линейный тренд с последним значением
    trend_forecast = alpha * trend_forecast + (1 - alpha) * trend_forecast[-1]

    shift = trend_forecast - pred_rolling_mean
    corrected_predictions = pred_array + shift

    # === ВЫРАВНИВАНИЕ СРЕДНЕГО ПО ХВОСТУ ===
    tail_window = min(len(trend_train), 14)  # 14 точек 
    target_level = np.nanmean(trend_train[-tail_window:])  # усредняем последние точки

    corrected_mean = np.nanmean(corrected_predictions)
    mean_shift = target_level - corrected_mean
    corrected_predictions = corrected_predictions + mean_shift


    # # КЛИППИНГ
    # train_min, train_max = y_train.min(), y_train.max()
    # corrected_predictions = np.clip(corrected_predictions, train_min, train_max)

    # ФИНАЛЬНОЕ СГЛАЖИВАНИЕ 
    df_final = pd.DataFrame({col_target: corrected_predictions})
    df_final[col_target] = df_final[col_target].rolling(window=5, center=True).median().bfill().ffill()
    final_predictions = df_final[col_target].values


    # итоговый датафрейм с прогнозами
    forecast_dates = pd.date_range(
        start=df_train[time_column].iloc[-1] + pd.Timedelta(
            seconds=calculate_time_interval(df_train, time_column)
        ),
        periods=forecast_steps,
        freq=f"{int(calculate_time_interval(df_train, time_column))}s"
    )

    df_predictions = pd.DataFrame({
        time_column: forecast_dates,
        col_target: final_predictions
    })

     # СОЗДАЁМ ДАТАФРЕЙМ ТРЕНДА
    trend_dates = list(df_train_sorted[time_column]) + list(forecast_dates)
    trend_values = list(trend_train) + list(trend_forecast)

    df_trend = pd.DataFrame({
        time_column: trend_dates,
        "trend": trend_values
    })

    return df_predictions, df_trend

def lag_selection_catboost(
    df_init: pd.DataFrame,
    time_column: str,
    col_target: str,
    lag_search_depth: int,
    catboost_params: dict
) -> Dict[str, int]:
    if len(df_init) < 2:
        raise ValueError("Для подбора лага требуется минимум 2 строки.")

    df_init[time_column] = pd.to_datetime(df_init[time_column], errors="coerce")
    df_init = df_init.sort_values(by=time_column).reset_index(drop=True)
    df_init[col_target] = df_init[col_target].apply(clean_numeric_column)
    df_init = df_init.dropna(subset=[col_target])

    optimal_evaluation_points = min(300, len(df_init) // 2)
    if len(df_init) <= optimal_evaluation_points:
        raise ValueError("Недостаточно данных для разделения на обучающую и тестовую выборки.")

    df_evaluation = df_init[-optimal_evaluation_points:].copy()
    df_train = df_init[:-optimal_evaluation_points].copy()

    max_lag = min(len(df_train) - 1, lag_search_depth)
    max_lag = max(1, max_lag)

    best_lag = 1
    best_mape = float('inf')

    print('[INFO] lag evaluation is working')

    for lag in tqdm(range(1, max_lag + 1), bar_format='{l_bar}{n_fmt}/{total_fmt} ({percentage:3.0f}%)'):
        lag_features = list(range(1, lag + 1))
        try:
            df_pred = forecast_catboost_method(
                df_train=df_train,
                time_column=time_column,
                col_target=col_target,
                lag_features=lag_features,
                forecast_steps=len(df_evaluation),
                catboost_params=catboost_params
            )

            if len(df_pred) != len(df_evaluation):
                continue

            y_true = df_evaluation[col_target].values
            y_pred = df_pred[col_target].values

            min_len = min(len(y_true), len(y_pred))
            y_true, y_pred = y_true[:min_len], y_pred[:min_len]

            valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            if not valid_mask.any():
                continue

            y_true, y_pred = y_true[valid_mask], y_pred[valid_mask]
            if len(y_true) == 0:
                continue

            metrics = calculate_metrics(y_true, y_pred)
            mape = metrics["MAPE"]

            print(f">>> CURRENT MAPE = {round(mape, 3)}  CURRENT LAG = {lag} | BEST MAPE = {round(best_mape, 3)} BEST LAG = {best_lag}")
            logger.info(f">>> CURRENT MAPE = {round(mape, 3)}  CURRENT LAG = {lag} | BEST MAPE = {round(best_mape, 3)} BEST LAG = {best_lag}")

            if mape < best_mape:
                best_mape = mape
                best_lag = lag

        except Exception as e:
            print(f"Error with lag {lag}: {str(e)}")
            continue

    return {"best_lag": best_lag, "best_mape": best_mape}


def user_predict_catboost(
    df: pd.DataFrame,
    time_column: str,
    col_target: str,
    forecast_horizon_time: str,
    lag_search_depth: int = 1
) -> dict:
    validate_input_data(df)

    forecast_horizon_time = standardize_datetime(forecast_horizon_time)
    forecast_horizon_time = pd.to_datetime(forecast_horizon_time)

    catboost_params = {
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 6,
        'random_seed': 42,
        'verbose': False,
        'loss_function': 'RMSE'
    }

    print('[INFO] >>>> lag_selection_catboost is working (method: level)')

    if lag_search_depth is None or lag_search_depth <= 1 or lag_search_depth > 21 or lag_search_depth < 0:
        lag = 1
        errors = {"mape": "unknown"}
    else:
        data_lag = lag_selection_catboost(
            df_init=df,
            time_column=time_column,
            col_target=col_target,
            lag_search_depth=lag_search_depth,
            catboost_params=catboost_params
        )
        lag = data_lag["best_lag"]
        errors = {"mape": data_lag["best_mape"]}

    lag_features = list(range(1, lag + 1))

    df = df.copy()
    df = df[[time_column, col_target]]
    df[col_target] = df[col_target].apply(clean_numeric_column)
    df = df.dropna(subset=[col_target])
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(by=time_column, ascending=True).reset_index(drop=True)

    time_point_interval = abs(calculate_time_interval(df, time_column))
    last_time = df[time_column].max()
    forecast_duration = (forecast_horizon_time - last_time).total_seconds()
    n_forecast_points = int(forecast_duration / time_point_interval)
    if n_forecast_points <= 0:
        raise ValueError(
            f"Невозможно построить прогноз: горизонта ('{forecast_horizon_time}') недостаточно после последней известной даты ('{last_time}')."
        )

    df_predictions, df_trend = forecast_catboost_method(
        df_train=df,
        time_column=time_column,
        col_target=col_target,
        lag_features=lag_features,
        forecast_steps=n_forecast_points,
        catboost_params=catboost_params
    )

    map_data = df_predictions.rename(columns={
        time_column: "timestamp",
        col_target: "value"
    }).to_dict(orient='records')

    trend_data = df_trend.rename(columns={
        time_column: "timestamp",
        "trend": "value"
    }).to_dict(orient='records')

    return {
        "map_data": map_data,
        "trend_data": trend_data,  # <-- новое
        "errors": errors
    }