import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from src.normalization.time2vec import Time2Vec
from src.utils.possible_forecast_date import calculate_time_interval
from src.utils.date_utils import standardize_datetime
from src.utils.possible_cols import load_possible_cols
from src.utils.metrics import calculate_metrics
from typing import List, Tuple, Dict
from tqdm import tqdm
import logging

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
    df_features['diff_24'] = df_features[col_target].diff(24)

    df_features['pct_change_1'] = df_features[col_target].pct_change(1, fill_method=None)
    df_features['pct_change_24'] = df_features[col_target].pct_change(24, fill_method=None)

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

    # Отношения скользящих средних
    df_features['mean_24_div_mean_7'] = df_features['rolling_mean_24'] / (df_features['rolling_mean_7'] + 1e-9)
    df_features['std_24_div_std_7'] = df_features['rolling_std_24'] / (df_features['rolling_std_7'] + 1e-9)

    # Тренд
    df_features['trend_30'] = df_features[col_target].rolling(window=30, min_periods=1).mean()

    # Экспоненциальное сглаживание
    df_features['ewm_0.9'] = df_features[col_target].ewm(alpha=0.9).mean()
    df_features['ewm_0.5'] = df_features[col_target].ewm(alpha=0.5).mean()

    # Убираем колонки, которые состоят только из NaN
    df_features = df_features.loc[:, df_features.isna().mean() < 1.0]

    return df_features


def prepare_data_for_catboost(df: pd.DataFrame, time_column: str, col_target: str, lag_features: List[int]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df_features = create_features(df, time_column, col_target, lag_features)
    feature_columns = [
        c for c in df_features.columns
        if c not in [time_column, col_target] and df_features[c].dtype in ['int64', 'float64', 'int32', 'float32']
    ]
    
    for col in feature_columns:
        df_features[col] = df_features[col].apply(clean_numeric_column)

    df_clean = df_features.dropna(subset=feature_columns + [col_target])
    if df_clean.empty:
        raise ValueError("После очистки данных не осталось строк для обучения")
    X = df_clean[feature_columns]
    y = df_clean[col_target]
    return X, y, feature_columns


def prepare_data_for_catboost_diff(df: pd.DataFrame, time_column: str, col_target: str, lag_features: List[int]):
    df2 = df.copy()
    df2[col_target] = df2[col_target].apply(clean_numeric_column)
    df2 = df2.sort_values(time_column).reset_index(drop=True)

    df2['target_delta'] = df2[col_target].diff(1)
    df_feat = create_features(df2, time_column, col_target, lag_features)

    feature_columns = [c for c in df_feat.columns if c not in [time_column, col_target, 'target_delta']]

    for col in feature_columns:
        df_feat[col] = df_feat[col].apply(clean_numeric_column)

    print(f"[DEBUG] feature_columns: {len(feature_columns)}")
    print(f"[DEBUG] df_feat.shape: {df_feat.shape}")
    print(f"[DEBUG] target_delta NaN count: {df_feat['target_delta'].isna().sum()}")

    feature_columns = [c for c in feature_columns if not df_feat[c].isna().all()]
    print(f"[DEBUG] feature_columns after removing all-NaN: {len(feature_columns)}")

    for col in feature_columns:
        nan_count = df_feat[col].isna().sum()
        if nan_count > 0:
            print(f"[DEBUG] Feature {col} has {nan_count} NaNs")

    df_final = df_feat.dropna(subset=['target_delta'] + feature_columns, how='any')
    print(f"[DEBUG] df_final.shape after dropna: {df_final.shape}")

    if df_final.empty:
        print("[ERROR] No rows left after dropna")
        raise ValueError("Нет строк для обучения после создания дельт/фич")

    X = df_final[feature_columns]
    y = df_final['target_delta']
    return X, y, feature_columns, df_feat


def forecast_catboost_method(
    df_train: pd.DataFrame,
    time_column: str,
    col_target: str,
    lag_features: List[int],
    forecast_steps: int,
    catboost_params: dict
) -> pd.DataFrame:
    # --- Обучение модели ---
    X_train, y_train, feature_columns = prepare_data_for_catboost(
        df_train, time_column, col_target, lag_features
    )

    model = CatBoostRegressor(
        iterations=catboost_params.get('iterations', 500),  # Снижено
        learning_rate=catboost_params.get('learning_rate', 0.03),
        depth=catboost_params.get('depth', 6),  # Снижено
        l2_leaf_reg=catboost_params.get('l2_leaf_reg', 3),
        random_seed=catboost_params.get('random_seed', 42),
        verbose=catboost_params.get('verbose', False),
        loss_function=catboost_params.get('loss_function', 'RMSE')
    )
    model.fit(X_train, y_train)

    # --- Прогноз ---
    df_extended = df_train.copy().reset_index(drop=True)
    df_extended["__target_used"] = df_extended[col_target]

    predictions = []

    for step in range(forecast_steps):
        temp = create_features(df_extended, time_column, "__target_used", lag_features)
        last_row = temp.iloc[[-1]][feature_columns]

        # Заполнение NaN только локально
        last_row = last_row.ffill().fillna(X_train.median())

        pred = model.predict(last_row)[0]
        predictions.append(pred)

        next_time = df_extended[time_column].iloc[-1] + pd.Timedelta(
            seconds=calculate_time_interval(df_extended, time_column)
        )

        new_row = pd.DataFrame({
            time_column: [next_time],
            col_target: [np.nan],
            "__target_used": [pred],
        })
        df_extended = pd.concat([df_extended, new_row], ignore_index=True)

    forecast_dates = pd.date_range(
        start=df_train[time_column].iloc[-1] + pd.Timedelta(seconds=calculate_time_interval(df_train, time_column)),
        periods=forecast_steps,
        freq=f"{int(calculate_time_interval(df_train, time_column))}s"
    )

    return pd.DataFrame({
        time_column: forecast_dates,
        col_target: predictions
    })


def forecast_catboost_method_diff(
    df_train: pd.DataFrame,
    time_column: str,
    col_target: str,
    lag_features: List[int],
    forecast_steps: int,
    catboost_params: dict
) -> pd.DataFrame:
    X_train, y_train, feature_columns, df_feat = prepare_data_for_catboost_diff(df_train, time_column, col_target, lag_features)

    model = CatBoostRegressor(
        iterations=catboost_params.get('iterations', 500),  # Снижено
        learning_rate=catboost_params.get('learning_rate', 0.03),
        depth=catboost_params.get('depth', 6),  # Снижено
        l2_leaf_reg=catboost_params.get('l2_leaf_reg', 1),
        random_seed=catboost_params.get('random_seed', 42),
        verbose=catboost_params.get('verbose', False),
        loss_function=catboost_params.get('loss_function', 'MAE')
    )
    model.fit(X_train, y_train)

    df_extended = df_train.sort_values(time_column).reset_index(drop=True).copy()
    df_extended['__pred_delta'] = np.nan
    predictions = []
    last_level = df_extended[col_target].dropna().iloc[-1]

    for step in range(forecast_steps):
        temp = create_features(df_extended.rename(columns={col_target: "__target_used"}), time_column, "__target_used", lag_features)
        
        # Подстановка лагов как в первом скрипте
        for lag in lag_features:
            lag_col = f"lag_{lag}"
            if lag_col in temp.columns:
                fact_vals = df_extended[col_target].shift(lag)
                pred_vals = df_extended["__target_used"].shift(lag)  # Используем __target_used вместо __pred_delta
                temp[lag_col] = fact_vals.combine_first(pred_vals)

        available_features = [c for c in feature_columns if c in temp.columns]
        last_row = temp.iloc[[-1]][available_features]

        if last_row.isna().any().any():
            last_row = last_row.fillna(X_train.mean())

        pred_delta = model.predict(last_row)[0]
        predictions.append(pred_delta)
        last_level += pred_delta

        next_time = df_extended[time_column].iloc[-1] + pd.Timedelta(seconds=calculate_time_interval(df_extended, time_column))
        new_row = pd.DataFrame({
            time_column: [next_time],
            col_target: [np.nan],
            '__pred_delta': [pred_delta]
        })
        df_extended = pd.concat([df_extended, new_row], ignore_index=True)

    forecast_dates = pd.date_range(
        start=df_train[time_column].iloc[-1] + pd.Timedelta(seconds=calculate_time_interval(df_train, time_column)),
        periods=forecast_steps,
        freq=f"{int(calculate_time_interval(df_train, time_column))}s"
    )
    cumulative = np.cumsum(predictions) + df_train.sort_values(time_column)[col_target].dropna().iloc[-1]
    return pd.DataFrame({time_column: forecast_dates, col_target: cumulative})

def lag_selection_catboost(
    df_init: pd.DataFrame,
    time_column: str,
    col_target: str,
    lag_search_depth: int,
    catboost_params: dict,
    forecast_method: str = "delta"
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

    max_lag = min(len(df_train) - 10, lag_search_depth)  
    max_lag = max(1, max_lag)

    best_lag = 1
    best_mape = float('inf')

    print('[INFO] lag evaluation is working')

    for lag in tqdm(range(1, max_lag + 1), bar_format='{l_bar}{n_fmt}/{total_fmt} ({percentage:3.0f}%)'):
        lag_features = list(range(1, lag + 1))
        try:
            if forecast_method == "level":
                df_pred = forecast_catboost_method(
                    df_train=df_train,
                    time_column=time_column,
                    col_target=col_target,
                    lag_features=lag_features,
                    forecast_steps=len(df_evaluation),
                    catboost_params=catboost_params
                )
            elif forecast_method == "delta":
                df_pred = forecast_catboost_method_diff(
                    df_train=df_train,
                    time_column=time_column,
                    col_target=col_target,
                    lag_features=lag_features,
                    forecast_steps=len(df_evaluation),
                    catboost_params=catboost_params
                )
            else:
                raise ValueError("forecast_method must be 'level' or 'delta'")

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
    lag_search_depth: int = 10,  
    forecast_method: str = "level"  
) -> dict:
    validate_input_data(df)

    forecast_horizon_time = standardize_datetime(forecast_horizon_time)
    forecast_horizon_time = pd.to_datetime(forecast_horizon_time)

    # Установка параметров CatBoost с учетом нового метода
    if forecast_method == "level":
        catboost_params = {
            'iterations': 500,  
            'learning_rate': 0.03,
            'depth': 6,  # Снижено
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': False,
            'loss_function': 'RMSE'
        }
    else:  # delta
        catboost_params = {
            'iterations': 500,  
            'learning_rate': 0.03,
            'depth': 6,  
            'l2_leaf_reg': 1,
            'random_seed': 42,
            'verbose': False,
            'loss_function': 'MAE'
        }

    print(f'[INFO] >>>> lag_selection_catboost is working (method: {forecast_method})')

    if lag_search_depth is None or lag_search_depth <= 1 or lag_search_depth > 50 or lag_search_depth < 0:
        lag = 1
        errors = {"mape": "unknown"}
    else:
        data_lag = lag_selection_catboost(
            df_init=df,
            time_column=time_column,
            col_target=col_target,
            lag_search_depth=lag_search_depth,
            catboost_params=catboost_params,
            forecast_method=forecast_method
        )
        lag = data_lag["best_lag"]
        errors = {"mape": data_lag["best_mape"]}

    lag_features = list(range(1, lag + 1))

    df = df.copy()
    df[col_target] = df[col_target].apply(clean_numeric_column)
    df = df.dropna(subset=[col_target])
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(by=time_column, ascending=True).reset_index(drop=True)

    time_point_interval = abs(calculate_time_interval(df, time_column))
    last_time = df[time_column].max()
    last_value = df[df[time_column] == last_time][[time_column, col_target]].iloc[0]

    forecast_duration = (forecast_horizon_time - last_time).total_seconds()
    n_forecast_points = int(forecast_duration / time_point_interval)
    if n_forecast_points <= 0:
        raise ValueError(
            f"Невозможно построить прогноз: горизонта ('{forecast_horizon_time}') недостаточно после последней известной даты ('{last_time}')."
        )

    if forecast_method == "level":
        df_predictions = forecast_catboost_method(
            df_train=df,
            time_column=time_column,
            col_target=col_target,
            lag_features=lag_features,
            forecast_steps=n_forecast_points,
            catboost_params=catboost_params
        )
    elif forecast_method == "delta":
        df_predictions = forecast_catboost_method_diff(
            df_train=df,
            time_column=time_column,
            col_target=col_target,
            lag_features=lag_features,
            forecast_steps=n_forecast_points,
            catboost_params=catboost_params
        )
    else:
        raise ValueError("forecast_method must be 'level' or 'delta'")

    new_row = pd.DataFrame({
        time_column: [pd.to_datetime(last_value[time_column])],
        col_target: [float(last_value[col_target])],
    })
    df_real_predict = pd.concat([new_row, df_predictions]).reset_index(drop=True)
    predictions = df_real_predict[[time_column, col_target]].to_dict(orient="records")

    return {
        "map_data": {
            "data": {"predictions": predictions},
            "errors": errors,
            "last_know_data": last_time,
            "title": f"CatBoost прогноз {col_target} ({forecast_method})",
            "legend": {
                "last_know_data_line": {"text": {"en": "Last known date", "ru": "Последняя известная дата"}, "color": "#A9A9A9"},
                "real_data_line": {"text": {"en": "Real data", "ru": "Реальные данные"}, "color": "#0000FF"},
            },
        },
    }