# src/services/catboost/runners.py
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.services.catboost.catboost_forecast import user_predict_catboost
import plotly.graph_objects as go
from src.utils.metrics import calculate_metrics
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt
from src.utils.possible_forecast_date import calculate_time_interval
from statsmodels.tsa.seasonal import seasonal_decompose
import logging

logger = logging.getLogger(__name__)

func_to_predict = user_predict_catboost

home_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
data_path = os.path.join(home_path, "src", "data")
experiment_path = os.path.join(home_path, "src", "services", "catboost", "experiments")

os.makedirs(experiment_path, exist_ok=True)

collection_data = {
    "italy": {
        "df": pd.read_csv(os.path.join(data_path, "load_consumption_2025.csv")),
        "time_column": "datetime",
        "col_target": "load_consumption",
        "lag_search_depth": 0,
        "points_to_predict": 288
    },
    "ats": {
        "df": pd.read_csv(os.path.join(data_path, "DEKENERG_ZONE2_S_PAMURENE.csv")),
        "time_column": "datetime",
        "col_target": "VC_факт",
        "lag_search_depth": 0,
        "points_to_predict": 300
    }
}

def test_response_structure(response):
    assert "map_data" in response, "Ответ должен содержать ключ 'map_data'"
    assert isinstance(response["map_data"], list), "'map_data' должен быть списком"
    assert len(response["map_data"]) > 0, "'map_data' не должен быть пустым"
    assert "errors" in response, "Ответ должен содержать ключ 'errors'"

    first = response["map_data"][0]
    assert "timestamp" in first, "Каждый элемент должен содержать 'timestamp'"
    assert "value" in first, "Каждый элемент должен содержать 'value'"


if __name__ == "__main__":
    for data_name_to_experiment in collection_data:
        df = collection_data[data_name_to_experiment]["df"]
        time_column = collection_data[data_name_to_experiment]["time_column"]
        col_target = collection_data[data_name_to_experiment]["col_target"]
        points_to_predict = collection_data[data_name_to_experiment]["points_to_predict"]
        lag_search_depth = collection_data[data_name_to_experiment]["lag_search_depth"]

        df = df.sort_values(by=time_column, ascending=False)
        forecast_horizon_time = df[time_column].iloc[0]

        df_true = df[:points_to_predict]
        df_train = df[points_to_predict:]
        df_to_show = df_train[:int(points_to_predict * 1.5)]

        # === ПОСТРОЕНИЕ ТРЕНДА ЧЕРЕЗ seasonal_decompose ===
        df_train_for_trend = df_train.copy().sort_values(time_column).reset_index(drop=True)
        df_train_for_trend = df_train_for_trend.set_index(time_column)
        df_train_for_trend = df_train_for_trend[[col_target]].dropna()

        # Убедимся, что индекс имеет частоту
        try:
            if df_train_for_trend.index.freq is None:
                inferred_freq = pd.infer_freq(df_train_for_trend.index)
                if inferred_freq is None:
                    interval_sec = calculate_time_interval(df_train, time_column)
                    if interval_sec == 3600:
                        inferred_freq = 'H'
                    elif interval_sec == 1800:
                        inferred_freq = '30T'
                    elif interval_sec == 900:
                        inferred_freq = '15T'
                    elif interval_sec == 60:
                        inferred_freq = 'T'
                    else:
                        inferred_freq = f'{int(interval_sec)}S'
                df_train_for_trend = df_train_for_trend.asfreq(inferred_freq)
        except Exception as e:
            logger.warning(f"Не удалось определить частоту: {e}. Используем rolling mean.")
            trend_series = df_train_for_trend[col_target].rolling(window=24, center=True).mean()
            df_trend_line = trend_series.reset_index()
            df_trend_line.columns = [time_column, "trend_value"]
        else:
            # Период для seasonal_decompose
            n_obs = len(df_train_for_trend.dropna())
            period = min(24, max(2, n_obs // 2)) if n_obs > 4 else 2

            try:
                decomposition = seasonal_decompose(
                    df_train_for_trend[col_target],
                    model='additive',
                    period=period,
                    extrapolate_trend='freq'
                )
                trend_series = decomposition.trend
            except Exception as e:
                logger.warning(f"seasonal_decompose не сработал: {e}. Используем rolling mean.")
                trend_series = df_train_for_trend[col_target].rolling(window=min(24, n_obs // 2), center=True).mean()

            # Преобразуем тренд в датафрейм для графика
            df_trend_line = trend_series.reset_index()
            df_trend_line.columns = [time_column, "trend_value"]

        result = func_to_predict(
            df=df_train,
            time_column=time_column,
            col_target=col_target,
            forecast_horizon_time=forecast_horizon_time,
            lag_search_depth=lag_search_depth
        )

        test_response_structure(result)

        df_pred = pd.DataFrame(result["map_data"])
        df_pred = df_pred.rename(columns={"timestamp": time_column, "value": col_target})
        df_pred = df_pred.sort_values(by=time_column, ascending=False).reset_index(drop=True)

        df_true = df_true.sort_values(by=time_column, ascending=False).reset_index(drop=True)

        # --- СРАЗУ приводим всё к datetime ---
        df_true[time_column] = pd.to_datetime(df_true[time_column])
        df_pred[time_column] = pd.to_datetime(df_pred[time_column])
        df_trend_line[time_column] = pd.to_datetime(df_trend_line[time_column])

        y_pred = np.array(df_pred[col_target].tolist())
        y_pred = y_pred[1:]
        y_true = np.array(df_true[col_target].tolist())

        df_trend_line_clean = df_trend_line.dropna()
        if not df_trend_line_clean.empty:

            tail_size = max(10, len(df_trend_line_clean) // 5)
            df_tail = df_trend_line_clean.tail(tail_size)

            df_tail[time_column] = pd.to_datetime(df_tail[time_column])
            X = df_tail[time_column].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
            y = df_tail["trend_value"].values


            model = Holt(y, exponential=False, damped_trend=False)  
            fit = model.fit(smoothing_level=0.3, smoothing_trend=0.2)  

            max_time = max(df_true[time_column].max(), df_pred[time_column].max())
            future_range = pd.date_range(
                start=df_trend_line_clean[time_column].max(),
                end=max_time,
                freq=pd.infer_freq(df_trend_line_clean[time_column]) or "D"
            )

            if len(future_range) > 0:
                X_future = future_range.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
                y_future = fit.forecast(len(future_range))

                df_future_trend = pd.DataFrame({
                    time_column: future_range,
                    "trend_value": y_future
                })

                df_trend_line = pd.concat([df_trend_line, df_future_trend], ignore_index=True)

        try:
            metrics = calculate_metrics(
                y_true=y_true,
                y_pred=y_pred
            )
            file_path = os.path.join(experiment_path, f"{data_name_to_experiment}_metrics.md")
            with open(file_path, "w") as f:
                f.write(f"# Метрики для данных {data_name_to_experiment}\n\n")
                f.write("| Метрика | Значение |\n")
                f.write("|---------|---------|\n")
                for k, v in metrics.items():
                    f.write(f"| {k} | {float(v):.3f} |\n")
        except Exception as e:
            print(e)



        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_to_show[time_column],
            y=df_to_show[col_target],
            mode='lines',
            name='Train',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=df_true[time_column],
            y=df_true[col_target],
            mode='lines',
            name='True',
            line=dict(color='red')
        ))

        fig.add_trace(go.Scatter(
            x=df_pred[time_column],
            y=df_pred[col_target],
            mode='lines',
            name='Predicted',
            line=dict(color='orange')
        ))

        # === ДОБАВЛЯЕМ ЛИНИЮ ТРЕНДА ===
        fig.add_trace(go.Scatter(
            x=df_trend_line[time_column],
            y=df_trend_line["trend_value"],
            mode='lines',
            name='Trend (decompose)',
            line=dict(color='green', dash='dot', width=2)
        ))

        # Определяем начало обучающих данных
        x_start = pd.Timestamp('2016-05-30')
        x_end = pd.Timestamp('2016-07-01')

        fig.update_layout(
            title='Сравнение Train / True / Predicted с трендом',
            xaxis_title='Time',
            yaxis_title=col_target,
            legend_title='Dataset',
            template='plotly_white',
            xaxis_range=[x_start, x_end]
        )

        path_to_save_fig = os.path.join(experiment_path, f"chart_{data_name_to_experiment}.html")
        fig.write_html(path_to_save_fig)
