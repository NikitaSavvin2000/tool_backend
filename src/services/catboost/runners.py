import os
import numpy as np
import pandas as pd
from src.services.catboost.catboost_forecast6 import user_predict_XGBoost 
import plotly.graph_objects as go
from src.utils.metrics import calculate_metrics

func_to_predict = user_predict_XGBoost

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
    map_data = response["map_data"]

    assert "data" in map_data, "В 'map_data' отсутствует ключ 'data'"
    assert "predictions" in map_data["data"], "В 'data' отсутствует ключ 'predictions'"

    assert "errors" in map_data, "В 'map_data' отсутствует ключ 'errors'"
    assert "last_know_data" in map_data, "В 'map_data' отсутствует ключ 'last_know_data'"
    assert "title" in map_data, "В 'map_data' отсутствует ключ 'title'"
    assert "legend" in map_data, "В 'map_data' отсутствует ключ 'legend'"

    legend = map_data["legend"]
    assert "last_know_data_line" in legend, "В 'legend' отсутствует ключ 'last_know_data_line'"
    assert "real_data_line" in legend, "В 'legend' отсутствует ключ 'real_data_line'"

    for line in ["last_know_data_line", "real_data_line"]:
        assert "text" in legend[line], f"В '{line}' отсутствует ключ 'text'"
        assert "en" in legend[line]["text"], f"В '{line}.text' отсутствует ключ 'en'"
        assert "ru" in legend[line]["text"], f"В '{line}.text' отсутствует ключ 'ru'"
        assert "color" in legend[line], f"В '{line}' отсутствует ключ 'color'"

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
        df_to_show = df_train[:int(points_to_predict*1.5)]

        result = func_to_predict(
            df=df_train,
            time_column=time_column,
            col_target=col_target,
            forecast_horizon_time=forecast_horizon_time,
            lag_search_depth=lag_search_depth,
        )

        test_response_structure(result)

        df_pred = pd.DataFrame(result["map_data"]["data"]["predictions"])

        df_pred = df_pred.sort_values(by=time_column, ascending=False).reset_index(drop=True)
        df_true = df_true.sort_values(by=time_column, ascending=False).reset_index(drop=True)
        y_pred = np.array(df_pred[col_target].tolist())
        y_pred = y_pred[1:] 
        y_true = np.array(df_true[col_target].tolist())

        try:
            metrics = calculate_metrics(
                y_true=y_true,
                y_pred=y_pred
            )
            file_path = os.path.join(experiment_path, f"{data_name_to_experiment}_metrics.md")
            with open(file_path, "w") as f:
                f.write(f"# Метрики для для данных {data_name_to_experiment}\n\n")
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

        fig.update_layout(
            title='Сравнение Train / True / Predicted',
            xaxis_title='Time',
            yaxis_title=col_target,
            legend_title='Dataset',
            template='plotly_white'
        )

        path_to_save_fig = os.path.join(experiment_path, f"chart_{data_name_to_experiment}.html")
        fig.write_html(path_to_save_fig)