import requests

import pandas as pd
import plotly.graph_objects as go

df_init = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQ1WwtR82Xu30hLodJ3nU_f060vXdzveNh7ctZzajHgNfo2YrK7je_hB7NpMA37og/pub?gid=247784248&single=true&output=csv")
df_init = df_init.rename(columns={"День": "date", "Сумма заказов минус комиссия WB, руб.": "target"})
df_init = df_init[["date", "target"]]

df_init['date'] = pd.to_datetime(df_init['date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d %H:%M:%S')

print(df_init.head(-5))


time_column = "date"
col_target = "target"
forecast_horizon_time = '2025-12-14 00:00:00'


base_url = "http://0.0.0.0:7078/backend/v1"

def func_generate_forecast(df: pd.DataFrame, time_column: str, col_target: str, forecast_horizon_time: str):
    url = f"{base_url}/generate_forecast"
пше
    # df = df.loc[:10]

    df_records = df.to_dict(orient='records')
    print(df_records)

    data = {
        "df": df_records,
        "time_column": time_column,
        "col_target": col_target,
        "forecast_horizon_time": forecast_horizon_time
    }

    try:
        response = requests.post(url, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Ошибка при запросе: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе: {e}")
        return None



predict_dict = func_generate_forecast(
    df=df_init,
    time_column=time_column,
    col_target=col_target,
    forecast_horizon_time=forecast_horizon_time
)


predictions= predict_dict["predictions"]
df_predictions = pd.DataFrame(predictions)


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=pd.to_datetime(df_init[time_column]),
    y=df_init[col_target],
    mode='lines',
    name='Forecast',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=pd.to_datetime(df_predictions[time_column]),
    y=df_predictions[col_target],
    mode='lines',
    name='Forecast',
    line=dict(color='orange')
))

fig.show()