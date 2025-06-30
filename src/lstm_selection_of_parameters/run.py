import plotly.graph_objects as go
import pandas as pd
import os

from src.lstm_selection_of_parameters.cols_selection import user_predict_LSTM, calculate_metrics
home_path = os.getcwd()



# "Morocco Zone 1": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSgwB47qVFZcr1Aq--UWxZ6fDi9CGLZm-1i8QoMgfdaHUbV8EqSli3ayPxYYxD8kqfYYHD41uuNxbjZ/pub?gid=1952392108&single=true&output=csv",
#     "Morocco Zone 2": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQT1DfqAB5Yec8MIQ_E5A8w-SXNcRmTwbXsv2W-ZT1ZcXN_G83BHlb6QBgnWkO-MpH3oVgfLoE0SnLx/pub?gid=1952392108&single=true&output=csv",
#     "Morocco Zone 3": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQSHw5k7n3_RM6ksGbvdQJsa1i9-zF-18CFLCFnXFkCxQwqLcQ4Wu2_8EF2H1lF02ih2NLL9BDecFzQ/pub?gid=1952392108&single=true&output=csv",
# wildberies: "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJrlRwIHeCwf3DUiu_WkG_bwgKcyOmXKv8aKN5GSjTbqCvae9OiTSkHoaMpMfOstTvRvGnj6-3gtRk/pub?gid=1448488998&single=true&output=csv"

df_data = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vSgwB47qVFZcr1Aq--UWxZ6fDi9CGLZm-1i8QoMgfdaHUbV8EqSli3ayPxYYxD8kqfYYHD41uuNxbjZ/pub?gid=1952392108&single=true&output=csv")


time_column = 'Datetime'
col_target = 'consumption'
print(df_data)

df_data = df_data[[time_column, col_target]]

print(df_data)

forecast_horizon_time = '2017-12-30 23:45:00 '

test_len = 96
df_to_predict = df_data.iloc[:-test_len]
df_test = df_data.iloc[-test_len:]


predict_dict = user_predict_LSTM(
    df=df_to_predict,
    time_column=time_column,
    col_target=col_target,
    forecast_horizon_time=forecast_horizon_time
)

predictions= predict_dict["map_data"]["data"]["predictions"]
df_predictions = pd.DataFrame(predictions)
df_predictions = df_predictions.iloc[1:]

print('=============================== ИТОГ - df_test')
print(df_test)
print('=============================== ИТОГ - df_predictions')

print(df_predictions)

y_true = df_test[col_target].reset_index(drop=True)
y_pred = df_predictions[col_target].reset_index(drop=True)

rmse, r2, mae, mape, wmape = calculate_metrics(y_true=y_true, y_pred=y_pred)


print('=============== METRIX =================')
print(f'MAPE = {mape}')
print(f'R2 = {r2}')
print(f'RMSE = {rmse}')
print(f'MAE = {mae}')
print(f'WMAPE = {wmape}')
print('========================================')

fig = go.Figure()

fig.add_trace(go.Scatter(
    y=y_true,
    mode='lines',
    name='Реальные данные',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    y=y_pred,
    mode='lines',
    name='Прогноз',
    line=dict(color='orange')
))

fig.update_layout(
    title='Сравнение прогноза и реальных значений',
    xaxis_title='Индекс',
    yaxis_title='Значение',
    legend=dict(x=0, y=1)
)

fig.show()
