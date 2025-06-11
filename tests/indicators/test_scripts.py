import requests
import pandas as pd


def func_generate_forecast(df: pd.DataFrame, time_column: str, col_target: str, forecast_horizon_time: str):
    # url = "http://0.0.0.0:7071/backend/v1/generate_forecast"

    url = "http://0.0.0.0:7071/backend/v1/backend/v1/pipeline/generate_forecast"
    

    df[time_column] = df[time_column].astype(str)
    df_records = df.to_dict(orient='records')


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

""" Тестовые данные

    "Australia Bundoora": "https://docs.google.com/spreadsheets/d/e/2PACX-1vTl3ZMKUEqYeXJe1b8A4IbfYIKjWlm0lR61glDoXOEfHxsmDUv1ZZ2IK2GpjkH2fZ6fvX3NaCOryqzW/pub?gid=751874949&single=true&output=csv",
    "Australia Albury-Wodonga": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQmRJCXCBp-qsY4LQrf8x_zJax_5FAnZDl6-sv1zje9m0pCM7hore-cjS3zlzJezgHIm6h81KY1hsEz/pub?gid=1184660391&single=true&output=csv",
    "Australia Bendigo": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQgXCJsm0V7ylsqvzRzK_LHZzky0lABeXvRuiqRWzDumN1Y8i8xul-Ih1ERIU1v-C46AKISnOOzBmtb/pub?gid=1902219272&single=true&output=csv",
    "Morocco Zone 1": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSgwB47qVFZcr1Aq--UWxZ6fDi9CGLZm-1i8QoMgfdaHUbV8EqSli3ayPxYYxD8kqfYYHD41uuNxbjZ/pub?gid=1952392108&single=true&output=csv",
    "Morocco Zone 2": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQT1DfqAB5Yec8MIQ_E5A8w-SXNcRmTwbXsv2W-ZT1ZcXN_G83BHlb6QBgnWkO-MpH3oVgfLoE0SnLx/pub?gid=1952392108&single=true&output=csv",
    "Morocco Zone 3": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQSHw5k7n3_RM6ksGbvdQJsa1i9-zF-18CFLCFnXFkCxQwqLcQ4Wu2_8EF2H1lF02ih2NLL9BDecFzQ/pub?gid=1952392108&single=true&output=csv",
    "load_consumption_temp": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRFF6SXvGbQgQG1bh0hwbXgVWpUU_UG8OQAVhHcNAcAFT5x-XoIYxMAeF-goym6_wNhJEwQd3iGHp9b/pub?gid=791211028&single=true&output=csv",

"""

df = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vSgwB47qVFZcr1Aq--UWxZ6fDi9CGLZm-1i8QoMgfdaHUbV8EqSli3ayPxYYxD8kqfYYHD41uuNxbjZ/pub?gid=1952392108&single=true&output=csv")

time_column = 'Datetime'
col_target = 'consumption'
forecast_horizon_time = '2018-01-10 05:00:00'
df[time_column] = pd.to_datetime(df[time_column])

response = func_generate_forecast(df, time_column, col_target, forecast_horizon_time)

print(response['map_data'].keys()) # dict_keys(['data', 'last_know_data', 'title', 'legend'])

df = pd.DataFrame(response['map_data']['data']['last_real_data'])
df_predict = pd.DataFrame(response['map_data']['data']['predictions'])

print(df_predict)