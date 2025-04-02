import os
import requests
import pandas as pd



url_backend = os.getenv("BACKEND_URL", 'http://0.0.0.0:7071/backend/v1')

docs = 'http://0.0.0.0:7071/backend/v1/'



df = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQT1DfqAB5Yec8MIQ_E5A8w-SXNcRmTwbXsv2W-ZT1ZcXN_G83BHlb6QBgnWkO-MpH3oVgfLoE0SnLx/pub?gid=1952392108&single=true&output=csv")


def call_cols_to_chose(df: pd.DataFrame):
    url_backend = "http://0.0.0.0:7071/backend/v1"
    url = url_backend + '/cols_to_chose'
    df_records = df.to_dict(orient='records')
    data = {
        "df": df_records
    }
    try:
        response = requests.post(url, json=data)
        return response
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе: {e}")
        return None


def func_convert_time_to_datetime(df: pd.DataFrame, time_column: str):

    url = url_backend + '/convert_time_to_datetime'
    df_records = df.to_dict(orient='records')

    data = {
        "df": df_records,
        "time_column": time_column
    }

    try:
        response = requests.post(url, json=data)
        return response

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе: {e}")
        return None


def func_generate_possible_date(df: pd.DataFrame, time_column: str):

    url = url_backend + '/generate_possible_date'
    df_records = df.to_dict(orient='records')

    data = {
        "df": df_records,
        "time_column": time_column
    }

    try:
        response = requests.post(url, json=data)
        return response

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе: {e}")
        return None



df = df[['Datetime']]

time_column = "Datetime"
response = func_generate_possible_date(df=df, time_column=time_column)

if response.status_code == 200:
    print(response.json())




