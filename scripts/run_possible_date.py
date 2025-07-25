#scripts/run_possible_date.py
import pandas as pd
import requests


def func_generate_possible_date(df: pd.DataFrame, time_column: str, token: str):
    url_backend = ''
    url = url_backend + '/generate_possible_date'
    df_records = df.to_dict(orient='records')

    data = {
        "df": df_records,
        "time_column": time_column
    }

    headers = {
        "Authorization": f"Bearer {token}"
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе: {e}")
        return None

df = pd.read_csv("")
time_column = "Datetime"
token = ""
response = func_generate_possible_date(df, time_column, token)
print(response)
