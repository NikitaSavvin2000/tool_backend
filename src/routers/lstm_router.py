import os
import pandas as pd
import logging
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from src.lstm_selection_of_parameters.main import user_predict_LSTM
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
home_path = os.getcwd()

example_df = pd.read_csv(f'{home_path}/src/examples_data/example_data.csv')
example_df = example_df.drop(columns=["Unnamed: 0"])
example_df_long = example_df[:1000]

example_df_json_long = example_df_long.to_dict(orient="records")


class PredictRequest(BaseModel):
    df: List[Dict]
    time_column: str
    col_target: str
    forecast_horizon_time: str


@router.post("/predict-lstm", response_model=dict)
async def predict_xgboost(request: PredictRequest = Body(...,
                                                         example={
                                                             "time_column": "time",
                                                             "col_target": "load_consumption",
                                                             "forecast_horizon_time": "2022-09-10 05:55:00",
                                                             "df": example_df_json_long
                                                         }
                                                         )):
    """
    Генерирует прогноз временного ряда с использованием pipeline Horizon на базе LSTmM.

    Описание:
    ----------
    Функция принимает временной ряд, нормализует данные, формирует будущий интервал времени и
    выполняет прогнозирование. Возвращает результат в формате JSON, который включает последние известные
    данные и прогнозируемые значения для визуализации на фронтенде.

    Параметры:
    ----------
    1. `df` (pd.DataFrame) — Исходный DataFrame, содержащий временной ряд с целевой переменной.
    2. `time_column` (str) — Название столбца, содержащего временные метки.
    3. `col_target` (str) — Название целевой переменной, по которой строится прогноз.
    4. `forecast_horizon_time` (str) — Временная граница прогнозирования в формате **yyyy-mm-dd hh:mm:ss** .

    Возвращает:
    ----------
    - map_data (dict) - словапь данных для отрисовки в словаре содержаться:
    1. `data` (dict):
            - **last_real_data** (list[dict]) — Последние известны еданные пользователя
            - **predictions** (list[dict]) — Данные предсказания.
    2. `errors` (dict): - процент ошибки на тестировании
            - **mape** float— mape
    3. `last_know_data_line` (dict) — линия, обозначающая последнюю известную дату:
        - **text** (dict):
            - **en** (str) — описание на английском языке.
            - **ru** (str) — описание на русском языке.
        - **color** (str) — цвет линии, разделяющей реальные данные и прогноз.
    4. `real_data_line` (dict) — линия реальных данных:
        - **text** (dict):
            - **en** (str) — описание на английском языке.
            - **ru** (str) — описание на русском языке.
        - **color** (str) — цвет линии реальных данных на графике.
    5. `predict_data_line` (dict) — линия прогнозируемых данных:
        - **text** (dict):
            - **en** (str) — описание на английском языке.
            - **ru** (str) — описание на русском языке.
        - **color** (str) — цвет линии прогнозируемых данных.


    Пример вызова API python:
    ------------------
    ```
    import requests
    import pandas as pd

    def func_generate_forecast(df: pd.DataFrame, time_column: str, col_target: str, forecast_horizon_time: str):
        url = "http://your_backend_url/backend/v1/generate_forecast"

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

    df = pd.read_csv(<here yor data>)

    time_column = 'time'
    col_target = 'load_consumption'
    forecast_horizon_time = '2022-09-10 05:00:00'
    df[time_column] = pd.to_datetime(df[time_column])

    response = func_generate_forecast(df, time_column, col_target, forecast_horizon_time)
    ```
    """


    logger.info("Received request for prediction")

    try:
        df = pd.DataFrame(request.df)

        result = await user_predict_LSTM(
            df=df,
            time_column=request.time_column,
            col_target=request.col_target,
            forecast_horizon_time=request.forecast_horizon_time
        )

        return result

    except Exception as e:
        logger.error("🔥 Ошибка во время предсказания:")
        traceback.print_exc()
        logger.error("📋 Сообщение:", str(e))
        raise HTTPException(status_code=400, detail=str(e))
