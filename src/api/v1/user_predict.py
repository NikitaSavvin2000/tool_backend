# src/api/v1/user_predict.py

from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, Body, HTTPException

from src.core.logger import logger
from src.models.schemes import UserPredictRequest
from src.services.user_predict_service import run_user_forecast

router = APIRouter()

@router.post("/user_forecast")
async def user_predict(body: UserPredictRequest = Body(...)) -> Dict[str, Any]:
    """
    Эндпоинт для пользовательского прогнозирования временных рядов.
    
    Description:
    - Принимает данные в формате JSON и выполняет прогнозирование с использованием выбранных параметров.
    - Возвращает результаты прогноза, включая последние известные данные и предсказания.
    
    Parameters:
    - **body (UserPredictRequest)**: Схема запроса, содержащая следующие поля:
        - df (List[Dict]): Входной DataFrame в формате JSON.
        - time_column (str): Название временной колонки.
        - col_target (str): Название целевой колонки.
        - forecast_horizon_time (str): Горизонт прогнозирования в формате `YYYY-MM-DD HH:MM:SS`.
        - col_for_train (List[str]): Список колонок для обучения модели.
    
    Returns:
    - **dict**: Словарь с результатами прогноза:
        - map_data (dict): Данные для отрисовки графика.
        - last_know_data (str): Последнее известное значение.
        - title (str): Заголовок графика.
        - legend (dict): Легенда графика.
    
    Example Request:
    ```json
    {
        "df": [
            {"Datetime": "2017-01-01 00:00:00", "Temperature": 6.4865, "Humidity": 74.15},
            {"Datetime": "2017-01-01 00:15:00", "Temperature": 6.5, "Humidity": 73.9}
        ],
        "time_column": "Datetime",
        "col_target": "Temperature",
        "forecast_horizon_time": "2017-01-01 01:00:00",
        "col_for_train": ["Datetime", "Humidity"]
    }
    ```
    
    Example Response:
    ```json
    {
        "map_data": {
            "data": {
                "last_real_data": [
                    {"Datetime": "2017-01-01 00:15:00", "Temperature": 6.5, "Humidity": 73.9}
                ],
                "predictions": [
                    {"Datetime": "2017-01-01 00:30:00", "Temperature": 6.55, "Humidity": 73.8}
                ]
            },
            "last_know_data": "2017-01-01 00:15:00",
            "title": "Реальный прогноз Temperature",
            "legend": {
                "last_know_data_line": {
                    "text": {"en": "Last known date", "ru": "Последняя известная дата"},
                    "color": "#A9A9A9"
                },
                "real_data_line": {
                    "text": {"en": "Real data", "ru": "Реальные данные"},
                    "color": "#0000FF"
                },
                "predict_data_line": {
                    "text": {"en": "Current forecast", "ru": "Актуальный прогноз"},
                    "color": "#FF0000"
                }
            }
        }
    }
    ```
    
    Raises:
    - **HTTPException 400**: Если входные данные некорректны или произошла ошибка при прогнозировании.
    """
    try:
        # Преобразование входных данных в DataFrame
        df = pd.DataFrame(body.df)
        
        # Проверка на пустые данные
        if df.empty:
            raise HTTPException(status_code=400, detail="Входной DataFrame не может быть пустым.")
        
        # Выполнение пользовательского прогноза
        result = run_user_forecast(
            df=df,
            time_column=body.time_column,
            col_target=body.col_target,
            forecast_horizon_time=body.forecast_horizon_time,
            col_for_train=body.col_for_train,
        )
        
        return result

    except Exception as e:
        logger.error(f"Ошибка в эндпоинте /user_forecast: {e}")
        raise HTTPException(status_code=400, detail=str(e))