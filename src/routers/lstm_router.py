# src/routers/lstm_router.py

from typing import Any, Dict
from fastapi import APIRouter, Body

from src.models.schemes import LSTMPredictRequest 
from src.core.base_handler import BaseHandler
from src.core.decorators.log_decorators import log_endpoint
from src.core.decorators.exception_decorators import handle_exceptions
from src.services.lstm_service import run_lstm_forecast

router = APIRouter()
base_handler = BaseHandler()

@router.post("/", response_model=dict)
@log_endpoint() 
@handle_exceptions 
async def predict_lstm(body: LSTMPredictRequest = Body(...)) -> Dict[str, Any]:
    """
    Эндпоинт для прогнозирования временного ряда с использованием LSTM.
    
    Description:
    - Принимает временной ряд, нормализует данные, формирует будущий интервал времени
      и выполняет прогнозирование.
    - Возвращает результат в формате JSON, который включает последние известные данные
      и предсказания.
    
    Parameters:
    - **body (PredictRequest)**: Схема запроса, содержащая следующие поля:
        - df (List[Dict]): Входной DataFrame в формате JSON.
        - time_column (str): Название временной колонки.
        - col_target (str): Название целевой колонки.
        - forecast_horizon_time (str): Горизонт прогнозирования.
    
    Returns:
    - **dict**: Результат прогнозирования, содержащий:
        - map_data (dict): Данные для отрисовки графика.
        - last_know_data (str): Последнее известное значение.
        - title (str): Заголовок графика.
        - legend (dict): Легенда графика.
    
    Example Request:
    ```json
    {
        "df": [
            {"Datetime": "2017-01-01 00:00:00", "Temperature": 6.4865},
            {"Datetime": "2017-01-01 00:15:00", "Temperature": 6.5}
        ],
        "time_column": "Datetime",
        "col_target": "Temperature",
        "forecast_horizon_time": "2017-01-01 01:00:00"
    }
    ```
    
    Example Response:
    ```json
    {
        "map_data": {
            "data": {
                "last_real_data": [
                    {"Datetime": "2017-01-01 00:15:00", "Temperature": 6.5}
                ],
                "predictions": [
                    {"Datetime": "2017-01-01 00:30:00", "Temperature": 6.55}
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
    """
    df = base_handler.parse_and_validate_dataframe(body.df, df_name="Input DataFrame")

    result = run_lstm_forecast(
        df=df,
        time_column=body.time_column,
        col_target=body.col_target,
        forecast_horizon_time=body.forecast_horizon_time,
    )

    return result