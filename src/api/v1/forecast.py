# src/api/v1/forecast.py

from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException

from src.core.logger import logger
from src.models.schemes import ForecastRequest
from src.services.forecast_service import run_forecast

router = APIRouter()

@router.post("/forecast", response_model=dict, tags=["Forecast"])
async def forecast_endpoint(body: ForecastRequest = Body(...)) -> Dict[str, Any]:
    """
    Эндпоинт для прогнозирования временных рядов.
    
    Description:
    - Принимает данные временного ряда, параметры модели и выполняет прогнозирование.
    - Возвращает результаты прогноза в формате JSON.
    
    Parameters:
    - **body (ForecastRequest)**: Схема запроса, содержащая следующие поля:
        - col_target (str): Название целевой колонки.
        - evaluation_index (int): Индекс последней известной точки данных.
        - last_know_index (int): Индекс последней известной точки времени.
        - epochs (int): Количество эпох обучения.
        - lag (int): Значение лага для прогнозирования.
        - activation (str): Функция активации.
        - optimizer (str): Оптимизатор.
        - dropout_count (float): Коэффициент Dropout.
        - model_architecture_params (List[Dict]): Параметры архитектуры модели.
        - json_list_df_all_data_norm (List[Dict]): Нормализованный DataFrame в формате JSON.
    
    Returns:
    - **dict**: Результат прогнозирования, содержащий:
        - map_data (dict): Данные для отрисовки графика.
        - last_know_data (str): Последнее известное значение.
        - title (str): Заголовок графика.
        - legend (dict): Легенда графика.
    
    Example Request:
    ```json
    {
        "col_target": "load_consumption",
        "evaluation_index": 7,
        "last_know_index": 9,
        "epochs": 5,
        "lag": 1,
        "activation": "relu",
        "optimizer": "adam",
        "dropout_count": 0.01,
        "model_architecture_params": [
            {"layer": 1, "type": "Bi-LSTM", "neurons": 2},
            {"layer": 2, "type": "Bi-LSTM", "neurons": 4},
            {"layer": 3, "type": "Bi-LSTM", "neurons": 8}
        ],
        "json_list_df_all_data_norm": [
            {"load_consumption": 0.6800409376, "year": 0.984, "week": 0.6274509804},
            {"load_consumption": 0.6800409376, "year": 0.984, "week": 0.6274509804}
        ]
    }
    ```
    
    Example Response:
    ```json
    {
        "map_data": {
            "data": {
                "last_real_data": [0.6800409376, 0.6800409376],
                "predictions": [0.6800409376, 0.6800409376]
            },
            "last_know_data": "2023-01-01 00:00:00",
            "title": "Прогноз load_consumption",
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
    - **HTTPException 400**: Если входные данные пустые или произошла ошибка при прогнозировании.
    """
    try:
        # Выполнение прогнозирования
        result = run_forecast(
            col_target=body.col_target,
            df_all_data_norm=body.json_list_df_all_data_norm,
            evaluation_index=body.evaluation_index,
            last_know_index=body.last_know_index,
            epochs=body.epochs,
            lag=body.lag,
            activation=body.activation,
            optimizer=body.optimizer,
            dropout_count=body.dropout_count,
            model_architecture_params=body.model_architecture_params,
        )
        return result

    except Exception as e:
        logger.error(f"Ошибка в эндпоинте /forecast: {e}")
        raise HTTPException(status_code=400, detail=str(e))

