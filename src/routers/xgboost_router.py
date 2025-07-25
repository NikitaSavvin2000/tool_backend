# src/routers/xgboost_router.py

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from src.models.schemes import PredictRequest
from src.services.xgboost_service import run_xgboost_forecast
from src.core.logger import logger
import traceback

router = APIRouter()

@router.post("/predict-xgboost", response_model=dict)
async def predict_xgboost(body: PredictRequest = Body(...)) -> Dict[str, Any]:
    """
    Эндпоинт для прогнозирования временного ряда с использованием XGBoost.
    
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
        - lag_search_depth (Optional[int]): Глубина поиска лагов.
    
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
        "forecast_horizon_time": "2017-01-01 01:00:00",
        "lag_search_depth": 2
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
    
    Raises:
    - **HTTPException 400**: Если переданы некорректные данные или произошла ошибка при прогнозировании.
    """
    try:
        # Логирование входного запроса
        logger.info("Received request for XGBoost prediction")

        # Преобразование входных данных в DataFrame
        df = pd.DataFrame(body.df)

        # Выполнение прогноза
        result = run_xgboost_forecast(
            df=df,
            time_column=body.time_column,
            col_target=body.col_target,
            forecast_horizon_time=body.forecast_horizon_time,
            lag_search_depth=body.lag_search_depth,
        )

        return result

    except Exception as e:
        logger.error("🔥 Ошибка во время XGBoost предсказания:")
        traceback.print_exc()
        logger.error(f"📋 Сообщение: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))