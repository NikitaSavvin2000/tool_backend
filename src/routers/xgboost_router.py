# src/routers/xgboost_router.py

import pandas as pd
import logging
from fastapi import APIRouter, Body
from pydantic import BaseModel
from typing import List, Dict
from src.xgboost_selection_of_parameters.main import user_predict_XGBoost

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class PredictRequest(BaseModel):
    df: List[Dict]
    time_column: str
    col_target: str
    forecast_horizon_time: str

@router.post("/predict-xgboost/", response_model=dict)
async def predict_xgboost(request: PredictRequest = Body(...,
    example={
        "df": [
            {"Дата": "2023-01-01", "Цена": 100},
            {"Дата": "2023-01-02", "Цена": 105}
        ],
        "time_column": "Дата",
        "col_target": "Цена",
        "forecast_horizon_time": "2024-01-01"
    }
)):
    """
    Эндпоинт для прогнозирования временного ряда с использованием XGBoost.

    Args:
        request (PredictRequest): Запрос на прогнозирование
            - df (List[Dict]): Исходные данные в формате списка словарей
            - time_column (str): Название колонки с временными метками
            - col_target (str): Название целевой переменной
            - forecast_horizon_time (str): Временная граница прогнозирования

    Returns:
        dict: Результаты прогнозирования
            - map_data (dict): Данные для построения графиков
                - data (dict):
                    - predictions (List[Dict]): Прогнозные значения
                - last_know_data (str): Последняя известная дата
                - title (str): Заголовок графика
                - legend (dict): Легенда графика
    """

    logger.info("Received request for prediction")
    
    try:
            # Преобразование входных данных в DataFrame
            df = pd.DataFrame(request.df)

            # Выполнение прогнозирования
            result = user_predict_XGBoost(
                df=df,
                time_column=request.time_column,
                col_target=request.col_target,
                forecast_horizon_time=request.forecast_horizon_time
            )

            return result
    except Exception as e:
        # Логирование ошибки и возврат понятного сообщения
        print(f"Error during prediction: {e}")
        return {"error": "An error occurred during prediction. Please check the input data."}