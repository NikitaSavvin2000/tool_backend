# src/api/v1/metrix.py

from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, Body, HTTPException

from src.core.logger import logger
from src.models.schemes import MetricsRequest
from src.services.metrix_service import run_metrix_all

router = APIRouter()

@router.post("/all-metrics", response_model=dict, tags=["Metrics"])
async def calculate_all_metrics(body: MetricsRequest = Body(...)) -> Dict[str, Any]:
    """
    Эндпоинт для расчёта метрик между двумя наборами данных.
    
    Description:
    - Принимает два DataFrame (основной и сравнительный), вычисляет метрики качества прогноза
      и возвращает результаты.
    
    Parameters:
    - **body (MetricsRequest)**: Схема запроса, содержащая следующие поля:
        - col_time (str): Название временной колонки.
        - col_target (str): Название целевой колонки.
        - df_evaluation (List[Dict]): Основной DataFrame в формате JSON.
        - df_comparative (List[Dict]): Сравнительный DataFrame в формате JSON.
    
    Returns:
    - **dict**: Результаты расчёта метрик, содержащие:
        - metrics (Dict[str, float]): Значения метрик (например, RMSE, MAE, MAPE).
        - df_metrics (Dict): DataFrame с детализацией метрик.
    
    Example Request:
    ```json
    {
        "col_time": "time",
        "col_target": "value",
        "df_evaluation": [
            {"time": "2023-01-01 00:00:00", "value": 10},
            {"time": "2023-01-01 00:15:00", "value": 20}
        ],
        "df_comparative": [
            {"time": "2023-01-01 00:00:00", "value": 12},
            {"time": "2023-01-01 00:15:00", "value": 22}
        ]
    }
    ```
    
    Example Response:
    ```json
    {
        "metrics": {
            "RMSE": 1.414,
            "MAE": 1.0,
            "MAPE": 5.0
        },
        "df_metrics": {
            "time": ["2023-01-01 00:00:00", "2023-01-01 00:15:00"],
            "value_true": [10, 20],
            "value_pred": [12, 22]
        }
    }
    ```
    
    Raises:
    - **HTTPException 400**: Если входные данные пустые или произошла ошибка при расчёте метрик.
    """
    try:
        # Преобразование входных данных в DataFrame
        df_evaluation = pd.DataFrame(body.df_evaluation)
        df_comparative = pd.DataFrame(body.df_comparative)

        # Проверка на пустые данные
        if df_evaluation.empty or df_comparative.empty:
            raise HTTPException(
                status_code=400,
                detail="Входные DataFrame не могут быть пустыми."
            )

        # Выполнение расчёта метрик
        result = run_metrix_all(
            col_time=body.col_time,
            col_target=body.col_target,
            df_evaluation=df_evaluation,
            df_comparative=df_comparative
        )
        return result

    except Exception as e:
        logger.error(f"Ошибка в эндпоинте /all-metrics: {e}")
        raise HTTPException(status_code=400, detail=str(e))