# src/routers/metrix_router.py
"""
Роутер для вычисления метрик (MAE, RMSE, MAPE и др.)
"""
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from src.core.logger import logger
from src.services.metrix_service import run_metrix_all

router = APIRouter(tags=["Metrics"])

class MetrixRequest(BaseModel):
    col_time: str # Добавлено недостающее поле
    col_target: str
    df_true: List[Dict[Any, Any]]
    df_pred: List[Dict[Any, Any]]

    class Config:
        json_schema_extra = {
            "example": {
                "col_time": "Дата", # Пример для нового поля
                "col_target": "Цена",
                "df_true": [
                    {"Дата": "2023-01-01", "Цена": 100},
                    {"Дата": "2023-01-02", "Цена": 105}
                ],
                "df_pred": [
                    {"Дата": "2023-01-01", "Цена": 102},
                    {"Дата": "2023-01-02", "Цена": 104}
                ]
            }
        }


# Изменен путь эндпоинта с "/" на "/all-metrics"
@router.post("/", response_model=Dict[str, Any]) 
async def calculate_metrics(body: MetrixRequest = Body(...)):
    """
    Вычисляет метрики качества прогноза: MAE, RMSE, MAPE, WMAPE, R².
    """
    try:
        df_true = pd.DataFrame(body.df_true)
        df_pred = pd.DataFrame(body.df_pred)
        # Исправлен вызов функции с правильными аргументами
        result = run_metrix_all(
            col_time=body.col_time,
            col_target=body.col_target,
            df_evaluation=df_true,
            df_comparative=df_pred
        )
        return result
    except Exception as e:
        logger.error(f"Ошибка при вычислении метрик: {e}")
        raise HTTPException(status_code=400, detail=f"Не удалось рассчитать метрики: {str(e)}")