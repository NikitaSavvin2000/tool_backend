# src/routers/metrix_router.py
"""
Роутер для вычисления метрик (MAE, RMSE, MAPE и др.)
"""
import pandas as pd
from typing import List, Dict, Any
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
from src.core.logger import logger
from src.services.metrix_service import run_metrix_all

router = APIRouter(tags=["Metrics"])

class MetrixRequest(BaseModel):
    df_true: List[Dict[Any, Any]]
    df_pred: List[Dict[Any, Any]]
    col_target: str

    class Config:
        json_schema_extra = {
            "example": {
                "df_true": [
                    {"Дата": "2023-01-01", "Цена": 100},
                    {"Дата": "2023-01-02", "Цена": 105}
                ],
                "df_pred": [
                    {"Дата": "2023-01-01", "Цена": 102},
                    {"Дата": "2023-01-02", "Цена": 104}
                ],
                "col_target": "Цена"
            }
        }


@router.post("/", response_model=Dict[str, Any])
async def calculate_metrics(body: MetrixRequest = Body(...)):
    """
    Вычисляет метрики качества прогноза: MAE, RMSE, MAPE, WMAPE, R².
    """
    try:
        df_true = pd.DataFrame(body.df_true)
        df_pred = pd.DataFrame(body.df_pred)
        result = run_metrix_all(df_true, df_pred, body.col_target)
        return result
    except Exception as e:
        logger.error(f"Ошибка при вычислении метрик: {e}")
        raise HTTPException(status_code=400, detail=f"Не удалось рассчитать метрики: {str(e)}")