# src/routers/metrix_router.py
"""
Роутер для вычисления метрик (MAE, RMSE, MAPE и др.)
"""
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, Body, HTTPException

from src.core.logger import logger
from src.services.metrix_service import run_metrix_all
from src.models.schemes import MetrixRequest

router = APIRouter(tags=["Metrics"])

@router.post("/", response_model=Dict[str, Any]) 
async def calculate_metrics(body: MetrixRequest = Body(...)):
    """
    Вычисляет метрики качества прогноза: MAE, RMSE, MAPE, WMAPE, R².
    """
    try:
        df_true = pd.DataFrame(body.df_true)
        df_pred = pd.DataFrame(body.df_pred)
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