"""
Роутер для эндпоинта /all-metrics (расчёт метрик между двумя DataFrame).
"""

from typing import List, Dict, Any
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
import pandas as pd

from src.core.logger import logger
from src.services.metrix_service import run_metrix_all

router = APIRouter()

class AllMetricsRequest(BaseModel):
    col_time: str
    col_target: str
    df_evaluation: List[Dict[str, Any]]
    df_comparative: List[Dict[str, Any]]

    class Config:
        json_schema_extra = {
            "example": {
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
        }

@router.post("/all-metrics/", response_model=Dict[str, Any], tags=["Metrics"])
async def calculate_all_metrics(body: AllMetricsRequest = Body(...)) -> Dict[str, Any]:
    """
    Эндпоинт для расчёта метрик между двумя наборами данных (df_evaluation и df_comparative).
    """
    try:
        df_eval = pd.DataFrame(body.df_evaluation)
        df_comp = pd.DataFrame(body.df_comparative)

        if df_eval.empty or df_comp.empty:
            raise HTTPException(status_code=400, detail="Оба DataFrame должны быть непустыми.")

        result = run_metrix_all(
            col_time=body.col_time,
            col_target=body.col_target,
            df_evaluation=df_eval,
            df_comparative=df_comp
        )
        return result

    except Exception as e:
        logger.error(f"Ошибка в /all-metrics: {e}")
        raise HTTPException(status_code=400, detail=str(e))
