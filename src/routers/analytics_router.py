# src/routers/analytics_router.py
from typing import Any, Dict

from fastapi import APIRouter, Body

from src.models.schemes import AnalyticsRequest
from src.services.analytics_service import run_analytics_dfs

router = APIRouter()

@router.post("/analyticsdfs", response_model=dict, tags=["Analytics"])
async def get_analytics_dfs(body: AnalyticsRequest = Body(...)) -> Dict[str, Any]:
    """
    Эндпоинт для анализа множества DataFrame'ов.
    Принимает список JSON-представлений DataFrame'ов.
    Возвращает общее сообщение и словарь с количеством NaN по колонкам для каждого DF.
    """
    try:
        result = run_analytics_dfs(body.dataframes)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))