# src/routers/analytics_router.py
from typing import Any, Dict

from fastapi import APIRouter, Body

from src.core.decorators.exception_decorators import handle_exceptions
from src.core.decorators.log_decorators import log_endpoint
from src.models.schemes import AnalyticsRequest
from src.services.analytics_service import run_analytics_dfs

router = APIRouter()

@router.post("/", response_model=Dict[str, Any], tags=["Analytics"])
@log_endpoint()
@handle_exceptions
async def get_analytics_dfs(body: AnalyticsRequest = Body(...)) -> Dict[str, Any]:
    """
    Эндпоинт для анализа множества DataFrame'ов.
    Принимает список JSON-представлений DataFrame'ов.
    Возвращает словарь с количеством NaN по колонкам для каждого DF.
    """
    return run_analytics_dfs(body.dataframes)
