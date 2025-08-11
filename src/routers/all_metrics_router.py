# src/routers/all_metrics_router.py
"""
Роутер для эндпоинта /all-metrics (расчёт метрик между двумя DataFrame).
"""

from typing import Any, Dict

from fastapi import APIRouter, Body

from src.core.base_handler import BaseHandler
from src.core.decorators.exception_decorators import handle_exceptions
from src.core.decorators.log_decorators import log_endpoint
from src.services.metrix_service import run_metrix_all
from src.models.schemes import AllMetricsRequest

router = APIRouter()
base_handler = BaseHandler()


@router.post("/", response_model=Dict[str, Any], tags=["Metrics"])
@log_endpoint()
@handle_exceptions
async def calculate_all_metrics(body: AllMetricsRequest = Body(...)) -> Dict[str, Any]:
    """
    Эндпоинт для расчёта метрик между двумя наборами данных (df_evaluation и df_comparative).
    """
    df_eval = base_handler.parse_and_validate_dataframe(body.df_evaluation)
    df_comp = base_handler.parse_and_validate_dataframe(body.df_comparative)

    result = run_metrix_all(
        col_time=body.col_time,
        col_target=body.col_target,
        df_evaluation=df_eval,
        df_comparative=df_comp
    )

    return result
