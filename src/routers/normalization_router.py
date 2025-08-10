# src/routers/normalization_router.py
import pandas as pd
from fastapi import APIRouter, Body
from typing import Dict, Any

from src.models.schemes import NormalizationRequest

from src.core.base_handler import BaseHandler
from src.core.decorators.log_decorators import log_endpoint
from src.core.decorators.exception_decorators import handle_exceptions
from src.services.normalization_service import run_normalization

router = APIRouter(tags=["Normalization"])
base_handler = BaseHandler() 

@router.post("/", response_model=Dict[str, Any]) 
@log_endpoint() 
@handle_exceptions 
async def normalize_data(body: NormalizationRequest = Body(...)) -> Dict[str, Any]: 
    """
    Эндпоинт для нормализации данных временного ряда.

    Parameters:
    - **body (NormalizationRequest)**: Схема запроса, содержащая следующие поля:
        - json_list_df (List[Dict]): Входной DataFrame в формате JSON.
        - col_time (str): Название колонки с временными метками.
        - col_target (str): Название целевой колонки.

    Returns:
    - **dict**: Результат нормализации.
    """
    df = base_handler.parse_and_validate_dataframe(body.json_list_df, df_name="Input DataFrame")

    result = run_normalization(df, body.col_time, body.col_target)
    
    return result