# src/routers/reverse_vectorization_router.py
from typing import Dict, Any

from fastapi import APIRouter, Body

from src.models.schemes import ReverseNormalizationRequest

from src.core.base_handler import BaseHandler
from src.core.decorators.log_decorators import log_endpoint
from src.core.decorators.exception_decorators import handle_exceptions
from src.services.normalization_service import run_reverse_normalization

router = APIRouter()
base_handler = BaseHandler() 

@router.post("/", response_model=Dict[str, Any])
@log_endpoint() 
@handle_exceptions 
async def reverse_vectorization_data(body: ReverseNormalizationRequest = Body(...)) -> Dict[str, Any]:
    """
    Эндпоинт для обратной нормализации (денормализации) временного ряда.

    Description:
    -------------
    Принимает нормализованный DataFrame и проводит обратную нормализацию (денормализацию)
    по указанной временной и целевой колонкам, используя предоставленные min/max значения.

    Parameters:
    -----------
    - **body (ReverseNormalizationRequest)**: Схема запроса, содержащая:
        - json_list_norm_df (List[Dict]): Нормализованные данные в формате списка словарей.
        - col_time (str): Название временной колонки.
        - col_target (str): Название целевой колонки для денормализации.
        - min_val (float): Минимальное значение, использованное при нормализации.
        - max_val (float): Максимальное значение, использованное при нормализации.

    Returns:
    --------
    - **dict**: Результат денормализации, содержащий:
        - df_denorm (List[Dict]): Денормализованный DataFrame в формате списка словарей.
        - col_time (str): Название временной колонки.
        - col_target (str): Название целевой колонки.

    Example Request:
    ----------------
    ```json
    {
        "json_list_norm_df": [
            {"Datetime": "2017-01-01 00:00:00", "consumption_norm": 0.0},
            {"Datetime": "2017-01-01 00:15:00", "consumption_norm": 0.001}
        ],
        "col_time": "Datetime",
        "col_target": "consumption",
        "min_val": 28349.81,
        "max_val": 34120.55
    }
    ```

    Example Response:
    -----------------
    ```json
    {
        "df_denorm": [
            {"Datetime": "2017-01-01 00:00:00", "consumption": 31935.19},
            {"Datetime": "2017-01-01 00:15:00", "consumption": 31846.25}
        ],
        "col_time": "Datetime",
        "col_target": "consumption"
    }
    ```

    Raises:
    -------
    - **HTTPException 400**: Если входной DataFrame пустой или произошла ошибка при денормализации.
    """
    df = base_handler.parse_and_validate_dataframe(body.json_list_norm_df, df_name="Normalized Input DataFrame")

    result = run_reverse_normalization(
        df=df, # pd.DataFrame
        col_time=body.col_time,
        col_target=body.col_target,
        min_val=body.min_val,
        max_val=body.max_val
    )
    return result