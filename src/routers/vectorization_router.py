# src/routers/vectorization_router.py
from typing import Dict, Any

from fastapi import APIRouter, Body

from src.models.schemes import NormalizationRequest

from src.core.base_handler import BaseHandler
from src.core.decorators.log_decorators import log_endpoint
from src.core.decorators.exception_decorators import handle_exceptions
from src.services.normalization_service import run_normalization

router = APIRouter()
base_handler = BaseHandler()

@router.post("/", response_model=Dict[str, Any]) 
@log_endpoint()
@handle_exceptions
async def normalize_data(body: NormalizationRequest = Body(...)) -> Dict[str, Any]:
    """
    Эндпоинт для нормализации временного ряда.

    Description:
    -------------
    Принимает DataFrame и проводит его нормализацию по указанной временной и целевой колонкам.
    Использует Min-Max нормализацию для целевой колонки и Time2Vec для преобразования временных данных.

    Parameters:
    -----------
    - **body (NormalizationRequest)**: Схема запроса, содержащая:
        - json_list_df (List[Dict]): Входные данные в формате списка словарей.
        - col_time (str): Название временной колонки.
        - col_target (str): Название целевой колонки для нормализации.

    Returns:
    --------
    - **dict**: Результат нормализации, содержащий:
        - df_all_data_norm (List[Dict]): Нормализованный DataFrame.
        - min_val (float): Минимальное значение целевой колонки.
        - max_val (float): Максимальное значение целевой колонки.

    Example Request:
    ----------------
    ```json
    {
        "json_list_df": [
            {"Datetime": "2017-01-01 00:00:00", "consumption": 31935.19},
            {"Datetime": "2017-01-01 00:15:00", "consumption": 31846.25}
        ],
        "col_time": "Datetime",
        "col_target": "consumption"
    }
    ```

    Example Response:
    -----------------
    ```json
    {
        "df_all_data_norm": [
            {"Datetime": "2017-01-01 00:00:00", "consumption": 0.0, "year": 0.5, ...},
            {"Datetime": "2017-01-01 00:15:00", "consumption": 0.001, "year": 0.5, ...}
        ],
        "min_val": 28349.81,
        "max_val": 34120.55
    }
    ```

    Raises:
    -------
    - **HTTPException 400**: Если входной DataFrame пустой или произошла ошибка при нормализации.
    (Обрабатывается декоратором @handle_exceptions)
    """
    df = base_handler.parse_and_validate_dataframe(body.json_list_df, df_name="Input DataFrame")

    result = run_normalization(df, body.col_time, body.col_target)
    
    return result