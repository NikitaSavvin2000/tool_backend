# src/routers/pipeline_router.py
"""
Роутер для пайплайна обработки данных.
Содержит эндпоинт для подготовки данных: нормализация, разбиение на train/test.
"""
from typing import Dict, List, Any

from fastapi import APIRouter, Body

from src.models.schemes import PipelineRequest

from src.core.base_handler import BaseHandler
from src.core.decorators.log_decorators import log_endpoint
from src.core.decorators.exception_decorators import handle_exceptions
from src.services.pipeline_service import prepare_data_for_pipeline

router = APIRouter(tags=["Pipeline"])
base_handler = BaseHandler()

@router.post("/", response_model=Dict[str, List[Dict[str, Any]]])
@log_endpoint()
@handle_exceptions
async def prepare_pipeline_data(body: PipelineRequest = Body(...)) -> Dict[str, List[Dict[str, Any]]]:
    """
    Подготовка данных для пайплайна:
    - Нормализация (если norm_values=True)
    - Векторизация времени
    - Разделение на обучающую и тестовую выборки (80/20)

    Parameters:
    - **body (PipelineRequest)**: Схема запроса, содержащая следующие поля:
        - df (List[Dict]): Входной DataFrame в формате JSON.
        - time_column (str): Название колонки с временными метками.
        - col_target (str): Название целевой колонки.
        - norm_values (bool): Флаг необходимости нормализации.

    Returns:
    - **dict**: Словарь с подготовленными данными:
        - df_train (List[Dict]): Обучающая выборка.
        - df_test (List[Dict]): Тестовая выборка.

    Example Request:
    ```json
    {
      "df": [
        {"Дата": "2023-01-01 00:00:00", "Цена": 100},
        {"Дата": "2023-01-01 00:15:00", "Цена": 105}
      ],
      "time_column": "Дата",
      "col_target": "Цена",
      "norm_values": true
    }
    ```

    Example Response:
    ```json
    {
      "df_train": [
        {"Дата": "2023-01-01 00:00:00", "Цена": 0.0, "year": 0.5, ...},
        ...
      ],
      "df_test": [
        {"Дата": "2023-01-01 00:15:00", "Цена": 0.1, "year": 0.5, ...},
        ...
      ]
    }
    ```
    """
    df = base_handler.parse_and_validate_dataframe(body.df, df_name="Input DataFrame")

    result = prepare_data_for_pipeline(
        df=df, 
        time_column=body.time_column,
        col_target=body.col_target,
        norm_values=body.norm_values
    )

    return {
        "df_train": result["df_train"].to_dict(orient="records"),
        "df_test": result["df_test"].to_dict(orient="records")
    }