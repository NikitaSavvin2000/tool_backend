# src/api/v1/normalization.py

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from src.models.schemes import NormalizationRequest, ReverseNormalizationRequest
from src.services.normalization_service import (
    run_normalization,
    run_reverse_normalization,
)
from src.core.logger import logger

router = APIRouter()

@router.post("/normalize", response_model=dict, tags=["Normalization"])
async def normalize_data(body: NormalizationRequest = Body(...)) -> Dict[str, Any]:
    """
    Эндпоинт для нормализации временного ряда.
    
    Description:
    - Принимает DataFrame и проводит его нормализацию по указанной временной и целевой колонкам.
    - Использует Time2Vec для преобразования временных данных.
    
    Parameters:
    - **body (NormalizationRequest)**: Схема запроса, содержащая следующие поля:
        - json_list_df (List[Dict]): Входные данные в формате списка словарей.
        - col_time (str): Название временной колонки.
        - col_target (str): Название целевой колонки для нормализации.
    
    Returns:
    - **dict**: Результат нормализации, содержащий:
        - df_all_data_norm (dict): Нормализованный DataFrame.
        - min_val (float): Минимальное значение целевой колонки.
        - max_val (float): Максимальное значение целевой колонки.
    
    Example Request:
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
    ```json
    {
        "df_all_data_norm": {
            "Datetime": ["2017-01-01 00:00:00", "2017-01-01 00:15:00"],
            "consumption": [0.0, 0.001]
        },
        "min_val": 28349.81,
        "max_val": 34120.55
    }
    ```
    
    Raises:
    - **HTTPException 400**: Если входной DataFrame пустой или произошла ошибка при нормализации.
    """
    try:
        # Преобразование входных данных в DataFrame
        df = pd.DataFrame(body.json_list_df)

        # Проверка на пустые данные
        if df.empty:
            raise HTTPException(status_code=400, detail="Входной DataFrame не может быть пустым.")

        # Выполнение нормализации
        result = run_normalization(df, body.col_time, body.col_target)
        return result

    except Exception as e:
        logger.error(f"Ошибка в эндпоинте /normalize: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/denormalize", response_model=dict, tags=["Normalization"])
async def denormalize_data(body: ReverseNormalizationRequest = Body(...)) -> Dict[str, Any]:
    """
    Эндпоинт для денормализации временного ряда.
    
    Description:
    - Принимает нормализованный DataFrame и восстанавливает исходные значения целевой колонки.
    
    Parameters:
    - **body (ReverseNormalizationRequest)**: Схема запроса, содержащая следующие поля:
        - json_list_norm_df (List[Dict]): Нормализованные данные в формате списка словарей.
        - col_time (str): Название временной колонки.
        - col_target (str): Название целевой колонки для денормализации.
        - min_val (float): Минимальное значение целевой колонки до нормализации.
        - max_val (float): Максимальное значение целевой колонки до нормализации.
    
    Returns:
    - **dict**: Результат денормализации, содержащий:
        - df_all_data_reverse_norm (dict): Денормализованный DataFrame.
    
    Example Request:
    ```json
    {
        "json_list_norm_df": [
            {"Datetime": "2017-01-01 00:00:00", "consumption": 0.0},
            {"Datetime": "2017-01-01 00:15:00", "consumption": 0.001}
        ],
        "col_time": "Datetime",
        "col_target": "consumption",
        "min_val": 28349.81,
        "max_val": 34120.55
    }
    ```
    
    Example Response:
    ```json
    {
        "df_all_data_reverse_norm": {
            "Datetime": ["2017-01-01 00:00:00", "2017-01-01 00:15:00"],
            "consumption": [28349.81, 28350.81]
        }
    }
    ```
    
    Raises:
    - **HTTPException 400**: Если входной DataFrame пустой или произошла ошибка при денормализации.
    """
    try:
        # Преобразование входных данных в DataFrame
        df = pd.DataFrame(body.json_list_norm_df)

        # Проверка на пустые данные
        if df.empty:
            raise HTTPException(status_code=400, detail="Входной DataFrame не может быть пустым.")

        # Выполнение денормализации
        result = run_reverse_normalization(
            df=df,
            col_time=body.col_time,
            col_target=body.col_target,
            min_val=body.min_val,
            max_val=body.max_val,
        )
        return result

    except Exception as e:
        logger.error(f"Ошибка в эндпоинте /denormalize: {e}")
        raise HTTPException(status_code=400, detail=str(e))