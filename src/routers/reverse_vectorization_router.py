# src/routers/reverse_vectorization_router.py
from typing import Annotated, Dict, List

import pandas as pd
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from src.core.logger import logger
from src.examples_fastapi.examples import example_reverse_norm_data
from src.services.normalization_service import run_reverse_normalization

router = APIRouter()

class ReverseNormalizationRequest(BaseModel):
    col_time: str
    col_target: str
    json_list_norm_df: List[Dict]
    min_val: float
    max_val: float

@router.post("/")
async def reverse_vectorization_data(body: Annotated[
    ReverseNormalizationRequest, Body(
        example={
            "col_time": example_reverse_norm_data['col_time'],
            "col_target": example_reverse_norm_data['col_target'],
            "json_list_norm_df": example_reverse_norm_data['json_list_norm_df'],
            "min_val": example_reverse_norm_data['min_val'],
            "max_val": example_reverse_norm_data['max_val']
        })]):
    """
    Эндпоинт для нормализации временного ряда.

    Description:
    -------------
    Принимает DataFrame и проводит его нормализацию по указанной временной и целевой колонкам.
    Использует Time2Vec для преобразования временных данных.

    Parameters:
    -----------
    - **body (NormalizationRequest)**: Схема запроса, содержащая:
        - json_list_df (List[Dict]): Входные данные в формате списка словарей.
        - col_time (str): Название временной колонки.
        - col_target (str): Название целевой колонки для нормализации.

    Returns:
    --------
    - **dict**: Результат нормализации, содержащий:
        - df_all_data_norm (dict): Нормализованный DataFrame.
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
        "df_all_data_norm": {
            "Datetime": ["2017-01-01 00:00:00", ...],
            "consumption": [0.0, 0.001, ...]
        },
        "min_val": 28349.81,
        "max_val": 34120.55
    }
    ```

    Raises:
    -------
    - **HTTPException 400**: Если входной DataFrame пустой или произошла ошибка при нормализации.
    """

    try:
        col_time = body.col_time
        col_target = body.col_target
        json_list_norm_df = body.json_list_norm_df
        min_val = body.min_val
        max_val = body.max_val
        df = pd.DataFrame(json_list_norm_df)

        if not df.empty:
            result = run_reverse_normalization(df, col_time, col_target, min_val, max_val)
            return result
        else:
            logger.error("Something happened during creation of the search table")
            raise HTTPException(
                status_code=400,
                detail="Bad Request",
                headers={"X-Error": "Something happened during creation of the search table"},
            )
    except Exception as ApplicationError:
        logger.error(ApplicationError.__repr__())
        raise HTTPException(
            status_code=400,
            detail="Unknown Error",
            headers={"X-Error": f"{ApplicationError.__repr__()}"},
        )
