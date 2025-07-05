import pandas as pd
from fastapi import APIRouter, Body, HTTPException
from src.services.normalization_service import run_normalization
from src.config import logger
from src.examples_fastapi.examples import example_not_norm_data
from pydantic import BaseModel
from typing import List, Dict


router = APIRouter()

class NormalizationRequest(BaseModel):
    col_time: str
    col_target: str
    json_list_df: List[Dict]


@router.post("/vectorization")
async def normalize_data(body: NormalizationRequest = Body(
    example={
        "col_time": example_not_norm_data['col_time'],
        "col_target": example_not_norm_data['col_target'],
        "json_list_df": example_not_norm_data['json_list_df'],
    }
)):
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
        df = pd.DataFrame(body.json_list_df)
        result = run_normalization(df, body.col_time, body.col_target)
        # TODO:   Когда я писал эту схему видно был не в себе. Такой формат json - это абсурд,
        #         но на этой схеме уже работает связка с другими микросервисами.
        #         Позже нужно сделать нормальную схему и поправить ее где она используется.
        #         Ниже конвертация в нормальный формат.

        # df_like_json = result["df_all_data_norm"]
        # normalized = []
        # for i in range(len(next(iter(df_like_json.values())))):
        #     row = {key: df_like_json[key][i] for key in df_like_json}
        #     normalized.append(row)
        # result["df_all_data_norm"] = normalized

        return result
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))