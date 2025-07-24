import pandas as pd
from fastapi import APIRouter, Body, HTTPException
from src.models.schemes import NormalizationRequest
from src.services.normalization_service import run_normalization
from src.core.logger import logger

router = APIRouter()

@router.post("/normalization")
async def normalize_data(body: NormalizationRequest = Body(...)):
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
        return result
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))