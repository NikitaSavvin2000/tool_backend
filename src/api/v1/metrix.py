from fastapi import APIRouter, Body, HTTPException
from src.models.schemes import MenrixAllRequest
from src.services.metrix_service import run_metrix_all
from src.config import logger
import pandas as pd

router = APIRouter()

@router.post("/all_metrix")
async def get_all_metrix(body: MenrixAllRequest = Body(...)):
    """
    Эндпоинт для расчёта метрик между двумя наборами данных.

    Description:
    -------------
    Данный эндпоинт принимает два DataFrame (основной и сравнительный), 
    вычисляет метрики качества прогноза и возвращает результаты.
    
    Parameters:
    -----------
    - **col_time** (str): Название временной колонки.
    - **col_target** (str): Название целевой колонки.
    - **json_list_df_reverse_evaluation** (List[Dict]): Список словарей — оценочные данные.
    - **json_list_df_reverse_comparative** (List[Dict]): Сравнительные данные.

    Returns:
    --------
    - **dict**: Результат выполнения функции `run_metrix_all`, содержащий:
        - metrics: dict — метрики (например, MAE, RMSE).
        - df_metrics: dict — DataFrame с детализацией ошибок по точкам.

    Example Request:
    ----------------
    ```json
    {
        "col_time": "Datetime",
        "col_target": "consumption",
        "json_list_df_reverse_evaluation": [
            {"Datetime": "2018-01-10 05:00:00", "consumption": 34000.5},
            ...
        ],
        "json_list_df_reverse_comparative": [
            {"Datetime": "2018-01-10 05:00:00", "consumption": 34000.0},
            ...
        ]
    }
    ```

    Example Response:
    -----------------
    ```json
    {
        "metrics": {
            "MAE": 12.3,
            "RMSE": 15.6,
            "R2": 0.98
        },
        "df_metrics": {
            "index": ["2018-01-10 05:00:00", ...],
            "consumption": [12.3, ...]
        }
    }
    ```

    Raises:
    -------
    - **HTTPException 400**: Если произошла ошибка при преобразовании данных или при вычислении метрик.
    """
    try:
        col_time = body.col_time
        col_target = body.col_target
        df_evaluation = pd.DataFrame(body.json_list_df_reverse_evaluation)
        df_comparative = pd.DataFrame(body.json_list_df_reverse_comparative)

        result = run_metrix_all(col_time, col_target, df_evaluation, df_comparative)
        return result
    except Exception as e:
        logger.error(f"Error in all_metrix endpoint: {e}")
        raise HTTPException(status_code=400, detail="Unknown Error")