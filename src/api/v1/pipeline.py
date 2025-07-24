from fastapi import APIRouter, Body, HTTPException
from src.models.schemes import ColsToChose, ConvertRequest, PredictRequest 
from src.services.pipeline_service import cols_to_chose, convert_df_to_datetime, generate_possible_date, all_available_forecast
from src.core.logger import logger
from src.core.utils import handle_exceptions
import pandas as pd

router = APIRouter()

@router.post("/cols_to_chose")
async def get_cols_to_chose(body: ColsToChose = Body(...)):
    """
    Эндпоинт для получения доступных колонок из DataFrame.

    Description:
    -------------
    Принимает список записей (JSON), преобразует их в DataFrame и возвращает список доступных колонок.

    Parameters:
    -----------
    - **body (ColsToChose)**: Схема запроса, содержащая:
        - df (List[Dict]): Входные данные в формате JSON.

    Returns:
    --------
    - Dict: Словарь с ключом `"columns"` и списком названий колонок.

    Example Request:
    ----------------
    ```json
    {
        "df": [
            {"col1": 1, "col2": "a"},
            {"col1": 2, "col2": "b"}
        ]
    }
    ```

    Example Response:
    -----------------
    ```json
    {
        "columns": ["col1", "col2"]
    }
    ```

    Raises:
    -------
    - **HTTPException 400**: Если входной DataFrame пустой или произошла ошибка при его обработке.
    """
    try:
        df = pd.DataFrame(body.df)
        return cols_to_chose(df)
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate_forecast")
@handle_exceptions
async def generate_forecast(body: PredictRequest = Body(...)):
    """
    Эндпоинт для генерации прогноза на основе временного ряда.

    Description:
    -------------
    Принимает нормализованный DataFrame, параметры целевой колонки и горизонта прогнозирования.
    Возвращает результат работы функции прогнозирования (например, предсказания, метрики, график и т.п.).

    Parameters:
    -----------
    - **body (PredictRequest)**: Схема запроса, содержащая:
        - df (List[Dict]): Данные в формате JSON (DataFrame).
        - time_column (str): Название временной колонки.
        - col_target (str): Название целевой колонки.
        - forecast_horizon_time (str): Горизонт прогнозирования в формате `YYYY-MM-DD HH:MM:SS`.

    Returns:
    --------
    - Dict: Результат прогнозирования, например:
        - map_data: Данные для построения графика.
        - last_know_data: Последнее известное значение.
        - title: Заголовок графика.
        - legend: Подписи для легенды.

    Example Request:
    ----------------
    ```json
    {
        "df": [
            {"Datetime": "2017-01-01 00:00:00", "consumption": 31935.18987},
            {"Datetime": "2017-01-01 00:15:00", "consumption": 31846.25}
        ],
        "time_column": "Datetime",
        "col_target": "consumption",
        "forecast_horizon_time": "2018-01-10 05:00:00"
    }
    ```

    Example Response:
    -----------------
    ```json
    {
        "map_data": {
            "data": {
                "last_real_data": [...],
                "predictions": [...]
            },
            "last_know_data": "2017-12-30 23:45:00",
            "title": "Consumption Forecast",
            "legend": ["Actual", "Prediction"]
        }
    }
    ```

    Raises:
    -------
    - **HTTPException 400**: Если переданы некорректные данные (например, отсутствует колонка времени или горизонт прогнозирования не соответствует требованиям).
    """
    df = pd.DataFrame(body.df)
    result = all_available_forecast(
        df=df,
        time_column=body.time_column,
        col_target=body.col_target,
        forecast_horizon_time=body.forecast_horizon_time
    )
    return result