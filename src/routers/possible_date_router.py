# src/routers/possible_date_router.py
from typing import Dict, Any # Добавлен импорт Any

from fastapi import APIRouter, Body

from src.models.schemes import ConvertRequest

from src.core.base_handler import BaseHandler
from src.core.decorators.log_decorators import log_endpoint
from src.core.decorators.exception_decorators import handle_exceptions
from src.utils.possible_forecast_date import generate_possible_date

router = APIRouter()
base_handler = BaseHandler() 

@router.post("/", response_model=Dict[str, Any])
@log_endpoint()
@handle_exceptions 
async def func_generate_possible_date(body: ConvertRequest = Body(...)) -> Dict[str, Any]:
    """
    Генерирует возможный диапазон дат и времени для фронта на основе временного столбца DataFrame.

    Описание:
    ----------
    Функция вычисляет интервал времени между записями и прогнозирует возможный диапазон дат.
    Минимальная дата (`min`) — это `last_know_date`, взятая из первой записи столбца `time_column`.
    Максимальная дата (`max`) определяется как `5%` от длины DataFrame в будущем, с учётом вычисленного интервала времени.
    Также возвращается параметр `min_forecast_horizon_time`, который обозначает минимально возможную дату для выбора пользователем.

    Parameters:
    ----------
    - **body (ConvertRequest)**: Схема запроса, содержащая:
        - df (List[Dict]): Входной DataFrame в формате JSON.
        - time_column (str): Название столбца, содержащего временные метки.

    Возвращает:
    ----------
    - **dict**:
        - `date`: словарь с минимальной (`min`) и максимальной (`max`) датами.
        - `min_forecast_horizon_time`: минимально возможная дата для выбора пользователем.
        - `time_hour`: список возможных значений часов (от `0` до `23`).
        - `time_minute`: список возможных значений минут (от `0` до `59`).

    Формат возвращаемого JSON:
    --------------------------
    ```json
    {
        "date": {
            "min": "2024-01-01",
            "max": "2024-02-15"
        },
        "min_forecast_horizon_time": "2024-01-01",
        "time_hour": [0, 1, 2, ..., 23],
        "time_minute": [0, 1, 2, ..., 59]
    }
    ```

    Пример использования:
    ---------------------
    ```python
    import requests

    def func_generate_possible_date(df: pd.DataFrame, time_column: str):
        url = url_backend + '/generate_possible_date'
        df_records = df.to_dict(orient='records')

        data = {
            "df": df_records,
            "time_column": time_column
        }

        try:
            response = requests.post(url, json=data)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе: {e}")
            return None

    response = func_generate_possible_date(df, time_column)
    print(response)
    ```
    """
    df = base_handler.parse_and_validate_dataframe(body.df, df_name="Input DataFrame")

    response = generate_possible_date(df=df, time_column=body.time_column)
    return response