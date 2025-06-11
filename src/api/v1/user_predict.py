from fastapi import APIRouter, Body, HTTPException
from src.models.schemes import PredictRequest
from src.services.user_predict_service import user_forecast
from src.config import logger

router = APIRouter()

@router.post("/user_forecast")
async def predict_user_forecast(body: PredictRequest = Body(...)):
    """
    Эндпоинт для пользовательского прогнозирования.

    Description:
    -------------
    Этот эндпоинт используется для установки списка колонок, которые будут использоваться при обучении модели.
    Принимает список названий колонок и вызывает функцию `user_forecast` из сервисного слоя.

    Parameters:
    -----------
    - **body (PredictRequest)**: Схема запроса, содержащая:
        - df (List[Dict]): Входной DataFrame в формате JSON.
        - time_column (str): Название временной колонки.
        - col_target (str): Название целевой колонки.
        - forecast_horizon_time (str): Горизонт прогнозирования в формате `YYYY-MM-DD HH:MM:SS`.

    Returns:
    --------
    - Dict: Результат выполнения функции `user_forecast`, например:
        - map_data: данные для отрисовки графика.
        - last_know_data: последнее известное значение.
        - title: заголовок графика.
        - legend: легенда графика.

    Example Request:
    ----------------
    ```json
    {
        "col_for_train": ["year", "month", "hour", "is_weekend"]
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
            "last_know_data": "2018-01-10 05:00:00",
            "title": "User Forecast",
            "legend": ["Actual", "Prediction"]
        }
    }
    ```

    Raises:
    -------
    - **HTTPException 400**: Если произошла ошибка при обработке данных или валидации.
    """
    try:
        col_for_train = body.col_for_train
        response = user_forecast(new_cols_for_train=col_for_train)
        return response
    except Exception as e:
        logger.error(f"Error in /user_forecast: {e}")
        raise HTTPException(status_code=400, detail="Unknown Error")