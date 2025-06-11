# src/api/v1/forecast.py

from fastapi import APIRouter
from src.models.schemes import ForecastRequest
from src.services.forecast_service import run_forecast
from src.config import logger

router = APIRouter()

@router.post("/forecast")
async def get_concepts(body: ForecastRequest):
    """
    Эндпоинт для прогнозирования временных рядов.

    Parameters:
    - **body (ForecastRequest)**: Схема запроса, содержащая следующие поля:
        - col_target (str): Название целевой колонки.
        - json_list_df_all_data_norm (List[Dict]): Нормализованный DataFrame в формате JSON.
        - evaluation_index (int): Индекс начала тестовой выборки.
        - last_know_index (int): Индекс последнего известного значения.
        - epochs (int): Количество эпох обучения.
        - lag (int): Размер лага.
        - activation (str): Функция активации.
        - optimizer (str): Оптимизатор.
        - dropout_count (float): Вероятность отсева нейронов.
        - model_architecture_params (List[Dict]): Параметры архитектуры модели.

    Returns:
    - Dict: Словарь с результатами прогноза, содержащий:
        - df_evaluetion: Оценочные данные.
        - df_true_all_col: Реальные значения целевой переменной.
        - loss_list: Список значений потерь.
        - df_real_predict: Предсказанные значения.
        - response_code: Код ответа.
        - response_massage: Сообщение об ошибке (если есть).

    Example Request:
    ```json
    {
        "col_target": "consumption",
        "json_list_df_all_data_norm": [
            {"Datetime": "2017-01-01 00:00:00", "Temperature": 6.4865, "Humidity": 74.15, ...},
            ...
        ],
        "evaluation_index": 100,
        "last_know_index": 200,
        "epochs": 50,
        "lag": 12,
        "activation": "relu",
        "optimizer": "adam",
        "dropout_count": 0.2,
        "model_architecture_params": [{"layer_type": "dense", "units": 64}]
    }
    ```

    Example Response:
    ```json
    {
        "df_evaluetion": {...},
        "df_true_all_col": {...},
        "loss_list": [0.01, 0.009, ...],
        "df_real_predict": {...},
        "response_code": 200,
        "response_massage": "Success"
    }
    ```
    """
    result = run_forecast(
        col_target=body.col_target,
        df_all_data_norm=body.json_list_df_all_data_norm,
        evaluation_index=body.evaluation_index,
        last_know_index=body.last_know_index,
        epochs=body.epochs,
        lag=body.lag,
        activation=body.activation,
        optimizer=body.optimizer,
        dropout_count=body.dropout_count,
        model_architecture_params=body.model_architecture_params,
    )
    return result