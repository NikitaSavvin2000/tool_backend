# src/models/schemes.py
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,  # Только этот способ для V2
    Field,
)


class ColsToChoseRequest(BaseModel):
    all_possible_cols: List[str]


class DataFrameRequest(BaseModel):
    """
    Базовая модель для передачи списка словарей (аналог DataFrame).
    Используется в normalization, vectorization и других роутерах.
    """
    data: List[Dict[str, Any]] = Field(..., description="Список записей (аналог строк DataFrame)")
    time_column: str = Field(..., description="Название колонки с временными метками")
    target_column: str = Field(..., description="Название целевой колонки")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "data": [
                        {"Дата": "2023-01-01", "Цена": 100},
                        {"Дата": "2023-01-02", "Цена": 105},
                        {"Дата": "2023-01-03", "Цена": 110}
                    ],
                    "time_column": "Дата",
                    "target_column": "Цена"
                }
            ]
        }
    )


class ForecastRequest(BaseModel):
    """
    Запрос для запуска прогноза (используется в /forecast/run).
    """
    data: List[Dict[str, Any]] = Field(..., description="Нормализованные данные временного ряда")
    time_column: str = Field(..., description="Имя временной колонки")
    target_column: str = Field(..., description="Имя целевой колонки")
    forecast_horizon: str = Field(..., description="Горизонт прогноза, например '7D', '30D', '1H'")
    model_type: Optional[str] = Field(default="LSTM", description="Тип модели: LSTM, XGBoost и т.д.")
    col_for_train: Optional[List[str]] = Field(
        default=None,
        description="Дополнительные колонки для обучения"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "data": [
                        {"temperature": 0.1, "year": 0.5, "month": 0.2},
                        {"temperature": 0.2, "year": 0.5, "month": 0.3}
                    ],
                    "time_column": "timestamp",
                    "target_column": "temperature",
                    "forecast_horizon": "7D",
                    "model_type": "LSTM",
                    "col_for_train": ["year", "month"]
                }
            ]
        }
    )


class PredictRequest(BaseModel):
    """
    Модель для запроса прогноза (LSTM/XGBoost/User Forecast).
    """
    json_list_df_all_data_norm: List[Dict[str, Any]] = Field(
        ...,
        description="Нормализованные данные временного ряда"
    )
    time_column: str
    col_target: str = Field(..., description="Целевая колонка")
    evaluation_index: int = Field(..., ge=1, description="Индекс начала оценки")
    last_know_index: int = Field(..., ge=0, description="Последний известный индекс")
    epochs: int = Field(..., ge=1, le=1000, description="Количество эпох обучения")
    lag: int = Field(..., ge=1, description="Количество шагов для lookback")
    activation: str = Field(default="relu", description="Функция активации")
    optimizer: str = Field(default="adam", description="Оптимизатор")
    dropout_count: float = Field(default=0.2, ge=0.0, le=0.5, description="Dropout rate") 

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "json_list_df_all_data_norm": [
                        {"temperature": 0.1, "year": 0.5},
                        {"temperature": 0.2, "year": 0.5}
                    ],
                    "col_target": "temperature",
                    "evaluation_index": 10,
                    "last_know_index": 12,
                    "epochs": 5,
                    "lag": 2,
                    "activation": "relu",
                    "optimizer": "adam",
                    "dropout_count": 0.2,
                    "model_architecture_params": [
                        {"layer": 1, "type": "LSTM", "neurons": 64}
                    ]
                }
            ]
        }
    )


class UserPredictRequest(PredictRequest):
    """
    Расширенная версия PredictRequest с явным указанием колонок.
    Используется в /user_forecast.
    """
    col_for_train: List[str] = Field(..., description="Колонки, используемые для обучения")

    forecast_horizon: str = Field(
        ..., 
        description="Горизонт прогнозирования, например '7D', '30D', '1H' или конкретная дата"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "json_list_df_all_data_norm": [
                        {"temperature": 0.1, "year": 0.5, "month": 0.2}
                    ],
                    "col_target": "temperature",
                    "evaluation_index": 10,
                    "last_know_index": 12,
                    "epochs": 5,
                    "lag": 2,
                    "activation": "relu",
                    "optimizer": "adam",
                    "dropout_count": 0.2,
                    "col_for_train": ["year", "month"],
                    "forecast_horizon": "1H"
                }
            ]
        }
    )


class MetricsResponse(BaseModel):
    RMSE: float
    R2: float
    MAE: float
    MAPE: float
    WMAPE: float


class ForecastResponse(BaseModel):
    df_train: List[Dict[str, Any]]
    df_test: List[Dict[str, Any]]
    df_predict: List[Dict[str, Any]]
    metrics: MetricsResponse
    loss: List[float]
    response_code: int
    response_message: str


class AnalyticsRequest(BaseModel):
    dataframes: List[List[Dict[str, Any]]] = Field(..., description="Список DataFrame")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "dataframes": [
                        [{"x": 1, "y": 2}, {"x": 2, "y": 3}],
                        [{"a": 10, "b": 20}]
                    ]
                }
            ]
        }
    )


class AnalyticsResponse(BaseModel):
    message: str
    nan_counts: Dict[str, Dict[str, int]] = Field(default={}, description="Количество NaN по колонкам")


class BenchmarkRequest(BaseModel):
    dataset: List[Dict[str, Any]]
    models: List[str]
    target_column: str
    forecast_horizon: str

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "dataset": [{"value": 100}, {"value": 105}],
                    "models": ["LSTM", "XGBoost"],
                    "target_column": "value",
                    "forecast_horizon": "7D"
                }
            ]
        }
    )


class ModelMetric(BaseModel):
    model: str
    RMSE: float
    MAPE: float
    R2: float


class BenchmarkResponse(BaseModel):
    results: List[ModelMetric]
    best_model: str
    timestamp: str