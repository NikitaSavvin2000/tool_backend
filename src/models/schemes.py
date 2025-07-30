# src/models/schemes.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DataFrameRequest(BaseModel):
    """
    Базовая модель для передачи списка словарей (аналог DataFrame).
    Используется в normalization, vectorization и других роутерах.
    """
    data: List[Dict[str, Any]] = Field(..., description="Список записей (аналог строк DataFrame)")
    time_column: str = Field(..., description="Название колонки с временными метками")
    target_column: str = Field(..., description="Название целевой колонки")

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"Дата": "2023-01-01", "Цена": 100},
                    {"Дата": "2023-01-02", "Цена": 105},
                    {"Дата": "2023-01-03", "Цена": 110}
                ],
                "time_column": "Дата",
                "target_column": "Цена"
            }
        }


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

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictRequest(BaseModel):
    """
    Запрос для предсказания модели (LSTM/XGBoost).
    """
    data: List[Dict[str, Any]] = Field(..., description="Нормализованные данные для предсказания")
    target_column: str = Field(..., description="Целевая колонка")
    evaluation_index: int = Field(..., description="Индекс начала оценки")
    last_known_index: int = Field(..., description="Последний известный индекс")
    epochs: int = Field(..., ge=1, le=1000, description="Количество эпох обучения")
    lag: int = Field(..., ge=1, description="Количество шагов для lookback")
    activation: str = Field(default="relu", description="Функция активации")
    optimizer: str = Field(default="adam", description="Оптимизатор")
    dropout: float = Field(default=0.2, ge=0.0, le=0.5, description="Dropout rate")
    model_architecture: List[Dict[str, Any]] = Field(
        ...,
        description="Архитектура модели (список слоёв)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"temperature": 0.1, "year": 0.5, "month": 0.2},
                    {"temperature": 0.2, "year": 0.5, "month": 0.3},
                    {"temperature": 0.3, "year": 0.5, "month": 0.4}
                ],
                "target_column": "temperature",
                "evaluation_index": 10,
                "last_known_index": 12,
                "epochs": 50,
                "lag": 3,
                "activation": "relu",
                "optimizer": "adam",
                "dropout": 0.2,
                "model_architecture": [
                    {"layer": 1, "type": "LSTM", "neurons": 64, "return_sequences": True},
                    {"layer": 2, "type": "Dense", "neurons": 1}
                ]
            }
        }


class UserPredictRequest(PredictRequest):
    """
    Расширенная версия PredictRequest с явным указанием колонок.
    Используется в /user-forecast.
    """
    col_for_train: List[str] = Field(..., description="Колонки, используемые для обучения")


class MetricsResponse(BaseModel):
    """
    Ответ с метриками качества модели.
    """
    RMSE: float
    R2: float
    MAE: float
    MAPE: float
    WMAPE: float


class ForecastResponse(BaseModel):
    """
    Стандартный ответ прогноза.
    """
    df_train: List[Dict[str, Any]] = Field(..., description="Обучающая выборка")
    df_test: List[Dict[str, Any]] = Field(..., description="Тестовая выборка")
    df_predict: List[Dict[str, Any]] = Field(..., description="Прогноз")
    metrics: MetricsResponse = Field(..., description="Метрики качества")
    loss: List[float] = Field(..., description="Значения функции потерь по эпохам")
    response_code: int = Field(..., description="Код ответа")
    response_message: str = Field(..., description="Сообщение об успешности")


class AnalyticsRequest(BaseModel):
    """
    Запрос для анализа нескольких DataFrame.
    """
    dataframes: List[List[Dict[str, Any]]] = Field(..., description="Список DataFrame (в формате list of dicts)")

    class Config:
        json_schema_extra = {
            "example": {
                "dataframes": [
                    [{"x": 1, "y": 2}, {"x": 2, "y": 3}],
                    [{"a": 10, "b": 20}]
                ]
            }
        }


class AnalyticsResponse(BaseModel):
    """
    Ответ сервиса анализа.
    """
    message: str
    nan_counts: Dict[str, Dict[str, int]] = Field(default={}, description="Количество NaN по колонкам")


class BenchmarkRequest(BaseModel):
    """
    Запрос для бенчмарка моделей.
    """
    dataset: List[Dict[str, Any]] = Field(..., description="Тестовый датасет")
    models: List[str] = Field(..., description="Список моделей для сравнения")
    target_column: str = Field(..., description="Целевая колонка")
    forecast_horizon: str = Field(..., description="Горизонт прогноза")


class ModelMetric(BaseModel):
    """
    Метрики одной модели в бенчмарке.
    """
    model: str
    RMSE: float
    MAPE: float
    R2: float


class BenchmarkResponse(BaseModel):
    """
    Ответ бенчмарка.
    """
    results: List[ModelMetric]
    best_model: str
    timestamp: str