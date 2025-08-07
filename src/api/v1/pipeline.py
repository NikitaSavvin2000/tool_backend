# src/api/v1/pipeline.py

from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, Body, HTTPException

from src.backend.all_available_forecast import all_available_forecast
from src.core.logger import logger
from src.models.schemes import ColsToChoseRequest
from src.routers.lstm_router import PredictRequest
from src.routers.possible_date_router import ConvertRequest
from src.services.pipeline_service import (
    cols_to_chose,
    convert_df_to_datetime,
    generate_possible_date,
)

router = APIRouter()

@router.post("/cols-to-chose", response_model=dict, tags=["Pipeline"])
async def get_cols_to_chose(body: ColsToChoseRequest = Body(...)) -> Dict[str, Any]:
    """
    Эндпоинт для получения доступных колонок для выбора.
    
    Description:
    - Принимает список всех возможных колонок и возвращает те, которые подходят для анализа.
    
    Parameters:
    - **body (ColsToChoseRequest)**: Схема запроса, содержащая следующие поля:
        - all_possible_cols (List[str]): Список всех возможных колонок.
    
    Returns:
    - **dict**: Результат выбора колонок, содержащий:
        - available_cols (List[str]): Доступные колонки для анализа.
    
    Example Request:
    ```json
    {
        "all_possible_cols": ["year", "month", "day", "hour", "minute"]
    }
    ```
    
    Example Response:
    ```json
    {
        "available_cols": ["year", "month", "day", "hour"]
    }
    ```
    
    Raises:
    - **HTTPException 400**: Если произошла ошибка при обработке данных.
    """
    try:
        # Выполнение выбора колонок
        result = cols_to_chose(body.all_possible_cols)
        return {"available_cols": result}

    except Exception as e:
        logger.error(f"Ошибка в эндпоинте /cols-to-chose: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/convert-datetime", response_model=dict, tags=["Pipeline"])
async def convert_datetime(body: ConvertRequest = Body(...)) -> Dict[str, Any]:
    """
    Эндпоинт для конвертации временной колонки в формат datetime.
    
    Description:
    - Принимает DataFrame и преобразует временную колонку в формат datetime.
    
    Parameters:
    - **body (ConvertRequest)**: Схема запроса, содержащая следующие поля:
        - df (List[Dict]): Входной DataFrame в формате JSON.
        - time_column (str): Название временной колонки.
    
    Returns:
    - **dict**: Преобразованный DataFrame.
    
    Example Request:
    ```json
    {
        "df": [
            {"time": "2023-01-01 00:00:00", "value": 10},
            {"time": "2023-01-01 00:15:00", "value": 20}
        ],
        "time_column": "time"
    }
    ```
    
    Example Response:
    ```json
    {
        "df": [
            {"time": "2023-01-01 00:00:00", "value": 10},
            {"time": "2023-01-01 00:15:00", "value": 20}
        ]
    }
    ```
    
    Raises:
    - **HTTPException 400**: Если входной DataFrame пустой или произошла ошибка при конвертации.
    """
    try:
        # Преобразование входных данных в DataFrame
        df = pd.DataFrame(body.df)

        # Проверка на пустые данные
        if df.empty:
            raise HTTPException(status_code=400, detail="Входной DataFrame не может быть пустым.")

        # Выполнение конвертации
        result = convert_df_to_datetime(df, body.time_column)
        return {"df": result.to_dict(orient="records")}

    except Exception as e:
        logger.error(f"Ошибка в эндпоинте /convert-datetime: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate-possible-date", response_model=dict, tags=["Pipeline"])
async def generate_possible_date_endpoint(body: PredictRequest = Body(...)) -> Dict[str, Any]:
    """
    Эндпоинт для генерации возможных дат прогнозирования.
    
    Description:
    - Принимает временной ряд и горизонт прогнозирования, возвращает возможные даты.
    
    Parameters:
    - **body (PredictRequest)**: Схема запроса, содержащая следующие поля:
        - df (List[Dict]): Входной DataFrame в формате JSON.
        - time_column (str): Название временной колонки.
        - col_target (str): Название целевой колонки.
        - forecast_horizon_time (str): Горизонт прогнозирования.
    
    Returns:
    - **dict**: Возможные даты прогнозирования.
    
    Example Request:
    ```json
    {
        "df": [
            {"time": "2023-01-01 00:00:00", "value": 10},
            {"time": "2023-01-01 00:15:00", "value": 20}
        ],
        "time_column": "time",
        "col_target": "value",
        "forecast_horizon_time": "2023-01-01 01:00:00"
    }
    ```
    
    Example Response:
    ```json
    {
        "possible_dates": ["2023-01-01 00:30:00", "2023-01-01 00:45:00", "2023-01-01 01:00:00"]
    }
    ```
    
    Raises:
    - **HTTPException 400**: Если входной DataFrame пустой или произошла ошибка при генерации дат.
    """
    try:
        # Преобразование входных данных в DataFrame
        df = pd.DataFrame(body.df)

        # Проверка на пустые данные
        if df.empty:
            raise HTTPException(status_code=400, detail="Входной DataFrame не может быть пустым.")

        # Генерация возможных дат
        possible_dates = generate_possible_date(
            df=df,
            time_column=body.time_column,
            col_target=body.col_target,
            forecast_horizon_time=body.forecast_horizon_time,
        )
        return {"possible_dates": possible_dates}

    except Exception as e:
        logger.error(f"Ошибка в эндпоинте /generate-possible-date: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/all-available-forecast", response_model=dict, tags=["Pipeline"])
async def all_available_forecast_endpoint(body: PredictRequest = Body(...)) -> Dict[str, Any]:
    """
    Эндпоинт для выполнения прогнозирования всеми доступными моделями.
    
    Description:
    - Принимает временной ряд, выполняет прогнозирование всеми доступными моделями.
    
    Parameters:
    - **body (PredictRequest)**: Схема запроса, содержащая следующие поля:
        - df (List[Dict]): Входной DataFrame в формате JSON.
        - time_column (str): Название временной колонки.
        - col_target (str): Название целевой колонки.
        - forecast_horizon_time (str): Горизонт прогнозирования.
    
    Returns:
    - **dict**: Результаты прогнозирования всеми моделями.
    
    Example Request:
    ```json
    {
        "df": [
            {"time": "2023-01-01 00:00:00", "value": 10},
            {"time": "2023-01-01 00:15:00", "value": 20}
        ],
        "time_column": "time",
        "col_target": "value",
        "forecast_horizon_time": "2023-01-01 01:00:00"
    }
    ```
    
    Example Response:
    ```json
    {
        "predictions": {
            "Horizon": [15, 16, 17],
            "LSTM": [14, 15, 16],
            "XGBoost": [13, 14, 15]
        }
    }
    ```
    
    Raises:
    - **HTTPException 400**: Если входной DataFrame пустой или произошла ошибка при прогнозировании.
    """
    try:
        # Преобразование входных данных в DataFrame
        df = pd.DataFrame(body.df)

        # Проверка на пустые данные
        if df.empty:
            raise HTTPException(status_code=400, detail="Входной DataFrame не может быть пустым.")

        # Выполнение прогнозирования
        predictions = all_available_forecast(
            df=df,
            time_column=body.time_column,
            col_target=body.col_target,
            forecast_horizon_time=body.forecast_horizon_time,
        )
        return {"predictions": predictions}

    except Exception as e:
        logger.error(f"Ошибка в эндпоинте /all-available-forecast: {e}")
        raise HTTPException(status_code=400, detail=str(e))