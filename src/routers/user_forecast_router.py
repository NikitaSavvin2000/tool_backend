import pandas as pd
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
from typing import Any, List, Dict
from src.core.logger import logger
from src.services.user_predict_service import run_user_forecast
from src.models.schemes import PredictRequest as UserPredictRequest

# Создаем экземпляр роутера
router = APIRouter()

@router.post("/", response_model=Dict[str, Any], tags=["User Forecast"])
async def user_forecast(body: UserPredictRequest = Body(...)) -> Dict[str, Any]:
    """
    Эндпоинт для пользовательского прогнозирования временных рядов.
    
    Description:
    - Принимает данные в формате JSON и выполняет прогнозирование с использованием выбранных параметров.
    - Возвращает результаты прогноза, включая последние известные данные и предсказания.
    
    Parameters:
    - **body (UserPredictRequest)**: Схема запроса, содержащая следующие поля:
        - df (List[Dict]): Входной DataFrame в формате JSON.
        - time_column (str): Название временной колонки.
        - col_target (str): Название целевой колонки.
        - forecast_horizon_time (str): Горизонт прогнозирования.
        - col_for_train (List[str]): Список колонок для обучения.
        - model_architecture_params (Optional[List[Dict]]): Параметры модели XGBoost.
        - lag (Optional[int]): Размер лага для прогнозирования.
    
    Returns:
    - **dict**: Результат прогнозирования, содержащий:
        - map_data (dict): Данные для отрисовки графика.
    """
    try:
        # Логирование входных данных
        logger.info("Получен запрос на прогнозирование:")
        logger.info(f"Колонки во входных данных: {list(body.df[0].keys()) if body.df else 'Нет данных'}")
        logger.info(f"Целевая колонка: {body.col_target}")
        logger.info(f"Горизонт прогнозирования: {body.forecast_horizon_time}")
        logger.info(f"Колонки для обучения: {body.col_for_train}")
        logger.info(f"Параметры модели: {body.model_architecture_params}")
        logger.info(f"Размер лага: {body.lag}")

        # Выполнение прогноза с использованием сервисной функции
        result = run_user_forecast(
            df=body.df,
            time_column=body.time_column,
            col_target=body.col_target,
            forecast_horizon_time=body.forecast_horizon_time,
            col_for_train=body.col_for_train,
            model_architecture_params=body.model_architecture_params,
            lag=body.lag
        )

        return result

    except ValueError as ve:
        # Логирование ошибки и возврат HTTP-исключения для некорректных данных
        logger.error(f"Ошибка валидации данных: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        # Логирование ошибки и возврат HTTP-исключения для других ошибок
        logger.error(f"Неожиданная ошибка во время прогнозирования: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")