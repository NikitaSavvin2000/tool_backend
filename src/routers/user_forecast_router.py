from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException

from src.core.logger import logger
from src.models.schemes import UserPredictRequest
from src.services.user_predict_service import run_user_forecast

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
        - json_list_df_all_data_norm (List[Dict]): Входной DataFrame в формате JSON.
        - time_column (str): Название временной колонки.
        - col_target (str): Название целевой колонки.
        - forecast_horizon (str): Горизонт прогнозирования. (Исправлено: без _time)
        - col_for_train (List[str]): Список колонок для обучения.
        - model_architecture_params (Optional[List[Dict]]): Параметры модели XGBoost.
        - lag (Optional[int]): Размер лага для прогнозирования.
    
    Returns:
    - **dict**: Результат прогнозирования, содержащий:
        - map_data (dict): Данные для отрисовки графика.
    """
    try:
            logger.info("Получен запрос на прогнозирование:")
            data_list = body.json_list_df_all_data_norm
            logger.info(f"Колонки во входных данных: {list(data_list[0].keys()) if data_list else 'Нет данных'}")
            logger.info(f"Целевая колонка: {body.col_target}")
            
            # === Теперь это работает! ===
            logger.info(f"Горизонт прогнозирования: {body.forecast_horizon}")
            
            logger.info(f"Колонки для обучения: {body.col_for_train}")
            
            logger.info(f"Размер лага: {body.lag}")

            result = run_user_forecast(
                df=body.json_list_df_all_data_norm, 
                time_column=body.time_column, 
                col_target=body.col_target,
                forecast_horizon_time=body.forecast_horizon, # Передаем значение
                col_for_train=body.col_for_train,
                lag=body.lag
            )

            return result

    except ValueError as ve:
        logger.error(f"Ошибка валидации данных: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        logger.error(f"Неожиданная ошибка во время прогнозирования: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")