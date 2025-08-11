# src/routers/user_forecast_router.py
from typing import Dict, Any

from fastapi import APIRouter, Body

from src.models.schemes import UserPredictRequest

from src.core.base_handler import BaseHandler
from src.core.decorators.log_decorators import log_endpoint
from src.core.decorators.exception_decorators import handle_exceptions
from src.services.user_predict_service import run_user_forecast

router = APIRouter()
base_handler = BaseHandler() 

@router.post("/", response_model=Dict[str, Any], tags=["User Forecast"])
@log_endpoint() 
@handle_exceptions 
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
        - forecast_horizon (str): Горизонт прогнозирования.
        - col_for_train (List[str]): Список колонок для обучения.
        - lag (int): Размер лага для прогнозирования.
        # Остальные поля из UserPredictRequest игнорируются сервисом run_user_forecast
    
    Returns:
    - **dict**: Результат прогнозирования, содержащий:
        - map_data (dict): Данные для отрисовки графика.
        - ... (другие поля, возвращаемые run_user_forecast)
    """
    df = base_handler.parse_and_validate_dataframe(body.json_list_df_all_data_norm, df_name="Input DataFrame")

    result = run_user_forecast(
        df=df, # pd.DataFrame
        time_column=body.time_column, 
        col_target=body.col_target,
        forecast_horizon_time=body.forecast_horizon,
        col_for_train=body.col_for_train,
        lag=body.lag,
    )

    return result