# src/routers/legacy_router.py

from fastapi import APIRouter, Body
from src.models.schemes import ForecastRequestXGBoost
from src.xgboost_selection_of_parameters.main import user_predict_XGBoost
import pandas as pd

router = APIRouter()

@router.post("/predict-old/")
async def predict_old(request: ForecastRequestXGBoost):
    """
    Старый эндпоинт прогнозирования (оставлен для совместимости)
    """
    df = pd.DataFrame(request.json_list_df_all_data_norm)
    
    result = user_predict_XGBoost(
        df=df,
        time_column="Дата",  # Или используйте значение из request
        col_target=request.col_target,
        forecast_horizon_time="2025-01-01"  # Или используйте значение из request
    )
    
    return result