# src/api/v1/forecast.py

from fastapi import APIRouter
from src.models.schemes import ForecastRequest
from src.services.forecast_service import run_forecast
from src.config import logger

router = APIRouter()

@router.post("/forecast")
async def get_concepts(body: ForecastRequest):
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