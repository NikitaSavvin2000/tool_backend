# src/routers/forecast_router.py

from typing import Any, Dict

from fastapi import APIRouter, Body

from src.backend.forecast import forecast
from src.core.base_handler import BaseHandler
from src.core.decorators.log_decorators import log_endpoint
from src.core.decorators.exception_decorators import handle_exceptions
from src.models.schemes import PredictRequest

router = APIRouter()
base_handler = BaseHandler()

@router.post("/", response_model=Dict[str, Any], tags=["Forecast"])
@log_endpoint()
@handle_exceptions
async def run_forecast(req: PredictRequest = Body(...)) -> Dict[str, Any]:
    """
    Запуск LSTM-прогноза на основе входных данных.
    """
    df = base_handler.parse_and_validate_dataframe(req.json_list_df_all_data_norm)

    df_eval, df_true, loss_list, df_predict, code, message = forecast(
        col_target=req.col_target,
        df_all_data_norm=df,
        evaluation_index=req.evaluation_index,
        last_know_index=req.last_know_index,
        epochs=req.epochs,
        lag=req.lag,
        activation=req.activation,
        optimizer=req.optimizer,
        dropout_count=req.dropout_count,
        model_architecture_params=req.model_architecture_params,
    )

    return {
        "df_evaluation": df_eval.to_dict(orient="records"),
        "df_true_all_col": df_true.to_dict(orient="records"),
        "df_real_predict": df_predict.to_dict(orient="records"),
        "response_code": code,
        "response_message": message,
    }