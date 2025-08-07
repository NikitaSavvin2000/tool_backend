import traceback
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, Body, HTTPException

from src.backend.forecast import forecast
from src.models.schemes import PredictRequest

router = APIRouter()
@router.post("/forecast/", response_model=Dict[str, Any])
async def run_forecast(req: PredictRequest = Body(...)):
    """
    Запуск LSTM-прогноза на основе входных данных.
    """
    try:
        df = pd.DataFrame(req.json_list_df_all_data_norm) 
        # Запуск forecast-функции
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
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Ошибка прогноза: {str(e)}")