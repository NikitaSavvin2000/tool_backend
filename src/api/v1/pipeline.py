# src/api/v1/pipeline.py

from fastapi import APIRouter, Body, HTTPException
from src.models.schemes import ColsToChose, ConvertRequest, PredictRequest 
from src.services.pipeline_service import cols_to_chose, convert_df_to_datetime, generate_possible_date, all_available_forecast
from src.config import logger
from src.core.utils import handle_exceptions
import pandas as pd

router = APIRouter()

@router.post("/cols_to_chose")
async def get_cols_to_chose(body: ColsToChose = Body(...)):
    try:
        df = pd.DataFrame(body.df)
        return cols_to_chose(df)
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate_forecast")
@handle_exceptions
async def generate_forecast(body: PredictRequest = Body(...)):
    df = pd.DataFrame(body.df)
    result = all_available_forecast(
        df=df,
        time_column=body.time_column,
        col_target=body.col_target,
        forecast_horizon_time=body.forecast_horizon_time
    )
    return result