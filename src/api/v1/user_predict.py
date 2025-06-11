from fastapi import APIRouter, Body, HTTPException
from src.models.schemes import PredictRequest
from src.services.user_predict_service import user_forecast
from src.config import logger

router = APIRouter()

@router.post("/user_forecast")
async def predict_user_forecast(body: PredictRequest = Body(...)):
    try:
        col_for_train = body.col_for_train
        response = user_forecast(new_cols_for_train=col_for_train)
        return response
    except Exception as e:
        logger.error(f"Error in /user_forecast: {e}")
        raise HTTPException(status_code=400, detail="Unknown Error")