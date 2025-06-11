from fastapi import APIRouter, Body, HTTPException
from src.models.schemes import UpdateColRequest
from src.services.col_update_service import update_col_for_train, update_col_for_train_lstm
from src.config import logger

router = APIRouter()

@router.post("/update_col_for_train")
async def update_col_train(body: UpdateColRequest = Body(...)):
    try:
        col_for_train = body.col_for_train
        response = update_col_for_train(new_cols_for_train=col_for_train)
        return response
    except Exception as e:
        logger.error(f"Error in /update_col_for_train: {e}")
        raise HTTPException(status_code=400, detail="Unknown Error")

@router.post("/update_col_for_train_lstm")
async def update_col_train_lstm(body: UpdateColRequest = Body(...)):
    try:
        col_for_train = body.col_for_train
        response = update_col_for_train_lstm(new_cols_for_train=col_for_train)
        return response
    except Exception as e:
        logger.error(f"Error in /update_col_for_train_lstm: {e}")
        raise HTTPException(status_code=400, detail="Unknown Error")