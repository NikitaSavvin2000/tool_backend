# src/api/v1/normalization.py
import pandas as pd

from fastapi import APIRouter, Body, HTTPException
from src.models.schemes import NormalizationRequest, ReverseNormalizationRequest
from src.services.normalization_service import run_normalization, run_reverse_normalization
from src.config import logger

router = APIRouter()

@router.post("/normalization")
async def normalize_data(body: NormalizationRequest = Body(...)):
    try:
        df = pd.DataFrame(body.json_list_df)
        result = run_normalization(df, body.col_time, body.col_target)
        return result
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))