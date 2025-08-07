# src/routers/normalization_router.py
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from src.core.logger import logger
from src.services.normalization_service import run_normalization

router = APIRouter(tags=["Normalization"])

class NormalizationRequest(BaseModel):
    col_time: str
    col_target: str
    json_list_df: List[Dict[Any, Any]]

    class Config:
        json_schema_extra = {
            "example": {
                "col_time": "Дата",
                "col_target": "Цена",
                "json_list_df": [
                    {"Дата": "2023-01-01", "Цена": 100},
                    {"Дата": "2023-01-02", "Цена": 105},
                    {"Дата": "2023-01-03", "Цена": 110}
                ]
            }
        }

@router.post("/")
async def normalize_data(body: NormalizationRequest = Body(...)):
    try:
        df = pd.DataFrame(body.json_list_df)
        result = run_normalization(df, body.col_time, body.col_target)
        return result
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))