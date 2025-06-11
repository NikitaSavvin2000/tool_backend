# src/api/v1/router.py

from fastapi import APIRouter
from src.api.v1 import (
    forecast,
    normalization,
    metrix,
    pipeline,
    analytics,
    col_update,
    user_predict,
)

router = APIRouter(prefix="/backend/v1")

router.include_router(forecast.router, prefix="/forecast")
router.include_router(normalization.router, prefix="/normalization")
router.include_router(metrix.router, prefix="/metrix")
router.include_router(pipeline.router, prefix="/pipeline")
router.include_router(analytics.router, prefix="/analytics")
router.include_router(col_update.router, prefix="/col_update")
router.include_router(user_predict.router, prefix="/user_predict")