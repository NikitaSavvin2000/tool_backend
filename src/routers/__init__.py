from fastapi import APIRouter

router = APIRouter()

# Импорт роутеров
from .xgboost_router import router as xgboost_router
from .legacy_router import router as legacy_router

# Добавление роутеров
router.include_router(xgboost_router, prefix="/api/v1", tags=["XGBoost"])
router.include_router(legacy_router, prefix="/api/v1", tags=["Legacy"])