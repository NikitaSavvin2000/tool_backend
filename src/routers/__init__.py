# src/routers/__init__.py
from fastapi import APIRouter

from src.routers.lstm_router import router as lstm_router

from .benchmark_router import router as benchmark_router
from .possible_date_router import router as possible_date_router
from .reverse_vectorization_router import router as reverse_vectorization_router
from .vectorization_router import router as vectorization_router
from .xgboost_router import router as xgboost_router

router = APIRouter()



router.include_router(benchmark_router, tags=["Benchmarks"])
router.include_router(vectorization_router, tags=["Vectorization"])
router.include_router(reverse_vectorization_router, tags=["Reverse Vectorization"])
router.include_router(possible_date_router, tags=["Generate possible date"])
router.include_router(xgboost_router, tags=["XGBoost"])
router.include_router(lstm_router, tags=["LSTM"])