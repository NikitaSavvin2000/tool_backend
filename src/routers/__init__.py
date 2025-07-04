from fastapi import APIRouter

router = APIRouter()

from .xgboost_router import router as xgboost_router
from .benchmark_router import router as benchmark_router
from .vectorization_router import router as vectorization_router
from .reverse_vectorization_router import router as reverse_vectorization_router
from .possible_date_router import router as possible_date_router



router.include_router(benchmark_router, prefix="/api/v1", tags=["Benchmarks"])
router.include_router(vectorization_router, prefix="/api/v1", tags=["Vectorization"])
router.include_router(reverse_vectorization_router, prefix="/api/v1", tags=["Reverse Vectorization"])
router.include_router(possible_date_router, prefix="/api/v1", tags=["Generate possible date"])
router.include_router(xgboost_router, prefix="/api/v1", tags=["XGBoost"])





