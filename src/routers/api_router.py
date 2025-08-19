"""
src/routers/api_router.py

Главный роутер API v1.
Собирает все маршруты из отдельных модулей.
Маршруты доступны по префиксу /api/v1/...
"""
from fastapi import APIRouter

# Импортируем все роутеры напрямую
from src.routers.forecast_router import router as forecast_router
from src.routers.normalization_router import router as normalization_router
from src.routers.metrix_router import router as metrix_router
from src.routers.pipeline_router import router as pipeline_router
from src.routers.lstm_router import router as lstm_router
from src.routers.xgboost_router import router as xgboost_router
from src.routers.benchmark_router import router as benchmark_router
from src.routers.possible_date_router import router as possible_date_router
from src.routers.vectorization_router import router as vectorization_router
from src.routers.reverse_vectorization_router import router as reverse_vectorization_router
from src.routers.user_forecast_router import router as user_forecast_router
from src.routers.col_update_router import router as col_update_router
from src.routers.analytics_router import router as analytics_router
from src.routers.all_metrics_router import router as all_metrics_router

# Создаём главный роутер с префиксом API
api_router = APIRouter()

# Подключаем все роутеры с их префиксами и тегами
# api_router.include_router(forecast_router, tags=["Forecast"])
# api_router.include_router(normalization_router, prefix="/normalization")
# api_router.include_router(metrix_router, prefix="/metrix", tags=["Metrics"])
# api_router.include_router(pipeline_router, prefix="/pipeline", tags=["Pipeline"])
# api_router.include_router(lstm_router, prefix="/predict-lstm", tags=["LSTM"])
api_router.include_router(xgboost_router, prefix="/predict-xgboost", tags=["XGBoost"])
api_router.include_router(benchmark_router, prefix="/benchmark", tags=["Benchmark"])
api_router.include_router(possible_date_router, prefix="/possible-date", tags=["Generate possible date"])
# api_router.include_router(vectorization_router, prefix="/vectorization", tags=["Vectorization"])
# api_router.include_router(reverse_vectorization_router, prefix="/reverse-vectorization", tags=["Reverse Vectorization"])
# api_router.include_router(user_forecast_router, prefix="/user_forecast")
# api_router.include_router(col_update_router, prefix="", tags=["Col Update"])
# api_router.include_router(analytics_router, tags=["Analytics"])
# api_router.include_router(all_metrics_router, prefix="", tags=["Metrics"])
