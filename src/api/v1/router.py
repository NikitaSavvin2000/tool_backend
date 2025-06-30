"""
Маршруты API версии v1

Этот модуль содержит главный роутер для всех API-эндпоинтов версии v1.
Все эндпоинты разделены по отдельным модулям:
- /forecast — прогнозирование
- /normalization — нормализация данных
- /metrix — вычисление метрик
- /pipeline — пайплайновые функции 
- /analytics — аналитика по входным данным
- /col_update — обновление списка колонок для обучения
- /user_predict — пользовательские прогнозы
- /benchmark_metrics — структура метрик моделей на разных датасетах
"""

from fastapi import APIRouter
from src.api.v1 import (
    forecast,
    normalization,
    metrix,
    pipeline,
    analytics,
    col_update,
    user_predict,
    benchmark_metrics
)

router = APIRouter(prefix="")

router.include_router(forecast.router, prefix="/forecast")
router.include_router(normalization.router, prefix="/normalization")
router.include_router(metrix.router, prefix="/metrix")
router.include_router(pipeline.router, prefix="/pipeline")
router.include_router(analytics.router, prefix="/analytics")
router.include_router(col_update.router, prefix="/col_update")
router.include_router(user_predict.router, prefix="/user_predict")
router.include_router(benchmark_metrics.router, prefix="/benchmark_metrics")
