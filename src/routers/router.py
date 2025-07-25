#src/routers/router.py
"""
Маршруты API версии v1

Этот модуль содержит главный роутер для всех API-эндпоинтов версии v1.
Все эндпоинты разделены по отдельным модулям и сгруппированы по функциональности:

- `/forecast` — прогнозирование временных рядов.
- `/normalization` — нормализация данных для обработки.
- `/metrix` — вычисление метрик качества прогнозов (например, MAE, RMSE, MAPE).
- `/pipeline` — пайплайновые функции для автоматизации процессов.
- `/analytics` — аналитика по входным данным (статистика, распределения и т.д.).
- `/col-update` — обновление списка колонок для обучения модели.
- `/user-predict` — пользовательские прогнозы на основе предоставленных данных.
- `/benchmark-metrics` — метрики различных моделей на разных датасетах.
- `/horizon` — прогнозирование с использованием модели Horizon.
- `/lstm` — прогнозирование с использованием LSTM (Long Short-Term Memory).
- `/xgboost` — прогнозирование с использованием XGBoost.
- `/benchmark` — бенчмарки для сравнения моделей.

Каждый эндпоинт снабжен тегами для удобства навигации в документации Swagger.
"""
from fastapi import APIRouter

from . import __init__

router = APIRouter(prefix="/api/v1")

# Группировка эндпоинтов по функциональности
router.include_router(__init__.forecast.router, prefix="/forecast", tags=["Forecast"])
router.include_router(__init__.normalization.router, prefix="/normalization", tags=["Normalization"])
router.include_router(__init__.metrix.router, prefix="/metrix", tags=["Metrics"])
router.include_router(__init__.pipeline.router, prefix="/pipeline", tags=["Pipeline"])
router.include_router(__init__.analytics.router, prefix="/analytics", tags=["Analytics"])
router.include_router(__init__.col_update.router, prefix="/col-update", tags=["Column Update"])
router.include_router(__init__.user_predict.router, prefix="/user-predict", tags=["User Predict"])
router.include_router(__init__.benchmark_metrics.router, prefix="/benchmark-metrics", tags=["Benchmark Metrics"])
router.include_router(__init__.horizon_router, prefix="/horizon", tags=["Horizon"])
router.include_router(__init__.lstm_router, prefix="/lstm", tags=["LSTM"])
router.include_router(__init__.xgboost_router, prefix="/xgboost", tags=["XGBoost"])
router.include_router(__init__.benchmark_router, prefix="/benchmark", tags=["Benchmark"])