# src/api/v1/benchmark_metrics.py

from fastapi import APIRouter
from pydantic import BaseModel, HttpUrl
from typing import List, Dict
from src.models.schemes import BenchmarkResponse, DatasetBenchmark, ModelMetric
from src.services.benchmark_metrics_service import get_mock_benchmark_data
from src.config import logger

router = APIRouter()

@router.get("/", response_model=BenchmarkResponse, tags=["Benchmark"])
async def get_benchmark_metrics():
    """
    Эндпоинт для получения метрик бенчмарка моделей по датасетам.
    
    Description:
    - Возвращает метрики качества прогнозов различных моделей на разных датасетах.
    - Метрики включают значения ошибок (например, MAE, RMSE, MAPE) и относительное улучшение/ухудшение по сравнению с Horizon.
    - Также включает ссылки на Google Colab для каждой модели.
    
    Returns:
    - **JSON**: Метрики моделей по каждому датасету + ссылки на Google Colab.
    
    Example Request:
    ```bash
    GET http://localhost:8000/api/v1/benchmark-metrics
    ```
    
    Example Response:
    ```json
    {
        "datasets": [
            {
                "name": "Consumption Morocco",
                "source_url": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSgwB47qVFZcr1Aq--UWxZ6fDi9CGLZm-1i8QoMgfdaHUbV8EqSli3ayPxYYxD8kqfYYHD41uuNxbjZ/pub?gid=1952392108&single=true&output=csv",
                "models": [
                    {
                        "name": "Horizon",
                        "metrics": {
                            "MAE": 2.512,
                            "RMSE": 1012.696,
                            "MAPE": 3.2
                        },
                        "relative_to_horizon": {
                            "MAE": 0.0,
                            "RMSE": 0.0,
                            "MAPE": 0.0
                        }
                    },
                    {
                        "name": "CatBoost",
                        "metrics": {
                            "MAE": 6.8,
                            "RMSE": 7.2,
                            "MAPE": 9.3
                        },
                        "relative_to_horizon": {
                            "MAE": -17.2,
                            "RMSE": -18.0,
                            "MAPE": -190.6
                        }
                    }
                ]
            }
        ],
        "colab_links": {
            "Horizon": "https://colab.research.google.com/drive/1jdo3EBuHgpBtvp0xsFXIGFDQOidWlCRa?usp=sharing",
            "CatBoost": "https://colab.research.google.com/drive/1KT1xtx5EZR_oLo8bsgSqb7OplqqLEi8q?usp=sharing",
            "XGBoost": "https://colab.research.google.com/drive/1qY7iGi5XDj8f5y0ROZOwbVRFqiFcEmkq?usp=sharing"
        }
    }
    ```
    
    Raises:
    - **HTTPException 400**: Если произошла ошибка при формировании данных.
    """
    try:
        # Получение моковых данных для бенчмарков
        mock_data = get_mock_benchmark_data()
        return mock_data

    except Exception as e:
        logger.error(f"Ошибка в эндпоинте /benchmark-metrics: {e}")
        raise HTTPException(status_code=400, detail="Не удалось получить данные бенчмарков.")