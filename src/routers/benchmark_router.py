from fastapi import APIRouter
from pydantic import BaseModel, HttpUrl
from typing import List, Dict

router = APIRouter()


class ModelMetric(BaseModel):
    name: str
    metrics: Dict[str, float]
    relative_to_horizon: Dict[str, float]


class DatasetBenchmark(BaseModel):
    name: str
    source_url: HttpUrl
    models: List[ModelMetric]


class BenchmarkResponse(BaseModel):
    datasets: List[DatasetBenchmark]
    colab_links: Dict[str, HttpUrl]

@router.get("/benchmarks", response_model=BenchmarkResponse)
async def get_benchmark_metrics():
    """
    Получить метрики бенчмарка моделей по датасетам.

    Метрики включают значения ошибок (например, MAE, RMSE, MAPE) и относительное улучшение/ухудшение по сравнению с Horizon.

    **Пример вызова на Python**:
    ```python
    import requests

    response = requests.get("http://localhost:8000/api/v1/benchmark-metrics")
    data = response.json()
    ```

    Returns:
        JSON: Метрики моделей по каждому датасету + ссылки на Google Colab.
    """
    mock_data = BenchmarkResponse(
        datasets=[
            DatasetBenchmark(
                name="consumption",
                source_url="https://example.com/dataset/consumption",
                models=[
                    ModelMetric(
                        name="Horizon",
                        metrics={"MAE": 5.8, "RMSE": 6.1, "MAPE": 12.0},
                        relative_to_horizon={"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0}
                    ),
                    ModelMetric(
                        name="CatBoost",
                        metrics={"MAE": 6.8, "RMSE": 7.2, "MAPE": 14.0},
                        relative_to_horizon={"MAE": -17.2, "RMSE": -18.0, "MAPE": -16.7}
                    ),
                    ModelMetric(
                        name="XGBoost",
                        metrics={"MAE": 7.8, "RMSE": 8.5, "MAPE": 15.5},
                        relative_to_horizon={"MAE": -25.8, "RMSE": -28.0, "MAPE": -22.5}
                    ),
                    ModelMetric(
                        name="H2O",
                        metrics={"MAE": 8.1, "RMSE": 8.9, "MAPE": 16.2},
                        relative_to_horizon={"MAE": -28.4, "RMSE": -31.1, "MAPE": -26.3}
                    ),
                    ModelMetric(
                        name="ARIMA",
                        metrics={"MAE": 9.0, "RMSE": 9.8, "MAPE": 18.0},
                        relative_to_horizon={"MAE": -35.1, "RMSE": -37.1, "MAPE": -33.3}
                    ),
                    ModelMetric(
                        name="SARIMA",
                        metrics={"MAE": 10.2, "RMSE": 10.8, "MAPE": 20.0},
                        relative_to_horizon={"MAE": -43.5, "RMSE": -44.3, "MAPE": -40.0}
                    )
                ]
            )
        ],
        colab_links={
            "Horizon": "https://colab.research.google.com/horizon",
            "CatBoost": "https://colab.research.google.com/catboost",
            "XGBoost": "https://colab.research.google.com/xgboost",
            "H2O": "https://colab.research.google.com/h2o",
            "ARIMA": "https://colab.research.google.com/arima",
            "SARIMA": "https://colab.research.google.com/sarima"
        }
    )
    return mock_data
