# src/routers/benchmark_router.py
from fastapi import APIRouter, Request

from src.models.schemes import BenchmarkStaticResponse, BenchmarkDataset, BenchmarkModelMetric
from src.core.decorators.log_decorators import log_endpoint

router = APIRouter()


@router.get("/", response_model=BenchmarkStaticResponse)
@log_endpoint()
async def get_benchmark_metrics(request: Request):
    """
    Получить метрики бенчмарка моделей по датасетам.

    Метрики включают значения ошибок (например, MAE, RMSE, MAPE) и относительное улучшение/ухудшение по сравнению с Horizon.

    **Пример вызова на Python**:
    ```python
    import requests

    response = requests.get("http://localhost:7070/api/v1/benchmark")
    data = response.json()
    ```

    Returns:
        JSON: Метрики моделей по каждому датасету + ссылки на Google Colab.
    """
    mock_data = BenchmarkStaticResponse(
        datasets=[
            BenchmarkDataset(
                name="Consumption Morocco",
                source_url="https://docs.google.com/spreadsheets/d/e/2PACX-1vSgwB47qVFZcr1Aq--UWxZ6fDi9CGLZm-1i8QoMgfdaHUbV8EqSli3ayPxYYxD8kqfYYHD41uuNxbjZ/pub?gid=1952392108&single=true&output=csv",
                models=[
                    BenchmarkModelMetric(
                        name="Horizon",
                        metrics={"MAE": 2.512, "RMSE": 1012.696, "MAPE": 3.20},
                        relative_to_horizon={"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0}
                    ),
                    BenchmarkModelMetric(
                        name="CatBoost",
                        metrics={"MAE": 6.8, "RMSE": 7.2, "MAPE": 9.3},
                        relative_to_horizon={"MAE": -17.2, "RMSE": -18.0, "MAPE": -190.6}
                    ),
                    BenchmarkModelMetric(
                        name="XGBoost",
                        metrics={"MAE": 7.8, "RMSE": 8.5, "MAPE": 6.91},
                        relative_to_horizon={"MAE": -25.8, "RMSE": -28.0, "MAPE": -115.9}
                    ),
                    BenchmarkModelMetric(
                        name="H2O",
                        metrics={"MAE": 8.1, "RMSE": 8.9, "MAPE": 8.4},
                        relative_to_horizon={"MAE": -28.4, "RMSE": -31.1, "MAPE": -162.5}
                    ),
                    BenchmarkModelMetric(
                        name="ARIMA",
                        metrics={"MAE": 9.0, "RMSE": 9.8, "MAPE": 0},
                        relative_to_horizon={"MAE": -35.1, "RMSE": -37.1, "MAPE": -0}
                    ),
                    BenchmarkModelMetric(
                        name="SARIMA",
                        metrics={"MAE": 10.2, "RMSE": 10.8, "MAPE": 0},
                        relative_to_horizon={"MAE": -43.5, "RMSE": -44.3, "MAPE": -0}
                    )
                ]
            )
        ],
        colab_links={
            "Horizon": "https://colab.research.google.com/drive/1jdo3EBuHgpBtvp0xsFXIGFDQOidWlCRa?usp=sharing",
            "CatBoost": "https://colab.research.google.com/drive/1KT1xtx5EZR_oLo8bsgSqb7OplqqLEi8q?usp=sharing",
            "XGBoost": "https://colab.research.google.com/drive/1qY7iGi5XDj8f5y0ROZOwbVRFqiFcEmkq?usp=sharing",
            "H2O": "https://colab.research.google.com/drive/1BxUqQUynZq6umWDIikbI-G1Zme8Wg-dp?usp=sharing#scrollTo=8aPlQwQC8flo",
            "ARIMA": "https://colab.research.google.com/arima",
            "SARIMA": "https://colab.research.google.com/drive/1dB0UIs2CEx_BNH10UHR05JYMx769msXH?usp=sharing"
        }
    )
    return mock_data