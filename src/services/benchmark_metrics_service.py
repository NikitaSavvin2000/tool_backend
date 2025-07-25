# src/services/benchmark_metrics_service.py

from src.models.schemes import BenchmarkResponse, DatasetBenchmark, ModelMetric


def get_mock_benchmark_data() -> BenchmarkResponse:
    """
    Формирует моковые данные для бенчмарков.
    
    :return: Схема BenchmarkResponse с данными бенчмарков.
    """
    return BenchmarkResponse(
        datasets=[
            DatasetBenchmark(
                name="Consumption Morocco",
                source_url="https://docs.google.com/spreadsheets/d/e/2PACX-1vSgwB47qVFZcr1Aq--UWxZ6fDi9CGLZm-1i8QoMgfdaHUbV8EqSli3ayPxYYxD8kqfYYHD41uuNxbjZ/pub?gid=1952392108&single=true&output=csv",
                models=[
                    ModelMetric(
                        name="Horizon",
                        metrics={"MAE": 2.512, "RMSE": 1012.696, "MAPE": 3.2},
                        relative_to_horizon={"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0},
                    ),
                    ModelMetric(
                        name="CatBoost",
                        metrics={"MAE": 6.8, "RMSE": 7.2, "MAPE": 9.3},
                        relative_to_horizon={"MAE": -17.2, "RMSE": -18.0, "MAPE": -190.6},
                    ),
                    ModelMetric(
                        name="XGBoost",
                        metrics={"MAE": 7.8, "RMSE": 8.5, "MAPE": 6.91},
                        relative_to_horizon={"MAE": -25.8, "RMSE": -28.0, "MAPE": -115.9},
                    ),
                ],
            )
        ],
        colab_links={
            "Horizon": "https://colab.research.google.com/drive/1jdo3EBuHgpBtvp0xsFXIGFDQOidWlCRa?usp=sharing",
            "CatBoost": "https://colab.research.google.com/drive/1KT1xtx5EZR_oLo8bsgSqb7OplqqLEi8q?usp=sharing",
            "XGBoost": "https://colab.research.google.com/drive/1qY7iGi5XDj8f5y0ROZOwbVRFqiFcEmkq?usp=sharing",
        },
    )