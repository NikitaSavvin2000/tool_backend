import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.v1.benchmark_metrics import router as benchmark_router

app = FastAPI()
app.include_router(benchmark_router, prefix="/api/v1/benchmark-metrics")

client = TestClient(app)

# Добавил более строгие проверки на типы и структуру данных в ответе benchmark-metrics.
def test_benchmark_metrics_response():
    response = client.get("/api/v1/benchmark-metrics")
    assert response.status_code == 200

    data = response.json()
    
    # Проверка наличия ключей
    assert "datasets" in data
    assert "colab_links" in data

    # Проверка типов
    assert isinstance(data["datasets"], list)
    for dataset in data["datasets"]:
        assert "name" in dataset
        assert isinstance(dataset["name"], str)
        
        assert "source_url" in dataset
        assert isinstance(dataset["source_url"], str)

        assert "models" in dataset
        assert isinstance(dataset["models"], list)

        for model in dataset["models"]:
            assert "name" in model
            assert isinstance(model["name"], str)
            
            assert "metrics" in model
            assert isinstance(model["metrics"], dict)

            assert "relative_to_horizon" in model
            assert isinstance(model["relative_to_horizon"], dict)

    # Проверка colab_links
    assert isinstance(data["colab_links"], dict)
    for key, link in data["colab_links"].items():
        assert isinstance(key, str)
        assert isinstance(link, str)
