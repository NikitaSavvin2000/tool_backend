import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.v1.benchmark_metrics import router as benchmark_router

app = FastAPI()
app.include_router(benchmark_router, prefix="/api/v1/benchmark-metrics")

client = TestClient(app)

def test_benchmark_metrics_response():
    response = client.get("/api/v1/benchmark-metrics/")
    assert response.status_code == 200

    data = response.json()
    assert "datasets" in data
    assert isinstance(data["datasets"], list)
    assert "colab_links" in data
    assert isinstance(data["colab_links"], dict)

    # Проверим хотя бы один dataset и модель
    dataset = data["datasets"][0]
    assert "name" in dataset
    assert "source_url" in dataset
    assert "models" in dataset
    assert isinstance(dataset["models"], list)

    model = dataset["models"][0]
    assert "name" in model
    assert "metrics" in model
    assert "relative_to_horizon" in model
