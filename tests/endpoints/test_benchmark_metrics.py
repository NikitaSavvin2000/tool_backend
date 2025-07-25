#tests/endpoints/test_benchmark_metrics.py
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


from api.v1.benchmark_metrics import router as benchmark_router

app = FastAPI()
app.include_router(benchmark_router, prefix="/api/v1/benchmark-metrics")

client = TestClient(app)

# Строгие проверки структуры, типов и сообщений к assert
def test_benchmark_metrics_response():
    response = client.get("/api/v1/benchmark-metrics")
    assert response.status_code == 200, "Response status is not 200 OK"

    data = response.json()

    # Проверка ключей верхнего уровня
    assert "datasets" in data, "Key 'datasets' is missing in the response"
    assert "colab_links" in data, "Key 'colab_links' is missing in the response"

    # Проверка структуры datasets
    assert isinstance(data["datasets"], list), "'datasets' should be a list"
    for dataset in data["datasets"]:
        assert "name" in dataset, "Each dataset must have a 'name'"
        assert isinstance(dataset["name"], str), "'name' must be a string"

        assert "source_url" in dataset, "Each dataset must have a 'source_url'"
        assert isinstance(dataset["source_url"], str), "'source_url' must be a string"

        assert "models" in dataset, "Each dataset must have a 'models' list"
        assert isinstance(dataset["models"], list), "'models' must be a list"

        for model in dataset["models"]:
            assert "name" in model, "Each model must have a 'name'"
            assert isinstance(model["name"], str), "'model name' must be a string"

            assert "metrics" in model, "Each model must have a 'metrics' dict"
            assert isinstance(model["metrics"], dict), "'metrics' must be a dict"
            for k, v in model["metrics"].items():
                assert isinstance(k, str), "Metric name must be a string"
                assert isinstance(v, float), f"Metric value for '{k}' must be a float"

    # Проверка colab_links
    assert isinstance(data["colab_links"], dict), "'colab_links' must be a dict"
    for key, link in data["colab_links"].items():
        assert isinstance(key, str), "Key in 'colab_links' must be a string"
        assert isinstance(link, str), f"Link for '{key}' must be a string"
