# tests/api/test_benchmark.py
import pytest
import os
import yaml
from httpx import AsyncClient, ASGITransport
from fastapi import status
from src.server import app

def load_valid_tokens():
    """Загружает валидные токены из пути, указанного в переменной окружения TOKEN_LIST."""
    try:
        tokens_link = os.getenv("TOKEN_LIST")
        if not tokens_link:
            raise ValueError("Environment variable TOKEN_LIST is not set or empty.")

        with open(tokens_link, "r", encoding="utf-8") as file:
            tokens_data = yaml.safe_load(file)

        if not tokens_data or "tokens" not in tokens_data:
            raise ValueError("Tokens file is empty or does not contain the 'tokens' key.")

        valid_tokens = [
            token["token"]
            for token in tokens_data["tokens"]
            if token.get("source") == "tool_backend"
        ]

        if not valid_tokens:
            raise ValueError("No valid tokens found for 'tool_backend' source.")

        return valid_tokens

    except Exception as e:
        raise RuntimeError(f"Failed to load tokens: {e}")

@pytest.mark.asyncio
async def test_benchmark_metrics_success():
    """Тест: успешное получение метрик бенчмарка."""
    valid_tokens = load_valid_tokens()
    VALID_TOKEN = valid_tokens[0]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://0.0.0.0:7070/") as ac:
        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
        response = await ac.get("/api/v1/benchmark/", headers=headers)

    # 1. Проверка статуса ответа
    assert response.status_code == status.HTTP_200_OK

    # 2. Парсинг JSON-ответа
    json_response = response.json()

    # 3. Проверка корневой структуры
    assert "datasets" in json_response
    assert "colab_links" in json_response

    # 4. Проверка типов
    assert isinstance(json_response["datasets"], list)
    assert isinstance(json_response["colab_links"], dict)

    # 5. Проверка наличия данных (для мок-данных)
    assert len(json_response["datasets"]) > 0, "Список datasets не должен быть пустым"
    assert len(json_response["colab_links"]) > 0, "Словарь colab_links не должен быть пустым"

    # 6. Проверка структуры первого элемента в datasets (глубже)
    first_dataset = json_response["datasets"][0]
    assert "name" in first_dataset
    assert "source_url" in first_dataset
    assert "models" in first_dataset
    assert isinstance(first_dataset["models"], list)
    assert len(first_dataset["models"]) > 0, "Список models в dataset не должен быть пустым"

    # 7. Проверка структуры первой модели
    first_model = first_dataset["models"][0]
    assert "name" in first_model
    assert "metrics" in first_model
    assert "relative_to_horizon" in first_model
    assert isinstance(first_model["metrics"], dict)
    assert isinstance(first_model["relative_to_horizon"], dict)

    assert "MAE" in first_model["metrics"]
    assert "RMSE" in first_model["metrics"]

    model_names_in_dataset = {model["name"] for model in first_dataset["models"]}
    colab_link_keys = set(json_response["colab_links"].keys())
    assert len(colab_link_keys) >= len(model_names_in_dataset) - 2 