# tests/api/test_metrics.py
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
async def test_all_metrics_success():
    """Тест: успешный расчет всех метрик."""
    # Загрузка токена
    valid_tokens = load_valid_tokens()
    VALID_TOKEN = valid_tokens[0]

    # Подготовка тестовых данных (пример из схемы и документации)
    payload = {
        "col_time": "time",
        "col_target": "value",
        "df_evaluation": [
            {"time": "2023-01-01 00:00:00", "value": 10},
            {"time": "2023-01-01 00:15:00", "value": 20},
            {"time": "2023-01-01 00:30:00", "value": 30}
        ],
        "df_comparative": [
            {"time": "2023-01-01 00:00:00", "value": 12},
            {"time": "2023-01-01 00:15:00", "value": 18},
            {"time": "2023-01-01 00:30:00", "value": 32}
        ]
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://0.0.0.0:7070/") as ac:
        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
        # Путь из api_router.py: prefix="/metrix" + route "/all-metrics"
        response = await ac.post("/api/v1/metrix/all-metrics", json=payload, headers=headers)

    # 1. Проверка статуса ответа
    assert response.status_code == status.HTTP_200_OK

    # 2. Парсинг JSON-ответа
    json_response = response.json()

    # 3. Проверка корневой структуры
    assert "metrics" in json_response
    assert "df_metrics" in json_response

    # 4. Проверка типов
    assert isinstance(json_response["metrics"], dict)
    assert isinstance(json_response["df_metrics"], dict)

    # 5. Проверка наличия ключевых метрик в 'metrics'
    metrics = json_response["metrics"]
    assert "RMSE" in metrics
    assert "MAE" in metrics
    assert "MAPE" in metrics
    # Проверим, что значения являются числами (float)
    assert isinstance(metrics["RMSE"], (int, float))
    assert isinstance(metrics["MAE"], (int, float))
    assert isinstance(metrics["MAPE"], (int, float))

    # 6. Проверка структуры 'df_metrics'
    df_metrics = json_response["df_metrics"]
    # Проверим наличие ключей, соответствующих входным данным
    assert "time" in df_metrics
    assert "value_true" in df_metrics
    assert "value_pred" in df_metrics
    # Проверим, что это списки
    assert isinstance(df_metrics["time"], list)
    assert isinstance(df_metrics["value_true"], list)
    assert isinstance(df_metrics["value_pred"], list)
    # Проверим длину (должна совпадать с длиной входных данных)
    expected_length = len(payload["df_evaluation"])
    assert len(df_metrics["time"]) == expected_length
    assert len(df_metrics["value_true"]) == expected_length
    assert len(df_metrics["value_pred"]) == expected_length
