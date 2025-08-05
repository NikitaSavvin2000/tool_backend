# tests/api/test_analytics.py
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
async def test_analytics_dfs_success():
    """Тест: успешный анализ DataFrame'ов."""
    valid_tokens = load_valid_tokens()
    VALID_TOKEN = valid_tokens[0]

    # Используем правильное имя поля
    payload = {
        "dataframes": [
            # Первый DataFrame
            [
                {"Datetime": "2017-01-01 00:00:00", "Temperature": 6.4865, "Humidity": 74.15},
                {"Datetime": "2017-01-01 00:15:00", "Temperature": 6.5, "Humidity": 74.2},
                {"Datetime": "2017-01-01 00:30:00", "Temperature": None, "Humidity": 75.0}
            ],
            # Второй DataFrame
            [
                {"SensorID": "S1", "Value": 100},
                {"SensorID": "S2", "Value": None},
                {"SensorID": "S3", "Value": 102}
            ]
        ]
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://0.0.0.0:7070/") as ac:
        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
        response = await ac.post("/api/v1/analyticsdfs", json=payload, headers=headers)  # Без слеша

    # Проверки
    assert response.status_code == status.HTTP_200_OK

    json_response = response.json()
    assert "message" in json_response
    assert "nan_counts" in json_response
    assert json_response["message"] == "Hello Backend"

    nan_counts = json_response["nan_counts"]
    assert "df_1" in nan_counts
    assert "df_2" in nan_counts
    assert nan_counts["df_1"]["Temperature"] == 1
    assert nan_counts["df_1"]["Humidity"] == 0
    assert nan_counts["df_2"]["Value"] == 1
    assert nan_counts["df_2"]["SensorID"] == 0


@pytest.mark.asyncio
async def test_analytics_dfs_empty_list():
    """Тест: пустой список DataFrame'ов."""
    valid_tokens = load_valid_tokens()
    VALID_TOKEN = valid_tokens[0]

    payload = {
        "dataframes": [] 
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://0.0.0.0:7070/") as ac:
        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
        response = await ac.post("/api/v1/analyticsdfs", json=payload, headers=headers)  # Без слеша

    assert response.status_code == status.HTTP_200_OK

    json_response = response.json()
    assert "message" in json_response
    assert "nan_counts" in json_response
    assert json_response["message"] == "Hello Backend"
    assert json_response["nan_counts"] == {}  # nan_counts должен быть пустым словарем
