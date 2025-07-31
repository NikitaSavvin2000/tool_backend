# tests/api/test_forecast.py
import pytest
import os
import yaml
from httpx import AsyncClient, ASGITransport
from fastapi import status
from src.server import app

def load_valid_tokens():
    try:
        tokens_link = os.getenv("TOKEN_LIST")
        if not tokens_link:
            raise ValueError("Environment variable TOKEN_LIST is not set or empty.")

        with open(tokens_link, 'r', encoding='utf-8') as file:
            tokens_data = yaml.safe_load(file)

        if not tokens_data or "tokens" not in tokens_data:
            raise ValueError("Tokens file is empty or does not contain the 'tokens' key.")

        valid_tokens = [
            token["token"] for token in tokens_data["tokens"]
            if token.get("source") == "tool_backend"
        ]

        if not valid_tokens:
            raise ValueError("No valid tokens found for 'tool_backend' source.")

        return valid_tokens

    except Exception as e:
        raise RuntimeError(f"Failed to load tokens: {e}")    

@pytest.mark.asyncio
async def test_forecast_valid_request():
    # Загрузка допустимых токенов
    valid_tokens = load_valid_tokens()
    VALID_TOKEN = valid_tokens[0] 

    # Подготовка валидных тестовых данных
    payload = {
        "col_target": "temperature",
        "evaluation_index": 10,
        "last_know_index": 12,
        "epochs": 5,
        "lag": 2,
        "activation": "relu",
        "optimizer": "adam",
        "dropout_count": 0.2,
        "model_architecture_params": [
            {"layer": 1, "type": "LSTM", "neurons": 64}
        ],
        "json_list_df_all_data_norm": [
            {"temperature": 0.1, "year": 0.5, "month": 0.2},
            {"temperature": 0.2, "year": 0.5, "month": 0.3},
            {"temperature": 0.3, "year": 0.5, "month": 0.4},
            {"temperature": 0.4, "year": 0.5, "month": 0.5},
            {"temperature": 0.5, "year": 0.5, "month": 0.6},
            {"temperature": 0.6, "year": 0.5, "month": 0.7},
            {"temperature": 0.7, "year": 0.5, "month": 0.8},
            {"temperature": 0.8, "year": 0.5, "month": 0.9},
            {"temperature": 0.9, "year": 0.5, "month": 1.0},
            {"temperature": 1.0, "year": 0.5, "month": 1.1},
            {"temperature": 1.1, "year": 0.5, "month": 1.2},
            {"temperature": 1.2, "year": 0.5, "month": 1.3},
            {"temperature": 1.3, "year": 0.5, "month": 1.4},
            {"temperature": 1.4, "year": 0.5, "month": 1.5},
            {"temperature": 1.5, "year": 0.5, "month": 1.6}
        ]
    }

    # Создаем асинхронный клиент для тестирования
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://0.0.0.0:7070/") as ac:
        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
        response = await ac.post("/api/v1/forecast/", json=payload, headers=headers)

    # Проверки
    assert response.status_code == status.HTTP_200_OK

    json_resp = response.json()
    # Проверяем наличие ключа "df_real_predict"
    assert "df_real_predict" in json_resp
    assert isinstance(json_resp["df_real_predict"], list)
    assert len(json_resp["df_real_predict"]) > 0

    for pred in json_resp["df_real_predict"]:
        assert isinstance(pred, dict)
        assert "temperature" in pred
        assert isinstance(pred["temperature"], float)

@pytest.mark.asyncio
async def test_forecast_invalid_request():
    # Загрузка допустимых токенов
    valid_tokens = load_valid_tokens()
    VALID_TOKEN = valid_tokens[0]  # Берем первый допустимый токен

    # Подготовка невалидных тестовых данных
    invalid_payload = {
        "col_target": "",  # пустая строка - некорректно
        "evaluation_index": -5,  # отрицательное значение - некорректно
        "last_know_index": 15,
        "epochs": 0,  # ноль эпох - некорректно
        "lag": -1,  # отрицательный лаг - некорректно
        "activation": "unknown",
        "optimizer": "invalid",
        "dropout_count": 1.5,  # больше 1 - некорректно
        "model_architecture_params": [],
        "json_list_df_all_data_norm": []
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://0.0.0.0:7070/") as ac:
        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
        response = await ac.post("/api/v1/forecast/", json=invalid_payload, headers=headers)

    # Проверяем, что получаем ошибку 400
    assert response.status_code == status.HTTP_400_BAD_REQUEST

    error_details = response.json()
    assert "detail" in error_details