# tests/api/test_col_update.py

import pytest
import os
import yaml
from httpx import AsyncClient, ASGITransport
from fastapi import status
from src.server import app


def load_valid_tokens():
    """
    Загружает валидные токены из пути, указанного в переменной окружения TOKEN_LIST.
    Используется тот же подход, что и в test_forecast.py.
    """
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
async def test_update_col_for_train_success():
    """
    Тест: успешное обновление списка колонок для обучения.
    Проверяет, что возвращаются ожидаемые поля: message и columns.
    """
    # Загрузка токена
    valid_tokens = load_valid_tokens()
    VALID_TOKEN = valid_tokens[0]
    headers = {"Authorization": f"Bearer {VALID_TOKEN}"}

    payload = {
        "col_for_train": ["year", "month", "hour", "is_weekend"]
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://0.0.0.0:7070") as ac:
        response = await ac.post("/api/v1/update_col_for_train", json=payload, headers=headers)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert "message" in data
    assert "columns" in data

    # Проверяем, что все переданные колонки вернулись
    assert set(data["columns"]) == set(payload["col_for_train"])
    assert "updated successfully" in data["message"].lower()


@pytest.mark.asyncio
async def test_update_col_for_train_invalid_columns():
    """
    Тест: передача несуществующих колонок.
    Ожидается ошибка 400 с описанием недопустимых колонок.
    """
    valid_tokens = load_valid_tokens()
    VALID_TOKEN = valid_tokens[0]
    headers = {"Authorization": f"Bearer {VALID_TOKEN}"}

    payload = {
        "col_for_train": ["invalid_col1", "invalid_col2", "year"]
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://0.0.0.0:7070") as ac:
        response = await ac.post("/api/v1/update_col_for_train", json=payload, headers=headers)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()

    assert "detail" in data
    detail = data["detail"]

    # Проверяем, что сообщение содержит упоминание недопустимых колонок
    assert "invalid_col1" in detail
    assert "invalid_col2" in detail
    assert "не могут быть заданы" in detail or "not exist" in detail.lower()


@pytest.mark.asyncio
async def test_update_col_for_train_empty_list():
    """
    Тест: пустой список колонок.
    Должен вернуть 400, так как список не может быть пустым.
    """
    valid_tokens = load_valid_tokens()
    VALID_TOKEN = valid_tokens[0]
    headers = {"Authorization": f"Bearer {VALID_TOKEN}"}

    payload = {
        "col_for_train": []
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://0.0.0.0:7070") as ac:
        response = await ac.post("/api/v1/update_col_for_train", json=payload, headers=headers)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "detail" in data