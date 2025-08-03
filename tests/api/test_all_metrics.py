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
    """Базовый тест: успешный расчёт метрик между двумя DataFrame'ами"""
    valid_tokens = load_valid_tokens()
    VALID_TOKEN = valid_tokens[0]

    payload = {
        "col_time": "time",
        "col_target": "value",
        "df_evaluation": [
            {"time": "2023-01-01 00:00:00", "value": 10},
            {"time": "2023-01-01 00:15:00", "value": 20}
        ],
        "df_comparative": [
            {"time": "2023-01-01 00:00:00", "value": 12},
            {"time": "2023-01-01 00:15:00", "value": 22}
        ]
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://0.0.0.0:7070/") as ac:
        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
        response = await ac.post("/api/v1/all-metrics", json=payload, headers=headers)

    assert response.status_code == status.HTTP_200_OK

    result = response.json()

    assert "metrics" in result
    assert "df_metrics" in result
    assert isinstance(result["df_metrics"], list)

    expected_keys = {"RMSE", "MAE", "MAPE", "R2", "WMAPE"}
    assert expected_keys.issubset(result["metrics"].keys())
