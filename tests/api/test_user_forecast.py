# tests/api/test_user_forecast.py

import pytest
import os
import yaml

from datetime import datetime, timedelta
from httpx import AsyncClient, ASGITransport
from fastapi import status
from src.server import app
from typing import Any, List, Dict, Optional

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
async def test_user_forecast_valid_request():
    valid_tokens = load_valid_tokens()
    VALID_TOKEN = valid_tokens[0]

    payload = {
    "json_list_df_all_data_norm": [
        {
            "time": (datetime(2023, 1, 1, 0, 0) + timedelta(minutes=15 * i)).strftime("%Y-%m-%d %H:%M:%S"),
            "value": float(i * 5 + 10),
        }
        for i in range(45)  
    ],
        "time_column": "time",
        "col_target": "value",
        "evaluation_index": 10,        
        "last_know_index": 12,            
        "epochs": 5,                     
        "lag": 40,
        "activation": "relu",
        "optimizer": "adam",
        "dropout_count": 0.2,
        "col_for_train": ["value"],
        "forecast_horizon": "2023-01-01 04:00:00"
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://0.0.0.0:7070/") as ac:
        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
        response = await ac.post("/api/v1/user_forecast/", json=payload, headers=headers)

    assert response.status_code == status.HTTP_200_OK

    # Проверяем структуру JSON-ответа
    json_resp = response.json()
    assert "map_data" in json_resp
    assert "data" in json_resp["map_data"]
    assert "predictions" in json_resp["map_data"]["data"]

    # Проверяем содержимое predictions
    predictions = json_resp["map_data"]["data"]["predictions"]
    assert isinstance(predictions, list)
    assert len(predictions) > 0

    for pred in predictions:
        assert isinstance(pred, dict)
        assert "time" in pred
        assert "value" in pred
        assert isinstance(pred["time"], str)
        assert isinstance(pred["value"], float)

@pytest.mark.asyncio
async def test_user_forecast_invalid_request():
    valid_tokens = load_valid_tokens()
    VALID_TOKEN = valid_tokens[0]  

    invalid_payload = {
        "df": [],
        "time_column": "",
        "col_target": "",
        "forecast_horizon_time": "",
        "col_for_train": []
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://0.0.0.0:7070/") as ac:
        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
        response = await ac.post("/api/v1/user_forecast/", json=invalid_payload, headers=headers)

    assert response.status_code == status.HTTP_400_BAD_REQUEST

    error_details = response.json()
    assert "detail" in error_details