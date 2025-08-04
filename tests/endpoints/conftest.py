# tests/endpoints/conftest.py
import os

import pandas as pd
import pytest
import yaml
from fastapi import FastAPI
from httpx import AsyncClient

from src.api.v1.pipeline import router
from src.routers.normalization_router import NormalizationRequest

home_path = os.getcwd()


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


@pytest.fixture(scope="session")
def valid_token():
    """Фикстура для получения валидного токена"""
    try:
        return load_valid_tokens()[0]
    except Exception as e:
        pytest.skip(f"Skipping tests due to token loading error: {e}")


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
async def async_client(app, valid_token):
    headers = {"Authorization": f"Bearer {valid_token}"}
    async with AsyncClient(
        app=app,
        base_url="http://0.0.0.0:7070/",
        headers=headers
    ) as client:
        yield client


@pytest.fixture()
def mock_normalization_data():
    df = pd.read_csv(f'{home_path}/src/examples_data/example_data.csv')
    df = df.loc[:2]

    df_list = df[['time', 'load_consumption']].to_dict('records')

    result_dict = {
        "json_list_df" : df_list,
        "col_time": "time",
        "col_target": "load_consumption"
    }

    data = NormalizationRequest(**result_dict)

    return data


@pytest.fixture
def forecast_data():
    return {
        'standard_case': {
            'input': {
                "df": [
                    { "time": "2023-01-01 00:00:00", "value": 4 },
                    { "time": "2023-01-01 00:15:00", "value": 5 },
                    { "time": "2023-01-01 00:30:00", "value": 6 },
                    { "time": "2023-01-01 00:45:00", "value": 7 },
                    { "time": "2023-01-01 01:00:00", "value": 8 },
                    { "time": "2023-01-01 01:15:00", "value": 9 },
                    { "time": "2023-01-01 01:30:00", "value": 10 },
                    { "time": "2023-01-01 01:45:00", "value": 11 },
                    { "time": "2023-01-01 01:50:00", "value": 12 },
                    { "time": "2023-01-01 01:55:00", "value": 13 },
                    { "time": "2023-01-01 02:30:00", "value": 14 },
                    { "time": "2023-01-01 02:34:00", "value": 15 },
                    { "time": "2023-01-01 02:50:00", "value": 16 }
                ],
                "time_column": "time",
                "col_target": "value",
                "forecast_horizon_time": "2023-01-02 03:00:00"
                }
        },
        'empty_df': {
            'input': {
                "df": [],
                "time_column": "time",
                "col_target": "value",
                "forecast_horizon_time": "2023-01-01 01:00:00"
            }
        },
        'unknown_column': {
            'input': {
                "df": [{"time": "2023-01-01 00:00:00", "value": 10}],
                "time_column": "datetime",
                "col_target": "value",
                "forecast_horizon_time": "2023-01-01 01:00:00"
            }
        },
        'wrong_format_forecast_horizon_time': {
            'input': {
                "df": [{"time": "2023-01-01 00:00:00", "value": 10}],
                "time_column": "datetime",
                "col_target": "value",
                "forecast_horizon_time": "invalid_date"
            }
        },
        'wrong_df_data_type': {
            'input': {
                "df": "not_a_list",
                "time_column": "time",
                "col_target": "value",
                "forecast_horizon_time": "2023-01-01 01:00:00"
            }
        },
    }


@pytest.fixture
def convert_datetime_data():
    return {
        'standard_case': {
            'input': {
                "df": [{"time": "2023-01-01", "value": 10}],
                "time_column": "time"
            },
            'output': {"df": [{"time": "2023-01-01T00:00:00", "value": 10}]}
        },
        'empty_df': {
            'input': {
                "df": [],
                "time_column": "time"
            },
        },
        'already_datetime': {
            'input': {
                "df": [{"time": "2023-01-01T00:00:00", "val": 1}],
                "time_column": "time"
            },
            'output': {"df": [{"time": "2023-01-01T00:00:00", "val": 1}]}
        },
        'wrong_column': {
            'input': {
                "df": [{"time": "2023-01-01T00:00:00", "val": 1}],
                "time_column": "datetime"
            },
        },
        'missing_time_key': {
            'input': {
                "df": [{"time": "2023-01-01", "val": 1}],
            },
        },
        'wrong_df_data_type': {
            'input': {
                "df": 'not_a_list',
                "time_column": "time"
            },
        }
        
    }


@pytest.fixture
def cols_to_chose_data():
    return {
        'typical_case': {
            'input': {"all_possible_cols": ["year_2020", "month_01", "day_15", "temp"]},
            'output': {"available_cols": ["year_2020", "month_01", "day_15"]}
        },
        'empty_list': {
            'input': {"all_possible_cols": []},
            'output': {"available_cols": []}
        },
        'no_matching_cols': {
            'input': {"all_possible_cols": ["temp", "humidity"]},
            'output': {"available_cols": []}
        },
        'check_register': {
            'input': {"all_possible_cols": ["temp", "humidity"]},
            'output': {"available_cols": []}
        },
        'wrong_possible_cols_data_type': {
            'input': {'all_possible_cols': ["year_2020", 2021, None]},
        },
        'wrong_field': {
            'input': {'wrong_field': ["year_2020"]},
        }
    }


@pytest.fixture
def test_norm_error_fixture():
    return {
        'wrong_df_data_type': {
            'input': {
                "json_list_df": "not_a_list",
                "col_time": 123,
                "col_target": 456
            },
        },
        'missing_col': {
            'input': {
                "json_list_df": [{"wrong_time": "2020-01-01", "value": 10.5}],
                "col_time": "time",
                "col_target": "load_consumption"
            }
        },
        'empty_df': {
            'input': {
                "json_list_df": [],
                "col_time": "time",
                "col_target": "load_consumption"
            }
        },
    }


@pytest.fixture
def test_denorm_error_fixture():
    return {
        'invalid_data': {
            'input': {
                "json_list_df": [{"time": "2020-01-01", "load_consumption": 0.5}],
                "col_time": "time",
                "col_target": "load_consumption",
                "min_val": "invalid",
                "max_val": 1.0
            },
        },
        'missing_col_data': {
            'input': {
                "json_list_df": [{"wrong_col": "2020-01-01", "value": 0.5}],
                "col_time": "time",
                "col_target": "load_consumption",
                "min_val": 0.0,
                "max_val": 1.0
            }
        },
        'empty_df': {
            'input': {
                "json_list_norm_df": [],
                "col_time": "time",
                "col_target": "load_consumption",
                "min_val": 0.0,
                "max_val": 1.0
            }
        },
    }
