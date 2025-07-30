# tests/endpoints/test_normalization.py
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api.v1.normalization import router


app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_normalization_response_structure(mock_normalization_data):
    response = client.post('/normalize', json=mock_normalization_data.model_dump())
    
    assert response.status_code == 200
    data = response.json()
    
    assert 'df_all_data_norm' in data
    assert 'min_val' in data
    assert 'max_val' in data
    assert data['max_val'] >= data['min_val']


def test_normalization_errors():
    # Тест на отсутствие обязательных полей
    response = client.post('/normalize', json={})
    assert response.status_code == 422
    assert "field required" in response.text.lower()
    
    # Тест на неверный тип данных
    invalid_data = {
        "json_list_df": "not_a_list",
        "col_time": 123,
        "col_target": 456
    }
    response = client.post('/normalize', json=invalid_data)
    assert response.status_code == 422
    
    # Тест на отсутствие колонок в данных
    missing_col_data = {
        "json_list_df": [{"wrong_time": "2020-01-01", "value": 10.5}],
        "col_time": "time",
        "col_target": "load_consumption"
    }
    response = client.post('/normalize', json=missing_col_data)
    assert response.status_code == 400
    
    # Тест на пустые данные
    empty_data = {
        "json_list_df": [],
        "col_time": "time",
        "col_target": "load_consumption"
    }
    response = client.post('/normalize', json=empty_data)
    assert response.status_code == 400
    


def test_denormalization_response_structure(mock_normalization_data):
    response_norm = client.post('/normalize', json=mock_normalization_data.model_dump())
    assert response_norm.status_code == 200, f"Normalization failed with status {response_norm.status_code}"

    data_norm = response_norm.json()
    denorm_request = {
        'json_list_norm_df': data_norm['df_all_data_norm'],
        "col_time": "time",
        "col_target": "load_consumption",
        'min_val': data_norm['min_val'],
        'max_val': data_norm['max_val']
    }

    denorm_response = client.post('/denormalize', json=denorm_request)
    assert denorm_response.status_code == 200

    data = denorm_response.json()
    
    assert 'df_all_data_reverse_norm' in data


def test_denormalization_errors(mock_normalization_data):
    # Тест на невалидные min_val/max_val
    invalid_data = {
        "json_list_df": [{"time": "2020-01-01", "load_consumption": 0.5}],
        "col_time": "time",
        "col_target": "load_consumption",
        "min_val": "invalid",
        "max_val": 1.0
    }
    response = client.post('/denormalize', json=invalid_data)
    assert response.status_code == 422
    
    # Тест на отсутствие колонок
    missing_col_data = {
        "json_list_df": [{"wrong_col": "2020-01-01", "value": 0.5}],
        "col_time": "time",
        "col_target": "load_consumption",
        "min_val": 0.0,
        "max_val": 1.0
    }
    response = client.post('/denormalize', json=missing_col_data)
    assert response.status_code == 422

    empty_data = {
        "json_list_norm_df": [],
        "col_time": "time",
        "col_target": "load_consumption",
        "min_val": 0.0,
        "max_val": 1.0
    }
    response = client.post('/denormalize', json=empty_data)
    assert response.status_code == 400