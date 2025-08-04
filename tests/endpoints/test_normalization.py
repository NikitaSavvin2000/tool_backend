# tests/endpoints/test_normalization.py
import pytest


@pytest.mark.asyncio
def test_normalization_response_structure(async_client, mock_normalization_data):
    response = async_client.post('/api/v1/vectorization', json=mock_normalization_data.model_dump())
    
    assert response.status_code == 200
    data = response.json()
    
    assert 'df_all_data_norm' in data
    assert 'min_val' in data
    assert 'max_val' in data
    assert data['max_val'] >= data['min_val']


@pytest.mark.asyncio
def test_normalization_errors(async_client, test_norm_error_fixture):
    # Тест на отсутствие обязательных полей
    response = async_client.post('/api/v1/vectorization', json={})
    assert response.status_code == 400
    assert "field required" in response.text.lower()
    
    # Тест на неверный тип данных
    response = async_client.post('/api/v1/vectorization', json=test_norm_error_fixture['wrong_df_data_type']['input'])
    assert response.status_code == 400
    
    # Тест на отсутствие колонок в данных
    response = async_client.post('/api/v1/vectorization', json=test_norm_error_fixture['missing_col']['input'])
    assert response.status_code == 400
    
    # Тест на пустые данные
    response = async_client.post('/api/v1/vectorization', json=test_norm_error_fixture['empty_df']['input'])
    assert response.status_code == 400
    

@pytest.mark.asyncio
def test_denormalization_response_structure(async_client, mock_normalization_data):
    response_norm = async_client.post('/api/v1/vectorization', json=mock_normalization_data.model_dump())
    assert response_norm.status_code == 200, f"Normalization failed with status {response_norm.status_code}"

    data_norm = response_norm.json()
    denorm_request = {
        'json_list_norm_df': data_norm['df_all_data_norm'],
        "col_time": "time",
        "col_target": "load_consumption",
        'min_val': data_norm['min_val'],
        'max_val': data_norm['max_val']
    }

    denorm_response = async_client.post('/api/v1/reverse-vectorization', json=denorm_request)
    assert denorm_response.status_code == 200

    data = denorm_response.json()
    
    assert 'df_all_data_reverse_norm' in data


@pytest.mark.asyncio
def test_denormalization_errors(async_client, test_denorm_error_fixture):
    # Тест на невалидные min_val/max_val
    response = async_client.post('/api/v1/reverse-vectorization', json=test_denorm_error_fixture['invalid_data']['input'])
    assert response.status_code == 400
    
    # Тест на отсутствие колонок
    response = async_client.post('/api/v1/reverse-vectorization', json=test_denorm_error_fixture['missing_col_data']['input'])
    assert response.status_code == 400

    response = async_client.post('/api/v1/reverse-vectorization', json=test_denorm_error_fixture['empty_df']['input'])
    assert response.status_code == 400