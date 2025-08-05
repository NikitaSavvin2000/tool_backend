import pytest


@pytest.mark.asyncio
def test_cols_to_chose_endpoint(async_client, cols_to_chose_data):
    # Типичный случай
    response = async_client.post('/cols-to-chose', json=cols_to_chose_data['typical_case']['input'])
    data = response.json()
    assert response.status_code == 200
    assert 'available_cols' in data
    assert isinstance(data['available_cols'], list)
    assert data == cols_to_chose_data['typical_case']['output']
    
    # Пустой список
    response = async_client.post('/cols-to-chose', json=cols_to_chose_data['empty_list']['input'])
    assert response.json() == cols_to_chose_data['empty_list']['output']
    
    # Нет подходящих колонок
    response = async_client.post('/cols-to-chose', json=cols_to_chose_data['no_matching_cols']['input'])
    assert response.json() == cols_to_chose_data['no_matching_cols']['output']

    # Проверка регистра
    response = async_client.post('/cols-to-chose', json=cols_to_chose_data['check_register']['input'])
    assert response.json() == cols_to_chose_data['check_register']['output']

    # Неправильный тип данных в массиве
    response = async_client.post('/cols-to-chose', json=cols_to_chose_data['wrong_possible_cols_data_type']['input'])
    assert response.status_code == 422
    
    # Отсутствие тела запроса
    response = async_client.post('/cols-to-chose')
    assert response.status_code == 422
    
    # Неправильное поле в запросе
    response = async_client.post('/cols-to-chose', json=cols_to_chose_data['wrong_field']['input'])
    assert response.status_code == 422


@pytest.mark.asyncio
def test_convert_datetime_endpoint(async_client, convert_datetime_data):
    # Стандартные форматы дат
    response = async_client.post("/convert-datetime", json=convert_datetime_data['standard_case']['input'])

    data = response.json()
    assert response.status_code == 200
    assert 'df' in data
    assert isinstance(data['df'], list)
    assert data == convert_datetime_data['standard_case']['output']
    
    # Пустой DataFrame (должен возвращать ошибку)
    response = async_client.post("/convert-datetime", json=convert_datetime_data['empty_df']['input'])
    assert response.status_code == 400
    assert "не может быть пустым" in response.json()["detail"]
    
    # Колонка уже в datetime формате
    response = async_client.post("/convert-datetime", json=convert_datetime_data['already_datetime']['input'])
    assert response.status_code == 200
    assert response.json() == convert_datetime_data['already_datetime']['output']
    
    # Несуществующая колонка
    response = async_client.post("/convert-datetime", json=convert_datetime_data['wrong_column']['input'])
    assert response.status_code == 400
    
    # Отсутствует обязательное поле
    response = async_client.post("/convert-datetime", json=convert_datetime_data['missing_time_key']['input'])
    assert response.status_code == 422
    
    # Неправильный тип данных
    response = async_client.post("/convert-datetime", json=convert_datetime_data['wrong_df_data_type']['input'])
    assert response.status_code == 422


@pytest.mark.asyncio
def test_all_available_forecast_endpoint(async_client, forecast_data):
    # Стандартный запрос
    response = async_client.post("/all-available-forecast", json=forecast_data['standard_case']['input'])
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], dict)
    assert 'map_data' in data['predictions']
    assert 'data' in data['predictions']['map_data']
    assert 'predictions' in data['predictions']['map_data']['data']
    assert isinstance(data['predictions']['map_data']['data']['predictions'], list)

    # Пустой DataFrame
    response = async_client.post("/all-available-forecast", json=forecast_data['empty_df']['input'])
    assert response.status_code == 400
    assert "не может быть пустым" in response.json()["detail"]

    # Несуществующая временная колонка
    response = async_client.post("/all-available-forecast", json=forecast_data['unknown_column']['input'])
    assert response.status_code == 400

    # Неправильный формат горизонта прогнозирования
    response = async_client.post("/all-available-forecast", json=forecast_data['wrong_format_forecast_horizon_time']['input'])
    assert response.status_code == 400

    # Неправильный тип данных (например, строка вместо списка)
    response = async_client.post("/all-available-forecast", json=forecast_data['wrong_df_data_type']['input'])
    assert response.status_code == 422
