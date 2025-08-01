# tests/endpoints/test_pipeline.py
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api.v1.pipeline import router


app = FastAPI()
app.include_router(router)
client = TestClient(app)

def test_cols_to_chose_endpoint():
    # 1. Базовые случаи
    # Типичный случай
    response = client.post('/cols-to-chose', json={
        'all_possible_cols': ["year_2020", "month_01", "day_15", "temp"]
    })
    assert response.status_code == 200
    assert response.json() == {"available_cols": ["year_2020", "month_01", "day_15"]}
    
    # Пустой список
    response = client.post('/cols-to-chose', json={
        'all_possible_cols': []
    })
    assert response.json() == {"available_cols": []}
    
    # Нет подходящих колонок
    response = client.post('/cols-to-chose', json={
        'all_possible_cols': ["temp", "humidity"]
    })
    assert response.json() == {"available_cols": []}

    # 2. Специальные форматы данных
    # Проверка регистра (должны игнорироваться)
    response = client.post('/cols-to-chose', json={
        'all_possible_cols': ["YEAR_2020", "Month_01", "dAy_15"]
    })
    assert response.json() == {"available_cols": []}

    # 3. Ошибочные сценарии
    # Неправильный тип данных в массиве
    response = client.post('/cols-to-chose', json={
        'all_possible_cols': ["year_2020", 2021, None]
    })
    assert response.status_code == 422  # Unprocessable Entity
    
    # Отсутствие тела запроса
    response = client.post('/cols-to-chose')
    assert response.status_code == 422
    
    # Неправильное поле в запросе
    response = client.post('/cols-to-chose', json={
        'wrong_field': ["year_2020"]
    })
    assert response.status_code == 422


def test_convert_datetime_endpoint():
    # 1. Успешные сценарии
    # Стандартные форматы дат
    response = client.post("/convert-datetime", json={
        "df": [
            {"time": "2023-01-01", "value": 10},
            {"time": "2023-01-02", "value": 20}
        ],
        "time_column": "time"
    })
    assert response.status_code == 200
    assert response.json()["df"][0]["time"] == "2023-01-01T00:00:00"
    
    # 2. Пограничные случаи
    # Пустой DataFrame (должен возвращать ошибку)
    response = client.post("/convert-datetime", json={
        "df": [],
        "time_column": "time"
    })
    assert response.status_code == 400
    assert "не может быть пустым" in response.json()["detail"]
    
    # Колонка уже в datetime формате
    response = client.post("/convert-datetime", json={
        "df": [
            {"time": "2023-01-01T00:00:00", "val": 1}
        ],
        "time_column": "time"
    })
    assert response.status_code == 200
    assert response.json()["df"][0]["time"] == "2023-01-01T00:00:00"
    
    # 3. Ошибочные сценарии
    # Несуществующая колонка
    response = client.post("/convert-datetime", json={
        "df": [
            {"time": "2023-01-01", "val": 1}
        ],
        "time_column": "datetime"
    })
    assert response.status_code == 400
    
    # 4. Неправильный запрос
    # Отсутствует обязательное поле
    response = client.post("/convert-datetime", json={
        "df": [{"time": "2023-01-01"}]
    })
    assert response.status_code == 422
    
    # Неправильный тип данных
    response = client.post("/convert-datetime", json={
        "df": "not_a_list",
        "time_column": "time"
    })
    assert response.status_code == 422


def test_generate_possible_date_endpoint():
    # 1. Успешный сценарий
    valid_data = {
        "df": [
            {"time": "2023-01-01 00:00:00", "value": 10},
            {"time": "2023-01-01 00:15:00", "value": 20}
        ],
        "time_column": "time",
        "col_target": "value",
        "forecast_horizon_time": "2023-01-01 01:00:00"
    }
    response = client.post("/generate-possible-date", json=valid_data)
    assert response.status_code == 200
    assert 'possible_dates' in response.json()
    assert isinstance(response.json()["possible_dates"], list)
    
    # 2. Ошибочные сценарии
    # Пустой DataFrame
    response = client.post("/generate-possible-date", json={
        "df": [],
        "time_column": "time",
        "col_target": "value",
        "forecast_horizon_time": "2023-01-01 01:00:00"
    })
    assert response.status_code == 400
    assert "не может быть пустым" in response.json()["detail"]
    
    # Несуществующая временная колонка
    response = client.post("/generate-possible-date", json={
        "df": [
            {"time": "2023-01-01 00:00:00", "value": 10}
        ],
        "time_column": "datetime",
        "col_target": "value",
        "forecast_horizon_time": "2023-01-01 01:00:00"
    })
    assert response.status_code == 400
    
    # Неправильный формат горизонта прогнозирования
    response = client.post("/generate-possible-date", json={
        "df": [
            {"time": "2023-01-01 00:00:00", "value": 10}
        ],
        "time_column": "time",
        "col_target": "value",
        "forecast_horizon_time": "invalid_date"
    })
    assert response.status_code == 400
    
    # 3. Неправильный запрос
    # Отсутствует обязательное поле
    response = client.post("/generate-possible-date", json={
        "df": [{"time": "2023-01-01"}],
        "time_column": "time",
        "col_target": "value"
    })
    assert response.status_code == 422
    
    # Неправильный тип данных
    response = client.post("/generate-possible-date", json={
        "df": "not_a_list",
        "time_column": "time",
        "col_target": "value",
        "forecast_horizon_time": "2023-01-01 01:00:00"
    })
    assert response.status_code == 422