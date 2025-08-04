# tests/endpoints/test_pipeline.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.v1.pipeline import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)

def test_cols_to_chose_endpoint():
    # 1. Успешные сценарии
    # Типичный случай
    response = client.post('/cols-to-chose', json={
        'all_possible_cols': ["year_2020", "month_01", "day_15", "temp"]
    })

    data = response.json()
    assert response.status_code == 200
    assert 'available_cols' in data
    assert isinstance(data['available_cols'], list)
    assert data == {"available_cols": ["year_2020", "month_01", "day_15"]}
    
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
    # Проверка регистра
    response = client.post('/cols-to-chose', json={
        'all_possible_cols': ["YEAR_2020", "Month_01", "dAy_15"]
    })
    assert response.json() == {"available_cols": []}

    # 3. Ошибочные сценарии
    # Неправильный тип данных в массиве
    response = client.post('/cols-to-chose', json={
        'all_possible_cols': ["year_2020", 2021, None]
    })
    assert response.status_code == 422
    
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
            {"time": "2023-01-02", "value": 20},
            {"time": "2023-01-02", "value": 30},
            {"time": "2023-01-02", "value": 50},
            {"time": "2023-01-02", "value": 12}
        ],
        "time_column": "time"
    })

    data = response.json()
    assert response.status_code == 200
    assert 'df' in data
    assert isinstance(data['df'], list)
    assert data["df"][0]["time"] == "2023-01-01T00:00:00"
    
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


def test_all_available_forecast_endpoint():
    # 1. Успешные сценарии
    # Стандартный запрос
    response = client.post("/all-available-forecast", json={
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
            })
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], dict)
    print(data)
    assert 'map_data' in data['predictions']
    assert 'data' in data['predictions']['map_data']
    assert 'predictions' in data['predictions']['map_data']['data']
    assert isinstance(data['predictions']['map_data']['data']['predictions'], list)

    # 2. Ошибочные сценарии
    # Пустой DataFrame
    response = client.post("/all-available-forecast", json={
        "df": [],
        "time_column": "time",
        "col_target": "value",
        "forecast_horizon_time": "2023-01-01 01:00:00"
    })
    assert response.status_code == 400
    assert "не может быть пустым" in response.json()["detail"]

    # Несуществующая временная колонка
    response = client.post("/all-available-forecast", json={
        "df": [{"time": "2023-01-01 00:00:00", "value": 10}],
        "time_column": "datetime",
        "col_target": "value",
        "forecast_horizon_time": "2023-01-01 01:00:00"
    })
    assert response.status_code == 400

    # Неправильный формат горизонта прогнозирования
    response = client.post("/all-available-forecast", json={
        "df": [{"time": "2023-01-01 00:00:00", "value": 10}],
        "time_column": "time",
        "col_target": "value",
        "forecast_horizon_time": "invalid_date"
    })
    assert response.status_code == 400

    # Неправильный тип данных (например, строка вместо списка)
    response = client.post("/all-available-forecast", json={
        "df": "not_a_list",
        "time_column": "time",
        "col_target": "value",
        "forecast_horizon_time": "2023-01-01 01:00:00"
    })
    assert response.status_code == 422
