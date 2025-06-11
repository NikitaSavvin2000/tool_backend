import requests
import pandas as pd
import time
import pytest

# URL = "http://localhost:7071/backend/v1/generate_forecast"
URL = "http://localhost:7071/backend/v1/backend/v1/pipeline/generate_forecast"


# Фикстура: ждёт, пока сервер запустится
@pytest.fixture(scope="module", autouse=True)
def wait_for_server():
    for _ in range(10):
        try:
            response = requests.get("http://localhost:7071")
            if response.status_code == 200:
                return
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    raise ConnectionError("Server did not start in time")

# Фикстура: подготовка тестового датафрейма
@pytest.fixture
def mock_dataframe():
    url = (
        "https://docs.google.com/spreadsheets/d/e/" 
        "2PACX-1vSgwB47qVFZcr1Aq--UWxZ6fDi9CGLZm-1i8QoMgfdaHUbV8EqSli3ayPxYYxD8kqfYYHD41uuNxbjZ/pub?"
        "gid=1952392108&single=true&output=csv"
    )
    df = pd.read_csv(url)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df

def prepare_payload(df, time_column, col_target, forecast_horizon_time):
    records = df.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if isinstance(value, pd.Timestamp):
                record[key] = str(value)
    return {
        "df": records,
        "time_column": time_column,
        "col_target": col_target,
        "forecast_horizon_time": forecast_horizon_time
    }

# Тест на структуру ответа
def test_generate_forecast_structure(mock_dataframe):
    payload = prepare_payload(
        mock_dataframe,
        time_column='Datetime',
        col_target='consumption',
        forecast_horizon_time='2018-01-10 05:00:00'
    )

    response = requests.post(URL, json=payload)
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"

    data = response.json()

    assert 'map_data' in data, "'map_data' key is missing in response"
    map_data = data['map_data']

    assert 'data' in map_data, "'data' key missing in map_data"
    assert 'last_real_data' in map_data['data'], "'last_real_data' missing in map_data.data"
    assert 'predictions' in map_data['data'], "'predictions' missing in map_data.data"

    assert isinstance(map_data['data']['last_real_data'], list), "last_real_data must be a list"
    assert isinstance(map_data['data']['predictions'], list), "predictions must be a list"

    assert 'last_know_data' in map_data, "'last_know_data' missing in map_data"
    assert 'title' in map_data, "'title' missing in map_data"
    assert 'legend' in map_data, "'legend' missing in map_data"

    print("Test passed: Response structure is correct")

# Тест: прогноз содержит корректные x и y
def test_predictions_content(mock_dataframe):
    payload = prepare_payload(
        mock_dataframe,
        time_column='Datetime',
        col_target='consumption',
        forecast_horizon_time='2018-01-10 05:00:00'
    )

    response = requests.post(URL, json=payload)
    assert response.status_code == 200

    data = response.json()
    predictions = data['map_data']['data']['predictions']

    assert len(predictions) > 0, "Predictions list is empty"
    for item in predictions:
        assert 'Datetime' in item, "Missing expected column 'Datetime'"
        assert 'consumption' in item, "Missing expected column 'consumption'"
        assert isinstance(item['Datetime'], str), "Expected 'Datetime' to be a string"
        assert isinstance(item['consumption'], (int, float)), "Expected 'consumption' to be numeric"

    print("Test passed: Predictions contain valid Datetime and consumption values")

# Тест: ошибка при слишком раннем горизонте прогнозирования
def test_invalid_forecast_horizon(mock_dataframe):
    last_date = mock_dataframe['Datetime'].max().strftime('%Y-%m-%d %H:%M:%S')
    payload = prepare_payload(
        mock_dataframe,
        time_column='Datetime',
        col_target='consumption',
        forecast_horizon_time=last_date  # Горизонта не хватает
    )

    response = requests.post(URL, json=payload)
    assert response.status_code == 400, "Expected 400 for invalid forecast horizon"

    print("Test passed: Server rejects invalid forecast horizon")

# Тест: ошибка при отсутствии колонки времени
def test_missing_time_column(mock_dataframe):
    bad_df = mock_dataframe.drop(columns=['Datetime'])
    payload = prepare_payload(
        bad_df,
        time_column='Datetime',
        col_target='consumption',
        forecast_horizon_time='2018-01-10 05:00:00'
    )

    response = requests.post(URL, json=payload)
    assert response.status_code == 400, "Expected 400 when time column is missing"

    print("Test passed: Server rejects request with missing time column")

# Параметризованный тест: разные горизонты прогноза
@pytest.mark.parametrize("forecast_time", [
    "2018-01-10 05:00:00",
    "2018-01-10 12:00:00",
])
def test_generate_forecast_with_different_times(mock_dataframe, forecast_time):
    payload = prepare_payload(
        mock_dataframe,
        time_column='Datetime',
        col_target='consumption',
        forecast_horizon_time=forecast_time
    )

    response = requests.post(URL, json=payload)
    assert response.status_code == 200, f"Failed with forecast time: {forecast_time}"

    data = response.json()
    predictions = data['map_data']['data']['predictions']
    assert len(predictions) > 0, "Predictions should not be empty"

    print(f"Test passed for forecast time: {forecast_time}")
