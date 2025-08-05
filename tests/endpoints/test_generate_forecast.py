#tests/endpoints/test_generate_forecast.py
import subprocess
import time
import pytest
import requests
import os
from dotenv import load_dotenv
import socket
import pandas as pd

load_dotenv()

home_path = os.getcwd()
URL = "http://localhost:7071"
TOKEN = os.getenv("TEST_TOKEN")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


@pytest.fixture(scope="session", autouse=True)
def start_server(request):
    port = 7071
    if is_port_in_use(port):
        print(f"[start_server] Port {port} is already in use, assuming server is running")
        yield
        return

    proc = subprocess.Popen(
        ["python", "-m", "src.server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for i in range(3):
        try:
            response = requests.get(f"http://localhost:{port}")
            if response.status_code == 200:
                print("[start_server] Server is up")
                break
        except Exception as e:
            print(f"[start_server] Attempt {i + 1} failed: {e}")
            time.sleep(1)
    else:
        stdout, _ = proc.communicate(timeout=2)
        print("[start_server] Server failed to start, logs:\n", stdout)
        proc.kill()
        raise RuntimeError("Server did not start in time")

    def teardown():
        if proc.poll() is None:
            print("[start_server] Terminating server process...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
                print("[start_server] Server stopped")
            except subprocess.TimeoutExpired:
                proc.kill()
                print("[start_server] Server force killed")

    request.addfinalizer(teardown)
    yield

@pytest.fixture(scope="session", autouse=True)
def wait_for_server():
    health_url = "http://localhost:7071/docs"
    for i in range(10):
        try:
            response = requests.get(health_url)
            if response.status_code in (200, 404):
                print("[wait_for_server] Server is up")
                return
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    raise ConnectionError("Server did not start in time")

@pytest.fixture
def mock_dataframe():
    df = pd.read_csv(f'{home_path}/src/examples_data/example_data.csv')
    df = df.loc[:1000]
    df['time'] = pd.to_datetime(df['time'])
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

def test_generate_forecast_structure(mock_dataframe):
    payload = prepare_payload(mock_dataframe, 'time', 'load_consumption', '2022-09-11 3:10:00')
    response = requests.post(URL, json=payload, headers=HEADERS)
    assert response.status_code == 200

    data = response.json()
    map_data = data['map_data']
    assert 'data' in map_data
    assert 'errors' in map_data
    assert 'last_real_data' in map_data['data']
    assert 'predictions' in map_data['data']
    assert isinstance(map_data['data']['last_real_data'], list)
    assert isinstance(map_data['data']['predictions'], list)
    assert 'last_know_data' in map_data
    assert 'title' in map_data
    assert 'legend' in map_data

def test_predictions_content(mock_dataframe):
    payload = prepare_payload(mock_dataframe, 'time', 'load_consumption', '2022-09-11 3:10:00')
    response = requests.post(URL, json=payload, headers=HEADERS)
    assert response.status_code == 200

    predictions = response.json()['map_data']['data']['predictions']
    assert len(predictions) > 0
    for item in predictions:
        assert 'time' in item
        assert 'load_consumption' in item
        assert isinstance(item['time'], str)
        assert isinstance(item['load_consumption'], (int, float))

def test_invalid_forecast_horizon(mock_dataframe):
    last_date = mock_dataframe['time'].max().strftime('%Y-%m-%d %H:%M:%S')
    payload = prepare_payload(mock_dataframe, 'time', 'load_consumption', last_date)
    response = requests.post(URL, json=payload, headers=HEADERS)
    assert response.status_code == 400

def test_missing_time_column(mock_dataframe):
    bad_df = mock_dataframe.drop(columns=['time'])
    payload = prepare_payload(bad_df, 'time', 'load_consumption', '2022-09-11 3:10:00')
    response = requests.post(URL, json=payload, headers=HEADERS)
    assert response.status_code == 400

@pytest.mark.parametrize("forecast_time", [
    "2022-09-9 3:10:00",
    "2022-09-10 2:10:00",
])
def test_generate_forecast_with_different_times(mock_dataframe, forecast_time):
    payload = prepare_payload(mock_dataframe, 'time', 'load_consumption', forecast_time)
    response = requests.post(URL, json=payload, headers=HEADERS)
    assert response.status_code == 200
    predictions = response.json()['map_data']['data']['predictions']
    assert len(predictions) > 0
