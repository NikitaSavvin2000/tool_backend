# tests/endpoints/conftest.py
import os
import pytest
import pandas as pd
from src.routers.normalization_router import NormalizationRequest


home_path = os.getcwd()


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
