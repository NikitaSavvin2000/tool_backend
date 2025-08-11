# src/utils/dataframe_utils.py

import pandas as pd

def parse_json_to_df(json_data: list[dict]) -> pd.DataFrame:
    try:
        return pd.DataFrame(json_data)
    except Exception as e:
        raise ValueError(f"Ошибка при преобразовании JSON в DataFrame: {e}")
