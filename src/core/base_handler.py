# src/core/base_handler.py

from typing import List, Dict
import pandas as pd
from fastapi import HTTPException

from src.utils.validation import validate_input_data, validate_dataframe
from src.utils.dataframe_utils import parse_json_to_df


class BaseHandler:
    def parse_and_validate_dataframe(
        self,
        json_data: List[Dict],
        required_columns: List[str] = None,
        df_name: str = "DataFrame"
    ) -> pd.DataFrame:
        """
        Преобразует JSON в DataFrame и валидирует его.
        """
        validate_input_data(json_data)

        df = parse_json_to_df(json_data)
        if df.empty:
            raise HTTPException(status_code=400, detail=f"{df_name} не может быть пустым.")
        
        if required_columns:
            validate_dataframe(df, required_columns)

        return df
