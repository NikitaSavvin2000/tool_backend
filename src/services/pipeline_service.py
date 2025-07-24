# src/services/pipeline_service.py
from src.backend.all_available_forecast import cols_to_chose as backend_cols_to_chose
from src.backend.all_available_forecast import convert_df_to_datetime as backend_convert_df_to_datetime
from src.backend.all_available_forecast import generate_possible_date as backend_generate_possible_date
from src.backend.all_available_forecast import all_available_forecast as backend_all_available_forecast

from src.core.logger import logger
import pandas as pd

def cols_to_chose(df):
    try:
        return backend_cols_to_chose(df)
    except Exception as e:
        logger.error(e)
        raise

def convert_df_to_datetime(df, time_column):
    try:
        return backend_convert_df_to_datetime(df, time_column)
    except Exception as e:
        logger.error(e)
        raise

def generate_possible_date(df, time_column):
    try:
        return backend_generate_possible_date(df, time_column)
    except Exception as e:
        logger.error(e)
        raise


def all_available_forecast(df, time_column, col_target, forecast_horizon_time):
    try:
        if df.empty:
            raise ValueError("DataFrame is empty")
        if time_column not in df.columns:
            raise ValueError(f"Time column '{time_column}' not found in data")
        if pd.isna(df[time_column]).any():
            raise ValueError(f"Time column '{time_column}' contains missing or invalid values")

        result = backend_all_available_forecast(
            df=df,
            time_column=time_column,
            col_target=col_target,
            forecast_horizon_time=forecast_horizon_time
        )
        return result

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise ve  # Это должно быть проброшено, чтобы @handle_exceptions поймало его
    except Exception as e:
        logger.error(f"Ошибка в all_available_forecast: {e}", exc_info=True)
        raise RuntimeError(f"Internal server error: {e}") from e