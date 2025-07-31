# src/services/user_predict_service.py

from typing import Any, Dict, List

import pandas as pd
import traceback

from src.backend.xgb import forecast_XGBoost_user
from src.core.logger import logger
from src.utils.date_utils import standardize_datetime


def run_user_forecast(
    df: List[Dict[str, Any]],
    time_column: str,
    col_target: str,
    forecast_horizon_time: str,
    col_for_train: List[str],
    lag: int
) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(df)

        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
        forecast_horizon_time = standardize_datetime(forecast_horizon_time)

        last_known_index = len(df) - 1

        df_all_data_norm = df[[time_column] + col_for_train]

        df_train, df_real_predict = forecast_XGBoost_user(
            col_target=col_target,
            time_column=time_column,
            df_all_data_norm=df_all_data_norm,
            last_known_index=last_known_index,
            lag=lag,
            model_architecture_params={"objective": "reg:squarederror"},
            col_for_train=col_for_train,
        )

        last_real_data = df.to_dict(orient="records")
        predictions = df_real_predict.to_dict(orient="records")

        return {
            "map_data": {
                "data": {
                    "last_real_data": last_real_data,
                    "predictions": predictions,
                },
                "last_know_data": df.iloc[last_known_index][time_column].strftime('%Y-%m-%d %H:%M:%S'),
                "title": f"Реальный прогноз {col_target}",
                "legend": {
                    "last_know_data_line": {
                        "text": {"en": "Last known date", "ru": "Последняя известная дата"},
                        "color": "#A9A9A9",
                    },
                    "real_data_line": {
                        "text": {"en": "Real data", "ru": "Реальные данные"},
                        "color": "#0000FF",
                    },
                    "predict_data_line": {
                        "text": {"en": "Current forecast", "ru": "Актуальный прогноз"},
                        "color": "#FF0000",
                    },
                },
            }
        }

    except Exception as e:
        logger.error(f"Ошибка в run_user_forecast: {e}")
        logger.error(traceback.format_exc()) 
        raise
