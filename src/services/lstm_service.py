# src/services/lstm_service.py

from typing import Any, Dict

import pandas as pd

from src.backend.lstm import forecast_LSTM
from src.core.logger import logger
from src.utils.date_utils import standardize_datetime


def run_lstm_forecast(
    df: pd.DataFrame,
    time_column: str,
    col_target: str,
    forecast_horizon_time: str,
) -> Dict[str, Any]:
    """
    Выполняет прогнозирование временных рядов с использованием LSTM.
    
    :param df: Исходный DataFrame.
    :param time_column: Название временной колонки.
    :param col_target: Название целевой колонки.
    :param forecast_horizon_time: Горизонт прогнозирования.
    :return: Словарь с результатами прогноза.
    """
    try:
        # Преобразование временной колонки в datetime
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
        forecast_horizon_time = standardize_datetime(forecast_horizon_time)

        # Определение последнего известного индекса
        last_known_index = len(df) - 1

        # Пример использования LSTM для прогнозирования
        df_evaluetion, df_true_all_col, loss_list, df_real_predict = (
            forecast_LSTM(
                col_target=col_target,
                time_column=time_column,
                df_all_data_norm=df,
                last_known_index=last_known_index,
                lag=12,  # Пример значения лага
                model_architecture_params={"architecture": [{"neurons": 32}]},
                forecast_type="predictions",
                norm_values=True,
            )
        )

        # Формирование результата
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
        logger.error(f"Ошибка в run_lstm_forecast: {e}")
        raise