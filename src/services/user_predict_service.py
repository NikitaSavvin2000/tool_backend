# src/services/user_predict_service.py

import pandas as pd
from src.backend.xgb import forecast_XGBoost_user
from src.backend.lstm import forecast_LSTM_user
from src.utils.date_utils import standardize_datetime
from src.core.logger import logger

def run_user_forecast(
    df: pd.DataFrame,
    time_column: str,
    col_target: str,
    forecast_horizon_time: str,
    col_for_train: List[str],
) -> Dict[str, Any]:
    """
    Выполняет пользовательское прогнозирование временных рядов.
    
    :param df: Исходный DataFrame.
    :param time_column: Название временной колонки.
    :param col_target: Название целевой колонки.
    :param forecast_horizon_time: Горизонт прогнозирования.
    :param col_for_train: Список колонок для обучения модели.
    :return: Словарь с результатами прогноза.
    """
    try:
        # Преобразование временной колонки в datetime
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
        forecast_horizon_time = standardize_datetime(forecast_horizon_time)

        # Определение последнего известного индекса
        last_known_index = len(df) - 1

        # Пример использования XGBoost для прогнозирования
        df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_massage = (
            forecast_XGBoost_user(
                col_target=col_target,
                time_column=time_column,
                df_all_data_norm=df[col_for_train],
                last_known_index=last_known_index,
                lag=12,  # Пример значения лага
                model_architecture_params={"objective": "reg:squarederror"},
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
        logger.error(f"Ошибка в run_user_forecast: {e}")
        raise