# services/forecast_service.py

import pandas as pd
from typing import Dict, Any, List
from src.utils.lstm_utils import split_sequence, create_x_input, make_predictions, inverse_transform_output
from src.services.model_handler import ModelHandler
from src.config import logger


def run_forecast(
    col_target: str,
    df_all_data_norm: List[Dict],
    evaluation_index: int,
    last_know_index: int,
    epochs: int,
    lag: int,
    activation: str,
    optimizer: str,
    dropout_count: float,
    model_architecture_params: List[Dict],
    model_type: str = "LSTM",
) -> Dict[str, Any]:
    """
    Выполняет прогнозирование временного ряда с использованием ModelHandler.

    Args:
        col_target (str): Название целевой колонки.
        df_all_data_norm (List[Dict]): Нормализованные данные временного ряда.
        evaluation_index (int): Индекс последней известной точки данных.
        last_know_index (int): Индекс последней известной точки времени.
        epochs (int): Количество эпох обучения.
        lag (int): Значение лага для прогнозирования.
        activation (str): Функция активации.
        optimizer (str): Оптимизатор.
        dropout_count (float): Коэффициент Dropout.
        model_architecture_params (List[Dict]): Параметры архитектуры модели.
        model_type (str): Тип модели ("LSTM" или "XGBoost").

    Returns:
        Dict[str, Any]: Словарь с результатами прогноза.
    """
    try:
        # Преобразование входных данных в DataFrame
        df = pd.DataFrame(df_all_data_norm)

        # Разделение данных на обучающую и тестовую выборки
        values = df[col_target].values
        X, y = split_sequence(values, lag)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Инициализация ModelHandler
        model_handler = ModelHandler(model_type=model_type)

        if model_type == "LSTM":
            # Создание и обучение LSTM модели
            input_shape = (lag, 1)
            architecture = model_architecture_params.get("architecture", [{"neurons": 32}])
            model_handler.build_lstm_model(
                input_shape=input_shape,
                architecture=architecture,
                dropout_count=dropout_count,
                optimizer=optimizer,
            )
            model_handler.train_model(X_train, y_train, epochs=epochs, batch_size=32)

            # Предсказание
            predictions = model_handler.predict(X_test)

        elif model_type == "XGBoost":
            # Обучение XGBoost модели
            xgboost_params = {"objective": "reg:squarederror", "activation": activation}
            model_handler.train_xgboost_model(X_train, y_train, params=xgboost_params)

            # Предсказание
            predictions = model_handler.forecast_xgboost(X_test)

        else:
            raise ValueError("Unsupported model type. Use 'LSTM' or 'XGBoost'.")

        # Обратное преобразование нормализованных данных
        scaler = MinMaxScaler()
        scaler.fit(df[[col_target]])
        df_predictions = inverse_transform_output(scaler, predictions, col_target)

        # Формирование результата
        last_real_data = df.iloc[last_know_index].to_dict()
        return {
            "map_data": {
                "data": {
                    "last_real_data": last_real_data,
                    "predictions": df_predictions.to_dict(orient="records"),
                },
                "last_know_data": df.iloc[last_know_index]["time_column"],
                "title": f"Прогноз {col_target}",
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
        logger.error(f"Ошибка в run_forecast: {e}")
        raise