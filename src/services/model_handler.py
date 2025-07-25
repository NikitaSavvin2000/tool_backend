# services/model_handler.py

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from sklearn.preprocessing import MinMaxScaler
from src.config import logger


class ModelHandler:
    """
    Класс для работы с различными моделями (LSTM, XGBoost).
    """

    def __init__(self, model_type: str = "LSTM"):
        """
        Инициализация ModelHandler.

        Args:
            model_type (str): Тип модели ("LSTM" или "XGBoost").
        """
        self.model_type = model_type
        self.model = None

    def build_lstm_model(self, input_shape: tuple, architecture: List[Dict], dropout_count: float, optimizer: str):
        """
        Создание архитектуры LSTM.

        Args:
            input_shape (tuple): Размерность входных данных.
            architecture (List[Dict]): Параметры архитектуры LSTM.
            dropout_count (float): Коэффициент Dropout.
            optimizer (str): Оптимизатор.

        Returns:
            Sequential: Модель LSTM.
        """
        try:
            model = Sequential()
            n_features = input_shape[-1]

            for i, layer in enumerate(architecture):
                neurons = layer["neurons"]
                activation = layer.get("activation", "relu")
                return_sequences = i < len(architecture) - 1

                if layer["type"] == "Bi-LSTM":
                    model.add(
                        Bidirectional(
                            LSTM(
                                neurons,
                                activation=activation,
                                return_sequences=return_sequences,
                                input_shape=input_shape if i == 0 else None,
                            )
                        )
                    )
                elif layer["type"] == "LSTM":
                    model.add(
                        LSTM(
                            neurons,
                            activation=activation,
                            return_sequences=return_sequences,
                            input_shape=input_shape if i == 0 else None,
                        )
                    )
                model.add(Dropout(dropout_count))

            model.add(Dense(1))
            model.compile(optimizer=optimizer, loss="mse")
            self.model = model
            return model
        except Exception as e:
            logger.error(f"Ошибка при создании LSTM модели: {e}")
            raise

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """
        Обучение модели.

        Args:
            X_train (np.ndarray): Входные данные.
            y_train (np.ndarray): Целевые данные.
            epochs (int): Количество эпох.
            batch_size (int): Размер батча.

        Returns:
            Any: Обученная модель.
        """
        try:
            if self.model is None:
                raise ValueError("Модель не инициализирована.")
            self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание модели.

        Args:
            X (np.ndarray): Входные данные.

        Returns:
            np.ndarray: Предсказанные значения.
        """
        try:
            if self.model is None:
                raise ValueError("Модель не обучена.")
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Ошибка при предсказании модели: {e}")
            raise

    def train_xgboost_model(self, X_train: np.ndarray, y_train: np.ndarray, params: Dict):
        """
        Обучение модели XGBoost.

        Args:
            X_train (np.ndarray): Входные данные.
            y_train (np.ndarray): Целевые данные.
            params (Dict): Параметры модели XGBoost.

        Returns:
            XGBRegressor: Обученная модель XGBoost.
        """
        try:
            xgb_model = XGBRegressor(**params)
            xgb_model.fit(X_train, y_train)
            self.model = xgb_model
            return xgb_model
        except Exception as e:
            logger.error(f"Ошибка при обучении XGBoost модели: {e}")
            raise

    def forecast_xgboost(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание модели XGBoost.

        Args:
            X (np.ndarray): Входные данные.

        Returns:
            np.ndarray: Предсказанные значения.
        """
        try:
            if self.model is None or not isinstance(self.model, XGBRegressor):
                raise ValueError("Модель XGBoost не обучена.")
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Ошибка при предсказании XGBoost модели: {e}")
            raise