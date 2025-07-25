# services/model_service.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from src.config import logger


class ModelHandler:
    """
    Класс для работы с моделями LSTM и XGBoost.
    """

    def __init__(self, model_type: str = "LSTM"):
        """
        Инициализация ModelHandler.

        Args:
            model_type (str): Тип модели ("LSTM" или "XGBoost").
        """
        self.model_type = model_type
        self.model = None

    def build_lstm_model(self, input_shape: tuple, neurons: List[int], dropout: float):
        """
        Создание архитектуры LSTM.

        Args:
            input_shape (tuple): Размерность входных данных.
            neurons (List[int]): Количество нейронов на каждом слое.
            dropout (float): Коэффициент Dropout.

        Returns:
            Sequential: Модель LSTM.
        """
        try:
            model = Sequential()
            for i, n in enumerate(neurons):
                if i == 0:
                    model.add(Bidirectional(LSTM(n, return_sequences=True), input_shape=input_shape))
                else:
                    model.add(Bidirectional(LSTM(n, return_sequences=True)))
                model.add(Dropout(dropout))
            model.add(Dense(1))
            model.compile(optimizer="adam", loss="mse")
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
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Ошибка при предсказании модели: {e}")
            raise