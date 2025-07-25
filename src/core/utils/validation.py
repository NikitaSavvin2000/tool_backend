# src/utils/validation.py

import pandas as pd
from typing import List, Dict, Any
from fastapi import HTTPException

def validate_input_data(data: List[Dict[str, Any]]) -> None:
    """
    Проверяет входные данные на соответствие минимальным требованиям.
    
    :param data: Входные данные в формате JSON (список словарей).
    :raises HTTPException: Если данные некорректны.
    """
    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="Входные данные должны быть списком.")
    
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Входные данные не могут быть пустыми.")
    
    for item in data:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="Каждый элемент входных данных должен быть словарем.")
        
        for key, value in item.items():
            if value is None or (isinstance(value, str) and value.strip() == ""):
                raise HTTPException(status_code=400, detail=f"Значение для ключа '{key}' не может быть пустым.")

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Проверяет DataFrame на наличие обязательных колонок и отсутствие NaN/None значений.
    
    :param df: DataFrame для проверки.
    :param required_columns: Список обязательных колонок.
    :raises HTTPException: Если DataFrame не проходит проверку.
    """
    if df.empty:
        raise HTTPException(status_code=400, detail="DataFrame не может быть пустым.")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Отсутствуют обязательные колонки: {missing_columns}.")
    
    if df.isna().any().any():
        nan_columns = df.columns[df.isna().any()].tolist()
        raise HTTPException(status_code=400, detail=f"Обнаружены NaN/None значения в колонках: {nan_columns}.")

def validate_forecast_parameters(
    col_target: str,
    evaluation_index: int,
    last_know_index: int,
    epochs: int,
    lag: int,
    activation: str,
    optimizer: str,
    dropout_count: float,
    model_architecture_params: List[Dict],
) -> None:
    """
    Проверяет параметры прогнозирования на корректность.
    
    :param col_target: Название целевой колонки.
    :param evaluation_index: Индекс начала тестовой выборки.
    :param last_know_index: Индекс последнего известного значения.
    :param epochs: Количество эпох обучения.
    :param lag: Размер лага.
    :param activation: Функция активации.
    :param optimizer: Оптимизатор.
    :param dropout_count: Вероятность отсева нейронов.
    :param model_architecture_params: Параметры архитектуры модели.
    :raises HTTPException: Если параметры некорректны.
    """
    if not col_target:
        raise HTTPException(status_code=400, detail="Целевая колонка не может быть пустой.")
    
    if evaluation_index < 0 or last_know_index < 0:
        raise HTTPException(status_code=400, detail="Индексы не могут быть отрицательными.")
    
    if evaluation_index > last_know_index:
        raise HTTPException(status_code=400, detail="Индекс начала тестовой выборки не может превышать индекс последнего известного значения.")
    
    if epochs <= 0:
        raise HTTPException(status_code=400, detail="Количество эпох должно быть больше 0.")
    
    if lag <= 0:
        raise HTTPException(status_code=400, detail="Размер лага должен быть больше 0.")
    
    if dropout_count < 0 or dropout_count > 1:
        raise HTTPException(status_code=400, detail="Вероятность отсева нейронов должна быть в диапазоне [0, 1].")
    
    if not isinstance(model_architecture_params, list) or len(model_architecture_params) == 0:
        raise HTTPException(status_code=400, detail="Параметры архитектуры модели должны быть непустым списком.")