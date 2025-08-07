# src/routers/pipeline_router.py
"""
Роутер для пайплайна обработки данных.
Содержит эндпоинт для подготовки данных: нормализация, разбиение на train/test.
"""
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from src.core.logger import logger
from src.services.pipeline_service import prepare_data_for_pipeline

router = APIRouter(tags=["Pipeline"])


class PipelineRequest(BaseModel):
    """
    Запрос для подготовки данных пайплайна.
    """
    df: List[Dict[Any, Any]]
    time_column: str
    col_target: str
    norm_values: bool = True  # По умолчанию — нормализация включена

    class Config:
        json_schema_extra = {
            "example": {
                "df": [
                    {"Дата": "2023-01-01", "Цена": 100},
                    {"Дата": "2023-01-02", "Цена": 105},
                    {"Дата": "2023-01-03", "Цена": 110}
                ],
                "time_column": "Дата",
                "col_target": "Цена",
                "norm_values": True
            }
        }


@router.post("/", response_model=Dict[str, List[Dict[str, Any]]])
async def prepare_pipeline_data(body: PipelineRequest = Body(...)):
    """
    Подготовка данных для пайплайна:
    - Нормализация (если norm_values=True)
    - Векторизация времени
    - Разделение на обучающую и тестовую выборки (80/20)

    Returns:
        {
          "df_train": [...],
          "df_test": [...]
        }
    """
    try:
        # Конвертируем JSON в DataFrame
        df = pd.DataFrame(body.df)

        # Логируем входные данные
        logger.info(f"Получено для обработки: {len(df)} строк")
        logger.info(f"Колонки: {list(df.columns)}")
        logger.info(f"Целевая колонка: {body.col_target}")

        # Выполняем подготовку данных
        result = prepare_data_for_pipeline(
            df=df,
            time_column=body.time_column,
            col_target=body.col_target,
            norm_values=body.norm_values
        )

        # Конвертируем DataFrame обратно в словари
        return {
            "df_train": result["df_train"].to_dict(orient="records"),
            "df_test": result["df_test"].to_dict(orient="records")
        }

    except Exception as e:
        logger.error(f"Ошибка при подготовке данных пайплайна: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Не удалось обработать данные: {str(e)}"
        )