# src/api/v1/analytics.py

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from src.models.schemes import AnalyticsDFsRequest
from src.services.analytics_service import run_analytics_dfs
from src.config import logger

router = APIRouter()

@router.post("/analyticsdfs", response_model=dict, tags=["Analytics"])
async def get_analytics_dfs(body: AnalyticsDFsRequest = Body(...)) -> Dict[str, Any]:
    """
    Эндпоинт для анализа множества DataFrame'ов.
    
    Description:
    - Принимает список JSON-представлений DataFrame'ов и выполняет их анализ.
    - Возвращает результат анализа в формате JSON.
    
    Parameters:
    - **body (AnalyticsDFsRequest)**: Схема запроса, содержащая следующие поля:
        - dfs_json_list (List[Dict]): Список JSON-представлений DataFrame'ов.
    
    Returns:
    - **dict**: Результат анализа, содержащий:
        - message: Сообщение "Hello Backend", если данные предоставлены.
    
    Example Request:
    ```json
    {
        "dfs_json_list": [
            {"Datetime": "2017-01-01 00:00:00", "Temperature": 6.4865, "Humidity": 74.15},
            {"Datetime": "2017-01-01 00:15:00", "Temperature": 6.5, "Humidity": 74.2}
        ]
    }
    ```
    
    Example Response:
    ```json
    {
        "message": "Hello Backend"
    }
    ```
    
    Raises:
    - **HTTPException 400**: Если входные данные пустые или произошла ошибка при анализе.
    """
    try:
        # Проверка на пустой список DataFrame'ов
        if not body.dfs_json_list:
            raise HTTPException(
                status_code=400,
                detail="Входной список DataFrame'ов не может быть пустым.",
                headers={"X-Error": "Empty input data provided"},
            )

        # Выполнение анализа
        result = run_analytics_dfs(body.dfs_json_list)
        return result

    except Exception as e:
        logger.error(f"Ошибка в эндпоинте /analyticsdfs: {e}")
        raise HTTPException(status_code=400, detail=str(e))