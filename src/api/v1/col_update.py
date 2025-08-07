# src/api/v1/col_update.py
from fastapi import APIRouter, Body, HTTPException

from src.core.logger import logger
from src.models.schemes import UpdateColRequest
from src.services.col_update_service import update_col_for_train, update_col_for_train_lstm

router = APIRouter()


@router.post("/update_col_for_train")
async def update_col_train(body: UpdateColRequest = Body(...)):
    """
    Эндпоинт для обновления списка колонок, используемых для обучения модели.

    Parameters:
    - **body (UpdateColRequest)**: Схема запроса, содержащая следующие поля:
        - col_for_train (List[str]): Список новых названий колонок для обучения.

    Returns:
    - Dict: Ответ от функции `update_col_for_train`, который может содержать подтверждение обновления или ошибку.

    Example Request:
    ```json
    {
        "col_for_train": ["year", "month", "hour", "is_weekend"]
    }
    ```

    Example Response (Success):
    ```json
    {
        "message": "Columns for training updated successfully",
        "columns": ["year", "month", "hour", "is_weekend"]
    }
    ```

    Raises:
    - **HTTPException 400**: Если произошла ошибка при обработке запроса.
    """
    try:
        col_for_train = body.col_for_train
        response = update_col_for_train(new_cols_for_train=col_for_train)
        return response
    except HTTPException:
        # Пробрасываем уже сформированное HTTP-исключение
        raise
    except Exception as e:
        logger.error(f"Error in /update_col_for_train: {e}")
        # Возвращаем реальное сообщение об ошибке, а не "Unknown Error"
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/update_col_for_train_lstm")
async def update_col_train_lstm(body: UpdateColRequest = Body(...)):
    """
    Эндпоинт для обновления списка колонок, используемых для обучения LSTM-модели.

    Parameters:
    - **body (UpdateColRequest)**: Схема запроса, содержащая следующие поля:
        - col_for_train (List[str]): Список новых названий колонок для LSTM-обучения.

    Returns:
    - Dict: Ответ от функции `update_col_for_train_lstm`, который может содержать подтверждение обновления или ошибку.

    Example Request:
    ```json
    {
        "col_for_train": ["year", "month", "hour", "season", "is_working_hours"]
    }
    ```

    Example Response (Success):
    ```json
    {
        "message": "LSTM columns updated successfully",
        "columns": ["year", "month", "hour", "season", "is_working_hours"]
    }
    ```

    Raises:
    - **HTTPException 400**: Если произошла ошибка при обработке запроса.
    """
    try:
        col_for_train = body.col_for_train
        response = update_col_for_train_lstm(new_cols_for_train=col_for_train)
        return response
    except HTTPException:
        # Пробрасываем уже сформированное HTTP-исключение
        raise
    except Exception as e:
        logger.error(f"Error in /update_col_for_train_lstm: {e}")
        # Возвращаем реальное сообщение об ошибке, а не "Unknown Error"
        raise HTTPException(status_code=400, detail=str(e))