from fastapi import APIRouter, Body, HTTPException
from src.models.schemes import AnalyticsDFsRequest
from src.config import logger

router = APIRouter()

@router.post("/analyticsdfs")
async def get_analytics_dfs(body: AnalyticsDFsRequest = Body(...)):
    """
    Эндпоинт для анализа множества DataFrame'ов.

    Parameters:
    - **body (AnalyticsDFsRequest)**: Схема запроса, содержащая следующие поля:
        - dfs_json_list (List[Dict]): Список JSON-представлений DataFrame'ов.

    Returns:
    - Dict: Словарь с результатом:
        - message: Сообщение "Hello Backend", если данные предоставлены.
        - В противном случае возвращает ошибку 400 Bad Request.

    Example Request:
    ```json
    {
        "dfs_json_list": [
            {"Datetime": "2017-01-01 00:00:00", "Temperature": 6.4865, "Humidity": 74.15, ...},
            ...
        ]
    }
    ```

    Example Response:
    ```json
    {
        "message": "Hello Backend"
    }
    ```
    """
    try:
        dfs_json_list = body.dfs_json_list
        if dfs_json_list:
            return {"message": "Hello Backend"}
        else:
            logger.error("Input data is empty during analyticsdfs request")
            raise HTTPException(
                status_code=400,
                detail="Bad Request",
                headers={"X-Error": "Empty input data provided"},
            )
    except Exception as e:
        logger.error(f"Error in analyticsdfs endpoint: {e}")
        raise HTTPException(status_code=400, detail="Unknown Error")