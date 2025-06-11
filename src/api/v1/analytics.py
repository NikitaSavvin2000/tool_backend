from fastapi import APIRouter, Body, HTTPException
from src.models.schemes import AnalyticsDFsRequest
from src.config import logger

router = APIRouter()

@router.post("/analyticsdfs")
async def get_analytics_dfs(body: AnalyticsDFsRequest = Body(...)):
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