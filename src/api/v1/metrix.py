from fastapi import APIRouter, Body, HTTPException
from src.models.schemes import MenrixAllRequest
from src.services.metrix_service import run_metrix_all
from src.config import logger

router = APIRouter()

@router.post("/all_metrix")
async def get_all_metrix(body: MenrixAllRequest = Body(...)):
    try:
        col_time = body.col_time
        col_target = body.col_target
        df_evaluation = pd.DataFrame(body.json_list_df_reverse_evaluation)
        df_comparative = pd.DataFrame(body.json_list_df_reverse_comparative)

        result = run_metrix_all(col_time, col_target, df_evaluation, df_comparative)
        return result
    except Exception as e:
        logger.error(f"Error in all_metrix endpoint: {e}")
        raise HTTPException(status_code=400, detail="Unknown Error")