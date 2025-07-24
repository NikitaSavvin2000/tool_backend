import pandas as pd
from src.backend.metrix import metrix_all
from src.core.logger import logger

def run_metrix_all(col_time, col_target, df_evaluation, df_comparative):
    try:
        metrics, df_metrics = metrix_all(col_time, col_target, df_evaluation, df_comparative)
        return {
            "metrics": metrics,
            "df_metrics": df_metrics.to_dict()
        }
    except Exception as e:
        logger.error(f"Error in metrix logic: {e}")
        raise