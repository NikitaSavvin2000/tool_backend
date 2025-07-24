from src.backend.all_available_forecast import all_available_forecast as backend_all_available_forecast
from src.core.logger import logger

def user_forecast(new_cols_for_train):
    try:
        return backend_all_available_forecast(df=None, time_column="", col_target="", forecast_horizon_time="")
    except Exception as e:
        logger.error(f"Error in user_forecast service: {e}")
        raise