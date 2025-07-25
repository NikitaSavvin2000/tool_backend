# src/services/col_update_service.py
from src.backend.update_col_for_train import update_col_for_train as backend_update_col_for_train
from src.backend.update_col_for_train import update_col_for_train_lstm as backend_update_col_for_train_lstm
from src.config import logger

def update_col_for_train(new_cols_for_train):
    try:
        return backend_update_col_for_train(new_cols_for_train=new_cols_for_train)
    except Exception as e:
        logger.error(f"Error in update_col_for_train service: {e}")
        raise

def update_col_for_train_lstm(new_cols_for_train):
    try:
        return backend_update_col_for_train_lstm(new_cols_for_train=new_cols_for_train)
    except Exception as e:
        logger.error(f"Error in update_col_for_train_lstm service: {e}")
        raise