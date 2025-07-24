# src/services/forecast_service.py

import pandas as pd
from typing import Dict, Any

from src.backend.forecast import forecast
from src.backend.xgb import forecast_XGBoost
from src.backend.lstm import forecast_LSTM
from src.backend.new_network import forecast_neural_networks

from src.core.logger import logger


def run_forecast(col_target, df_all_data_norm, evaluation_index, last_know_index, epochs, lag, activation, optimizer, dropout_count, model_architecture_params):
    try:
        df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_massage = forecast(
            col_target=col_target,
            df_all_data_norm=df_all_data_norm,
            evaluation_index=evaluation_index,
            last_know_index=last_know_index,
            epochs=epochs,
            lag=lag,
            activation=activation,
            optimizer=optimizer,
            dropout_count=dropout_count,
            model_architecture_params=model_architecture_params,
        )
        return {
            "df_evaluetion": df_evaluetion.to_dict(),
            "df_true_all_col": df_true_all_col.to_dict(),
            "df_real_predict": df_real_predict.to_dict(),
            "loss_list": loss_list,
            "response_code": response_code,
            "response_massage": response_massage,
        }
    except Exception as e:
        logger.error(f"Ошибка в run_forecast: {e}")
        raise


def run_xgboost_forecast(col_target, df_all_data_norm, evaluation_index, last_know_index, lag, model_architecture_params, forecast_type, norm_values):
    try:
        df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_massage = forecast_XGBoost(
            col_target=col_target,
            df_all_data_norm=df_all_data_norm,
            evaluation_index=evaluation_index,
            last_known_index=last_know_index,
            lag=lag,
            model_architecture_params=model_architecture_params,
            forecast_type=forecast_type,
            norm_values=norm_values
        )
        return {
            "df_evaluetion": df_evaluetion.to_dict(),
            "df_true_all_col": df_true_all_col.to_dict(),
            "df_real_predict": df_real_predict.to_dict(),
            "loss_list": loss_list,
            "response_code": response_code,
            "response_massage": response_massage,
        }
    except Exception as e:
        logger.error(f"Ошибка в run_xgboost_forecast: {e}")
        raise


def run_lstm_forecast(col_target, df_all_data_norm, evaluation_index, last_know_index, lag, model_architecture_params, forecast_type, norm_values):
    try:
        df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_massage = forecast_LSTM(
            col_target=col_target,
            df_all_data_norm=df_all_data_norm,
            evaluation_index=evaluation_index,
            last_know_index=last_know_index,
            lag=lag,
            model_architecture_params=model_architecture_params,
            type=forecast_type,
            norm_values=norm_values
        )
        return {
            "df_evaluetion": df_evaluetion.to_dict(),
            "df_true_all_col": df_true_all_col.to_dict(),
            "df_real_predict": df_real_predict.to_dict(),
            "loss_list": loss_list,
            "response_code": response_code,
            "response_massage": response_massage,
        }
    except Exception as e:
        logger.error(f"Ошибка в run_lstm_forecast: {e}")
        raise


def run_neural_network_forecast(col_target, df_all_data_norm, evaluation_index, last_know_index, model_architecture_params, forecast_type, norm_values):
    try:
        df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_massage = forecast_neural_networks(
            col_target=col_target,
            df_all_data_norm=df_all_data_norm,
            evaluation_index=evaluation_index,
            last_know_index=last_know_index,
            model_architecture_params=model_architecture_params,
            type=forecast_type,
            norm_values=norm_values
        )
        return {
            "df_evaluetion": df_evaluetion.to_dict(),
            "df_true_all_col": df_true_all_col.to_dict(),
            "df_real_predict": df_real_predict.to_dict(),
            "loss_list": loss_list,
            "response_code": response_code,
            "response_massage": response_massage,
        }
    except Exception as e:
        logger.error(f"Ошибка в run_neural_network_forecast: {e}")
        raise