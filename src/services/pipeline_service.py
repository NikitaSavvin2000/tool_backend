# src/services/pipeline_service.py

from src.utils.pipeline_utils import generate_possible_date, prepare_data_for_pipeline
from src.config import logger

def cols_to_chose(all_possible_cols: List[str]) -> List[str]:
    """
    Возвращает доступные колонки для анализа.
    
    :param all_possible_cols: Список всех возможных колонок.
    :return: Список доступных колонок.
    """
    try:
        return [col for col in all_possible_cols if col.startswith(("year", "month", "day"))]
    except Exception as e:
        logger.error(f"Ошибка в cols_to_chose: {e}")
        raise


def convert_df_to_datetime(df: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """
    Конвертирует временную колонку в формат datetime.
    
    :param df: Исходный DataFrame.
    :param time_column: Название временной колонки.
    :return: Преобразованный DataFrame.
    """
    try:
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
        return df
    except Exception as e:
        logger.error(f"Ошибка в convert_df_to_datetime: {e}")
        raise


def generate_possible_date_endpoint(
    df: pd.DataFrame, time_column: str, col_target: str, forecast_horizon_time: str
) -> List[str]:
    """
    Эндпоинт для генерации возможных дат прогнозирования.
    
    :param df: Исходный DataFrame.
    :param time_column: Название временной колонки.
    :param col_target: Название целевой колонки.
    :param forecast_horizon_time: Горизонт прогнозирования.
    :return: Список возможных дат.
    """
    try:
        return generate_possible_date(df, time_column, col_target, forecast_horizon_time)
    except Exception as e:
        logger.error(f"Ошибка в generate_possible_date_endpoint: {e}")
        raise


def prepare_data_for_pipeline_endpoint(
    df: pd.DataFrame, time_column: str, col_target: str, norm_values: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Эндпоинт для подготовки данных для пайплайна.
    
    :param df: Исходный DataFrame.
    :param time_column: Название временной колонки.
    :param col_target: Название целевой колонки.
    :param norm_values: Флаг для нормализации значений.
    :return: Словарь с подготовленными данными.
    """
    try:
        return prepare_data_for_pipeline(df, time_column, col_target, norm_values)
    except Exception as e:
        logger.error(f"Ошибка в prepare_data_for_pipeline_endpoint: {e}")
        raise