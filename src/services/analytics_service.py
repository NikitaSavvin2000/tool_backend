# src/services/analytics_service.py

from typing import Dict, List

import pandas as pd

from src.core.logger import logger


def run_analytics_dfs(dfs_json_list: List[Dict]) -> Dict[str, str]:
    """
    Выполняет анализ множества DataFrame'ов.
    
    :param dfs_json_list: Список JSON-представлений DataFrame'ов.
    :return: Результат анализа в формате JSON.
    """
    try:
        # Преобразование JSON-списков в DataFrame'ы
        dfs = [pd.DataFrame(df) for df in dfs_json_list]

        # Пример анализа: проверка на наличие NaN значений
        nan_counts = {}
        for i, df in enumerate(dfs):
            nan_counts[f"df_{i+1}"] = df.isnull().sum().to_dict()

        # Возвращаем результат
        return {
            "message": "Hello Backend",
            "nan_counts": nan_counts,
        }

    except Exception as e:
        logger.error(f"Ошибка в run_analytics_dfs: {e}")
        raise