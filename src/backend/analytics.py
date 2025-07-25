# src/backend/analytics.py

import pandas as pd


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет базовую обработку DataFrame'а.
    
    :param df: Исходный DataFrame.
    :return: Обработанный DataFrame.
    """
    try:
        # Пример простой обработки: удаление дубликатов
        df = df.drop_duplicates()
        
        # Можно добавить дополнительные шаги анализа
        # Например, расчет статистик, агрегация данных и т.д.
        
        return df

    except Exception as e:
        raise ValueError(f"Ошибка при обработке DataFrame: {e}")