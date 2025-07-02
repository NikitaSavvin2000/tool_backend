#src/notmalization/time2vec.py
import pandas as pd
import numpy as np

class Time2Vec:
    """
    Класс для преобразования временных меток в векторное представление.
    """
    def __init__(self, col_time: str, col_target: str):
        """
        Инициализация класса.
        
        :param col_time: Название колонки с временными метками.
        :param col_target: Название целевой переменной.
        """
        self.col_time = col_time
        self.col_target = col_target

    def vectorization(self, df: pd.DataFrame) -> tuple:
        """
        Преобразует DataFrame с временными метками в нормализованное векторное представление.
        
        :param df: Исходный DataFrame.
        :return: Кортеж из (преобразованный DataFrame, минимальное значение, максимальное значение).
        """
        df_copy = df.copy()
        df_copy[self.col_time] = pd.to_datetime(df_copy[self.col_time])
        
        # Создание новых признаков на основе времени
        df_copy['year'] = df_copy[self.col_time].dt.year
        df_copy['month'] = df_copy[self.col_time].dt.month
        df_copy['day'] = df_copy[self.col_time].dt.day
        df_copy['week'] = df_copy[self.col_time].dt.isocalendar().week
        df_copy['day_of_week'] = df_copy[self.col_time].dt.dayofweek
        df_copy['hour'] = df_copy[self.col_time].dt.hour
        df_copy['minute'] = df_copy[self.col_time].dt.minute
        df_copy['second'] = df_copy[self.col_time].dt.second
        
        # Тригонометрические преобразования
        df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24)
        df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24)
        df_copy['day_of_week_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
        df_copy['day_of_week_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)
        df_copy['week_sin'] = np.sin(2 * np.pi * df_copy['week'] / 52)
        df_copy['week_cos'] = np.cos(2 * np.pi * df_copy['week'] / 52)
        df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
        df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
        
        # Дополнительные признаки
        df_copy['part_of_day'] = (df_copy['hour'] % 24 + df_copy['minute'] / 60).astype(int)
        df_copy['is_night'] = ((df_copy['hour'] >= 0) & (df_copy['hour'] < 6)).astype(int)
        df_copy['is_weekend'] = (df_copy['day_of_week'] >= 5).astype(int)
        df_copy['day_of_year'] = df_copy[self.col_time].dt.dayofyear
        df_copy['is_working_hours'] = ((df_copy['hour'] >= 9) & (df_copy['hour'] <= 18)).astype(int)
        
        # Сезонность
        df_copy['season'] = (df_copy['month'] % 12 + 3) // 3
        df_copy['season_sin'] = np.sin(2 * np.pi * df_copy['season'] / 4)
        df_copy['season_cos'] = np.cos(2 * np.pi * df_copy['season'] / 4)
        
        # Кварталы
        df_copy['quarter'] = df_copy[self.col_time].dt.quarter
        df_copy['quarter_sin'] = np.sin(2 * np.pi * df_copy['quarter'] / 4)
        df_copy['quarter_cos'] = np.cos(2 * np.pi * df_copy['quarter'] / 4)
        
        # Фазы луны (примерная реализация)
        df_copy['moon_phase'] = (df_copy['day'] % 29.53).astype(int)
        
        # Нормализация целевой переменной
        min_val = df_copy[self.col_target].min()
        max_val = df_copy[self.col_target].max()
        df_copy[self.col_target] = (df_copy[self.col_target] - min_val) / (max_val - min_val)
        
        return df_copy, min_val, max_val

    def light_reverse_vectorization(self, df: pd.DataFrame, min_val: float, max_val: float) -> pd.DataFrame:
        """
        Обратное преобразование для целевой переменной.
        
        :param df: DataFrame с нормализованными данными.
        :param min_val: Минимальное значение до нормализации.
        :param max_val: Максимальное значение до нормализации.
        :return: DataFrame с денормализованной целевой переменной.
        """
        df_copy = df.copy()
        df_copy[self.col_target] = df_copy[self.col_target] * (max_val - min_val) + min_val
        return df_copy