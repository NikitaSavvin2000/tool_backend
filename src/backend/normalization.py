from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import holidays
from sklearn.preprocessing import MinMaxScaler


class TimeNormalization:

    def __init__(self, col_time, col_target):
        self.min_year = 1900
        self.max_year = 2027
        self.min_weak = 1
        self.max_weak = 52
        self.min_day_of_weak = 0
        self.max_day_of_weak = 6
        self.min_minute = 0
        self.max_minute = 59
        self.min_second = 0
        self.max_second = 59
        self.min_hour = 0
        self.max_hour = 23
        self.scaler = MinMaxScaler()
        self.col_time = col_time
        self.col_target = col_target


    def normalize_column(self, value_series, min_val, max_val):
        """
        Нормировка колонки с использованием Min-Max Scaling.

        Parameters:
        - value_series: pd.Series, колонка значений для нормировки.
        - min_val: float, минимальное значение для нормировки.
        - max_val: float, максимальное значение для нормировки.

        Returns:
        - pd.Series, нормированные значения.
        """
        normalized = value_series.apply(
            lambda x: "None" if pd.isna(x) else (x - min_val) / (max_val - min_val)
        )

        return normalized

    def inverse_normalize_column(self, normalized_value, min_val, max_val):
        """
        Обратная нормировка колонки.

        Parameters:
        - normalized_series: pd.Series, колонка нормированных значений.
        - min_val: float, минимальное значение для обратной нормировки.
        - max_val: float, максимальное значение для обратной нормировки.

        Returns:
        - pd.Series, оригинальные значения.
        """
        if pd.isna(normalized_value):
            original = "None"
        else:
            original = normalized_value * (max_val - min_val) + min_val

        return original


    def check_difrent_years(self):
        if self.min_year != self.max_year:
            return True
        else:
            return False
    def meta_date(self, df):
        df_with_meta = df.copy()
        df_with_meta[self.col_time] = pd.to_datetime(df_with_meta[self.col_time])
        df_with_meta.set_index(self.col_time, inplace=True)
        df_with_meta['year'] = df_with_meta.index.year
        df_with_meta['week'] = df_with_meta.index.isocalendar().week
        df_with_meta['day_of_week'] = df_with_meta.index.dayofweek
        df_with_meta['hour'] = df_with_meta.index.hour
        df_with_meta['minute'] = df_with_meta.index.minute
        df_with_meta['second'] = df_with_meta.index.second
        df_with_meta['hour_sin'] = np.sin(2 * np.pi * df_with_meta['hour'] / 24)
        df_with_meta['hour_cos'] = np.cos(2 * np.pi * df_with_meta['hour'] / 24)
        df_with_meta['day_of_week_sin'] = np.sin(2 * np.pi * df_with_meta['day_of_week'] / 7)
        df_with_meta['day_of_week_cos'] = np.cos(2 * np.pi * df_with_meta['day_of_week'] / 7)
        df_with_meta['week_sin'] = np.sin(2 * np.pi * df_with_meta['week'] / 52)
        df_with_meta['week_cos'] = np.cos(2 * np.pi * df_with_meta['week'] / 52)
        it_holidays = holidays.Italy(years=df_with_meta['year'].unique())
        df_with_meta['is_holiday'] = pd.Series(df_with_meta.index.date).isin(it_holidays).astype(int).values

        return df_with_meta


    def df_normalize_with_meta(self, df):
        min_val = df[self.col_target].min()*1.2
        max_val = df[self.col_target].max()*1.2

        df_with_meta = self.meta_date(df)
        normalized_dates = []
        for index, date in df_with_meta.iterrows():
            if self.check_difrent_years:
                year_norm = (date['year'] - self.min_year) / (self.max_year - self.min_year)
            else:
                year_norm = 1
            weak_norm = ((date['week']) - self.min_weak) / (self.max_weak - self.min_weak)
            day_of_weak_norm = (
                                       date['day_of_week'] - self.min_day_of_weak) / (
                                           self.max_day_of_weak - self.min_day_of_weak
                                           )
            minute_norm = (date['minute'] - self.min_minute) / (self.max_minute - self.min_minute)
            second_norm = (date['second'] - self.min_second) / (self.max_second - self.min_second)
            hour_norm = (date['hour'] - self.min_hour) / (self.max_hour - self.min_hour)
            normalized_date = [date[self.col_target], year_norm, weak_norm, day_of_weak_norm, hour_norm, minute_norm, second_norm,
                               date['hour_sin'], date['hour_cos'], date['day_of_week_sin'], date['day_of_week_cos'],
                               date['week_sin'], date['week_cos'], date['is_holiday']]
            normalized_dates.append(normalized_date)
        normalized_df = pd.DataFrame(normalized_dates,
                                     columns=[self.col_target, 'year', 'week', 'day_of_week', 'hour', 'minute', 'second',
                                              'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                                              'week_sin', 'week_cos', 'is_holiday']
                                     )

        normalized_df[self.col_target] = self.normalize_column(normalized_df[self.col_target], min_val, max_val)

        return normalized_df, min_val, max_val

    def df_denormalize_with_meta(self, df, min_val, max_val):
        df = df.sort_values(by=['year', 'week', 'day_of_week', 'hour', 'minute'], ascending=True)

        def _convert_date(date_str):
            parts = date_str.split()
            year, week, day_of_week = map(int, parts[0].split('-'))
            hour, minute, second = map(int, parts[1].split(':'))
            start_of_week = datetime.strptime(f'{year}-{week}-1', '%Y-%W-%w')
            target_date = start_of_week + timedelta(days=day_of_week)
            target_date = target_date.replace(hour=hour, minute=minute, second=second)
            return target_date.strftime('%Y-%m-%d %H:%M:%S')

        denormalized_dates = []
        for index, date in df.iterrows():
            year_denorm = date['year'] * (self.max_year - self.min_year) + self.min_year
            weak_denorm = date['week'] * (self.max_weak - self.min_weak) + self.min_weak
            day_of_weak_denorm = date['day_of_week'] * (self.max_day_of_weak - self.min_day_of_weak) + self.min_day_of_weak
            minute_denorm = date['minute'] * (self.max_minute - self.min_minute) + self.min_minute
            second_denorm = date['second'] * (self.max_second - self.min_second) + self.min_second
            hour_denorm = date['hour'] * (self.max_hour - self.min_hour) + self.min_hour
            denormalized_date = [date[self.col_target], year_denorm, weak_denorm, day_of_weak_denorm, hour_denorm,
                                 minute_denorm, second_denorm,
                                 date['hour_sin'], date['hour_cos'], date['day_of_week_sin'], date['day_of_week_cos'],
                                 date['week_sin'], date['week_cos'], date['is_holiday']
                                 ]
            denormalized_dates.append(denormalized_date)

        for i in range(len(denormalized_dates)):
            # denormalized_dates[i][0] = self.scaler.inverse_transform([[denormalized_dates[i][0]]])[0][0]
            denormalized_dates[i][0] = self.inverse_normalize_column(denormalized_dates[i][0], min_val, max_val)


        denormalized_df = pd.DataFrame(denormalized_dates,
                                       columns=[self.col_target, 'year', 'week', 'day_of_week', 'hour', 'minute', 'second',
                                                'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                                                'week_sin', 'week_cos', 'is_holiday']
                                       )

        #TODO: Это костыль нужно убрать!!! (проблема что в колонке hout приходит дробное значение к примеру 12.99 и без строк ниде оно округляется до 12
        import math
        denormalized_df['hour'] = denormalized_df['hour'].apply(lambda x: math.ceil(x))
        denormalized_df['week'] = denormalized_df['week'].apply(lambda x: math.ceil(x))
        denormalized_df['day_of_week'] = denormalized_df['day_of_week'].apply(lambda x: math.ceil(x))
        denormalized_df['minute'] = denormalized_df['minute'].apply(lambda x: math.ceil(x))
        denormalized_df['year'] = denormalized_df['year'].apply(lambda x: math.ceil(x))

        denormalized_df['time_str'] = (denormalized_df['year'].astype(str) + '-' +
                                       denormalized_df['week'].astype(str) + '-' +
                                       denormalized_df['day_of_week'].astype(str) + ' ' +
                                       denormalized_df['hour'].astype(str) + ':' +
                                       denormalized_df['minute'].astype(str) + ':' +
                                       denormalized_df['second'].astype(int).astype(str))

        denormalized_df[self.col_time] = denormalized_df['time_str'].apply(_convert_date)
        return denormalized_df
