import pandas as pd


def calculate_time_interval(df: pd.DataFrame, time_column: str) -> int:
    """
    Вычисляет средний временной интервал в минутах между записями.

    :param df: DataFrame с временными метками
    :param time_column: Название колонки с временными метками
    :return: Средний временной интервал в минутах
    """
    df[time_column] = pd.to_datetime(df[time_column])
    time_interval = df[time_column].diff().dt.total_seconds().mean() / 60
    return round(time_interval)

def generate_possible_date(df, time_column):

    df = df.sort_values(by=time_column, ascending=False).reset_index(drop=True)

    last_know_date = df[time_column].iloc[0]
    possible_len = int(round(len(df)*0.05, 0))
    time_interval = abs(calculate_time_interval(df, time_column))

    start_date = pd.to_datetime(last_know_date, format='%Y-%m-%d %H:%M:%S')

    date_range = pd.date_range(start=start_date, periods=possible_len, freq=f'{time_interval}T')
    min_forecast_horizon_time = date_range[2]
    max_date = date_range[-1].strftime('%Y-%m-%d')

    last_know_date_dt = pd.to_datetime(last_know_date, format='%Y-%m-%d %H:%M:%S')

    min_date = last_know_date_dt.strftime('%Y-%m-%d')

    date = {
        "min": min_date,
        "max": max_date
    }

    unique_hours = sorted(date_range.hour.unique())   # [2, 3]
    unique_minutes = sorted(date_range.minute.unique())

    datetime = {
        "date": date,
        "time_hour": unique_hours,
        "time_minute": unique_minutes,
        "min_forecast_horizon_time": min_forecast_horizon_time
    }

    return datetime