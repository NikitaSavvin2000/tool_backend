# src/utils/date_utils.py

from datetime import datetime, timezone
from dateutil.parser import parse
import re
from typing import Union

def standardize_datetime(input_date: Union[str, int, float, datetime]) -> str:
    """
    Приводит входные данные к стандартному формату даты и времени: '%Y-%m-%d %H:%M:%S'.
    
    :param input_date: Входная дата в различных форматах (строка, Unix timestamp, datetime).
    :return: Стандартизированная строка даты и времени.
    :raises ValueError: Если входной формат не поддерживается.
    """
    try:
        dt = None

        # Если входной формат — datetime
        if isinstance(input_date, datetime):
            dt = input_date

        # Если входной формат — число (Unix timestamp)
        elif isinstance(input_date, (int, float)):
            timestamp = input_date
            if timestamp > 10**8:  # Предполагаем, что это Unix timestamp
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            else:  # Иначе предполагаем, что это год
                dt = datetime(year=int(timestamp), month=1, day=1, tzinfo=timezone.utc)

        # Если входной формат — строка
        elif isinstance(input_date, str):
            input_date = input_date.strip()

            # Проверка на Unix timestamp в строковом формате
            if re.match(r"^\d+$", input_date):
                timestamp = int(input_date)
                if timestamp > 10**8:
                    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                else:
                    dt = datetime(year=timestamp, month=1, day=1, tzinfo=timezone.utc)

            # Явная обработка частичных форматов дат
            else:
                possible_formats = [
                    "%Y-%m",
                    "%Y-%m-%d",
                    "%d-%m-%Y",
                    "%Y-%m-%d %H:%M",
                    "%Y-%m-%d %I:%M %p",
                    "%B %Y",
                    "%B %d, %Y",
                    "%Y%m%d",
                ]
                for fmt in possible_formats:
                    try:
                        dt = datetime.strptime(input_date, fmt)
                        if fmt == "%Y-%m":
                            dt = dt.replace(day=1, hour=0, minute=0, second=0)
                        elif fmt in ["%Y-%m-%d", "%d-%m-%Y"]:
                            dt = dt.replace(hour=0, minute=0, second=0)
                        break
                    except ValueError:
                        continue
                else:
                    # Если ни один формат не подошел, пробуем парсинг через dateutil
                    dt = parse(input_date)

        # Если ни один из форматов не подходит
        if dt is None:
            raise ValueError("Unsupported input type")

        # Приводим к UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

        # Преобразуем datetime в строку
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    except Exception as e:
        raise ValueError(f"Error processing date: {e}")


def calculate_time_interval(df: pd.DataFrame, time_column: str) -> float:
    """
    Вычисляет интервал времени между последовательными записями в DataFrame.
    
    :param df: DataFrame с временным столбцом.
    :param time_column: Название временного столбца.
    :return: Интервал времени в секундах.
    """
    try:
        # Преобразование временного столбца в datetime
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')

        # Вычисление интервала времени
        time_diffs = df[time_column].diff().dropna()
        avg_interval = time_diffs.mean().total_seconds()

        return avg_interval

    except Exception as e:
        raise ValueError(f"Error calculating time interval: {e}")