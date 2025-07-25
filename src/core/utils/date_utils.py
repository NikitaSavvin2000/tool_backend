from datetime import datetime, timezone
from dateutil.parser import parse

def standardize_datetime(input_date: str) -> str:
    """
    Преобразует входную дату в единый формат ISO 8601 с временем в UTC.

    :param input_date: Входная строка с датой/временем в любом формате.
    :return: Стандартная строка в формате YYYY-MM-DD HH:MM:SS UTC.
    """
    try:
        # 1. Определяем тип входного значения
        if isinstance(input_date, datetime):
            dt = input_date
        elif isinstance(input_date, (int, float)):
            # Интерпретируем как Unix timestamp
            dt = datetime.fromtimestamp(input_date, tz=timezone.utc)
        elif isinstance(input_date, str):
            # Проверяем, является ли строка числом
            if input_date.isdigit():
                timestamp = int(input_date)
                # Если это похоже на Unix timestamp (длинное число), интерпретируем как timestamp
                if timestamp > 10**8:
                    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                else:
                    # Иначе предполагаем, что это год
                    dt = datetime.strptime(input_date, "%Y")
                    dt = dt.replace(month=1, day=1, hour=0, minute=0, second=0)
            else:
                # Явная обработка частичных форматов дат
                possible_formats = [
                    "%Y-%m",
                    "%Y-%m-%d",
                    "%d-%m-%Y",
                    "%Y-%m-%d %H:%M",
                    "%Y-%m-%d %I:%M %p",
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
        else:
            raise ValueError("Unsupported input type")

        # 2. Приводим к UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

        # 3. Преобразуем datetime в строку
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    except Exception as e:
        raise ValueError(f"Error processing date: {e}")
