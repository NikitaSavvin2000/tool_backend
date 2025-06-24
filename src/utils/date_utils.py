from datetime import datetime, timezone

def standardize_datetime(input_date: str) -> str:
    """
    Преобразует входную дату в единый формат yyyy-mm-dd hh:mm:ss UTC.
    
    :param input_date: Входная строка с датой/временем в любом формате.
    :return: Стандартная строка в формате yyyy-mm-dd hh:mm:ss UTC.
    """
    # Список возможных форматов для парсинга
    possible_formats = [
        "%Y-%m-%d %H:%M:%S UTC",  # Поддержка формата с " UTC"
        "%Y-%m-%d %H:%M:%S%z",   # Поддержка формата с временной зоной (например, "+00:00")
        "%Y",                    # Год (например, 2023)
        "%Y-%m",                 # Год-месяц (например, 2023-09)
        "%Y-%m-%d",              # Год-месяц-день (например, 2023-09-15)
        "%d-%m-%Y",              # День-месяц-год (например, 15-09-2023)
        "%Y-%m-%d %H:%M",        # Год-месяц-день часы:минуты (например, 2023-09-15 14:30)
        "%Y-%m-%d %I:%M %p",     # Год-месяц-день часы:минуты AM/PM (например, 2023-09-15 2:30 PM)
        "%d-%m-%Y %I:%M %p",     # День-месяц-год часы:минуты AM/PM (например, 15-09-2023 2:30 PM)
        "%d/%m/%Y",              # День/месяц/год (например, 15/09/2023)
        "%Y/%m/%d",              # Год/месяц/день (например, 2023/09/15)
        "%Y/%m",                 # Год/месяц (например, 2023/09)
        "%Y%m%d",                # ГодМесяцДень без разделителей (например, 20230915)
        "%Y.%m.%d",              # Год.месяц.день (например, 2023.09.15)
        "%Y-%m-%d %H:%M:%S",     # Полный формат с секундами (например, 2023-09-15 14:30:00)
        "%d-%m-%Y %H:%M:%S",     # День-месяц-год часы:минуты:секунды (например, 15-09-2023 14:30:00)
        "%d.%m.%Y",              # День.месяц.год (например, 06.06.2025)
    ]
    
    # Попытка распознать формат
    for fmt in possible_formats:
        try:
            # Если формат подходит, преобразуем в datetime
            parsed_date = datetime.strptime(str(input_date), fmt)

            # Заполняем недостающие части по умолчанию
            if fmt == "%Y":
                parsed_date = parsed_date.replace(month=1, day=1, hour=0, minute=0, second=0)
            elif fmt == "%Y-%m":
                parsed_date = parsed_date.replace(day=1, hour=0, minute=0, second=0)
            elif fmt in ["%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"]:
                parsed_date = parsed_date.replace(hour=0, minute=0, second=0)

            # Явно преобразуем к UTC
            parsed_date = parsed_date.replace(tzinfo=timezone.utc)
            return parsed_date.strftime("%Y-%m-%d %H:%M:%S UTC")
        
        except ValueError:
            continue
    
    # Отдельная обработка для Unix timestamp
    try:
        timestamp = int(input_date)
        parsed_date = datetime.fromtimestamp(timestamp, tz=timezone.utc)  # Используем UTC
        return parsed_date.strftime("%Y-%m-%d %H:%M:%S UTC") 
    except (ValueError, TypeError):
        pass
    
    # Если ни один формат не подошел, вызываем ошибку
    raise ValueError(f"Не удалось преобразовать дату: {input_date}")