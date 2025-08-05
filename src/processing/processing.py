# src/processing/processing.py
def to_float(value: str) -> float:
    if value is None:
        return value
    value = value.strip()

    if ',' in value and '.' in value:
        value = value.replace('.', '').replace(',', '.')
    elif ',' in value:
        value = value.replace(',', '.')
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"Невозможно преобразовать '{value}' в число")