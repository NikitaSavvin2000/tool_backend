import re
from typing import List, Dict, Optional

import pandas as pd
import pytest


def _get_conversion_func():
    """
    Пытается импортировать будущую функцию конвертации времени.
    Ожидаемая сигнатура: convert_time_column_to_iso_minutes(df: pd.DataFrame, column: str) -> Optional[pd.DataFrame]
    """
    try:
        # Функция, которая будет использоваться для конвертации времени
        from utils.date_utils import convert_time_column_to_iso_minutes  # type: ignore
        return convert_time_column_to_iso_minutes
    except Exception:
        pytest.xfail("Функция convert_time_column_to_iso_minutes ещё не реализована/не доступна для импорта.")


@pytest.fixture
def time_cases() -> List[Dict[str, str]]:
    """
    Заглушка для тестовых данных. Ожидается список словарей вида:
    {
      "input": <исходная дата в произвольном формате>,
      "expected": <строка в формате YYYY-MM-DDTHH:MM>
    }
    
    Пустой список приведёт к пропуску теста.
    """
    # Пример:
    # return [
    #     {"input": "2023-09-15 14:30:45", "expected": "2023-09-15T14:30"},
    #     {"input": "15/09/2023 2:30 PM", "expected": "2023-09-15T14:30"},
    #     {"input": 1694797800, "expected": "2023-09-15T17:10"},  # unix ts в секундах
    # ]
    return []


def _assert_iso_minutes_format(value: str):
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$"
    assert isinstance(value, str), f"Ожидалась строка в ISO-формате до минут, получено: {type(value)}"
    assert re.match(pattern, value) is not None, (
        f"Строка '{value}' не соответствует формату YYYY-MM-DDTHH:MM"
    )


def test_convert_time_column_to_iso_minutes(time_cases):
    if not time_cases:
        pytest.skip("Нет тестовых кейсов: добавьте/переопределите фикстуру 'time_cases'.")

    convert = _get_conversion_func()

    # Подготовка входного DataFrame из кейсов
    df_in = pd.DataFrame({"time": [case["input"] for case in time_cases]})

    # Поддерживаем два варианта реализации: in-place или с возвратом нового датафрейма
    maybe_df = convert(df_in.copy(), "time")
    if isinstance(maybe_df, pd.DataFrame):
        df_out = maybe_df
    else:
        df_out = df_in

    assert "time" in df_out.columns, "В результирующем датафрейме отсутствует столбец 'time'."
    assert len(df_out) == len(time_cases), "Число строк изменилось после конвертации."

    actual_values = list(df_out["time"])
    expected_values = [case["expected"] for case in time_cases]

    for idx, (src, actual, expected) in enumerate(zip([c["input"] for c in time_cases], actual_values, expected_values)):
        _assert_iso_minutes_format(actual)
        assert actual == expected, (
            f"Несоответствие на индексе {idx}: входное значение='{src}',\n"
            f"получено='{actual}', ожидалось='{expected}'"
        )


