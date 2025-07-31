# src/backend/update_col_for_train.py
import os
import yaml
from typing import List, Dict

home_path = os.getcwd()


class NotExistCol(Exception):
    def __init__(self, not_possible_cols: List[str], possible_cols: List[str]):
        message = (
            f'Колонки {not_possible_cols} не могут быть заданы, '
            f'так как не существуют в функции нормализации.\n'
            f'Возможные колонки: {possible_cols}'
        )
        super().__init__(message)


def update_col_for_train(new_cols_for_train: List[str]) -> Dict[str, List[str]]:
    """
    Обновляет список колонок для обучения (основной режим).
    Проверяет валидность колонок, записывает в YAML-файл.

    Args:
        new_cols_for_train (List[str]): Новые колонки для обучения.

    Returns:
        Dict[str, List[str]]: Успешный ответ с сообщением и списком колонок.

    Raises:
        ValueError: Если колонки невалидны или список пуст.
    """
    if not new_cols_for_train:
        raise ValueError("Список колонок для обучения не может быть пустым.")

    possible_cols = [
        "year", "month", "day", "week", "day_of_week", "hour", "minute", "second",
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
        "week_sin", "week_cos", "month_sin", "month_cos",
        "part_of_day", "is_night", "is_weekend", "day_of_year",
        "is_working_hours", "season", "season_sin", "season_cos",
        "quarter", "quarter_sin", "quarter_cos", "moon_phase",
        "time_trend", "fourier_time"
    ]

    not_possible_cols = [col for col in new_cols_for_train if col not in possible_cols]

    if not_possible_cols:
        raise ValueError(
            f"Колонки {not_possible_cols} не могут быть заданы, "
            f"так как не существуют в функции нормализации.\n"
            f"Возможные колонки: {possible_cols}"
        )

    file_path = f'{home_path}/src/backend/col_for_train.yaml'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            col_for_train = yaml.safe_load(f)
            print(f"Текущие колонки: {col_for_train['col_for_train']}")

        col_for_train['col_for_train'] = new_cols_for_train

        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(col_for_train, f, allow_unicode=True, default_flow_style=False)

        return {
            "message": "Columns for training updated successfully",
            "columns": new_cols_for_train
        }

    except Exception as e:
        raise ValueError(f"Ошибка при записи файла {file_path}: {e}")


def update_col_for_train_lstm(new_cols_for_train: List[str]) -> Dict[str, List[str]]:
    """
    Обновляет список колонок для обучения LSTM-модели.
    Проверяет валидность колонок, записывает в YAML-файл.

    Args:
        new_cols_for_train (List[str]): Новые колонки для LSTM.

    Returns:
        Dict[str, List[str]]: Успешный ответ с сообщением и списком колонок.

    Raises:
        ValueError: Если колонки невалидны или список пуст.
    """
    if not new_cols_for_train:
        raise ValueError("Список колонок для LSTM не может быть пустым.")

    possible_cols = [
        "year", "month", "day", "week", "day_of_week", "hour", "minute", "second",
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
        "week_sin", "week_cos", "month_sin", "month_cos",
        "part_of_day", "is_night", "is_weekend", "day_of_year",
        "is_working_hours", "season", "season_sin", "season_cos",
        "quarter", "quarter_sin", "quarter_cos", "moon_phase",
        "time_trend", "fourier_time"
    ]

    not_possible_cols = [col for col in new_cols_for_train if col not in possible_cols]

    if not_possible_cols:
        raise ValueError(
            f"Колонки {not_possible_cols} не могут быть заданы, "
            f"так как не существуют в функции нормализации.\n"
            f"Возможные колонки: {possible_cols}"
        )

    file_path = f'{home_path}/src/backend/col_for_train_lstm.yaml'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            col_for_train = yaml.safe_load(f)
            print(f"Текущие колонки для LSTM: {col_for_train['col_for_train']}")

        col_for_train['col_for_train'] = new_cols_for_train

        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(col_for_train, f, allow_unicode=True, default_flow_style=False)

        return {
            "message": "LSTM columns updated successfully",
            "columns": new_cols_for_train
        }

    except Exception as e:
        raise ValueError(f"Ошибка при записи файла {file_path}: {e}")