# src/backend/update_col_for_train.py
from pathlib import Path
from typing import Dict, List

import yaml

SCRIPT_DIR = Path(__file__).parent.resolve() 
POSSIBLE_COLS_FILE_PATH = SCRIPT_DIR.parent / "core" / "constants" / "possible_cols.yaml"

class NotExistCol(Exception):
    def __init__(self, not_possible_cols: List[str], possible_cols: List[str]):
        message = (
            f'Колонки {not_possible_cols} не могут быть заданы, '
            f'так как не существуют в функции нормализации.\n'
            f'Возможные колонки: {possible_cols}'
        )
        super().__init__(message)

def load_possible_columns() -> List[str]:
    """
    Загружает список всех возможных колонок из YAML-файла.

    Returns:
        List[str]: Список возможных колонок.

    Raises:
        ValueError: Если файл не найден или не может быть прочитан.
    """
    try:
        with open(POSSIBLE_COLS_FILE_PATH, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        possible_cols = data.get('all_possible_cols', [])
        if not isinstance(possible_cols, list):
             raise ValueError(f"Файл {POSSIBLE_COLS_FILE_PATH} не содержит корректный список 'all_possible_cols'.")
        return possible_cols
    except FileNotFoundError:
        raise ValueError(f"Файл с возможными колонками не найден: {POSSIBLE_COLS_FILE_PATH}")
    except yaml.YAMLError as e:
        raise ValueError(f"Ошибка при чтении файла {POSSIBLE_COLS_FILE_PATH}: {e}")
    except Exception as e:
        raise ValueError(f"Неизвестная ошибка при загрузке возможных колонок: {e}")


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

    possible_cols = load_possible_columns() 

    not_possible_cols = [col for col in new_cols_for_train if col not in possible_cols]

    if not_possible_cols:
        raise ValueError(
            f"Колонки {not_possible_cols} не могут быть заданы, "
            f"так как не существуют в функции нормализации.\n"
            f"Возможные колонки: {possible_cols}"
        )

    file_path = SCRIPT_DIR / "col_for_train.yaml" 
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            col_for_train = yaml.safe_load(f)

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

    possible_cols = load_possible_columns() 

    not_possible_cols = [col for col in new_cols_for_train if col not in possible_cols]

    if not_possible_cols:
        raise ValueError(
            f"Колонки {not_possible_cols} не могут быть заданы, "
            f"так как не существуют в функции нормализации.\n"
            f"Возможные колонки: {possible_cols}"
        )

    file_path = SCRIPT_DIR / "col_for_train_lstm.yaml"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            col_for_train = yaml.safe_load(f)

        col_for_train['col_for_train'] = new_cols_for_train

        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(col_for_train, f, allow_unicode=True, default_flow_style=False)

        return {
            "message": "LSTM columns updated successfully",
            "columns": new_cols_for_train
        }

    except Exception as e:
        raise ValueError(f"Ошибка при записи файла {file_path}: {e}")