# src/utils/possible_cols.py

from pathlib import Path
from typing import List

import yaml

from src.core.logger import logger

# Определение корневой директории проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # Корень проекта
CONFIG_DIR = PROJECT_ROOT / "src" / "configuration"

def load_possible_cols() -> List[str]:
    """
    Загружает список возможных колонок из конфигурационного файла.
    
    :return: Список возможных колонок.
    :raises FileNotFoundError: Если конфигурационный файл не найден.
    :raises ValueError: Если произошла ошибка при парсинге YAML.
    """
    config_path = CONFIG_DIR / "possible_cols.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return config.get('all_possible_cols', [])
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise ValueError(f"Error parsing YAML file: {e}")


def validate_columns(selected_cols: List[str]) -> None:
    """
    Проверяет, что выбранные колонки существуют в списке возможных колонок.
    
    :param selected_cols: Список выбранных колонок.
    :raises ValueError: Если одна или несколько колонок не существуют в списке возможных колонок.
    """
    possible_cols = load_possible_cols()
    invalid_cols = [col for col in selected_cols if col not in possible_cols]
    if invalid_cols:
        error_message = (
            f"Колонки {invalid_cols} не могут быть заданы, так как отсутствуют в списке возможных колонок. "
            f"Возможные колонки: {possible_cols}"
        )
        logger.error(error_message)
        raise ValueError(error_message)


def update_possible_cols(new_cols: List[str]) -> str:
    """
    Обновляет список возможных колонок в конфигурационном файле.
    
    :param new_cols: Новый список колонок.
    :return: Сообщение об успешном обновлении.
    :raises ValueError: Если произошла ошибка при записи YAML.
    """
    config_path = CONFIG_DIR / "possible_cols.yaml"
    try:
        # Валидация новых колонок (проверка на дубликаты и формат)
        unique_new_cols = list(set(new_cols))
        if len(unique_new_cols) != len(new_cols):
            raise ValueError("Список новых колонок содержит дубликаты.")
        
        # Запись новых колонок в файл
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump({"all_possible_cols": unique_new_cols}, file, allow_unicode=True, default_flow_style=False)
        
        return f"Обновили старые колонки на {unique_new_cols}"
    
    except Exception as e:
        logger.error(f"Ошибка при обновлении списка колонок: {e}")
        raise ValueError(f"Ошибка при обновлении списка колонок: {e}")