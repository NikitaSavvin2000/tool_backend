import os
import yaml


home_path = os.getcwd()


class NotExistCol(Exception):
    def __init__(self, not_possible_cols, possible_cols):
        message = f'Колонки {not_possible_cols} не могут быть заданны, тк не существуют в функции нормализации\nвозможные для указания колонки - {possible_cols}'
        super().__init__(message)


def update_col_for_train(new_cols_for_train):

    possible_cols = [
        "year", "month", "day", "week", "day_of_week", "hour", "minute", "second",
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
        "week_sin", "week_cos", "month_sin", "month_cos",
        "part_of_day", "is_night", "is_weekend", "day_of_year",
        "is_working_hours", "season", "season_sin", "season_cos",
        "quarter", "quarter_sin", "quarter_cos", "moon_phase",
        "time_trend", "fourier_time"
    ]

    not_possible_cols = []
    for new_col in new_cols_for_train:
        if new_col not in possible_cols:
            not_possible_cols.append(new_col)

    if len(not_possible_cols) > 0:
        message = f'Колонки {not_possible_cols} не могут быть заданны, тк не существуют в функции нормализации\nвозможные для указания колонки - {possible_cols}'

        return message
        # raise NotExistCol(not_possible_cols, possible_cols)

    else:
        file_path = f'{home_path}/src/backend/col_for_train.yaml'
        with open(file_path, 'r', encoding='utf-8') as f:
            col_for_train = yaml.safe_load(f)
            col_for_train['col_for_train'] = new_cols_for_train
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(col_for_train, f, allow_unicode=True, default_flow_style=False)

        return f'Обновили старые колонки на {new_cols_for_train}'


def update_col_for_train_lstm(new_cols_for_train):
    possible_cols = [
        "year", "month", "day", "week", "day_of_week", "hour", "minute", "second",
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
        "week_sin", "week_cos", "month_sin", "month_cos",
        "part_of_day", "is_night", "is_weekend", "day_of_year",
        "is_working_hours", "season", "season_sin", "season_cos",
        "quarter", "quarter_sin", "quarter_cos", "moon_phase",
        "time_trend", "fourier_time"
    ]

    not_possible_cols = []
    for new_col in new_cols_for_train:
        if new_col not in possible_cols:
            not_possible_cols.append(new_col)

    if len(not_possible_cols) > 0:
        message = f'Колонки {not_possible_cols} не могут быть заданны, тк не существуют в функции нормализации\nвозможные для указания колонки - {possible_cols}'

        return message
        # raise NotExistCol(not_possible_cols, possible_cols)

    else:
        file_path = f'{home_path}/src/backend/col_for_train_lstm.yaml'
        with open(file_path, 'r', encoding='utf-8') as f:
            col_for_train = yaml.safe_load(f)
            col_for_train['col_for_train'] = new_cols_for_train
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(col_for_train, f, allow_unicode=True, default_flow_style=False)

        return f'Обновили старые колонки на {new_cols_for_train}'


