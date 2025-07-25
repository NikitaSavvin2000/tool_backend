import os
import re
import pandas as pd

from src.backend.xgb import forecast_XGBoost
from src.backend.normalization import Time2Vec
from datetime import datetime
from fastapi.responses import JSONResponse
from src.core.utils.possible_forecast_date import calculate_time_interval
from src.configuration.xgboost_constants import MODEL_ARCHITECTURE_PARAMS



home_path = os.getcwd()

def all_available_forecast(
        df: pd.DataFrame,
        time_column: str,
        col_target: str,
        forecast_horizon_time: str,
        lag: int = 4,
        forecast_type: str = 'predictions',
        norm_values: bool = True
) -> dict:
    """
    Генерирует прогноз временного ряда с использованием XGBoost.

    :param df: Исходный DataFrame с временным рядом
    :param time_column: Название колонки с временными метками
    :param col_target: Название целевой переменной
    :param forecast_horizon_time: Временная граница прогнозирования
    :param lag: Количество временных лагов
    :param forecast_type: Тип прогноза
    :param norm_values: Флаг нормализации значений
    :return: Словарь с прогнозными данными
    """
    df = df.sort_values(by=time_column, ascending=False).reset_index(drop=True)

    last_value = df[col_target].iloc[0]
    last_known_data = df.iloc[0][time_column]
    time_point_interval = abs(calculate_time_interval(df, time_column))
    last_time = df[time_column].iloc[0]

    date_range = pd.date_range(start=last_time, end=forecast_horizon_time, freq=f'{int(time_point_interval)}T')
    date_range = date_range[1:]

    df_future = pd.DataFrame({time_column: date_range, col_target: [None] * len(date_range)})
    df_all_data = pd.concat([df, df_future], ignore_index=True)

    df_all_data = df_all_data.sort_values(by=time_column, ascending=True).reset_index(drop=True)

    idx_first_none = df_all_data[col_target].first_valid_index()


    evaluation_index = df_all_data.index[-1]
    last_known_index = len(df_all_data) - len(date_range) - 1

    t2v = Time2Vec(col_time=time_column, col_target=col_target)
    df_all_data_norm, min_val, max_val = t2v.vectorization(df_all_data)

    df_eval, df_true_all, loss_list, df_pred_vector, response_code, response_message = forecast_XGBoost(
        col_target=col_target,
        df_all_data_norm=df_all_data_norm,
        evaluation_index=evaluation_index,
        last_known_index=last_known_index,
        lag=lag,
        model_architecture_params=[MODEL_ARCHITECTURE_PARAMS],
        forecast_type=forecast_type,
        norm_values=norm_values
    )

    df_real_predict = t2v.reverse_vectorization(df_pred_vector, min_val, max_val)
    df_real_predict.iloc[-1, df_real_predict.columns.get_loc(col_target)] = last_value
    df_real_predict = df_real_predict[[time_column, col_target]]
    df[time_column] = df[time_column].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_real_predict[time_column] = df_real_predict[time_column].dt.strftime("%Y-%m-%d %H:%M:%S")
    last_real_data = df.to_dict(orient="records")
    predictions = df_real_predict.to_dict(orient="records")

    return {
        "map_data": {
            "data": {
                "last_real_data": last_real_data,
                "predictions": predictions,
            },
            "last_know_data": last_known_data,
            "title": f"Реальный прогноз {col_target}",
            "legend": {
                "last_know_data_line": {
                    "text": {
                        "en": "Last known date",
                        "ru": "Последняя известная дата"
                    },
                    "color": "#A9A9A9"
                },
                "real_data_line": {
                    "text": {
                        "en": "Real data",
                        "ru": "Реальные данные"
                    },
                    "color": "#0000FF"
                },
                "predict_data_line": {
                    "text": {
                        "en": "Current forecast",
                        "ru": "Актуальный прогноз"
                    },
                    "color": "#FF0000"
                },
            },
        }
    }



def cols_to_chose(df: pd.DataFrame):
    time_patterns = re.compile(r'\b(date|time|datetime|timestamp|year)\b', re.IGNORECASE)
    probable_time_cols = []

    if len(df.columns) < 2:
        return JSONResponse(
            status_code=422,
            content={"ru": "Для выполнения прогноза необходимо как минимум 2 колонки, колонка времени и колонка целевого значения.",
                        "en": "To make a forecast, you need at least 2 columns, a time column and a target value column."}
        )
    for col in df.columns:
        if time_patterns.search(col):
            probable_time_cols.append(col)
            continue

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            probable_time_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            pd.to_datetime(df[col], errors='raise')
            probable_time_cols.append(col)
        except:
            pass

    time_col = probable_time_cols[0] if probable_time_cols else None
    cols_to_chose = {
        "time_col": time_col,
        "all_col": list(df.columns)
    }
    return cols_to_chose


def check_time_format(df, time_column):
    try:
        pd.to_datetime(df[time_column], format='%Y-%m-%d %H:%M:%S', errors='raise')
        return True
    except ValueError:
        return False


def convert_to_standard_datetime(date_value):
    """
    Convert various date formats to a standard datetime format: "%Y-%m-%d %H:%M:%S".

    Parameters:
    date_value (int, str, or None): The value to be converted. It can be an integer representing a timestamp or a string in various date formats.

    Returns:
    str: The converted date as a string in the format "%Y-%m-%d %H:%M:%S".

    Raises:
    ValueError: If the date_value cannot be converted to a valid date.
    """
    if date_value is None:
        return None

    if isinstance(date_value, int):
        if 1000 <= date_value <= 3000:
            return datetime(date_value, 1, 1).strftime("%Y-%m-%d %H:%M:%S")
        return datetime.utcfromtimestamp(date_value).strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(date_value, str):
        date_value = date_value.strip()

    if isinstance(date_value, str):
        date_value = date_value.strip()

        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_value):
            try:
                return datetime.strptime(date_value, "%Y-%m-%d").strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        if re.match(r"^\d{4}-\d{2}-\d{2} \d{1,2}$", date_value):
            try:
                return datetime.strptime(date_value, "%Y-%m-%d %H").strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        if re.match(r"^\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}$", date_value):
            try:
                return datetime.strptime(date_value, "%Y-%m-%d %H:%M").strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        if re.match(r"^\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}:\d{2}$", date_value):
            try:
                return datetime.strptime(date_value, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass

        if re.match(r"([a-zA-Z]+[\s]*[\d]+|[\d]{1,2}-[a-zA-Z]+-[\d]+)", date_value):
            try:
                return datetime.strptime(date_value, "%B %Y").strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        if re.match(r"^\d{8}$", date_value):
            try:
                return datetime.strptime(date_value, "%Y%m%d").strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        if re.match(r"^\d{4}-\d{2}$", date_value):
            try:
                return datetime.strptime(date_value, "%Y-%m").strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        if re.match(r"^[a-zA-Z]+\s+\d{1,2},\s+\d{4}$", date_value):
            try:
                return datetime.strptime(date_value, "%B %d, %Y").strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        if re.match(r"^\d{1,2}-\d{2}-\d{4}$", date_value):
            try:
                return datetime.strptime(date_value, "%d-%m-%Y").strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H",
        "%Y-%m-%d",
        "%Y-%m",
        "%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_value, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return date_value

    raise ValueError(f"Невозможно преобразовать дату: {date_value}")


def convert_df_to_datetime(df, time_column):
    if check_time_format(df, time_column):
        return df.to_dict(orient='records')

    try:
        df[time_column] = df[time_column].apply(convert_to_standard_datetime)
    except Exception as e:
        return JSONResponse(
            status_code=422,
            content={
                "ru": "Ошибка конвертации времени. Ожидаемый формат колонки времени : %Y-%m-%d %H:%M:%S",
                "en": "Time conversion error. Expected format: %Y-%m-%d %H:%M:%S"
            }
        )

    if check_time_format(df, time_column):
        return df.to_dict(orient='records')
    else:
        return JSONResponse(
            status_code=422,
            content={
                "ru": "Не удалось конвертировать колонку времени в необходимый формат. Измените формат колонки времени на %Y-%m-%d %H:%M:%S и повторно загрузите данные",
                "en": "Failed to convert the time column to the required format. Change the format of the time column to %Y-%m-%d %H:%M:%S and re-upload the data"
            }
        )



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




if __name__ == "__main__":
    df = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQT1DfqAB5Yec8MIQ_E5A8w-SXNcRmTwbXsv2W-ZT1ZcXN_G83BHlb6QBgnWkO-MpH3oVgfLoE0SnLx/pub?gid=1952392108&single=true&output=csv")
    col_time = 'Datetime'
    col_target = 'consumption'
    forecast_horizon_time = '2017-12-31 23:45:00'

    # response = all_available_forecast(df, time_column, col_target, forecast_horizon_time)
    # cols_to_chose = cols_to_chose(df=df)
    # print(cols_to_chose)
    test_df = pd.DataFrame({
        'mixed_dates': [
            '2023_03', '2023-09-21 21:22', 'March 2023', '2025 May', 1679722200, '20230325', 2025, 'March 25, 2023', None, '25-03-2023'
        ]
    })

    # test_df['mixed_dates_new'] = test_df['mixed_dates'].apply(convert_to_standard_datetime)
    print(df[col_time])
    data = generate_possible_date(df, col_time)
    print(data)
