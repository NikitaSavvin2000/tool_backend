import uvicorn
import pandas as pd
import os

from typing import Annotated, List
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

from src.backend.metrix import metrix_all
from src.backend.forecast import forecast
from src.backend.lstm import forecast_LSTM
from src.backend.xgb import forecast_XGBoost
from src.config import logger, public_or_local
from src.processing.processing import to_float
from src.backend.normalization import Time2Vec
from src.backend.new_network import forecast_neural_networks
from src.backend.update_col_for_train import update_col_for_train, update_col_for_train_lstm
from src.models.schemes import (
    AnalyticsDFsRequest, NormalizationRequest, ForecastRequest,
    ReverseNormalizationRequest, MenrixAllRequest, ForecastRequestXGBoost,
    ForecastRequestNeuralNetworks, UpdateColRequest, ColsToChose, ConvertRequest, PredictRequest
)
from src.examples_fastapi.examples import (
    example_dfs_1, example_dfs_2, example_dfs_3,
    example_not_norm_data, example_reverse_norm_data,
    example_forecast_point, example_forecast_point_XGBoost,
    example_forecast_neural_networks, example_metrix_all,
)

from src.backend.all_available_forecast import cols_to_chose, convert_df_to_datetime, generate_possible_date, all_available_forecast
from dotenv import load_dotenv

load_dotenv()

home_path = os.getcwd()


example_df = pd.read_csv(f'{home_path}/src/examples/example_data.csv')
example_df_short = example_df[:10]
example_df_long = example_df[:1000]

example_df_json_short = example_df_short.to_dict(orient="records")
example_df_json_long = example_df_long.to_dict(orient="records")




if public_or_local == 'LOCAL':
    url = 'http://localhost'
else:
    url = 'http://77.37.136.11'

origins = [
    url
]
docs_url = "/backend/v1/"
app = FastAPI(docs_url=docs_url, openapi_url='/backend/v1/openapi.json')
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokens_link = os.getenv("TOKEN_LIST")

tokens_df = pd.read_csv(tokens_link)

VALID_TOKENS = tokens_df[tokens_df["source"] == "tool_backend"]["token"].tolist()

class TokenAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Unauthorized")
        token = auth.split(" ")[1]
        if token not in VALID_TOKENS:
            raise HTTPException(status_code=401, detail="Unauthorized")
        response = await call_next(request)
        return response

app.add_middleware(TokenAuthMiddleware)
#

@app.post("/backend/v1/analyticsdfs")
async def get_concepts(body: Annotated[
    AnalyticsDFsRequest, Body(
        example={"dfs_json_list": [example_dfs_1, example_dfs_2, example_dfs_3]})]):
    try:
        dfs_json_list = body.dfs_json_list
        if dfs_json_list:
            return 'Hello Backend'
        else:
            logger.error("Something happened during creation of the search table")
            raise HTTPException(
                status_code=400,
                detail="Bad Request",
                headers={"X-Error": "Something happened during creation of the search table"},
            )
    except Exception as ApplicationError:
        logger.error(ApplicationError.__repr__())
        raise HTTPException(
            status_code=400,
            detail="Unknown Error",
            headers={"X-Error": f"{ApplicationError.__repr__()}"},
        )

@app.post("/backend/v1/forecast")
async def get_concepts(body: Annotated[
    ForecastRequest, Body(
        example={
            "col_target": example_forecast_point['col_target'],
            "evaluation_index": example_forecast_point['evaluation_index'],
            "last_know_index": example_forecast_point['last_know_index'],
            "epochs": example_forecast_point['epochs'],
            "lag": example_forecast_point['lag'],
            "activation": example_forecast_point['activation'],
            "optimizer": example_forecast_point['optimizer'],
            "dropout_count": example_forecast_point['dropout_count'],
            "model_architecture_params": example_forecast_point['model_architecture_params'],
            "json_list_df_all_data_norm": example_forecast_point['json_list_df_all_data_norm'],
        })]):
    try:
        col_target = body.col_target
        epochs = body.epochs
        lag = body.lag
        activation = body.activation
        optimizer = body.optimizer
        dropout_count = body.dropout_count
        model_architecture_params = body.model_architecture_params
        json_list_df_all_data_norm = body.json_list_df_all_data_norm
        df_all_data_norm = pd.DataFrame(json_list_df_all_data_norm)
        df_all_data_norm['second'] = df_all_data_norm['second'].astype('int64')
        if not df_all_data_norm.empty:
            df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_massage = forecast(
                col_target=col_target,
                df_all_data_norm=df_all_data_norm,
                evaluation_index=body.evaluation_index,
                last_know_index=body.last_know_index,
                epochs=epochs,
                lag=lag,
                activation=activation,
                optimizer=optimizer,
                dropout_count=dropout_count,
                model_architecture_params=model_architecture_params,
            )
            response = {
                "df_evaluetion": df_evaluetion.to_dict(),
                "df_true_all_col": df_true_all_col.to_dict(),
                "df_real_predict": df_real_predict.to_dict(),
                "loss_list": loss_list,
                "response_code": response_code,
                "response_massage": response_massage,
            }
            return response
        else:
            logger.error("Something happened during creation of the search table")
            raise HTTPException(
                status_code=400,
                detail="Bad Request",
                headers={"X-Error": "Something happened during creation of the search table"},
            )
    except Exception as ApplicationError:
        logger.error(ApplicationError.__repr__())
        raise HTTPException(
            status_code=400,
            detail="Unknown Error",
            headers={"X-Error": f"{ApplicationError.__repr__()}"},
        )


@app.post("/backend/v1/forecast_XGBoost")
async def get_concepts(body: Annotated[
    ForecastRequestXGBoost, Body(
        example={
            "col_target": example_forecast_point_XGBoost['col_target'],
            "evaluation_index": example_forecast_point_XGBoost['evaluation_index'],
            "last_know_index": example_forecast_point_XGBoost['last_know_index'],
            "lag": example_forecast_point_XGBoost['lag'],
            "model_architecture_params": example_forecast_point_XGBoost['model_architecture_params'],
            "json_list_df_all_data_norm": example_forecast_point_XGBoost['json_list_df_all_data_norm'],
            "norm_values": example_forecast_point_XGBoost['norm_values'],
            "type": example_forecast_point_XGBoost['type']

        })]):

    try:
        evaluation_index = body.evaluation_index
        last_know_index = body.last_know_index
        col_target = body.col_target
        lag = body.lag
        model_architecture_params = body.model_architecture_params
        json_list_df_all_data_norm = body.json_list_df_all_data_norm
        df_all_data_norm = pd.DataFrame(json_list_df_all_data_norm)
        type=body.type
        norm_values = body.norm_values

        if norm_values:
            df_all_data_norm['second'] = df_all_data_norm['second'].astype('int64')

        if not df_all_data_norm.empty:

            df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_massage = forecast_XGBoost(
                col_target=col_target,
                df_all_data_norm=df_all_data_norm,
                evaluation_index=evaluation_index,
                last_known_index=last_know_index,
                lag=lag,
                model_architecture_params=model_architecture_params,
                forecast_type=type,
                norm_values=norm_values
            )
            response = {
                "df_evaluetion": df_evaluetion.to_dict(),
                "df_true_all_col": df_true_all_col.to_dict(),
                "df_real_predict": df_real_predict.to_dict(),
                "loss_list": loss_list,
                "response_code": response_code,
                "response_massage": response_massage,
            }
            return response
        else:
            logger.error("Something happened during creation of the search table")
            raise HTTPException(
                status_code=400,
                detail="Bad Request",
                headers={"X-Error": "Something happened during creation of the search table"},
            )
    except Exception as ApplicationError:
        logger.error(ApplicationError.__repr__())
        raise HTTPException(
            status_code=400,
            detail="Unknown Error",
            headers={"X-Error": f"{ApplicationError.__repr__()}"},
        )



@app.post("/backend/v1/all_metrix")
async def get_metrix_all(body: Annotated[
    MenrixAllRequest, Body(
        example={
            "col_time": example_metrix_all['col_time'],
            "col_target": example_metrix_all['col_target'],
            "json_list_df_reverse_evaluation": example_metrix_all['json_list_df_reverse_evaluation'],
            "json_list_df_reverse_comparative": example_metrix_all['json_list_df_reverse_comparative'],

        })]):
    try:
        col_time = body.col_time
        col_target = body.col_target
        json_list_df_reverse_evaluation = body.json_list_df_reverse_evaluation
        json_list_df_reverse_comparative = body.json_list_df_reverse_comparative

        df_evaluation = pd.DataFrame(json_list_df_reverse_evaluation)
        df_comparative = pd.DataFrame(json_list_df_reverse_comparative)

        if not df_evaluation.empty and not df_comparative.empty:

            metrics, df_metrics = metrix_all(col_time, col_target, df_evaluation, df_comparative)

            response = {
                "metrics": metrics,
                "df_metrics": df_metrics.to_dict()
            }
            return response
        else:
            logger.error("Something happened during creation of the search table")
            raise HTTPException(
                status_code=400,
                detail="Bad Request",
                headers={"X-Error": "Something happened during creation of the search table"},
            )
    except Exception as ApplicationError:
        logger.error(ApplicationError.__repr__())
        raise HTTPException(
            status_code=400,
            detail="Unknown Error",
            headers={"X-Error": f"{ApplicationError.__repr__()}"},
        )


@app.post("/backend/v1/LSTM_forecast")
async def LSTM_forecast(body: Annotated[
    ForecastRequestXGBoost, Body(
        example={
            "col_target": example_forecast_point_XGBoost['col_target'],
            "evaluation_index": example_forecast_point_XGBoost['evaluation_index'],
            "last_know_index": example_forecast_point_XGBoost['last_know_index'],
            "lag": example_forecast_point_XGBoost['lag'],
            "model_architecture_params": example_forecast_point_XGBoost['model_architecture_params'],
            "json_list_df_all_data_norm": example_forecast_point_XGBoost['json_list_df_all_data_norm'],
        })]):

    try:
        evaluation_index = body.evaluation_index
        last_know_index = body.last_know_index
        col_target = body.col_target
        lag = body.lag
        model_architecture_params = body.model_architecture_params
        json_list_df_all_data_norm = body.json_list_df_all_data_norm
        df_all_data_norm = pd.DataFrame(json_list_df_all_data_norm)
        type=body.type
        norm_values = eval(body.norm_values)
        if norm_values:
            df_all_data_norm['second'] = df_all_data_norm['second'].astype('int64')

        if not df_all_data_norm.empty:

            df_evaluetion, df_true_all_col, loss_list, df_real_predict, response_code, response_massage = forecast_LSTM(
                col_target=col_target,
                df_all_data_norm=df_all_data_norm,
                evaluation_index=evaluation_index,
                last_know_index=last_know_index,
                lag=lag,
                model_architecture_params=model_architecture_params,
                type=type,
                norm_values=norm_values
            )
            response = {
                "df_evaluetion": df_evaluetion.to_dict(),
                "df_true_all_col": df_true_all_col.to_dict(),
                "df_real_predict": df_real_predict.to_dict(),
                "loss_list": loss_list,
                "response_code": response_code,
                "response_massage": response_massage,
            }
            return response
        else:
            logger.error("Something happened during creation of the search table")
            raise HTTPException(
                status_code=400,
                detail="Bad Request",
                headers={"X-Error": "Something happened during creation of the search table"},
            )
    except Exception as ApplicationError:
        logger.error(ApplicationError.__repr__())
        raise HTTPException(
            status_code=400,
            detail="Unknown Error",
            headers={"X-Error": f"{ApplicationError.__repr__()}"},
        )


@app.post("/backend/v1/forecast_neural_networks")
async def request_forecast_neural_networks(body: Annotated[
    ForecastRequestNeuralNetworks, Body(
        example={
            "col_target": example_forecast_neural_networks['col_target'],
            "evaluation_index": example_forecast_neural_networks['evaluation_index'],
            "last_know_index": example_forecast_neural_networks['last_know_index'],
            "model_architecture_params": example_forecast_neural_networks['model_architecture_params'],
            "json_list_df_all_data_norm": example_forecast_neural_networks['json_list_df_all_data_norm'],
            "type": example_forecast_neural_networks['type'],
            "norm_values": example_forecast_neural_networks['norm_values']
        })]):

    try:
        col_target = body.col_target
        evaluation_index = body.evaluation_index
        last_know_index = body.last_know_index
        model_architecture_params = body.model_architecture_params
        json_list_df_all_data_norm = body.json_list_df_all_data_norm
        df_all_data_norm = pd.DataFrame(json_list_df_all_data_norm)
        type=body.type
        norm_values = eval(body.norm_values)
        if norm_values:
            df_all_data_norm['second'] = df_all_data_norm['second'].astype('int64')

        if not df_all_data_norm.empty:

            (df_evaluetion, df_true_all_col, loss_list, df_real_predict,
             response_code, response_massage) = forecast_neural_networks(
                col_target=col_target,
                df_all_data_norm=df_all_data_norm,
                evaluation_index=evaluation_index,
                last_know_index=last_know_index,
                model_architecture_params=model_architecture_params,
                type=type,
                norm_values=norm_values
            )
            response = {
                "df_evaluetion": df_evaluetion.to_dict(),
                "df_true_all_col": df_true_all_col.to_dict(),
                "df_real_predict": df_real_predict.to_dict(),
                "loss_list": loss_list,
                "response_code": response_code,
                "response_massage": response_massage,
            }
            return response
        else:
            logger.error("Something happened during creation of the search table")
            raise HTTPException(
                status_code=400,
                detail="Bad Request",
                headers={"X-Error": "Something happened during creation of the search table"},
            )
    except Exception as ApplicationError:
        logger.error(ApplicationError.__repr__())
        raise HTTPException(
            status_code=400,
            detail="Unknown Error",
            headers={"X-Error": f"{ApplicationError.__repr__()}"},
        )


@app.post("/backend/v1/normalization")
async def get_vectorization(body: Annotated[
    NormalizationRequest, Body(
        example={
            "col_time": example_not_norm_data['col_time'],
            "col_target": example_not_norm_data['col_target'],
            "json_list_df": example_not_norm_data['json_list_df'],
        })]):
    """
    Эндпоинт для нормализации данных с временной и целевой колонками.
    Принимает данные в виде списка словарей, преобразует их в DataFrame и нормализует с использованием
    указанного временного и целевого столбца. Возвращает нормализованные данные и метаинформацию.

    Параметры:
    - body (NormalizationRequest): Запрос с данными для нормализации:
        - col_time (str): Название столбца с временными метками.
        - col_target (str): Название столбца с целевыми значениями.
        - json_list_df (List[dict]): Данные в виде списка словарей для преобразования в DataFrame.

    Возвращает:
    - dict: Словарь с нормализованными данными (`df_all_data_norm`), минимальным значением (`min_val`),
      максимальным значением (`max_val`).

    Исключения:
    - HTTPException: В случае ошибки при обработке запроса или данных (например, пустой DataFrame или ошибка преобразования).
    """
    try:
        col_time = body.col_time
        col_target = body.col_target
        json_list_df = body.json_list_df

        df = pd.DataFrame(json_list_df)
        df[col_target] = df[col_target].replace("None", None)
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Input data is empty",
                headers={"X-Error": "Empty input data provided"}
            )
        try:
            try:
                df[col_target] = df[col_target].str.replace(',', '')
            except Exception as e:
                logger.error(e)
            finally:
                df[col_target] = df[col_target].astype(float)
        except Exception as e:
            logger.error(e)

            df[col_target] = df[col_target].apply(lambda x: to_float(x))
        # df[col_time] = pd.to_datetime(df[col_time], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df[col_time] = pd.to_datetime(df[col_time], errors='coerce')
        t2v = Time2Vec(col_time=col_time, col_target=col_target)
        df_all_data_norm, min_val, max_val = t2v.vectorization(df)
        response = {
            "df_all_data_norm": df_all_data_norm.to_dict(),
            "min_val": min_val,
            "max_val": max_val
        }
        return response

    except Exception as ApplicationError:
        logger.error(ApplicationError.__repr__())
        raise HTTPException(
            status_code=400,
            detail="Unknown Error",
            headers={"X-Error": f"{ApplicationError.__repr__()}"},
        )


@app.post("/backend/v1/reverse_normalization")
async def get_reverse_vectorization(body: Annotated[
    ReverseNormalizationRequest, Body(
        example={
            "col_time": example_reverse_norm_data['col_time'],
            "col_target": example_reverse_norm_data['col_target'],
            "json_list_norm_df": example_reverse_norm_data['json_list_norm_df'],
            "min_val": example_reverse_norm_data['min_val'],
            "max_val": example_reverse_norm_data['max_val']
        })]):
    try:
        col_time = body.col_time
        col_target = body.col_target
        json_list_norm_df = body.json_list_norm_df
        min_val = body.min_val
        max_val = body.max_val
        df = pd.DataFrame(json_list_norm_df)

        if not df.empty:

            t2v = Time2Vec(col_time=col_time, col_target=col_target)
            df_all_data_reverse_norm = t2v.reverse_vectorization(df, min_val, max_val)
            response = {
                "df_all_data_reverse_norm": df_all_data_reverse_norm.to_dict()
            }
            return response
        else:
            logger.error("Something happened during creation of the search table")
            raise HTTPException(
                status_code=400,
                detail="Bad Request",
                headers={"X-Error": "Something happened during creation of the search table"},
            )
    except Exception as ApplicationError:
        logger.error(ApplicationError.__repr__())
        raise HTTPException(
            status_code=400,
            detail="Unknown Error",
            headers={"X-Error": f"{ApplicationError.__repr__()}"},
        )


@app.post("/backend/v1/update_col_for_train")
async def update_col_for_train_request(body: Annotated[
    UpdateColRequest, Body(
        example={
            "col_for_train": ['year', 'month', 'day', 'day_of_year', 'week', 'day_of_week', 'hour', 'minute', 'second', 'part_of_day', 'is_night']
        })]):
    """
    Возможные колонки
        "year", "month", "day", "week", "day_of_week", "hour", "minute", "second",
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
        "week_sin", "week_cos", "month_sin", "month_cos",
        "part_of_day", "is_night", "is_weekend", "day_of_year",
        "is_working_hours", "season", "season_sin", "season_cos",
        "quarter", "quarter_sin", "quarter_cos", "moon_phase",
        "time_trend", "fourier_time"
    """
    try:
        col_for_train = body.col_for_train
        massage = update_col_for_train(new_cols_for_train=col_for_train)
        return massage

    except Exception as ApplicationError:
        logger.error(ApplicationError.__repr__())
        raise HTTPException(
            status_code=400,
            detail="Unknown Error",
            headers={"X-Error": f"{ApplicationError.__repr__()}"},
        )

@app.post("/backend/v1/update_col_for_train_lstm")
async def update_col_for_train_lstm_request(body: Annotated[
    UpdateColRequest, Body(
        example={
            "col_for_train": ['year', 'month', 'day', 'day_of_year', 'week', 'day_of_week', 'hour', 'minute', 'second', 'part_of_day', 'is_night']
        })]):
    """
    Возможные колонки
        "year", "month", "day", "week", "day_of_week", "hour", "minute", "second",
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
        "week_sin", "week_cos", "month_sin", "month_cos",
        "part_of_day", "is_night", "is_weekend", "day_of_year",
        "is_working_hours", "season", "season_sin", "season_cos",
        "quarter", "quarter_sin", "quarter_cos", "moon_phase",
        "time_trend", "fourier_time"
    """
    try:
        col_for_train = body.col_for_train
        massage = update_col_for_train_lstm(new_cols_for_train=col_for_train)
        return massage

    except Exception as ApplicationError:
        logger.error(ApplicationError.__repr__())
        raise HTTPException(
            status_code=400,
            detail="Unknown Error",
            headers={"X-Error": f"{ApplicationError.__repr__()}"},
        )


@app.post("/backend/v1/user_forecast")
async def func_user_forecast(body: Annotated[
    UpdateColRequest, Body(
        example={
            "df": [],
            "col_time": "Datetime",
            "col_target": "consumption",
            "forecast_horizon_time": "2017-12-31 23:45:00",
        })]):
    """
    Возможные колонки
        "year", "month", "day", "week", "day_of_week", "hour", "minute", "second",
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
        "week_sin", "week_cos", "month_sin", "month_cos",
        "part_of_day", "is_night", "is_weekend", "day_of_year",
        "is_working_hours", "season", "season_sin", "season_cos",
        "quarter", "quarter_sin", "quarter_cos", "moon_phase",
        "time_trend", "fourier_time"
    """
    try:
        col_for_train = body.col_for_train
        massage = update_col_for_train(new_cols_for_train=col_for_train)
        return massage

    except Exception as ApplicationError:
        logger.error(ApplicationError.__repr__())
        raise HTTPException(
            status_code=400,
            detail="Unknown Error",
            headers={"X-Error": f"{ApplicationError.__repr__()}"},
        )

# ================================ Новый бэкенд пайплайн =================================

@app.post("/backend/v1/cols_to_chose")
async def func_cols_to_chose(body: Annotated[
    ColsToChose, Body(
        example={
            "df": example_df_json_short,
        })]):
    """
        Определяет возможную колонку времени и возвращает список всех колонок в DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            Исходный DataFrame, содержащий данные.

        Returns:
        --------
        dict
            Словарь с ключами:
            - "time_col" (str | None): наиболее вероятная колонка времени или None, если подходящая колонка не найдена.
            - "all_col" (List[str]): список всех колонок в DataFrame.

        Raises:
        -------
        ValueError
            Если в DataFrame меньше двух колонок.
        """
    try:
        json_df = body.df
        df = pd.DataFrame(json_df)

        response = cols_to_chose(df)
        return response

    except Exception as ApplicationError:
        logger.error(ApplicationError.__repr__())
        raise HTTPException(
            status_code=400,
            detail="Unknown Error",
            headers={"X-Error": f"{ApplicationError.__repr__()}"},
        )


@app.post("/backend/v1/convert_time_to_datetime")
async def func_convert_time_to_datetime(body: Annotated[
    ConvertRequest, Body(
        example={
            "df": example_df_json_short,
            "time_column": "time"
        })]):
    """
    Конвертирует столбец времени в DataFrame в стандартный формат datetime.

    Описание:
    ----------
    Функция сначала проверяет, соответствует ли столбец `time_column` уже ожидаемому формату.
    Если да, то возвращает DataFrame в виде списка словарей.
    В противном случае выполняется попытка преобразования значений столбца в стандартный формат.
    Если преобразование не удаётся, возвращается JSON-ответ с кодом 422 и сообщением об ошибке.

    Параметры:
    ----------
    df : pd.DataFrame
        DataFrame, содержащий столбец времени.
    time_column : str
        Название столбца, содержащего временные данные.

    Возвращает:
    ----------
    list[dict] | JSONResponse
        - Список словарей, если конвертация успешна.
        - JSONResponse с кодом 422 и сообщением об ошибке, если конвертация не удалась.

    Обработка ошибок:
    -----------------
    В случае ошибки конвертации возвращается JSONResponse с HTTP-кодом 422.
    Сообщение пользователю передаётся в формате JSON и содержит два поля:
    - `ru`: сообщение на русском языке.
    - `en`: сообщение на английском языке.

    Возможные ошибки и их сообщения:
    --------------------------------
    1. Если столбец времени не соответствует ожидаемому формату и конвертация невозможна:
        ```json
        {
            "ru": "Ошибка конвертации времени. Ожидаемый формат колонки времени: %Y-%m-%d %H:%M:%S",
            "en": "Time conversion error. Expected format: %Y-%m-%d %H:%M:%S"
        }
        ```
    2. Если после попытки преобразования формат времени остаётся некорректным:
        ```json
        {
            "ru": "Не удалось конвертировать колонку времени в необходимый формат.",
            "en": "Failed to convert the time column to the required format."
        }
        ```
    Пример вызова:
    --------------------------------
    ```python
    import os
    import requests


    def call_cols_to_chose(df: pd.DataFrame):
    url_backend = "http://0.0.0.0:7071/backend/v1"
    url = url_backend + '/cols_to_chose'
    df_records = df.to_dict(orient='records')
    data = {
        "df": df_records
    }
    try:
        response = requests.post(url, json=data)
        return response
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе: {e}")
        return None

    response = func_convert_time_to_datetime(df=df, time_column=time_column)
    ```
    """
    try:
        json_df = body.df
        df = pd.DataFrame(json_df)
        time_column = body.time_column

        response = convert_df_to_datetime(df=df, time_column=time_column)
        return response

    except Exception as ApplicationError:
        logger.error(ApplicationError.__repr__())
        raise HTTPException(
            status_code=400,
            detail="Unknown Error",
            headers={"X-Error": f"{ApplicationError.__repr__()}"},
        )


@app.post("/backend/v1/generate_possible_date")
async def func_generate_possible_date(body: Annotated[
    ConvertRequest, Body(
        example={
            "df": example_df_json_long,
            "time_column": "time"
        })]):

    """
    Генерирует возможный диапазон дат и времени для фронта на основе временного столбца DataFrame.

    Описание:
    ----------
    Функция вычисляет интервал времени между записями и прогнозирует возможный диапазон дат.
    Минимальная дата (`min`) — это `last_know_date`, взятая из первой записи столбца `time_column`.
    Максимальная дата (`max`) определяется как `5%` от длины DataFrame в будущем, с учётом вычисленного интервала времени.
    Также возвращается параметр `min_forecast_horizon_time`, который обозначает минимально возможную дату для выбора пользователем.

    Параметры:
    ----------
    df : pd.DataFrame
        DataFrame, содержащий временные данные.
    time_column : str
        Название столбца, содержащего временные метки.

    Возвращает:
    ----------
    dict:
        - `date`: словарь с минимальной (`min`) и максимальной (`max`) датами.
        - `min_forecast_horizon_time`: минимально возможная дата для выбора пользователем.
        - `time_hour`: список возможных значений часов (от `0` до `23`).
        - `time_minute`: список возможных значений минут (от `0` до `59`).

    Формат возвращаемого JSON:
    --------------------------
    ```json
    {
        "date": {
            "min": "2024-01-01",
            "max": "2024-02-15"
        },
        "min_forecast_horizon_time": "2024-01-01",
        "time_hour": [0, 1, 2, ..., 23],
        "time_minute": [0, 1, 2, ..., 59]
    }
    ```

    Пример использования:
    ---------------------
    ```python
    import requests

    def func_generate_possible_date(df: pd.DataFrame, time_column: str):
        url = url_backend + '/generate_possible_date'
        df_records = df.to_dict(orient='records')

        data = {
            "df": df_records,
            "time_column": time_column
        }

        try:
            response = requests.post(url, json=data)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе: {e}")
            return None

    response = func_generate_possible_date(df, time_column)
    print(response)
    ```
    """
    try:
        json_df = body.df
        df = pd.DataFrame(json_df)
        time_column = body.time_column

        response = generate_possible_date(df=df, time_column=time_column)
        return response

    except Exception as ApplicationError:
        logger.error(ApplicationError.__repr__())
        raise HTTPException(
            status_code=400,
            detail="Unknown Error",
            headers={"X-Error": f"{ApplicationError.__repr__()}"},
        )


@app.post("/backend/v1/generate_forecast")
async def func_generate_forecast(body: Annotated[
    PredictRequest, Body(
        example={
            "df": example_df_json_long,
            "time_column": "time",
            "col_target": "load_consumption",
            "forecast_horizon_time": "2022-09-10 05:55:00"
        })]):

    """
    Генерирует прогноз временного ряда на основе исторических данных.

    Описание:
    ----------
    Функция принимает временной ряд, нормализует данные, формирует будущий интервал времени и
    выполняет прогнозирование. Возвращает результат в формате JSON, который включает последние известные
    данные и прогнозируемые значения для визуализации на фронтенде.

    Параметры:
    ----------
    1. `df` (pd.DataFrame) — Исходный DataFrame, содержащий временной ряд с целевой переменной.
    2. `time_column` (str) — Название столбца, содержащего временные метки.
    3. `col_target` (str) — Название целевой переменной, по которой строится прогноз.
    4. `forecast_horizon_time` (str) — Временная граница прогнозирования (последняя возможная дата прогноза).
    5. `lag` (int, optional, по умолчанию 4) — Количество временных лагов, используемых для предсказания.
    6. `forecast_type` (str, optional, по умолчанию 'predictions') — Тип прогнозирования, определяющий, какие предсказания будут сгенерированы.
    7. `norm_values` (bool, optional, по умолчанию True) — Флаг нормализации значений перед подачей в модель.

    Возвращает:
    ----------
    1. `map_data` (dict) - словапь данных для отрисовки
        - **data** (dict):
            - **last_real_data** (list[dict]) — Последние известны еданные пользователя
            - **predictions** (list[dict]) — Данные предсказания.

    2. `last_know_data_line` (dict) — линия, обозначающая последнюю известную дату:
        - **text** (dict):
            - **en** (str) — описание на английском языке.
            - **ru** (str) — описание на русском языке.
        - **color** (str) — цвет линии, разделяющей реальные данные и прогноз.
    3. `real_data_line` (dict) — линия реальных данных:
        - **text** (dict):
            - **en** (str) — описание на английском языке.
            - **ru** (str) — описание на русском языке.
        - **color** (str) — цвет линии реальных данных на графике.
    4. `predict_data_line` (dict) — линия прогнозируемых данных:
        - **text** (dict):
            - **en** (str) — описание на английском языке.
            - **ru** (str) — описание на русском языке.
        - **color** (str) — цвет линии прогнозируемых данных.


    Пример вызова API python:
    ------------------
    ```
    import requests
    import pandas as pd

    def func_generate_forecast(df: pd.DataFrame, time_column: str, col_target: str, forecast_horizon_time: str):
        url = "http://your_backend_url/backend/v1/generate_forecast"

        df_records = df.to_dict(orient='records')

        data = {
            "df": df_records,
            "time_column": time_column,
            "col_target": col_target,
            "forecast_horizon_time": forecast_horizon_time
        }

        try:
            response = requests.post(url, json=data)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Ошибка при запросе: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе: {e}")
            return None

    df = pd.read_csv(<here yor data>)

    time_column = 'time'
    col_target = 'load_consumption'
    forecast_horizon_time = '2022-09-10 05:00:00'
    df[time_column] = pd.to_datetime(df[time_column])

    response = func_generate_forecast(df, time_column, col_target, forecast_horizon_time)
    ```
    """

    try:
        json_df = body.df
        df = pd.DataFrame(json_df)
        time_column = body.time_column
        col_target = body.col_target
        forecast_horizon_time = body.forecast_horizon_time

        response = all_available_forecast(df, time_column, col_target, forecast_horizon_time)
        return response

    except Exception as ApplicationError:
        logger.error(ApplicationError.__repr__())
        raise HTTPException(
            status_code=400,
            detail="Unknown Error",
            headers={"X-Error": f"{ApplicationError.__repr__()}"},
        )


@app.get("/")
def read_root():
    return {"message": "Welcome to the indicators System API"}



if __name__ == "__main__":
    port = 7078
    print(f'🚀 Документация http://0.0.0.0:{port}{docs_url}')
    uvicorn.run("server:app", host="0.0.0.0", port=port, workers=10, log_level="debug")

