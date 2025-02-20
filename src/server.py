import uvicorn
import pandas as pd

from typing import Annotated, List
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

from src.backend.metrix import metrix_all
from src.backend.forecast import forecast
from src.backend.lstm import forecast_LSTM
from src.backend.xgb import forecast_XGBoost
from src.config import logger, public_or_local
from src.processing.processing import to_float
from src.backend.normalization import Time2Vec
from src.backend.new_network import forecast_neural_networks
from src.models.schemes import (
    AnalyticsDFsRequest, NormalizationRequest, ForecastRequest,
    ReverseNormalizationRequest, MenrixAllRequest, ForecastRequestXGBoost,
    ForecastRequestNeuralNetworks
)
from src.examples_fastapi.examples import (
    example_dfs_1, example_dfs_2, example_dfs_3,
    example_not_norm_data, example_reverse_norm_data,
    example_forecast_point, example_forecast_point_XGBoost,
    example_forecast_neural_networks, example_metrix_all,
    example_forecast_prophet
)


if public_or_local == 'LOCAL':
    url = 'http://localhost'
else:
    url = 'http://77.37.136.11'

origins = [
    url
]

app = FastAPI(docs_url="/backend/v1/", openapi_url='/backend/v1/openapi.json')
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        logger.info('I work')


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

        df[col_time] = pd.to_datetime(df[col_time], format='%Y-%m-%d %H:%M:%S', errors='coerce')
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

@app.get("/")
def read_root():
    return {"message": "Welcome to the indicators System API"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7070)
