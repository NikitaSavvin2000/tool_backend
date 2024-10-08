import json
from typing import Annotated, List

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

from src.backend.forecast import forecast
from src.backend.metrix import metrix_all
from src.backend.normalization import TimeNormalization
from src.config import logger, public_or_local
from src.models.statisticsrequest import AnalyticsDFsRequest, NormalizationRequest, ForecastRequest, \
    ReverseNormalizationRequest, MenrixAllRequest

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

example_dfs_1 = {
    "09:00:00": 10,
    "09:01:00": 12,
    "09:02:00": 15,
    "09:03:00": 14,
    "09:04:00": 13
        }

example_dfs_2 = {
    "10:00:00": 20,
    "10:01:00": 22,
    "10:02:00": 21,
    "10:03:00": 25,
    "10:04:00": 24
}

example_dfs_3 = {
    "11:00:00": 30,
    "11:01:00": 29,
    "11:02:00": 31,
    "11:03:00": 33,
    "11:04:00": 32
}

example_not_norm_data = {
    "col_time": "time",
    "col_target": "load_consumption",
    "json_list_df": [
        {"load_consumption":56797.7,"time":"2023-08-19 17:53:00"},
        {"load_consumption":58040.5,"time":"2023-08-19 17:58:00"}]
}

example_reverse_norm_data = {
    "col_time": "time",
    "col_target": "load_consumption",
    "json_list_norm_df": [{"load_consumption":0.8535390496253967,"week":0.9215686274509803,"day_of_week":0.6666666666666666,"hour":0.782608695652174,"minute":0.6610169491525424,"hour_sin":-1,"hour_cos":-1.8369701987210297e-16,"day_of_week_sin":-0.433883739117558,"day_of_week_cos":-0.9009688679024191,"week_sin":-0.4647231720437684,"week_cos":0.88545602565321,"is_holiday":0,"second":0,"year":0.968503937007874},
                     {"load_consumption":0.9265455603599548,"week":0.9215686274509803,"day_of_week":0.6666666666666666,"hour":0.782608695652174,"minute":0.7457627118644068,"hour_sin":-1,"hour_cos":-1.8369701987210297e-16,"day_of_week_sin":-0.433883739117558,"day_of_week_cos":-0.9009688679024191,"week_sin":-0.4647231720437684,"week_cos":0.88545602565321,"is_holiday":0,"second":0,"year":0.968503937007874},
                     {"load_consumption":1.0235118865966797,"week":0.9215686274509803,"day_of_week":0.6666666666666666,"hour":0.782608695652174,"minute":0.8305084745762712,"hour_sin":-1,"hour_cos":-1.8369701987210297e-16,"day_of_week_sin":-0.433883739117558,"day_of_week_cos":-0.9009688679024191,"week_sin":-0.4647231720437684,"week_cos":0.88545602565321,"is_holiday":0,"second":0,"year":0.968503937007874},
                     {"load_consumption":1.1619222164154053,"week":0.9215686274509803,"day_of_week":0.6666666666666666,"hour":0.782608695652174,"minute":0.9152542372881356,"hour_sin":-1,"hour_cos":-1.8369701987210297e-16,"day_of_week_sin":-0.433883739117558,"day_of_week_cos":-0.9009688679024191,"week_sin":-0.4647231720437684,"week_cos":0.88545602565321,"is_holiday":0,"second":0,"year":0.968503937007874},
                     {"load_consumption":1.3761029243469238,"week":0.9215686274509803,"day_of_week":0.6666666666666666,"hour":0.782608695652174,"minute":1,"hour_sin":-1,"hour_cos":-1.8369701987210297e-16,"day_of_week_sin":-0.433883739117558,"day_of_week_cos":-0.9009688679024191,"week_sin":-0.4647231720437684,"week_cos":0.88545602565321,"is_holiday":0,"second":0,"year":0.968503937007874}
                     ],
    "min_val": 0.1,
    "max_val": 200000.5,
}


example_forecast_point = {
    "col_target": "load_consumption",
    "evaluation_index": 7,
    "last_know_index": 9,
    "epochs": 5,
    "lag": 1,
    "activation": "relu",
    "optimizer": "adam",
    "dropout_count": 0.01,
    "model_architecture_params": [
        {"layer":1,"type":"Bi-LSTM","neurons":2},
        {"layer":2,"type":"Bi-LSTM","neurons":4},
        {"layer":3,"type":"Bi-LSTM","neurons":8}],

    "json_list_df_all_data_norm": [{"load_consumption":0.6800409376,"year":0.984,"week":0.6274509804,"day_of_week":0.8333333333,"hour":0.7391304348,"minute":0.8983050847,"second":0,"hour_sin":-0.9659258263,"hour_cos":-0.2588190451,"day_of_week_sin":-0.9749279122,"day_of_week_cos":-0.222520934,"week_sin":-0.7485107482,"week_cos":-0.6631226582,"is_holiday":0},
                                   {"load_consumption":0.6952882419,"year":0.984,"week":0.6274509804,"day_of_week":0.8333333333,"hour":0.7391304348,"minute":0.9830508475,"second":0,"hour_sin":-0.9659258263,"hour_cos":-0.2588190451,"day_of_week_sin":-0.9749279122,"day_of_week_cos":-0.222520934,"week_sin":-0.7485107482,"week_cos":-0.6631226582,"is_holiday":0},
                                   {"load_consumption":0.7110373283,"year":0.984,"week":0.6274509804,"day_of_week":0.8333333333,"hour":0.7826086957,"minute":0.0508474576,"second":0,"hour_sin":-1,"hour_cos":-1.836970199e-16,"day_of_week_sin":-0.9749279122,"day_of_week_cos":-0.222520934,"week_sin":-0.7485107482,"week_cos":-0.6631226582,"is_holiday":0},
                                   {"load_consumption":0.6988007163,"year":0.984,"week":0.6274509804,"day_of_week":0.8333333333,"hour":0.7826086957,"minute":0.1355932203,"second":0,"hour_sin":-1,"hour_cos":-1.836970199e-16,"day_of_week_sin":-0.9749279122,"day_of_week_cos":-0.222520934,"week_sin":-0.7485107482,"week_cos":-0.6631226582,"is_holiday":0},
                                   {"load_consumption":0.6925315077,"year":0.984,"week":0.6274509804,"day_of_week":0.8333333333,"hour":0.7826086957,"minute":0.2203389831,"second":0,"hour_sin":-1,"hour_cos":-1.836970199e-16,"day_of_week_sin":-0.9749279122,"day_of_week_cos":-0.222520934,"week_sin":-0.7485107482,"week_cos":-0.6631226582,"is_holiday":0},
                                   {"load_consumption":0.6344474735,"year":0.984,"week":0.6274509804,"day_of_week":0.8333333333,"hour":0.7826086957,"minute":0.3050847458,"second":0,"hour_sin":-1,"hour_cos":-1.836970199e-16,"day_of_week_sin":-0.9749279122,"day_of_week_cos":-0.222520934,"week_sin":-0.7485107482,"week_cos":-0.6631226582,"is_holiday":0},
                                   {"load_consumption":0.5210668107,"year":0.984,"week":0.6274509804,"day_of_week":0.8333333333,"hour":0.7826086957,"minute":0.3898305085,"second":0,"hour_sin":-1,"hour_cos":-1.836970199e-16,"day_of_week_sin":-0.9749279122,"day_of_week_cos":-0.222520934,"week_sin":-0.7485107482,"week_cos":-0.6631226582,"is_holiday":0},
                                   {"load_consumption":0.5220188471,"year":0.984,"week":0.6274509804,"day_of_week":0.8333333333,"hour":0.7826086957,"minute":0.4745762712,"second":0,"hour_sin":-1,"hour_cos":-1.836970199e-16,"day_of_week_sin":-0.9749279122,"day_of_week_cos":-0.222520934,"week_sin":-0.7485107482,"week_cos":-0.6631226582,"is_holiday":0},
                                   {"load_consumption":0.5324090483,"year":0.984,"week":0.6274509804,"day_of_week":0.8333333333,"hour":0.7826086957,"minute":0.5593220339,"second":0,"hour_sin":-1,"hour_cos":-1.836970199e-16,"day_of_week_sin":-0.9749279122,"day_of_week_cos":-0.222520934,"week_sin":-0.7485107482,"week_cos":-0.6631226582,"is_holiday":0},
                                   {"load_consumption":0.5249571553,"year":0.984,"week":0.6274509804,"day_of_week":0.8333333333,"hour":0.7826086957,"minute":0.6440677966,"second":0,"hour_sin":-1,"hour_cos":-1.836970199e-16,"day_of_week_sin":-0.9749279122,"day_of_week_cos":-0.222520934,"week_sin":-0.7485107482,"week_cos":-0.6631226582,"is_holiday":0}]
}


example_metrix_all = {
    "col_time": "time",
    "col_target": "load_consumption",
    "json_list_df_reverse_evaluation": [
        {"load_consumption":18256.488175171136,"time":"2023-12-17 00:07:00"},
        {"load_consumption":18557.543205893933,"time":"2023-12-17 00:12:00"},
        {"load_consumption":18732.26521594262,"time":"2023-12-17 00:17:00"},
        {"load_consumption":18786.505777776718,"time":"2023-12-17 00:22:00"},
        {"load_consumption":18778.05954055041,"time":"2023-12-17 00:27:00"},
        {"load_consumption":18707.63702002251,"time":"2023-12-17 00:32:00"},
        {"load_consumption":18585.811475117684,"time":"2023-12-17 00:37:00"},
        {"load_consumption":18423.762592223524,"time":"2023-12-17 00:42:00"},
        {"load_consumption":18245.625279249252,"time":"2023-12-17 00:47:00"},
        {"load_consumption":18069.144327852846,"time":"2023-12-17 00:52:00"}
    ],
    "json_list_df_reverse_comparative": [
        {"load_consumption":17574.4,"time":"2023-12-17 00:02:00"},
        {"load_consumption":17930,"time":"2023-12-17 00:07:00"},
        {"load_consumption":16916.699999999997,"time":"2023-12-17 00:12:00"},
        {"load_consumption":18008.1,"time":"2023-12-17 00:17:00"},
        {"load_consumption":17515.1,"time":"2023-12-17 00:22:00"},
        {"load_consumption":19243.9,"time":"2023-12-17 00:27:00"},
        {"load_consumption":15745.6,"time":"2023-12-17 00:32:00"},
        {"load_consumption":16434.9,"time":"2023-12-17 00:37:00"},
        {"load_consumption":16154.700000000003,"time":"2023-12-17 00:42:00"},
        {"load_consumption":15885.500000000002,"time":"2023-12-17 00:47:00"}
    ],
}

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

@app.post("/backend/v1/normalization")
async def get_normalization(body: Annotated[
    NormalizationRequest, Body(
        example={
            "col_time": example_not_norm_data['col_time'],
            "col_target": example_not_norm_data['col_target'],
            "json_list_df": example_not_norm_data['json_list_df'],
        })]):

    try:
        col_time = body.col_time
        col_target = body.col_target
        json_list_df = body.json_list_df

        df = pd.DataFrame(json_list_df)
        df[col_target] = df[col_target].replace("None", None)
        if not df.empty:

            df[col_time] = pd.to_datetime(df[col_time], errors='coerce')
            df[col_time] = df[col_time].apply(lambda x: x.replace(hour=x.hour or 0,
                                                                  minute=x.minute or 0,
                                                                  second=x.second or 0))

            tn = TimeNormalization(col_time, col_target)
            df_all_data_norm, min_val, max_val = tn.df_normalize_with_meta(df)
            df_all_data_norm.to_csv('/Users/nikitasavvin/Desktop/Учеба/tool_backend/experiments/df_all_data_norm.csv')
            response = {
                "df_all_data_norm": df_all_data_norm.to_dict(),
                "min_val": min_val,
                "max_val": max_val
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
        evaluation_index = body.evaluation_index
        last_know_index = body.last_know_index
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
            df_evaluetion, df_true_all_col, loss_list, df_real_predict = forecast(
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
                "loss_list": loss_list
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



@app.post("/backend/v1/reverse_normalization")
async def get_reverse_normalization(body: Annotated[
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
            tn = TimeNormalization(col_time, col_target)
            df_all_data_reverse_norm = tn.df_denormalize_with_meta(df, min_val, max_val)
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

        df_evaluation  = pd.DataFrame(json_list_df_reverse_evaluation)
        df_comparative  = pd.DataFrame(json_list_df_reverse_comparative)

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

@app.get("/")
def read_root():
    return {"message": "Welcome to the indicators System API"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7070)
    print('Start backend url - http://0.0.0.0:7070/backend/v1/')
