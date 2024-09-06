from typing import Annotated, List

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

from src.config import logger, public_or_local
from src.models.statisticsrequest import  AnalyticsDFsRequest

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


@app.get("/")
def read_root():
    return {"message": "Welcome to the indicators System API"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7070)
    print('Start backend url - http://0.0.0.0:7070/backend/v1/')
