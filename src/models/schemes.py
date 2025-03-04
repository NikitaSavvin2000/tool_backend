import json
from typing import Optional

from pydantic import BaseModel
from typing import List, Dict


class AnalyticsDFsRequest(BaseModel):
    dfs_json_list: list[dict]

class NormalizationRequest(BaseModel):
    col_time: str
    col_target: str
    json_list_df: List[Dict]

class ForecastRequest(BaseModel):
    col_target: str
    evaluation_index: int
    last_know_index: int
    epochs: int
    lag: int
    activation: str
    optimizer: str
    dropout_count: float
    model_architecture_params: List[Dict]
    json_list_df_all_data_norm: List[Dict]


class ReverseNormalizationRequest(BaseModel):
    col_time: str
    col_target: str
    json_list_norm_df: List[Dict]
    min_val: float
    max_val: float

class MenrixAllRequest(BaseModel):
    col_time: str
    col_target: str
    json_list_df_reverse_evaluation: List[Dict]
    json_list_df_reverse_comparative: List[Dict]


class ForecastRequestXGBoost(BaseModel):
    col_target: str
    evaluation_index: int
    last_know_index: int
    lag: int
    type: str
    model_architecture_params: List[Dict]
    json_list_df_all_data_norm: List[Dict]
    norm_values: str


class ForecastRequestLSTM(BaseModel):
    col_target: str
    evaluation_index: int
    last_know_index: int
    epochs: int
    lag: int
    activation: str
    optimizer: str
    dropout_count: float
    model_architecture_params: List[Dict]
    json_list_df_all_data_norm: List[Dict]
    norm_values: str
    type: str


class ForecastRequestNeuralNetworks(BaseModel):
    col_target: str
    evaluation_index: int
    last_know_index: int
    model_architecture_params: List[Dict]
    json_list_df_all_data_norm: List[Dict]
    norm_values: str
    type: str


class UpdateColRequest(BaseModel):
    col_for_train: List[str]