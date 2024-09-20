import json
from typing import Optional

from pydantic import BaseModel
from typing import List, Dict


class StatisticsRequest(BaseModel):
    doc_ids: list[str]
    years: list[int]


class MetaRequest(BaseModel):
    doc_ids: list[str]


class ConceptsRequest(BaseModel):
    doc_ids: list[str]
    level: Optional[int]
    rule: Optional[str]

class AnalyticsDFsRequest(BaseModel):
    dfs_json_list: list[dict]

class NormalizationRequest(BaseModel):
    col_time: str
    col_target: str
    json_list_df: List[Dict]
