from typing import Optional

from pydantic import BaseModel


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