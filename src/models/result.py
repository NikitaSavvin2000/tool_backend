# src/models/result.py
from pydantic import BaseModel


class AnalyticsDFs(BaseModel):
    dfs_metrix: list[dict]
