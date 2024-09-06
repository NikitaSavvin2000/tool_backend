from typing import Optional

from pydantic import BaseModel


class AnalyticsDFs(BaseModel):
    dfs_metrix: list[dict]
