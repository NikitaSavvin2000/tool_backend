# src/services/normalization_service.py

import pandas as pd
from src.backend.normalization import Time2Vec
from src.config import logger

def run_normalization(df, col_time, col_target):
    try:
        df[col_target] = df[col_target].replace("None", None).apply(lambda x: float(x) if x is not None else None)
        df[col_time] = pd.to_datetime(df[col_time], errors='coerce')
        t2v = Time2Vec(col_time=col_time, col_target=col_target)
        df_all_data_norm, min_val, max_val = t2v.vectorization(df)
        return {
            "df_all_data_norm": df_all_data_norm.to_dict(),
            "min_val": min_val,
            "max_val": max_val
        }
    except Exception as e:
        logger.error(f"Ошибка в run_normalization: {e}")
        raise


def run_reverse_normalization(df, col_time, col_target, min_val, max_val):
    try:
        t2v = Time2Vec(col_time=col_time, col_target=col_target)
        df_reversed = t2v.reverse_vectorization(df, min_val=min_val, max_val=max_val)

        # TODO:   Когда я писал эту схему видно был не в себе. Такой формат json - это абсурд,
        #         но на этой схеме уже работает связка с другими микросервисами.
        #         Позже нужно сделать нормальную схему и поправить ее где она используется.
        #         Ниже конвертация в нормальный формат.

        # reversed_json = df_reversed.to_dict(orient="records")

        return {
            "df_all_data_reverse_norm": df_reversed.to_dict()
        }
    except Exception as e:
        logger.error(f"Ошибка в run_reverse_normalization: {e}")
        raise