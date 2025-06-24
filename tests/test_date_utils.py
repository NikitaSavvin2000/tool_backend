# tests/test_date_utils.py

import pytest
from src.utils.date_utils import standardize_datetime

def test_standardize_datetime():
    assert standardize_datetime("2023") == "2023-01-01 00:00:00 UTC"
    assert standardize_datetime("2023-09") == "2023-09-01 00:00:00 UTC"
    assert standardize_datetime("2023-09-15") == "2023-09-15 00:00:00 UTC"
    assert standardize_datetime("15-09-2023") == "2023-09-15 00:00:00 UTC"
    assert standardize_datetime("2023-09-15 2:30 PM") == "2023-09-15 14:30:00 UTC"
    assert standardize_datetime("1694797200") == "2023-09-15 17:00:00 UTC" #Unix timestamp интерпретируется в UTC