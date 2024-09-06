import pandas as pd
import pytest

from indicators.core_indicators import core_indicators
from indicators.growth_rate import growth_rate


@pytest.mark.unit
def test_core_indicators(core_indicators_params, core_indicators_expected_result):
    doc_count, doc_term_counts_df, term_counts_df, corpus_counts_df = core_indicators_params
    result = core_indicators(doc_count=doc_count,
                             doc_term_counts_df=doc_term_counts_df,
                             term_counts_df=term_counts_df,
                             corpus_counts_df=corpus_counts_df)
    assert result.sort_index(inplace=True) == core_indicators_expected_result.sort_index(inplace=True)


@pytest.mark.parametrize("x,year,expected_result", [
    ({"rel_2021": 1, "rel_2022": 2}, 2022, 1),
    ({"rel_2021": 1, "rel_2023": 2}, 2022, -1),
    ({"rel_2021": 1, "rel_2022": 0}, 2022, -1),
    ({"rel_2021": 0, "rel_2022": 0}, 2022, 0),
    ({"rel_2021": 0, "rel_2020": 0}, 2022, 0),
    ({"rel_2020": 0, "rel_2022": 0}, 2022, 0),
    ({"rel_2020": 1, "rel_2022": 2}, 2022, 1)
])
@pytest.mark.unit
def test_growth_rate(x, year, expected_result):
    rel_freq_series = pd.Series(x)
    result = growth_rate(rel_freq_series=rel_freq_series, year=year)
    assert result == expected_result
