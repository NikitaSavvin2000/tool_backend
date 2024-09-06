import pandas as pd


def growth_rate(rel_freq_series: pd.Series, year: int):
    year_value = rel_freq_series.get(f"rel_{year}", 0)
    previous_year_value = rel_freq_series.get(f"rel_{year - 1}", 0)
    if previous_year_value == 0 and year_value == 0:
        return 0
    elif previous_year_value == 0:
        return 1
    else:
        return year_value / previous_year_value - 1

'''
def growth_rate(x, year):
    if x[f"rel_{year - 1}"] == 0:
        if x[f"rel_{year}"] == 0:
            return 0
        return 1
    return x[f"rel_{year}"] / x[f"rel_{year - 1}"] - 1
'''
