import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from itertools import product
from indicators.growth_rate import growth_rate


def sort_df(data: pd.DataFrame):
    """
    Входные данные:
        data.columns == ["term", "query_count", "spec", "aagr"]

    Логика:
        Сортировка терминов с учетом трёх индикаторов: query_count, aagr, spec.
        Нормируем query_count относительно медианного значения, aagr и spec - относительно нуля.
    """
    cols = data.columns

    median_freq, min_freq, max_freq = np.median(data["query_count"]), min(data["query_count"]), max(data["query_count"])
    mask = data["query_count"] < median_freq
    data["norm_freq_med"] = [0] * len(data)
    data.loc[mask, "norm_freq_med"] = (data.loc[mask, "query_count"] - min_freq) / (median_freq - min_freq) - 1
    data.loc[~mask, "norm_freq_med"] = (data.loc[~mask, "query_count"] - median_freq) / (max_freq - median_freq)

    for indicator in ("aagr", "spec"):
        mn, mx = min(data[indicator]), max(data[indicator])
        mask = data[indicator] < 0
        data[f"norm_{indicator}_zero"] = [0] * len(data)
        data.loc[mask, f"norm_{indicator}_zero"] = (data.loc[mask, indicator] - mn) / (-mn) - 1
        data.loc[~mask, f"norm_{indicator}_zero"] = data.loc[~mask, indicator] / mx

    data["med_freq_zero_aagr_zero_spec"] = data["norm_freq_med"] + data["norm_aagr_zero"] + data["norm_spec_zero"]
    return data.sort_values(by="med_freq_zero_aagr_zero_spec", ascending=False)[cols]


def core_indicators(
        doc_count: int,
        doc_term_counts_df: pd.DataFrame,
        term_counts_df: pd.DataFrame,
        corpus_counts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Входные данные:
        term_counts_df.columns == ["term", "count", "year"]
        corpus_counts_df.columns == ["year", "count"]

    Логика:
        Получаем список уникальных лет из таблицы терминов
        Генерируем период через определение минимальных и максимальных лет в списке уникальных лет
        Вычисляем общее число документов внутри корпуса
        Генерируем таблицу со всеми комбинациями термин-год
        Заполняем пропущенные года в таблице статистики корпуса
        Мёрджим таблицу со встречаемостью терминов и таблицу с комбинациями
        Вычисляем count, rel_freq, query_count, query_freq, spec, aagr
    """
    corpus_total_value = sum(corpus_counts_df["count"])

    corpus_counts_df = pd.DataFrame(
        [corpus_counts_df["count"].values],
        columns=corpus_counts_df["year"].values,
        index=["corpus"]
    )

    period = list(range(term_counts_df["year"].min(), term_counts_df["year"].max() + 1))

    for year in period:
        if year in corpus_counts_df:
            continue
        corpus_counts_df[year] = 0

    corpus_counts_df = corpus_counts_df.reindex(sorted(corpus_counts_df.columns), axis=1)

    all_years_for_every_term = pd.DataFrame(
        list(product(term_counts_df["term"].unique(), period)),
        columns=["term", "year"]
    )

    terms_with_all_years = pd.merge(all_years_for_every_term, term_counts_df, on=["term", "year"], how="left").fillna(0)

    terms_with_all_years = terms_with_all_years.drop_duplicates(subset=['term', 'year'], keep='first')


    terms_with_all_years.to_csv('/Users/nikitasavvin/Desktop/HSE_work/microservice_indicators/src/tests/terms_with_all_years.csv')

    terms_with_all_years = terms_with_all_years.pivot(index="term", columns="year", values="count").reset_index()


    terms_with_all_years["count"] = terms_with_all_years[period].sum(axis=1)
    for year in period:

        terms_with_all_years[f"rel_{year}"] = terms_with_all_years[year] / corpus_counts_df.at["corpus", year]

    terms_with_all_years.fillna(0, inplace=True)

    terms_with_all_years.fillna(0, inplace=True)

    terms_with_all_years.replace([np.inf, -np.inf], 0, inplace=True)

    terms_with_all_years["rel_freq"] = terms_with_all_years["count"] / corpus_total_value

    counts = doc_term_counts_df.groupby(by="term")["count"].sum().reset_index(name="query_count")


    counts["query_freq"] = counts["query_count"] / doc_count

    terms_with_all_years = pd.merge(terms_with_all_years, counts[["term", "query_count", "query_freq"]], how="left", on="term")


    terms_with_all_years["spec"] = np.log10(terms_with_all_years["query_freq"] / terms_with_all_years["rel_freq"])
    terms_with_all_years = terms_with_all_years.sort_values(by=["spec"], ascending=False).head(3000)

    terms_with_all_years["aagr"] = [
        sum([growth_rate(x, year) for year in period[1:]]) / (len(period) - 1)
        for _, x in terms_with_all_years.iterrows()
    ]
    cols = ["term", "count", "rel_freq", "query_count", "query_freq", "spec", "aagr"]


    return sort_df(terms_with_all_years[cols]).head(150).reset_index(drop=True)
    #return sort_df(sort_df(terms_with_all_years[cols]).head(1000)).head(150).reset_index(drop=True)
