# tests/conftest.py
import pytest
import pandas as pd


@pytest.fixture
def core_indicators_params():
    return 2, \
        pd.DataFrame([{'term': 'high_time_resolution', 'count': 1, 'year': 2019},
                      {'term': 'recent_demonstration', 'count': 1, 'year': 2019},
                      {'term': 'antenna_electric_field', 'count': 1, 'year': 2019},
                      {'term': 'real_time', 'count': 1, 'year': 2019},
                      {'term': 'objective', 'count': 1, 'year': 2017},
                      {'term': 'radio_astronomy', 'count': 1, 'year': 2019},
                      {'term': 'new_capability', 'count': 1, 'year': 2019},
                      {'term': 'abstract', 'count': 1, 'year': 2019},
                      {'term': 'real_time_direct_imaging_radio_interferometry_correlator', 'count': 1, 'year': 2019},
                      {'term': 'direct_wide_field_radio_imaging', 'count': 1, 'year': 2019}]), \
        pd.DataFrame([{'term': 'recent_demonstration', 'count': 3, 'year': 2023},
                      {'term': 'real_time', 'count': 823, 'year': 2023},
                      {'term': 'real_time', 'count': 8, 'year': 2024},
                      {'term': 'radio_astronomy', 'count': 22, 'year': 2023},
                      {'term': 'objective', 'count': 12990, 'year': 2023},
                      {'term': 'objective', 'count': 91, 'year': 2024},
                      {'term': 'new_capability', 'count': 30, 'year': 2023},
                      {'term': 'high_time_resolution', 'count': 8, 'year': 2023},
                      {'term': 'abstract', 'count': 5147, 'year': 2023},
                      {'term': 'abstract', 'count': 49, 'year': 2024}]), \
        pd.DataFrame([{'year': 2023, 'count': 1270766}, {'year': 2024, 'count': 18216}])


@pytest.fixture
def core_indicators_expected_result():
    return pd.DataFrame([{'term': 'objective', 'freq': 13081.0, 'rel_freq': 0.010148318595604904,
                          'spec': 1.6925759113616858, 'aagr': -0.5112972192724174},
                         {'term': 'abstract', 'freq': 5196.0, 'rel_freq': 0.004031088098980435,
                          'spec': 2.093547714621825, 'aagr': -0.33586788642287424},
                         {'term': 'real_time', 'freq': 831.0, 'rel_freq': 0.0006446948056683491,
                          'spec': 2.889615833238704, 'aagr': -0.3218859843615509},
                         {'term': 'new_capability', 'freq': 30.0, 'rel_freq': 2.3274180710048706e-05,
                          'spec': 4.332095602303152, 'aagr': -1.0},
                         {'term': 'radio_astronomy', 'freq': 22.0, 'rel_freq': 1.7067732520702385e-05,
                          'spec': 4.466794176200609, 'aagr': -1.0},
                         {'term': 'high_time_resolution', 'freq': 8.0, 'rel_freq': 6.206448189346321e-06,
                          'spec': 4.906126870030871, 'aagr': -1.0},
                         {'term': 'recent_demonstration', 'freq': 3.0, 'rel_freq': 2.3274180710048704e-06,
                          'spec': 5.332095602303152, 'aagr': -1.0}])
