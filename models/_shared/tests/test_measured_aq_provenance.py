"""Measured sample provenance follows completion-time alignment, including legacy inputs."""

import numpy as np
import pandas as pd
import pytest

from models._shared import data


@pytest.mark.parametrize("start", ["2024-01-31 17:00", "2024-07-31 17:00", "2024-12-31 23:00"])
@pytest.mark.parametrize("source", ["current", "legacy", "absent"])
def test_loader_aligns_nullable_pm25_provenance_without_mutating_raw_inputs(tmp_path, monkeypatch, start, source):
    datasets = tmp_path / "datasets"
    for directory in ("weather", "aqi", "visibility", "aq_measured"):
        (datasets / directory).mkdir(parents=True)
    times = pd.date_range(start, periods=4, freq="h")
    weather = pd.DataFrame({"time": times, **{col: np.arange(4, dtype=float) for col in data.WEATHER_COLS}})
    weather.to_csv(datasets / "weather/site_weather.csv", index=False)
    pd.DataFrame({"time": times, **{col: np.arange(4, dtype=float) for col in data.AQI_COLS}}).to_csv(
        datasets / "aqi/site_aqi.csv", index=False)
    pd.DataFrame({"time": times, "cat_lo": 17, "cat_hi": 17, "category": 17, "fog": False}).to_csv(
        datasets / "visibility/site_visibility_hourly.csv", index=False)
    measured = datasets / "aq_measured/site_aq_measured.csv"
    if source != "absent":
        aq = pd.DataFrame({"time": times[:2], "nox_ppb": [11., 999.], "nox_monitor": ["first", "second"],
                           "pm25_ugm3": [5., np.nan], "pm25_monitor": ["pm-first", None]})
        if source == "current":
            for column, value in (("pm25_parameter", 88101), ("pm25_poc", 2), ("pm25_method", 170)):
                aq[column] = pd.array([value, pd.NA], dtype="Int64")
        aq.to_csv(measured, index=False)
        original = measured.read_bytes()
    monkeypatch.setattr(data, "ROOT", tmp_path)
    loaded, _ = data.load_sources("site")
    for column in data.WEATHER_COLS + data.AQI_COLS:
        np.testing.assert_array_equal(loaded[column], np.arange(4, dtype=float))
    for column in ("pm25_parameter", "pm25_poc", "pm25_method"):
        assert str(loaded[column].dtype) == "Int64"
        if source == "current":
            assert loaded[column].notna().tolist() == [False, True, False, False]
        else:
            assert loaded[column].isna().all()
    if source == "absent":
        assert loaded.nox_ppb.isna().all() and loaded.pm25_ugm3.isna().all()
    else:
        assert measured.read_bytes() == original
        assert loaded.nox_ppb.iloc[1:3].tolist() == [11., 999.]
        assert loaded.nox_monitor.iloc[1:3].tolist() == ["first", "second"]
        assert loaded.nox_mon_first.tolist() == [0., 1., 0., 0.]
        assert loaded.nox_mon_second.tolist() == [0., 0., 1., 0.]
        assert loaded.pm25_ugm3.iloc[1] == 5.
        if source == "current":
            assert loaded.pm25_parameter.iloc[1] == 88101
            assert loaded.pm25_poc.iloc[1] == 2
            assert loaded.pm25_method.iloc[1] == 170
