"""Regression checks for the actual clock-consuming loader (no network/GPU)."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from models._shared import data
from models._shared.visibility import hourly_labels


@pytest.mark.parametrize('day', ['2024-01-15', '2024-07-15'])
def test_join_matches_physical_instant(monkeypatch, tmp_path, day):
    utc = pd.Timestamp(day + ' 12:00')
    times = pd.date_range(day, periods=24, freq='h')
    weather = pd.DataFrame({'time': times, **{c: np.arange(24) for c in data.WEATHER_COLS}})
    aqi = pd.DataFrame({'time': times, **{c: np.zeros(24) for c in data.AQI_COLS}})
    hourly = hourly_labels(pd.DataFrame({'time_utc': pd.to_datetime([utc], utc=True),
                                         'cat_lo': [2], 'cat_hi': [2]}))
    folder = tmp_path / 'datasets' / 'weather'
    folder.mkdir(parents=True)
    (folder / 'location_10_weather_appended.csv').touch()
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    original_read = pd.read_csv

    def read_csv(path, *args, **kwargs):
        if not isinstance(path, (str, Path)):
            return original_read(path, *args, **kwargs)
        name = Path(path).name
        if 'weather' in name:
            return weather.copy()
        if 'visibility' in name:
            assert name == 'location_10_visibility_hourly.csv'
            return hourly.copy()
        if 'aqi' in name:
            return aqi.copy()
        raise AssertionError(f'unexpected cache reuse: {path}')

    monkeypatch.setattr(pd, 'read_csv', read_csv)
    w, labels = data.load_sources('location_10')
    joined = labels.merge(w[['time', 'temperature_2m']], on='time')
    assert len(joined) == 1
    assert joined.temperature_2m.tolist() == [4]
    assert joined.time.iloc[0] + pd.Timedelta(hours=8) == utc
