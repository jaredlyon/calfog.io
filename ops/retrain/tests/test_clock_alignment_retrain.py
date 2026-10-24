"""Offline fixed-clock and ordinal-label regressions for operational loaders."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from ops.retrain import run_retrain as rr
from ops.shadow import build_serving as sb


@pytest.mark.parametrize('day', ['2024-01-15', '2024-07-15'])
def test_join_matches_physical_instant(monkeypatch, tmp_path, day):
    utc = pd.Timestamp(day + ' 12:00')
    times = pd.date_range(day, periods=24, freq='h')
    weather = pd.DataFrame({'time': times, **{c: np.arange(24) for c in rr.APPROVED}})
    monkeypatch.setattr(rr, 'DATA', tmp_path)
    monkeypatch.setattr(rr, 'OUT', tmp_path)

    class Response:
        status_code = 200
        url = 'https://example.invalid/test'
        content = b'synthetic'
        text = f'station,valid,vsby\nBFL,{utc},0.5\n'

        def json(self):
            return {'hourly': {c: weather[c].astype(str).tolist() if c == 'time'
                               else weather[c].tolist() for c in weather}}

    monkeypatch.setattr(rr, 'get', lambda *a: Response())
    w, labels, _ = rr.fetch_proxy_and_labels('location_10', rr.SITES['location_10'], [])
    joined = labels.merge(w[['time', 'temperature_2m']], on='time')
    assert len(joined) == 1
    assert joined.temperature_2m.tolist() == [4]
    assert joined.time.iloc[0] + pd.Timedelta(hours=8) == utc


def test_proxy_rejects_old_clock_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(rr, 'DATA', tmp_path)
    monkeypatch.setattr(rr, 'OUT', tmp_path)
    site = 'location_10'
    stem = 'historical_forecast_proxy'
    pd.DataFrame({'time': ['2024-07-15T05:00'], **{c: [999.] for c in rr.APPROVED}}).to_csv(
        tmp_path / f'{site}_{stem}.csv', index=False)
    pd.DataFrame({'time': ['2024-07-15T05:00'], 'visibility_meters': [1000.], 'fog': [1]}).to_csv(
        tmp_path / f'{site}_fog_season_iem_metar.csv', index=False)
    calls = []

    class Response:
        status_code = 200
        url = 'https://example.invalid/test'
        content = b'synthetic'
        text = 'station,valid,vsby\nBFL,2024-07-15 12:00,0.5\n'

        def json(self):
            return {'hourly': {'time': ['2024-07-15T04:00'], **{c: [1.] for c in rr.APPROVED}}}

    def get(url, params, *args):
        calls.append((url, params))
        if url == rr.HF_URL:
            assert params['timezone'] == 'Etc/GMT+8'
        return Response()

    monkeypatch.setattr(rr, 'get', get)
    rr.fetch_proxy_and_labels(site, rr.SITES[site], [])
    assert calls
    assert (tmp_path / f'{site}_{stem}_pst.csv').exists()


@pytest.mark.parametrize('has_null', [False, True])
def test_archive_uses_exact_local_columns_and_records_source(monkeypatch, tmp_path, has_null):
    monkeypatch.setattr(rr, 'ROOT', tmp_path)
    monkeypatch.setattr(rr, 'DATA', tmp_path / 'stale-cache')
    rr.DATA.mkdir()
    (rr.DATA / 'location_10_era5_archive_pst.csv').write_text('obsolete cache')
    source = tmp_path / 'datasets' / 'weather' / 'location_10_weather_appended.csv'
    source.parent.mkdir(parents=True)
    frame = pd.DataFrame({
        'time': pd.date_range('2025-08-27 22:00', periods=3, freq='h'),
        **{column: [1., 2., 3.] for column in rr.APPROVED},
        'unapproved': [np.nan, np.nan, np.nan],
    })
    if has_null:
        frame.loc[0, rr.APPROVED[0]] = np.nan
    frame.to_csv(source, index=False)

    def no_network(*args, **kwargs):
        raise AssertionError('archive loader must use local data')

    monkeypatch.setattr(rr, 'get', no_network)
    manifest = []
    if has_null:
        with pytest.raises(AssertionError, match='archive nulls'):
            rr.fetch_archive('location_10', rr.SITES['location_10'], manifest)
        return
    weather = rr.fetch_archive('location_10', rr.SITES['location_10'], manifest)
    assert list(weather.columns) == ['time'] + rr.APPROVED + rr.CALENDAR
    assert weather.time.tolist() == frame.time.iloc[:2].tolist()
    assert weather[rr.APPROVED].iloc[-1].tolist() == [2.] * len(rr.APPROVED)
    assert manifest == [{
        'kind': 'local_archive_source', 'site': 'location_10',
        'path': 'datasets/weather/location_10_weather_appended.csv',
        'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'rows': 2,
        'clock': 'fixed UTC-8 (Etc/GMT+8)',
    }]


def test_shadow_sources_use_shared_hourly_categories(monkeypatch):
    weather=pd.DataFrame({'time':pd.date_range('2024-01-01',periods=72,freq='h')})
    vis=pd.DataFrame({'time':pd.date_range('2024-01-02',periods=10,freq='h'),
                      'category':[2]+[17]*9,'fog':[True]+[False]*9,
                      'cat_lo':[2]+[17]*9,'cat_hi':[2]+[17]*9})
    seen=[]
    monkeypatch.setattr(sb,'load_sources',lambda site:(seen.append(site) or weather,vis))
    _,now,lead=sb.sources('location_10')
    assert seen==['location_10']
    assert now.category.tolist()==[2]+[17]*9
    assert lead.category.tolist()==[2]
    assert lead.fog.tolist()==[True]


def test_iem_proxy_loader_keeps_any_fog(monkeypatch, tmp_path):
    monkeypatch.setattr(rr, 'DATA', tmp_path)
    monkeypatch.setattr(rr, 'OUT', tmp_path)
    pd.DataFrame({'time': ['2025-12-01 00:00'], 'visibility_meters': [2. * 1609.344], 'fog': [0]}).to_csv(
        tmp_path / 'location_10_fog_season_iem_metar_pst.csv', index=False)

    class Response:
        status_code = 200
        url = 'https://example.invalid'
        content = b'synthetic'
        text = 'valid,vsby\n2025-12-01 08:10,0.5\n2025-12-01 08:59,2.0\n'

        def json(self):
            return {'hourly': {'time': ['2025-12-01T00:00'], **{c: [1.] for c in rr.APPROVED}}}

    monkeypatch.setattr(rr, 'get', lambda *a: Response())
    _, labels, _ = rr.fetch_proxy_and_labels('location_10', rr.SITES['location_10'], [])
    assert labels.fog.tolist() == [1]
    assert labels.category.tolist() == [2]
