"""Network-free acquisition, retained-input and pinned-winter regression cases."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_gen as gen
import append_inferred_variables as augmentation
import weather_era5_trend as era5


@pytest.fixture(autouse=True)
def isolated_acquisition(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gen.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(gen, "END_DATE", "2020-01-02")

    def no_network(*args, **kwargs):
        pytest.fail("unmocked network request")

    monkeypatch.setattr(gen.requests, "get", no_network)


def weather(times):
    times = pd.DatetimeIndex(times)
    return pd.DataFrame({"time": times, **{name: np.arange(len(times), dtype=float) + index
                                         for index, name in enumerate(gen.WEATHER_VARS)}})


def payload(variables, times):
    return {
        "latitude": 36.9, "longitude": -120.1, "elevation": 80,
        "timezone": "Etc/GMT+8", "utc_offset_seconds": -28800,
        "hourly_units": {name: "observed-unit" for name in variables},
        "hourly": {"time": [str(value) for value in times],
                   **{name: [float(index)] * len(times) for index, name in enumerate(variables)}},
    }


def response(data, status=200):
    result = requests.Response()
    result.status_code = status
    result.url = "https://example.invalid/resolved?timezone=Etc%2FGMT%2B8"
    result._content = json.dumps(data).encode("utf-8")
    return result


def empty_cams(*args):
    return pd.DataFrame(columns=["time"] + gen.AIR_QUALITY_VARS), {}


def test_response_bytes_and_observed_provenance(monkeypatch):
    data = payload(gen.WEATHER_VARS, ["2020-01-01T00:00"])
    observed = response(data)
    monkeypatch.setattr(gen.requests, "get", lambda *args, **kwargs: observed)
    parsed, record = gen.get_json("https://example.invalid/archive", {"timezone": gen.TIMEZONE})
    assert parsed == data
    assert Path(record["raw_path"]).read_bytes() == observed.content
    assert record["response_sha256"] == hashlib.sha256(observed.content).hexdigest()
    assert record["resolved_url"] == observed.url
    assert record["query"]["models"] is None and record["query"]["domains"] is None
    assert record["retrieval_utc"].endswith("+00:00")
    assert record["units"] == data["hourly_units"]
    assert record["upstream_version"] is None
    assert record["response_hash_scope"] == "HTTP response-content bytes"


@pytest.mark.parametrize("pinned", [False, True])
def test_weather_query_and_return_shape(monkeypatch, pinned):
    variables = era5.WEATHER_COLS if pinned else gen.WEATHER_VARS
    captured = []

    def get(url, params, **kwargs):
        captured.append(params)
        return response(payload(variables, ["2020-01-01T00:00"]))

    monkeypatch.setattr(gen.requests, "get", get)
    kwargs = {"model": "era5", "variables": era5.WEATHER_COLS} if pinned else {}
    frame, record = gen.fetch_weather_data(36, -120, "2020-01-01", "2020-01-01", **kwargs)
    query = captured[0]
    assert query["hourly"] == ",".join(variables)
    assert query["timezone"] == "Etc/GMT+8"
    assert frame.columns.tolist() == ["time"] + variables
    assert frame.time.iloc[0] == pd.Timestamp("2020-01-01")
    if pinned:
        assert query["models"] == "era5"
        assert query["temperature_unit"] == "celsius"
        assert query["wind_speed_unit"] == "kmh"
        assert query["precipitation_unit"] == "mm"
    else:
        assert "models" not in query
        assert record["query"]["models"] is None


def test_cams_empty_and_populated_return_shape(monkeypatch):
    for times in ([], ["2020-01-01T00:00"]):
        monkeypatch.setattr(gen.requests, "get", lambda *args, **kwargs: response(payload(gen.AIR_QUALITY_VARS, times)))
        frame, record = gen.fetch_air_quality_data(36, -120, "2013-01-01", "2020-01-01")
        assert len(frame) == len(times)
        assert frame.columns.tolist() == ["time"] + gen.AIR_QUALITY_VARS
        assert record["query"]["domains"] is None
        assert Path(record["raw_path"]).exists()


def test_appended_only_seeds_raw_offline_and_resumes_last_day(monkeypatch):
    coord = gen.COORDINATES[0]
    directory = Path("datasets/weather")
    directory.mkdir(parents=True)
    retained = weather(pd.date_range("2020-01-01", periods=25, freq="h"))
    augmentation.calculate_inferred_variables(retained).to_csv(directory / f"{coord['name']}_weather_appended.csv", index=False)
    calls = []

    def fetch(lat, lon, start, end):
        calls.append((start, end))
        return retained.copy(), {}

    monkeypatch.setattr(gen, "fetch_weather_data", fetch)
    monkeypatch.setattr(gen, "fetch_air_quality_data", empty_cams)
    assert gen.generate_dataset_for_location(coord)
    raw = pd.read_csv(directory / f"{coord['name']}_weather.csv", parse_dates=["time"])
    pd.testing.assert_frame_equal(raw, retained)
    assert raw.columns.tolist() == ["time"] + gen.WEATHER_VARS
    assert calls == [("2020-01-02", "2020-01-02")]
    entry = json.loads(gen.ACQUISITION_MANIFEST.read_text())["files"][f"datasets/weather/{coord['name']}_weather.csv"]
    assert entry["imported"]["retrieval_utc"] is None
    assert entry["imported"]["query"] is None
    assert entry["csv_sha256"] == gen.file_sha256(directory / f"{coord['name']}_weather.csv")


@pytest.mark.parametrize("defect", ["value", "coverage"])
def test_raw_appended_contradictions_fail_before_requests(monkeypatch, defect):
    coord = gen.COORDINATES[0]
    directory = Path("datasets/weather")
    directory.mkdir(parents=True)
    retained = weather(pd.date_range("2020-01-01", periods=3, freq="h"))
    retained.to_csv(directory / f"{coord['name']}_weather_appended.csv", index=False)
    raw = retained.iloc[1:].copy() if defect == "coverage" else retained.copy()
    if defect == "value":
        raw.loc[0, "temperature_2m"] += 1
    raw.to_csv(directory / f"{coord['name']}_weather.csv", index=False)
    monkeypatch.setattr(gen, "fetch_weather_data", lambda *args: pytest.fail("must fail before requests"))
    with pytest.raises(ValueError, match="raw.*weather"):
        gen.generate_dataset_for_location(coord)


def test_strict_incremental_retention_repeat_and_manifest():
    path = Path("datasets/weather/synthetic.csv")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"time,value\r\n2020-01-01 23:00:00,1.2345678901234567\r\n")
    original = path.read_bytes()
    frame = pd.DataFrame({"time": pd.to_datetime(["2020-01-01 23:00", "2020-01-02 00:00"]), "value": [999, 2.5]})
    fetch = lambda *args: (frame, {"query": {"models": None, "domains": None}})
    assert gen.pull(fetch, path, gen.COORDINATES[0])
    after = path.read_bytes()
    assert after.startswith(original)
    assert pd.read_csv(path).value.tolist() == [1.2345678901234567, 2.5]
    assert gen.pull(fetch, path, gen.COORDINATES[0])
    assert path.read_bytes() == after
    entry = json.loads(gen.ACQUISITION_MANIFEST.read_text())["files"][path.as_posix()]
    assert entry["requests"][0]["accepted_rows"] == 1
    assert entry["requests"][1]["accepted_rows"] == 0
    assert entry["requests"][0]["csv_sha256"] == gen.file_sha256(path)


def test_later_annual_failure_preserves_csv_and_checkpoints_response(monkeypatch):
    monkeypatch.setattr(gen, "END_DATE", "2020-01-02")
    path = Path("existing.csv")
    path.write_text("time,value\n2018-12-31 23:00:00,1\n")
    original = path.read_bytes()
    calls = []

    def fetch(lat, lon, start, end):
        calls.append((start, end))
        return None if start == "2020-01-01" else (pd.DataFrame({"time": [pd.Timestamp(start)], "value": [2]}), {})

    assert not gen.pull(fetch, path, gen.COORDINATES[0])
    assert path.read_bytes() == original
    assert calls == [("2018-12-31", "2018-12-31"), ("2019-01-01", "2019-12-31"), ("2020-01-01", "2020-01-02")]
    records = json.loads(gen.ACQUISITION_MANIFEST.read_text())["files"][path.as_posix()]["requests"]
    assert len(records) == 2 and all(record["csv_sha256"] is None for record in records)


def test_interrupted_materialization_catches_up_without_new_rows(monkeypatch):
    coord = gen.COORDINATES[0]
    directory = Path("datasets/weather")
    directory.mkdir(parents=True)
    retained = weather(pd.date_range("2020-01-01", periods=25, freq="h"))
    retained.to_csv(directory / f"{coord['name']}_weather.csv", index=False)
    augmentation.calculate_inferred_variables(retained.iloc[:12]).to_csv(directory / f"{coord['name']}_weather_appended.csv", index=False)
    monkeypatch.setattr(gen, "fetch_weather_data", lambda *args: (retained.copy(), {}))
    monkeypatch.setattr(gen, "fetch_air_quality_data", empty_cams)
    assert gen.generate_dataset_for_location(coord)
    preferred = pd.read_csv(directory / f"{coord['name']}_weather_appended.csv", parse_dates=["time"])
    pd.testing.assert_frame_equal(preferred[["time"] + gen.WEATHER_VARS], retained)
    assert {"dewpoint_depression", "previous_night_low", "cooling_rate_6h", "cooling_rate_12h"} <= set(preferred)
    monkeypatch.setattr(gen, "process_weather_dataset", lambda path: False)
    assert not gen.generate_dataset_for_location(coord)


def test_augmentation_failed_replace_keeps_preferred_file(monkeypatch):
    path = Path("synthetic_weather.csv")
    weather(pd.date_range("2020-01-01", periods=2, freq="h")).to_csv(path, index=False)
    preferred = Path("synthetic_weather_appended.csv")
    preferred.write_bytes(b"retained evidence\n")

    def fail_replace(*args):
        raise OSError("interrupted publication")

    monkeypatch.setattr(Path, "replace", fail_replace)
    assert not augmentation.process_weather_dataset(path)
    assert preferred.read_bytes() == b"retained evidence\n"


def winter_response(winter):
    return response(payload(era5.WEATHER_COLS, era5.winter_hours(winter)))


def test_era5_interruption_resumes_only_missing_requests(monkeypatch):
    roster = Path("winters.csv")
    pd.DataFrame({"site": ["location_6", "location_6"], "winter": [2020, 2021]}).to_csv(roster, index=False)
    calls = []

    def get(url, params, **kwargs):
        calls.append(params)
        return winter_response(2020) if params["end_date"] == "2020-03-31" else response({}, status=429)

    monkeypatch.setattr(gen.requests, "get", get)
    with pytest.raises(RuntimeError, match="HTTP 429.*resume"):
        era5.acquire(roster, Path("era5"))
    manifest = json.loads(Path("era5/manifest.json").read_text())
    assert list(manifest["requests"]) == ["location_6/2020"]
    assert calls[0]["start_date"] == "2019-11-01" and calls[0]["end_date"] == "2020-03-31"
    assert calls[0]["hourly"] == ",".join(era5.WEATHER_COLS)
    assert calls[0]["models"] == "era5" and calls[0]["timezone"] == "Etc/GMT+8"
    calls.clear()
    monkeypatch.setattr(gen.requests, "get", lambda url, params, **kwargs: (calls.append(params) or winter_response(2021)))
    era5.acquire(roster, Path("era5"))
    assert [query["end_date"] for query in calls] == ["2021-03-31"]
    saved = pd.read_csv("era5/location_6_weather_era5_trend.csv", parse_dates=["time"])
    assert saved.columns.tolist() == ["time"] + era5.WEATHER_COLS
    assert saved.time.is_monotonic_increasing and saved.time.is_unique
    record = json.loads(Path("era5/manifest.json").read_text())["requests"]["location_6/2021"]
    assert record["rows"] == len(era5.winter_hours(2021))
    assert record["retrieval_utc"] and record["response_sha256"] and record["units"]
    calls.clear()
    era5.acquire(roster, Path("era5"))
    assert calls == []


@pytest.mark.parametrize("defect", ["null", "missing", "hour", "clock"])
def test_era5_invalid_response_never_completes_request(monkeypatch, defect):
    roster = Path("winters.csv")
    pd.DataFrame({"site": ["location_6"], "winter": [2020]}).to_csv(roster, index=False)
    data = payload(era5.WEATHER_COLS, era5.winter_hours(2020))
    if defect == "null":
        data["hourly"][era5.WEATHER_COLS[0]][3] = None
    elif defect == "missing":
        del data["hourly"][era5.WEATHER_COLS[0]]
    elif defect == "hour":
        for values in data["hourly"].values():
            values.pop()
    else:
        data["utc_offset_seconds"] = -25200
    monkeypatch.setattr(gen.requests, "get", lambda *args, **kwargs: response(data))
    with pytest.raises(ValueError):
        era5.acquire(roster, Path("era5"))
    assert not Path("era5/manifest.json").exists()
    assert not Path("era5/location_6_weather_era5_trend.csv").exists()


def test_era5_completed_chunk_corruption_is_not_refetched(monkeypatch):
    roster = Path("winters.csv")
    pd.DataFrame({"site": ["location_6"], "winter": [2020]}).to_csv(roster, index=False)
    monkeypatch.setattr(gen.requests, "get", lambda *args, **kwargs: winter_response(2020))
    era5.acquire(roster, Path("era5"))
    path = Path("era5/location_6_weather_era5_trend.csv")
    frame = pd.read_csv(path)
    frame.loc[0, era5.WEATHER_COLS[0]] += 1
    frame.to_csv(path, index=False)
    monkeypatch.setattr(gen.requests, "get", lambda *args, **kwargs: pytest.fail("completed key must not be requested"))
    with pytest.raises(ValueError, match="chunk hash mismatch"):
        era5.acquire(roster, Path("era5"))


def test_augmentation_preserves_acquisition_float_values():
    path = Path("precise_weather.csv")
    retained = weather(pd.date_range("2020-01-01", periods=2, freq="h"))
    retained.loc[0, "temperature_2m"] = 1.0000000000000002
    retained.to_csv(path, index=False)
    assert augmentation.process_weather_dataset(path)
    preferred = pd.read_csv("precise_weather_appended.csv", parse_dates=["time"], float_precision="round_trip")
    pd.testing.assert_frame_equal(preferred[["time"] + gen.WEATHER_VARS], retained)


@pytest.mark.parametrize("periods", [25, 30, 48])
def test_previous_night_low_handles_partial_last_day_and_nonrange_index(periods):
    retained = weather(pd.date_range("2020-01-01", periods=periods, freq="h"))
    retained.index = np.arange(100, 100 + periods)
    result = augmentation.calculate_inferred_variables(retained)
    assert result.previous_night_low.iloc[:24].isna().all()
    assert result.previous_night_low.iloc[24:].eq(18).all()
    pd.testing.assert_frame_equal(result[["time"] + gen.WEATHER_VARS], retained)


def test_previous_night_low_selects_clock_hours_in_partial_days():
    retained = weather(pd.to_datetime([
        "2020-01-01 17:00", "2020-01-01 18:00", "2020-01-01 23:00",
        "2020-01-02 00:00", "2020-01-02 06:00", "2020-01-02 12:00",
        "2020-01-04 00:00", "2020-01-04 12:00",
    ]))
    retained["temperature_2m"] = [-100, 10, 9, 8, -200, -300, 20, -400]
    result = augmentation.calculate_inferred_variables(retained)
    assert result.previous_night_low.iloc[3:6].eq(8).all()
    assert result.previous_night_low.iloc[6:].eq(20).all()


def test_midnight_only_acquisition_writes_canonical_timestamps():
    path = Path("midnight_weather.csv")
    retained = weather(pd.date_range("2020-01-01", periods=2, freq="D"))
    fetch = lambda *args: (retained.copy(), {})
    assert gen.pull(fetch, path, gen.COORDINATES[0])
    expected = ["2020-01-01 00:00:00", "2020-01-02 00:00:00"]
    assert pd.read_csv(path).time.tolist() == expected
    pd.testing.assert_frame_equal(gen.read_hourly_csv(path), retained)
    before = path.read_bytes()
    assert gen.pull(fetch, path, gen.COORDINATES[0])
    assert path.read_bytes() == before
    assert augmentation.process_weather_dataset(path)
    assert pd.read_csv("midnight_weather_appended.csv").time.tolist() == expected


@pytest.mark.parametrize("timestamp", ["2020-01-02", "not-a-time"])
def test_hourly_csv_rejects_invalid_timestamp_before_request(timestamp):
    path = Path("invalid.csv")
    path.write_text(f"time,value\n2020-01-01 00:00:00,1\n{timestamp},2\n")

    def no_request(*args):
        pytest.fail("invalid retained timestamps must fail before acquisition")

    with pytest.raises(ValueError):
        gen.pull(no_request, path, gen.COORDINATES[0])
