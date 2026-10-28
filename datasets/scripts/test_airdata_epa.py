"""Synthetic, CPU-only, network-free checks for the measured EPA series."""
import csv
from datetime import datetime, timedelta
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from urllib.error import HTTPError
from zipfile import ZipFile

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("airdata_epa", Path(__file__).with_name("airdata_epa.py"))
epa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(epa)

# The genuine hourly CSV header recorded by the 2008 AirData probe.
HEADER = [
    "State Code", "County Code", "Site Num", "Parameter Code", "POC", "Latitude", "Longitude",
    "Datum", "Parameter Name", "Date Local", "Time Local", "Date GMT", "Time GMT",
    "Sample Measurement", "Units of Measure", "MDL", "Uncertainty", "Qualifier",
    "Method Type", "Method Code", "Method Name", "State Name", "County Name", "Date of Last Change",
]


def row(pst="2020-01-01 00:00", monitor="06-019-5001", parameter=42603, value=10, poc=1, method=99):
    utc = datetime.fromisoformat(pst) + timedelta(hours=8)
    state, county, site = monitor.split("-")
    return {
        "State Code": state, "County Code": county, "Site Num": site,
        "Parameter Code": str(parameter), "POC": str(poc),
        # Deliberately wrong local columns: the converter must use GMT.
        "Date Local": "2099-12-31", "Time Local": "23:00",
        "Date GMT": utc.strftime("%Y-%m-%d"), "Time GMT": utc.strftime("%H:%M"),
        "Sample Measurement": str(value), "Method Code": str(method),
        "Units of Measure": "Parts per billion" if parameter == 42603 else "Micrograms/cubic meter (LC)",
    }


def archive_bytes(rows, duration=False):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=HEADER + (["Sample Duration"] if duration else []))
    writer.writeheader()
    writer.writerows(rows)
    result = io.BytesIO()
    with ZipFile(result, "w") as archive:
        archive.writestr("hourly.csv", text.getvalue())
    return result.getvalue()


def filtered(tmp_path, rows, group, duration=False):
    archive = tmp_path / f"{group}.zip"
    archive.write_bytes(archive_bytes(rows, duration))
    destination = tmp_path / f"filtered_{group}_2020.csv"
    count = epa.filter_archive(archive, destination, group)
    return destination, count


def hourly_rows(start, count, **kwargs):
    return [row(pst=str(stamp), **kwargs) for stamp in pd.date_range(start, periods=count, freq="h")]


def test_month_priority_coverage_and_no_hourly_monitor_switching(tmp_path):
    near, far = "06-019-5001", "06-019-0008"
    rows = hourly_rows("2020-01-01", 372, monitor=near, value=10)
    rows += hourly_rows("2020-01-01", 500, monitor=far, value=20)
    # Leap-year February: 696 hours, so 348 is exactly the 50% threshold.
    rows += hourly_rows("2020-02-01", 347, monitor=near, value=11)
    rows += hourly_rows("2020-02-01", 348, monitor=far, value=21)
    # Neither qualifies in March: choose the one with most hours, not priority.
    rows += hourly_rows("2020-03-01", 1, monitor=near, value=12)
    rows += hourly_rows("2020-03-01", 2, monitor=far, value=22)
    # Duplicated POCs must not inflate the nearest monitor's hourly coverage.
    rows += [row("2020-02-01", monitor=near, value=999, poc=poc) for poc in range(2, 20)]
    path, count = filtered(tmp_path, rows, "NONOxNOy")
    assert count == len(rows)
    frame, choices = epa.splice_site(epa.read_observations([path]), "location_7")
    assert choices["nox_ppb"] == {"2020-01": near, "2020-02": far, "2020-03": far}
    frame = frame.set_index("time")
    assert frame.loc["2020-01-01 00:00", "nox_ppb"] == 10
    # Although the farther monitor has this hour, the whole month chose near.
    assert pd.isna(frame.loc["2020-01-20 00:00", "nox_ppb"])
    assert frame.loc["2020-01-20 00:00", "nox_monitor"] == ""
    assert frame.loc["2020-02-01 00:00", "nox_ppb"] == 21
    assert frame.loc["2020-03-01 01:00", "nox_ppb"] == 22


def test_pm_parameter_priority_poc_and_superseded_methods(tmp_path):
    rows_88101 = [
        row(parameter=88101, poc=9, value=19), row(parameter=88101, poc=3, value=13),
        row("2020-01-01 01:00", parameter=88101, poc=1, value=""),
        row("2020-01-01 01:00", parameter=88101, poc=3, value=8),
        row("2020-01-01 02:00", parameter=88101, method=236, value=999),
        row("2020-01-01 03:00", parameter=88101, method=238, value=999),
        row("2020-01-01 03:00", parameter=88101, method=738, value=33),
        row("2020-01-01 04:00", parameter=88101, method=236, value=999),
        row("2020-01-01 05:00", parameter=88101, value=0),
    ]
    rows_88502 = [row(parameter=88502, poc=1, value=90),
                  row("2020-01-01 01:00", parameter=88502, poc=1, value=90),
                  row("2020-01-01 02:00", parameter=88502, poc=1, value=22)]
    a, count = filtered(tmp_path, rows_88101, "88101")
    b, _ = filtered(tmp_path, rows_88502, "88502")
    assert count == 5  # Three superseded rows and one blank value removed.
    frame, _ = epa.splice_site(epa.read_observations([b, a]), "location_7")
    assert frame.pm25_ugm3.iloc[:4].tolist() == [13, 8, 22, 33]
    assert pd.isna(frame.pm25_ugm3.iloc[4])
    assert frame.pm25_monitor.iloc[4] == ""
    assert frame.pm25_ugm3.iloc[5] == 0


@pytest.mark.parametrize("month", [1, 7])
def test_gmt_to_fixed_pst_without_dst_and_ppm_to_ppb(tmp_path, month):
    observation = row(pst=f"2020-{month:02d}-01 00:00", value=0.025)
    assert observation["Time GMT"] == "08:00"
    observation["Units of Measure"] = "Parts per million"
    path, _ = filtered(tmp_path, [observation], "NONOxNOy")
    data = epa.read_observations([path])
    assert data.time.iloc[0] == pd.Timestamp(f"2020-{month:02d}-01 00:00")
    assert data.time.dt.tz is None
    assert data.value.iloc[0] == pytest.approx(25)


def test_contiguous_output_grid_and_empty_missing_values(tmp_path):
    rows = [row(value=4), row("2020-01-01 02:00", value=6)]
    # Wrong state/site and NO/NOy must not contaminate the NOx series.
    rows += [row(monitor="07-019-5001", value=999), row(parameter=42601, value=999)]
    path, count = filtered(tmp_path, rows, "NONOxNOy")
    assert count == 2
    frame, _ = epa.splice_site(epa.read_observations([path]), "location_7")
    assert frame.columns.tolist() == epa.OUTPUT_COLUMNS
    assert frame.time.tolist() == list(pd.date_range("2020-01-01", periods=3, freq="h"))
    assert frame.nox_ppb.iloc[[0, 2]].tolist() == [4, 6]
    assert pd.isna(frame.nox_ppb.iloc[1])
    assert frame.nox_monitor.tolist() == ["06-019-5001", "", "06-019-5001"]
    output = tmp_path / "out.csv"
    frame.to_csv(output, index=False, date_format="%Y-%m-%d %H:%M:%S")
    with output.open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert saved[1] == dict(zip(epa.OUTPUT_COLUMNS, ["2020-01-01 01:00:00", "", "", "", ""]))


def test_optional_sample_duration_drops_daily_filters(tmp_path):
    hourly = row(parameter=88502, value=4)
    hourly["Sample Duration"] = "1 HOUR"
    daily = row("2020-01-01 01:00", parameter=88502, value=999)
    daily["Sample Duration"] = "24 HOUR"
    path, count = filtered(tmp_path, [hourly, daily], "88502", duration=True)
    assert count == 1
    assert epa.read_observations([path]).value.tolist() == [4]


def test_resume_skips_existing_filtered_years_and_preserves_provenance(tmp_path, monkeypatch):
    cache, output = tmp_path / "raw", tmp_path / "out"
    cache.mkdir()
    output.mkdir()
    provenance = {}
    for group in epa.GROUPS:
        path = cache / f"filtered_{group}_2020.csv"
        if group == "NONOxNOy":
            source, _ = filtered(tmp_path, [row(value=7)], group)
            path.write_bytes(source.read_bytes())
        else:
            pd.DataFrame(columns=epa.FILTER_COLUMNS).to_csv(path, index=False)
        provenance[path.name] = {"url": "original-url", "retrieval_utc": "2020-02-01T00:00:00+00:00",
                                 "sha256": "original-sha", "row_count": int(group == "NONOxNOy")}
    (output / "aq_manifest.json").write_text(json.dumps({"downloads": provenance}), encoding="utf-8")

    def no_request(*args, **kwargs):
        pytest.fail("A completed filtered year must not issue another request")

    monkeypatch.setattr(epa, "urlopen", no_request)
    monkeypatch.setattr(epa, "filter_archive", no_request)
    epa.build([2020], cache, output)
    manifest = json.loads((output / "aq_manifest.json").read_text(encoding="utf-8"))
    assert manifest["downloads"] == provenance
    assert manifest["monitor_months"]["location_7"]["nox_ppb"] == {"2020-01": "06-019-5001"}
    assert manifest["row_counts"]["location_7"]["nox_ppb"] == 1
    saved = pd.read_csv(output / "location_7_aq_measured.csv")
    assert saved.columns.tolist() == epa.OUTPUT_COLUMNS
    assert saved.nox_ppb.tolist() == [7]


def test_download_stream_hash_throttle_raw_cleanup_and_404_resume(tmp_path, monkeypatch):
    payload = archive_bytes([row(value=14)])
    clock = [100.0]
    starts, reads = [], []

    class Response(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024
            reads.append(size)
            return super().read(size)

    def request(url, timeout):
        starts.append(clock[0])
        if "_2021.zip" in url:
            raise HTTPError(url, 404, "missing", {}, None)
        return Response(payload)

    monkeypatch.setattr(epa, "urlopen", request)
    monkeypatch.setattr(epa.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(epa.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    last = [-float("inf")]
    path, meta = epa.collect_year("NONOxNOy", 2020, tmp_path, False, last)
    assert path.exists() and meta["row_count"] == 1
    assert meta["sha256"] == hashlib.sha256(payload).hexdigest()
    assert meta["retrieval_utc"] and reads
    assert not (tmp_path / "hourly_NONOxNOy_2020.zip").exists()
    missing, absent = epa.collect_year("NONOxNOy", 2021, tmp_path, False, last)
    assert starts[1] - starts[0] >= 2.0
    assert missing.exists() and absent["status"] == "not_found"
    assert pd.read_csv(missing).empty
    assert epa.collect_year("NONOxNOy", 2021, tmp_path, False, last) == (missing, None)
    assert len(starts) == 2


def test_keep_raw_and_retained_archive_are_network_free(tmp_path, monkeypatch):
    payload = archive_bytes([row(value=3)])
    archive = tmp_path / "hourly_NONOxNOy_2020.zip"
    archive.write_bytes(payload)

    def no_request(*args, **kwargs):
        pytest.fail("Retained archives should be filtered without redownloading")

    monkeypatch.setattr(epa, "urlopen", no_request)
    _, meta = epa.collect_year("NONOxNOy", 2020, tmp_path, True, [-float("inf")])
    assert archive.read_bytes() == payload
    assert meta["sha256"] == hashlib.sha256(payload).hexdigest()
    assert meta["row_count"] == 1
