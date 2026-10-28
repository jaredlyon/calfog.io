"""Download EPA hourly NOx/PM2.5 and splice fixed-PST airport series (D-028/029/034).

National archives are downloaded in bounded chunks, then CSV rows are streamed
straight from the ZIP into the small site-filtered cache; no national CSV is
extracted. Completed filtered years and download provenance survive restarts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models._shared.data import utc_to_pst_naive

NOX_PRIORITY = {
    "location_6": ["06-039-0004", "06-019-0242"],
    "location_7": ["06-019-5001", "06-019-0008", "06-019-0011", "06-019-0005", "06-019-0242"],
    "location_8": ["06-107-2003", "06-107-2002"],
    "location_9": ["06-031-1004", "06-107-2002"],
    "location_10": ["06-029-0232", "06-029-0231", "06-029-0004", "06-029-0014"],
}
PM25_PRIORITY = {
    "location_6": ["06-039-2010"],
    "location_7": ["06-019-5001", "06-019-0008", "06-019-0011"],
    "location_8": ["06-107-2003", "06-107-2002"],
    "location_9": ["06-031-1004"],
    "location_10": ["06-029-0010", "06-029-0014"],
}
GROUPS = {"NONOxNOy": "42603", "88101": "88101", "88502": "88502"}
BASE_URL = "https://aqs.epa.gov/aqsweb/airdata/"
OUTPUT_COLUMNS = ["time", "nox_ppb", "nox_monitor", "pm25_ugm3", "pm25_monitor"]
# Actual hourly AirData header, verified by the 2008 archive probe (ObservedAQ).
FILTER_COLUMNS = [
    "State Code", "County Code", "Site Num", "Parameter Code", "POC",
    "Date GMT", "Time GMT", "Sample Measurement", "Units of Measure", "Method Code",
]
OBS_COLUMNS = ["time", "monitor", "variable", "value"]


def filter_archive(archive: Path, destination: Path, group: str) -> int:
    """Stream an hourly ZIP's CSV(s), retaining only the requested monitors."""
    priorities = NOX_PRIORITY if group == "NONOxNOy" else PM25_PRIORITY
    monitors = {monitor for values in priorities.values() for monitor in values}
    temporary = destination.with_suffix(".csv.part")
    count = 0
    with ZipFile(archive) as zipped, temporary.open("w", encoding="utf-8", newline="") as output:
        pd.DataFrame(columns=FILTER_COLUMNS).to_csv(output, index=False)
        members = [name for name in zipped.namelist() if name.lower().endswith(".csv")]
        if not members:
            raise ValueError(f"{archive}: no hourly CSV found")
        for member in members:
            with zipped.open(member) as source:
                for chunk in pd.read_csv(
                    source, dtype=str, chunksize=100000,
                    usecols=lambda column: column in FILTER_COLUMNS or column == "Sample Duration",
                ):
                    missing = set(FILTER_COLUMNS) - set(chunk.columns)
                    if missing:
                        raise ValueError(f"{archive}: missing columns {sorted(missing)}")
                    monitor = (chunk["State Code"].str.zfill(2) + "-"
                               + chunk["County Code"].str.zfill(3) + "-"
                               + chunk["Site Num"].str.zfill(4))
                    keep = monitor.isin(monitors) & chunk["Parameter Code"].eq(GROUPS[group])
                    if "Sample Duration" in chunk:
                        keep &= chunk["Sample Duration"].eq("1 HOUR")
                    if group != "NONOxNOy":
                        keep &= ~pd.to_numeric(chunk["Method Code"], errors="coerce").isin([236, 238])
                    values = pd.to_numeric(chunk["Sample Measurement"], errors="coerce")
                    keep &= np.isfinite(values)
                    selected = chunk.loc[keep, FILTER_COLUMNS]
                    selected.to_csv(output, index=False, header=False)
                    count += len(selected)
    temporary.replace(destination)
    return count


def read_observations(paths) -> pd.DataFrame:
    """Normalize units/GMT and resolve parameter and POC duplicates per hour."""
    pieces = []
    for path in paths:
        for chunk in pd.read_csv(path, dtype=str, chunksize=100000):
            if chunk.empty:
                continue
            parameter = pd.to_numeric(chunk["Parameter Code"], errors="raise")
            pm = parameter.isin([88101, 88502])
            keep = parameter.eq(42603) | pm
            keep &= ~(pm & pd.to_numeric(chunk["Method Code"], errors="coerce").isin([236, 238]))
            if "Sample Duration" in chunk:
                keep &= chunk["Sample Duration"].eq("1 HOUR")
            chunk = chunk.loc[keep].copy()
            if chunk.empty:
                continue
            parameter = parameter.loc[keep]
            units = chunk["Units of Measure"].str.strip().str.casefold()
            gas = parameter.eq(42603)
            ppm = units.eq("parts per million")
            valid_units = ((gas & (ppm | units.eq("parts per billion")))
                           | (~gas & units.isin(["micrograms/cubic meter (lc)", "micrograms/cubic meter"])))
            if not valid_units.all():
                raise ValueError(f"{path}: unexpected concentration units {units[~valid_units].unique().tolist()}")
            values = pd.to_numeric(chunk["Sample Measurement"], errors="coerce")
            values = values * np.where(gas & ppm, 1000.0, 1.0)
            times = utc_to_pst_naive(chunk["Date GMT"] + " " + chunk["Time GMT"])
            if not times.eq(times.dt.floor("h")).all():
                raise ValueError(f"{path}: non-hourly GMT timestamps")
            normalized = pd.DataFrame({
                "time": times,
                "monitor": (chunk["State Code"].str.zfill(2) + "-" + chunk["County Code"].str.zfill(3)
                            + "-" + chunk["Site Num"].str.zfill(4)),
                "variable": np.where(gas, "nox_ppb", "pm25_ugm3"),
                "value": values,
                "parameter": parameter,
                "poc": pd.to_numeric(chunk["POC"], errors="raise"),
            })
            pieces.append(normalized.loc[np.isfinite(normalized.value)])
    if not pieces:
        return pd.DataFrame(columns=OBS_COLUMNS)
    combined = pd.concat(pieces, ignore_index=True)
    # 88101 precedes 88502 even when its POC is larger. Within a parameter,
    # choose the first POC with a value, not the mean of co-located instruments.
    combined = combined.sort_values(["parameter", "poc"], kind="stable")
    return combined.drop_duplicates(["monitor", "time", "variable"])[OBS_COLUMNS]


def splice_site(observations: pd.DataFrame, site: str) -> tuple[pd.DataFrame, dict]:
    """Choose one monitor for a whole fixed-PST month, without gap filling."""
    candidates = observations.loc[
        (observations.variable.eq("nox_ppb") & observations.monitor.isin(NOX_PRIORITY[site]))
        | (observations.variable.eq("pm25_ugm3") & observations.monitor.isin(PM25_PRIORITY[site]))
    ]
    chosen_months = {"nox_ppb": {}, "pm25_ugm3": {}}
    if candidates.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), chosen_months
    grid = pd.date_range(candidates.time.min(), candidates.time.max(), freq="h", name="time")
    result = pd.DataFrame(index=grid)
    for variable, monitor_column, priorities in [
        ("nox_ppb", "nox_monitor", NOX_PRIORITY[site]),
        ("pm25_ugm3", "pm25_monitor", PM25_PRIORITY[site]),
    ]:
        result[variable] = np.nan
        result[monitor_column] = ""
        data = candidates.loc[candidates.variable.eq(variable)].copy()
        data["month"] = pd.to_datetime(data.time).dt.to_period("M")
        for month, rows in data.groupby("month", sort=True):
            counts = rows.groupby("monitor").time.nunique()
            qualifying = [monitor for monitor in priorities if counts.get(monitor, 0) >= month.days_in_month * 24 * 0.5]
            chosen = qualifying[0] if qualifying else max(priorities, key=lambda monitor: counts.get(monitor, 0))
            selected = rows.loc[rows.monitor.eq(chosen)].set_index("time")
            result.loc[selected.index, variable] = selected.value
            result.loc[selected.index, monitor_column] = chosen
            chosen_months[variable][str(month)] = chosen
    return result.reset_index()[OUTPUT_COLUMNS], chosen_months


def save_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def collect_year(group: str, year: int, cache: Path, keep_raw: bool, last_request: list) -> tuple[Path, dict | None]:
    """Return the completed cache; None metadata means no request was needed."""
    filtered = cache / f"filtered_{group}_{year}.csv"
    if filtered.exists():
        return filtered, None
    archive = cache / f"hourly_{group}_{year}.zip"
    url = BASE_URL + archive.name
    metadata = {"url": url, "group": group, "year": year}
    if archive.exists():
        # A retained archive can be re-filtered without another request.
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata.update(retrieval_utc=None, sha256=digest, status="cached_archive")
    else:
        time.sleep(max(0.0, 2.0 - (time.monotonic() - last_request[0])))
        last_request[0] = time.monotonic()
        metadata["retrieval_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = archive.with_suffix(".zip.part")
        digest = hashlib.sha256()
        try:
            with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    digest.update(block)
                    output.write(block)
        except HTTPError as exc:
            if exc.code != 404:
                raise
            temporary.unlink(missing_ok=True)
            # A header-only filtered cache also makes unavailable years resumable.
            empty = filtered.with_suffix(".csv.part")
            pd.DataFrame(columns=FILTER_COLUMNS).to_csv(empty, index=False)
            empty.replace(filtered)
            metadata.update(sha256=None, status="not_found", row_count=0)
            return filtered, metadata
        temporary.replace(archive)
        metadata.update(sha256=digest.hexdigest(), status="downloaded")
    metadata["row_count"] = filter_archive(archive, filtered, group)
    if not keep_raw:
        archive.unlink()
    return filtered, metadata


def build(years, cache: Path, output: Path, keep_raw=False) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "aq_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"downloads": {}}
    last_request = [-float("inf")]
    paths = []
    for year in years:
        for group in GROUPS:
            path, metadata = collect_year(group, year, cache, keep_raw, last_request)
            paths.append(path)
            if metadata is not None:
                manifest["downloads"][path.name] = metadata
            elif path.name not in manifest["downloads"]:
                # Never invent retrieval dates or ZIP hashes for an imported cache.
                with path.open(encoding="utf-8") as stream:
                    row_count = sum(1 for _ in stream) - 1
                manifest["downloads"][path.name] = {
                    "url": BASE_URL + f"hourly_{group}_{year}.zip", "group": group, "year": year,
                    "retrieval_utc": None, "sha256": None, "row_count": row_count, "status": "existing_filtered",
                }
            # Checkpoint provenance after every completed year/group, before a later failure.
            save_manifest(manifest_path, manifest)
    observations = read_observations(paths)
    manifest["monitor_months"] = {}
    manifest["row_counts"] = {}
    for site in NOX_PRIORITY:
        frame, choices = splice_site(observations, site)
        destination = output / f"{site}_aq_measured.csv"
        temporary = destination.with_suffix(".csv.part")
        frame.to_csv(temporary, index=False, date_format="%Y-%m-%d %H:%M:%S")
        temporary.replace(destination)
        manifest["monitor_months"][site] = choices
        manifest["row_counts"][site] = {
            "hours": len(frame), "nox_ppb": int(frame.nox_ppb.notna().sum()),
            "pm25_ugm3": int(frame.pm25_ugm3.notna().sum()),
        }
    save_manifest(manifest_path, manifest)


def parse_years(value: str) -> range:
    try:
        bounds = value.split("-")
        start = int(bounds[0])
        end = int(bounds[1]) if len(bounds) == 2 else start
        if len(bounds) > 2 or start < 1 or end > 9998 or end < start:
            raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError("years must be YYYY or YYYY-YYYY in ascending order") from exc
    return range(start, end + 1)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=parse_years, default=parse_years("1980-2026"))
    parser.add_argument("--cache", type=Path, default=ROOT / "datasets/aq_measured/raw")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args(argv)
    build(args.years, args.cache, ROOT / "datasets/aq_measured", args.keep_raw)


if __name__ == "__main__":
    main()
