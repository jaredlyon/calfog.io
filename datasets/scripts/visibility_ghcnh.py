"""Build full-history ordinal observations and fixed-PST hourly labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models._shared.visibility import LABEL_POLICIES, decode_ghcnh, hourly_labels, select_hourly_report

STATIONS = {
    "location_6": "USW00093242", "location_7": "USW00093193",
    "location_8": "USW00093144", "location_9": "USW00053119",
    "location_10": "USW00023155",
}
ICAO = dict(zip(STATIONS, ("KMAE", "KFAT", "KVIS", "KHJO", "KBFL")))
COLUMNS = ["DATE", "visibility", "visibility_Source_Code", "visibility_Report_Type", "REM",
           "visibility_Measurement_Code", "visibility_Quality_Code", "visibility_Source_Station_ID",
           "REM_Source_Code", "REM_Report_Type", "REM_Source_Station_ID"]
BASE_URL = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-station/"
ERAS = (("1980-1995", 1980, 1995), ("1996-2004", 1996, 2004),
        ("2005-2025", 2005, 2025), ("2026", 2026, 2026))


def _deduplicate(obs, seen):
    keys = zip(obs.time_utc.astype("int64").to_numpy(),
               obs.cat_lo.fillna(-1).to_numpy(dtype=int), obs.cat_hi.fillna(-1).to_numpy(dtype=int))
    keep = []
    for key in keys:
        keep.append(key not in seen)
        seen.add(key)
    return obs.loc[keep].copy()


def _combine_hours(hours):
    # Preserve the original global minima and first-report deduplication policy.
    if hours:
        combined = pd.concat(hours, ignore_index=True).groupby("time", as_index=False).agg(
            cat_lo=("cat_lo", "min"), cat_hi=("cat_hi", "min"), n_reports=("n_reports", "sum"),
        )
        combined["time_utc"] = (combined.time + pd.Timedelta(hours=8)).dt.tz_localize("UTC")
        hourly = hourly_labels(combined)
        hourly["n_reports"] = combined.n_reports.to_numpy()
    else:
        hourly = hourly_labels(pd.DataFrame(columns=["time_utc", "cat_lo", "cat_hi"]))
    return hourly.loc[hourly.time >= pd.Timestamp("1980-01-01")]


def _cadence_candidates(obs):
    # Only the best timestamp in each chunk can win globally; keep all its bounds.
    rows = obs.loc[obs.cat_lo.notna() & obs.cat_hi.notna(),
                   ["time_utc", "cat_lo", "cat_hi", "report_type"]].copy()
    local = (pd.to_datetime(rows.time_utc, utc=True) - pd.Timedelta(hours=8)).dt.tz_localize(None)
    rows["hour"] = local.dt.floor("h")
    rows["distance"] = (local - rows.hour - pd.Timedelta(minutes=55)).abs()
    winners = rows.sort_values(["hour", "distance", "time_utc"], kind="stable").drop_duplicates("hour")
    return rows.merge(winners[["hour", "time_utc"]], on=["hour", "time_utc"])[
        ["time_utc", "cat_lo", "cat_hi", "report_type"]]


def _comparison(site, hourly):
    base = hourly["baseline"].set_index("time").category
    rows = []
    for policy, table in hourly.items():
        paired = pd.concat([base.rename("old"), table.set_index("time").category.rename("new")], axis=1)
        for era, start, end in ERAS:
            cell = paired.loc[(paired.index.year >= start) & (paired.index.year <= end)]
            old, new = cell.old, cell.new
            both = old.notna() & new.notna()
            rows.append({"site": site, "era": era, "policy": policy,
                         "known_hours": int(new.notna().sum()), "fog_hours": int(new.le(4).sum()),
                         "changed_categories": int((both & old.ne(new)).sum()),
                         "gained_known_hours": int((old.isna() & new.notna()).sum()),
                         "lost_known_hours": int((old.notna() & new.isna()).sum()),
                         "fog_to_nonfog": int((both & old.le(4) & new.gt(4)).sum()),
                         "nonfog_to_fog": int((both & old.gt(4) & new.le(4)).sum())})
    return pd.DataFrame(rows)


def _write_csv(frame, path):
    temp = path.with_suffix(".csv.part")
    frame.to_csv(temp, index=False, date_format="%Y-%m-%d %H:%M:%S")
    temp.replace(path)


def convert_site(site: str, path: Path, output: Path, *, policy: str = "combined") -> None:
    if policy not in {"baseline", "combined"}:
        raise ValueError(f"unsupported output policy: {policy}")
    output.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    obs_path = output / f"{site}_visibility_ghcnh_obs.csv"
    obs_temp = obs_path.with_suffix(".csv.part")
    hours = {name: [] for name in LABEL_POLICIES}
    seen = {name: set() for name in LABEL_POLICIES}
    cadence, routine = [], []
    routine_seen = set()
    flags = Counter()
    with obs_temp.open("w", encoding="utf-8", newline="") as stream:
        first = True
        for chunk in pd.read_csv(path, sep="|", usecols=COLUMNS, dtype=str,
                                 keep_default_na=False, chunksize=50000):
            flags.update(zip(chunk.visibility_Source_Code, chunk.visibility_Measurement_Code,
                             chunk.visibility_Quality_Code))
            for name in LABEL_POLICIES:
                obs = _deduplicate(decode_ghcnh(chunk, station_icao=ICAO[site], policy=name), seen[name])
                hours[name].append(hourly_labels(obs))
                if name == policy:
                    obs["site"] = site
                    obs.to_csv(stream, index=False, header=first)
                    first = False
                    cadence.append(_cadence_candidates(obs))
            # Filter the raw element before decoding: specials cannot erase routine reports.
            raw_routine = chunk.loc[chunk.visibility_Report_Type.eq("FM15")]
            obs = _deduplicate(decode_ghcnh(raw_routine, station_icao=ICAO[site], policy=policy), routine_seen)
            obs = obs.loc[obs.time_utc.dt.minute.ge(45)]
            routine.append(_cadence_candidates(obs))
    hourly = {name: _combine_hours(parts) for name, parts in hours.items()}
    _write_csv(hourly[policy], output / f"{site}_visibility_hourly.csv")
    empty = pd.DataFrame(columns=["time_utc", "cat_lo", "cat_hi", "report_type"])
    for name, parts in (("one_per_hour", cadence), ("routine_only", routine)):
        selected = select_hourly_report(pd.concat(parts, ignore_index=True) if parts else empty)
        selected = selected.loc[selected.time >= pd.Timestamp("1980-01-01")]
        _write_csv(selected, output / f"{site}_visibility_hourly_{name}.csv")
    _write_csv(_comparison(site, hourly), output / f"{site}_label_policy_comparison.csv")
    manifest_path = output / "ghcnh_inputs.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest[site] = {"path": str(path.resolve()), "sha256": source_hash,
                      "flag_frequencies": [{"source_code": source, "measurement_code": measurement,
                                            "quality_code": quality, "count": count}
                                           for (source, measurement, quality), count in sorted(flags.items())]}
    temp = manifest_path.with_suffix(".json.part")
    temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(manifest_path)
    obs_temp.replace(obs_path)
    print(f"{site}: wrote {policy} labels ({len(hourly[policy])} hours), sidecars and policy comparison")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "datasets/visibility/ghcnh")
    parser.add_argument("--sites", nargs="+", choices=tuple(STATIONS), default=list(STATIONS))
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/visibility")
    parser.add_argument("--policy", choices=("baseline", "combined"), default="combined")
    args = parser.parse_args(argv)
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    last_download = -float("inf")
    for site in args.sites:
        filename = f"GHCNh_{STATIONS[site]}_por.psv"
        path = args.cache / filename
        if not path.exists():
            time.sleep(max(0.0, 2.0 - (time.monotonic() - last_download)))
            last_download = time.monotonic()
            temp = path.with_suffix(".psv.part")
            with urlopen(BASE_URL + filename, timeout=120) as response, temp.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            temp.replace(path)
        convert_site(site, path, args.output, policy=args.policy)


if __name__ == "__main__":
    main()
