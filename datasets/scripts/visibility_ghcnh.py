"""Build full-history ordinal observations and fixed-PST hourly labels."""
import argparse
from pathlib import Path
import shutil
import sys
import time
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models._shared.visibility import decode_ghcnh, hourly_labels

STATIONS = {
    "location_6": "USW00093242", "location_7": "USW00093193",
    "location_8": "USW00093144", "location_9": "USW00053119",
    "location_10": "USW00023155",
}
COLUMNS = ["DATE", "visibility", "visibility_Source_Code", "visibility_Report_Type", "REM"]
BASE_URL = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-station/"


def convert_site(site: str, path: Path, output: Path) -> None:
    obs_path = output / f"{site}_visibility_ghcnh_obs.csv"
    hourly_path = output / f"{site}_visibility_hourly.csv"
    obs_temp = obs_path.with_suffix(".csv.part")
    hourly_temp = hourly_path.with_suffix(".csv.part")
    hours = []
    seen = set()
    with obs_temp.open("w", encoding="utf-8", newline="") as stream:
        first = True
        for chunk in pd.read_csv(path, sep="|", usecols=COLUMNS, dtype=str, chunksize=50000):
            obs = decode_ghcnh(chunk)
            # decode_ghcnh removes within-chunk duplicates; retain the same
            # exact first-report policy when a duplicate crosses chunks.
            keys = zip(
                obs["time_utc"].astype("int64").to_numpy(),
                obs["cat_lo"].fillna(-1).to_numpy(dtype=int),
                obs["cat_hi"].fillna(-1).to_numpy(dtype=int),
            )
            keep = []
            for key in keys:
                keep.append(key not in seen)
                seen.add(key)
            obs = obs.loc[keep]
            obs["site"] = site
            obs.to_csv(stream, index=False, header=first)
            first = False
            hours.append(hourly_labels(obs))
    # Chunk boundaries can split an hour. Combine bounds and counts before
    # deriving the final category, rather than taking a last chunk's label.
    if hours:
        combined = pd.concat(hours, ignore_index=True).groupby("time", as_index=False).agg(
            cat_lo=("cat_lo", "min"), cat_hi=("cat_hi", "min"), n_reports=("n_reports", "sum"),
        )
        combined["time_utc"] = (combined["time"] + pd.Timedelta(hours=8)).dt.tz_localize("UTC")
        hourly = hourly_labels(combined)
        hourly["n_reports"] = combined["n_reports"].to_numpy()
    else:
        hourly = hourly_labels(pd.DataFrame(columns=["time_utc", "cat_lo", "cat_hi"]))
    hourly = hourly.loc[hourly["time"] >= pd.Timestamp("1980-01-01")]
    hourly.to_csv(hourly_temp, index=False, date_format="%Y-%m-%d %H:%M:%S")
    obs_temp.replace(obs_path)
    hourly_temp.replace(hourly_path)
    print(f"{site}: wrote {obs_path.name} and {hourly_path.name} ({len(hourly)} hours)")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "datasets/visibility/ghcnh")
    parser.add_argument("--sites", nargs="+", choices=tuple(STATIONS), default=list(STATIONS))
    args = parser.parse_args(argv)
    args.cache.mkdir(parents=True, exist_ok=True)
    output = ROOT / "datasets/visibility"
    output.mkdir(parents=True, exist_ok=True)
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
        convert_site(site, path, output)


if __name__ == "__main__":
    main()
