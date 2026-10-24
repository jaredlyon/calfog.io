"""Date-free reported-visibility recovery for the five US study airports."""
from bisect import bisect_left
from fractions import Fraction
from functools import lru_cache
import math
import re

import numpy as np
import pandas as pd

SM_TO_M = 1609.344
CATEGORY_SM: tuple[float, ...] = (
    0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0,
    4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
)
N_CATEGORIES = 18
FOG_MAX_CATEGORY = 4
CATEGORY_LABELS: tuple[str, ...] = (
    "<1/4", "1/4", "1/2", "3/4", "1", "1 1/4", "1 1/2", "1 3/4", "2",
    "2 1/2", "3", "4", "5", "6", "7", "8", "9", "10+",
)
CATEGORY_LOWER_M: tuple[float, ...] = tuple(q * SM_TO_M for q in CATEGORY_SM)
CATEGORY_UPPER_M: tuple[float, ...] = CATEGORY_LOWER_M[1:] + (math.inf,)
CATEGORY_NOMINAL_M: tuple[float, ...] = (math.nan,) + CATEGORY_LOWER_M[1:]

# FMH-1 manual + automated union, FMH-2 Table 4-4; see research/ReportableScale.
# Above 40 SM the manual scale continues as 40 + 5*k, without an arbitrary cap.
REPORTABLE_SM: tuple[float, ...] = (
    0.0, 1/16, 1/8, 3/16, 1/4, 5/16, 3/8, 1/2, 5/8, 3/4, 7/8,
    1.0, 9/8, 5/4, 11/8, 3/2, 13/8, 7/4, 15/8, 2.0, 9/4, 5/2, 11/4,
    3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
    15.0, 20.0, 25.0, 30.0, 35.0, 40.0,
)
US_SYNOP_VV: dict[float, int] = dict(zip(REPORTABLE_SM[1:], (
    1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    32, 36, 40, 44, 48, 56, 58, 60, 61, 63, 64, 66, 68, 69, 71, 73, 74,
    80, 82, 84, 85, 87,
)))
US_SYNOP_KM: dict[float, float] = {
    q: vv / 10 if vv <= 50 else float(vv - 50) if vv <= 80
    else float(30 + 5 * (vv - 80)) for q, vv in US_SYNOP_VV.items()
}
# Keep the authoritative master table above, while admitting observed fine-grid
# and intermediate whole-mile reports under the same category ceiling rule.
CANDIDATE_SM: tuple[float, ...] = tuple(sorted(
    set(REPORTABLE_SM) | {n / 16 for n in range(241)} | set(range(16, 101))
))
# Empirical transport profiles from ValueCensus source-grid-overview, not instrument
# precision guarantees. Source 343 includes the observed sporadic 0.01-km encoding.
SOURCE_PROFILES: dict[str, tuple[str, ...]] = {
    "220": ("integer_m", "truncate_m"),
    "221": ("integer_m", "truncate_m"),
    "223": ("integer_m", "truncate_m", "approx_1p6"),
    "314": ("approx_1p6",),
    "315": ("integer_m", "truncate_m"),
    "335": ("approx_1p6", "integer_m", "truncate_m"),
    "343": ("integer_m", "truncate_m", "round_2dp"),
    "412": ("us_synop",),
    "413": ("round_2dp",),
    "999": ("round_2dp",),
}
SYNOP_REPORT_TYPES = frozenset({"FM12", "FM13", "SYSA", "SYMT"})


def sm_to_category(q: float) -> int:
    """Ceiling on the common ordinal scale, with explicit low and high bins."""
    if math.isnan(q) or q < 0:
        raise ValueError("visibility must be nonnegative statute miles")
    if q < 0.25:
        return 0
    return min(bisect_left(CATEGORY_SM, q), N_CATEGORIES - 1)


# Adapted from scratch-RawRecovery/raw_recovery.py: only the prevailing body
# group is accepted; remarks, NEW payloads, RVR and metric groups are not SM.
_SM = re.compile(r"(?<!\S)([MP]?)(?:(\d+)\s+)?(\d+(?:/\d+)?)SM(?=\s|$|[;=])")
_SKY = r"(?:(?:-[X]|[X])\s+)?(?:CLR|SKC|(?:(?:[ME]?\d{1,3}(?:V\d{1,3})?|[ME]?\d{1,3}V)\s*(?:OVC|BKN|SCT)\s+)+)"
_SAO = re.compile(r"^" + _SKY + r"\s*([MP]?)(?:(\d+)\s+)?(\d+(?:/\d+)?)(\+?)(?=\s|$|[A-Z])")
_HEADER = re.compile(r"(?<![A-Z])([A-Z][A-Z0-9]{3})\s+(\d{6})Z\b")


def _raw_sm(qualifier: str, whole: str | None, part: str) -> float:
    try:
        q = float(Fraction(whole or "0") + Fraction(part))
    except (ValueError, ZeroDivisionError):
        return math.nan
    return 0.0 if qualifier == "M" else math.inf if qualifier == "P" else q


def parse_rem_prevailing_sm(rem) -> float:
    """Parse METAR/SAO prevailing SM, never sector/tower visibility in RMK."""
    body = re.sub(r"^(?:MET|AWY|SYN)\d{3}", "", str(rem).upper()).strip()
    if body.startswith("NEW"):
        return math.nan
    body = re.split(r"\bRMK\b", body, maxsplit=1)[0]
    header = _HEADER.search(body)
    if header:
        body = body[header.end():].strip()
        body = re.sub(r"((?:KT|MPS|KMH))(?=[MP]?\d+(?:/\d+)?SM)", r"\1 ", body)
        match = _SM.search(body)
        if match:
            try:
                if "/" in match[3] and Fraction(match[3]) >= 1:
                    return math.nan
            except (ValueError, ZeroDivisionError):
                return math.nan
            return _raw_sm(match[1], match[2], match[3])
        return math.nan
    body = re.sub(r"^[A-Z]{3,4}\s+(?:SA|SP|RS)\s+\d{4}\s+", "", body)
    body = re.sub(r"^AO[12][A-Z]?\s+", "", body)
    match = _SAO.match(body)
    if match:
        whole, part = match[2], match[3]
        if "/" in part:
            n, d = part.split("/")
            if int(d) == 0:
                return math.nan
            if int(n) >= int(d):
                if len(n) != 2 or int(n[-1]) >= int(d):
                    return math.nan
                whole, part = n[0], n[-1] + "/" + d
        return _raw_sm(match[1], whole, part)
    return math.nan


def _profiles(source: str, report: str) -> tuple[str, ...]:
    if report in SYNOP_REPORT_TYPES or report.startswith("SY"):
        return ("us_synop", "native_metric")
    return SOURCE_PROFILES.get(source, ())


def _encoded(q: float, profile: str) -> float:
    if profile == "integer_m":
        return round(q * SM_TO_M) / 1000
    if profile == "truncate_m":
        return math.floor(q * SM_TO_M) / 1000
    if profile == "round_2dp":
        return round(q * 1.609344, 2)
    if profile == "approx_1p6":
        return round(q * 1.6, 1)
    return US_SYNOP_KM.get(q, math.nan)


@lru_cache(maxsize=65536)
def _candidates(km: float, profiles: tuple[str, ...]) -> tuple[float, ...]:
    if km == 0:
        return (0.0, 1/16, 1/8)
    grid = list(CANDIDATE_SM)
    # Only a few extension points can encode to this value, even for huge q.
    if km > 60:
        for scale in (1.6, 1.609344):
            k = round((km / scale - 40) / 5)
            grid.extend(40.0 + 5 * j for j in range(max(1, k - 1), k + 2))
    candidates = {q for q in grid if any(
        abs(_encoded(q, p) - km) <= 0.000500000001 for p in profiles
        if p != "native_metric"
    )}
    if "native_metric" in profiles:
        candidates.add(km / 1.609344)
    return tuple(sorted(candidates))


def _summary(candidates: tuple[float, ...]) -> tuple[float, int, int, str]:
    if not candidates:
        return math.nan, 0, 17, "ambiguous"
    cats = [sm_to_category(q) for q in candidates]
    lo, hi = min(cats), max(cats)
    status = ("ambiguous" if lo != hi else "censored_low" if lo == 0 else
              "censored_high" if lo == 17 else "identified")
    return candidates[0] if len(candidates) == 1 else math.nan, lo, hi, status


def _decoded(index, q, lo, hi, status, rem) -> pd.DataFrame:
    """Vectorized bounds, validator and nullable output shared by both ingests."""
    q, lo, hi = np.asarray(q, float), np.asarray(lo, float), np.asarray(hi, float)
    rem = np.asarray(rem, float)
    valid_raw = ~np.isnan(rem)
    raw_cat = np.searchsorted(CATEGORY_SM, rem, side="left").clip(0, 17)
    raw_cat[rem < 0.25] = 0
    known = ~np.isnan(lo)
    agrees = (raw_cat >= lo) & (raw_cat <= hi)
    conflict = valid_raw & known & ~agrees
    lo[conflict] = np.minimum(lo[conflict], raw_cat[conflict])
    hi[conflict] = np.maximum(hi[conflict], raw_cat[conflict])
    status = np.asarray(status, object)
    status[conflict] = "ambiguous"
    q[conflict] = np.nan
    canonical = np.full(len(q), np.nan)
    unique_cat = known & (lo == hi)
    canonical[unique_cat] = np.asarray(CATEGORY_NOMINAL_M)[lo[unique_cat].astype(int)]
    unique_q = unique_cat & (q >= 0.25) & (q < 10)
    canonical[unique_q] = q[unique_q] * SM_TO_M
    lower, upper = np.full(len(q), np.nan), np.full(len(q), np.nan)
    lower[known] = np.asarray(CATEGORY_LOWER_M)[lo[known].astype(int)]
    upper[known] = np.asarray(CATEGORY_UPPER_M)[hi[known].astype(int)]
    out = pd.DataFrame({
        "rem_sm": rem, "q_sm": q, "cat_lo": pd.array(lo, dtype="Int64"),
        "cat_hi": pd.array(hi, dtype="Int64"), "status": status,
        "canonical_m": canonical, "lower_m": lower, "upper_m": upper,
        "fog": pd.array(np.where(hi <= FOG_MAX_CATEGORY, True,
                        np.where(lo > FOG_MAX_CATEGORY, False, None)), dtype="boolean"),
        "raw_agrees": pd.array(np.where(valid_raw & known, agrees, None), dtype="boolean"),
    }, index=index)
    return out


def _observation_times(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Use a nearby body report timestamp, including across month/year edges."""
    original = pd.to_datetime(frame["DATE"], utc=True, errors="coerce")
    body = frame["REM"].fillna("").astype(str).str.upper()
    body = body.str.replace(r"^(?:MET|AWY|SYN)\d{3}", "", regex=True).str.strip()
    body = body.mask(body.str.startswith("NEW"), "").str.split(r"\bRMK\b", n=1, regex=True).str[0]
    stamp = body.str.extract(_HEADER)[1]
    day = pd.to_numeric(stamp.str[:2], errors="coerce")
    hour = pd.to_numeric(stamp.str[2:4], errors="coerce")
    minute = pd.to_numeric(stamp.str[4:6], errors="coerce")
    valid = day.between(1, 31) & hour.between(0, 23) & minute.between(0, 59)
    month = original.dt.tz_localize(None).dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")
    delta = (pd.to_timedelta(day - 1, unit="D") + pd.to_timedelta(hour, unit="h")
             + pd.to_timedelta(minute, unit="m"))
    corrected = original.copy()
    best = pd.Series(pd.Timedelta(minutes=61), index=frame.index)
    time_from_rem = pd.Series(False, index=frame.index)
    for offset in (0, -1, 1):
        candidate = month + pd.DateOffset(months=offset) + delta
        distance = (candidate - original).abs()
        use = valid & candidate.dt.day.eq(day) & distance.le(pd.Timedelta(minutes=60)) & distance.lt(best)
        corrected.loc[use] = candidate.loc[use]
        best.loc[use] = distance.loc[use]
        time_from_rem.loc[use] = True
    return corrected, time_from_rem


def decode_ghcnh(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover candidates once per stored km/profile set, not once per report."""
    km = pd.to_numeric(frame["visibility"], errors="coerce").to_numpy(dtype=float)
    sources = frame["visibility_Source_Code"].fillna("").astype(str).str.strip()
    reports = frame["visibility_Report_Type"].fillna("").astype(str).str.strip()
    n = len(frame)
    q, lo, hi = (np.full(n, np.nan) for _ in range(3))
    status = np.full(n, "missing", dtype=object)
    groups = pd.DataFrame({"source": sources.to_numpy(), "report": reports.to_numpy()})
    for (source, report), positions in groups.groupby(["source", "report"]).indices.items():
        positions = positions[np.isfinite(km[positions]) & (km[positions] >= 0)]
        values, inverse = np.unique(km[positions], return_inverse=True)
        summaries = [_summary(_candidates(float(v), _profiles(source, report))) for v in values]
        if summaries:
            sq, sl, sh, ss = zip(*summaries)
            q[positions] = np.asarray(sq)[inverse]
            lo[positions] = np.asarray(sl)[inverse]
            hi[positions] = np.asarray(sh)[inverse]
            status[positions] = np.asarray(ss)[inverse]
    top = np.isfinite(km) & (km >= 15.95)
    lo[top], hi[top], status[top] = 17, 17, "censored_high"
    q[top & (q < 10)] = np.nan
    rem = frame["REM"].map(parse_rem_prevailing_sm).to_numpy(dtype=float)
    out = _decoded(frame.index, q, lo, hi, status, rem)
    out.insert(0, "visibility_km", km)
    out.insert(0, "source_code", sources)
    out.insert(0, "report_type", reports)
    corrected_time, time_from_rem = _observation_times(frame)
    out.insert(0, "time_utc", corrected_time)
    out["time_from_rem"] = time_from_rem
    return out.drop_duplicates(["time_utc", "cat_lo", "cat_hi"], keep="first")


@lru_cache(maxsize=4096)
def _statute_candidates(text: str) -> tuple[float, ...] | None:
    text = text.strip().upper().removesuffix("SM")
    if text in {"", "NAN", "NONE", "<NA>"}:
        return None
    qualifier = text[0] if text[0] in "MP<" else ""
    number = text[1:] if qualifier else text
    try:
        q = sum(float(Fraction(part)) for part in number.rstrip("+").split())
    except (ValueError, ZeroDivisionError):
        return None
    if q < 0 or not math.isfinite(q):
        return None
    if qualifier in {"M", "<"}:
        return (0.0, 1/16, 1/8) if q <= 0.25 else ()
    if qualifier == "P":
        return (math.inf,)
    if q == 0:
        return (0.0, 1/16, 1/8)
    if q in CANDIDATE_SM or q >= 10:
        return (q,)
    # IEM reports fractional SM to two decimals (e.g. 0.12 is 1/8 SM).
    return tuple(v for v in CANDIDATE_SM if abs(round(v, 2) - q) < 1e-9)


def decode_statute_miles(values_sm: pd.Series, raw: pd.Series | None = None) -> pd.DataFrame:
    candidates = values_sm.map(lambda v: None if pd.isna(v) else _statute_candidates(str(v)))
    summaries = [(_summary(c) if c is not None else
                  (math.nan, math.nan, math.nan, "missing")) for c in candidates]
    if summaries:
        q, lo, hi, status = zip(*summaries)
    else:
        q, lo, hi, status = [], [], [], []
    rem = (raw.reindex(values_sm.index).map(parse_rem_prevailing_sm).to_numpy(dtype=float)
           if raw is not None else np.full(len(values_sm), np.nan))
    return _decoded(values_sm.index, q, lo, hi, status, rem)


def hourly_labels(obs: pd.DataFrame) -> pd.DataFrame:
    """Minimum report category bounds per fixed UTC-8 hour; unknowns stay unknown."""
    valid = obs["cat_lo"].notna() & obs["cat_hi"].notna()
    if "status" in obs:
        valid &= obs["status"].ne("missing")
    rows = obs.loc[valid, ["time_utc", "cat_lo", "cat_hi"]].copy()
    rows["time"] = (pd.to_datetime(rows["time_utc"], utc=True) - pd.Timedelta(hours=8)).dt.tz_localize(None).dt.floor("h")
    out = rows.groupby("time", as_index=False).agg(
        cat_lo=("cat_lo", "min"), cat_hi=("cat_hi", "min"), n_reports=("cat_lo", "size"),
    )
    out["cat_lo"] = out["cat_lo"].astype("Int64")
    out["cat_hi"] = out["cat_hi"].astype("Int64")
    out["category"] = out["cat_lo"].where(out["cat_lo"].eq(out["cat_hi"])).astype("Int64")
    out["fog"] = pd.array(np.where(out["cat_hi"] <= FOG_MAX_CATEGORY, True,
                          np.where(out["cat_lo"] > FOG_MAX_CATEGORY, False, None)), dtype="boolean")
    return out[["time", "cat_lo", "cat_hi", "category", "fog", "n_reports"]]
