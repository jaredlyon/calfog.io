from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import pandas as pd
from .data import _require_columns, _as_timestamps
from .visibility import FOG_MAX_CATEGORY, N_CATEGORIES

@dataclass(frozen=True)
class CausalAudit:
    """Result of checking feature timestamps against their availability cutoff."""

    cutoff_column: str
    checked_columns: tuple[str, ...]
    violations: pd.DataFrame

    @property
    def ok(self) -> bool:
        return self.violations.empty

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    def __bool__(self) -> bool:
        return self.ok

    def raise_for_violations(self) -> None:
        if not self.ok:
            raise ValueError(
                f"causal feature audit found {self.violation_count} violation(s); "
                f"features must be available no later than {self.cutoff_column!r}"
            )

def make_lead_target(
    observations: pd.DataFrame,
    *,
    time_col: str = "time",
    target_col: str = "category",
    group_cols: Sequence[str] = ("location_id",),
    issue_hour: int = 18,
    verification_start_hour: int = 0,
    verification_end_hour: int = 9,
    output_target_col: str = "category",
    require_complete_hours: bool = True,
) -> pd.DataFrame:
    """Build one next-morning lead label per location and verification day.

    An issue at 18:00 on day D predicts the minimum visibility category on
    D+1 from 00:00 through 09:59. All ten distinct known hourly categories
    are required by default; unknown hours cannot manufacture a target.
    """
    groups = tuple(group_cols)
    _require_columns(observations, [time_col, target_col, *groups])
    if not (0 <= issue_hour <= 23):
        raise ValueError("issue_hour must be in 0..23")
    if not (0 <= verification_start_hour <= verification_end_hour <= 23):
        raise ValueError("verification hours must satisfy 0 <= start <= end <= 23")

    work = observations.copy()
    work[time_col] = _as_timestamps(work[time_col], time_col)
    target = pd.to_numeric(work[target_col], errors="raise")
    if not target.isin(range(N_CATEGORIES)).all():
        raise ValueError(f"column {target_col!r} must contain ordinal categories")
    work[target_col] = target.astype("int64")
    in_window = work[time_col].dt.hour.between(
        verification_start_hour, verification_end_hour, inclusive="both"
    )
    work = work.loc[in_window].copy()
    output_columns = [*groups, "issue_time", "verification_start", "verification_end", output_target_col]
    if work.empty:
        return pd.DataFrame(columns=output_columns)

    work["_verification_day"] = work[time_col].dt.normalize()
    work["_verification_hour"] = work[time_col].dt.floor("h")
    aggregate_keys = [*groups, "_verification_day"]
    hourly = work.groupby([*aggregate_keys,"_verification_hour"],as_index=False,sort=True)[target_col].min()
    agg = hourly.groupby(aggregate_keys,as_index=False,sort=True)[target_col].agg(["min","size"]).reset_index()
    if require_complete_hours:
        expected = verification_end_hour-verification_start_hour+1
        agg = agg.loc[agg["size"] == expected].copy()
    labels = agg.rename(columns={"min":target_col}).drop(columns="size")
    day = labels.pop("_verification_day")
    labels["verification_start"] = day + pd.to_timedelta(verification_start_hour, unit="h")
    # End is exclusive, making the hour-9 window end at 10:00 by default.
    labels["verification_end"] = day + pd.to_timedelta(verification_end_hour + 1, unit="h")
    labels["issue_time"] = day - pd.Timedelta(days=1) + pd.to_timedelta(issue_hour, unit="h")
    labels = labels.rename(columns={target_col: output_target_col})
    return labels.loc[:, output_columns].sort_values([*groups, "issue_time"], kind="mergesort").reset_index(drop=True)

def audit_causal_features(
    frame: pd.DataFrame,
    feature_time_cols: Sequence[str] | Mapping[str, str],
    *,
    task: str = "nowcast",
    label_time_col: str = "label_time",
    issue_time_col: str = "issue_time",
    raise_on_error: bool = False,
) -> CausalAudit:
    """Audit that every feature was available by the prediction cutoff.

    For a ``nowcast`` the cutoff is the label time.  For a ``lead`` forecast it
    is the issue time, not the future verification/label time.  A mapping may be
    supplied when human feature names differ from their timestamp-column names.
    Missing feature timestamps are violations because causality cannot then be
    demonstrated.
    """
    if task not in {"nowcast", "lead"}:
        raise ValueError("task must be 'nowcast' or 'lead'")
    cutoff_col = label_time_col if task == "nowcast" else issue_time_col
    if isinstance(feature_time_cols, Mapping):
        feature_times = dict(feature_time_cols)
    else:
        feature_times = {column: column for column in feature_time_cols}
    if not feature_times:
        raise ValueError("at least one feature timestamp column is required")
    _require_columns(frame, [cutoff_col, *feature_times.values()])

    cutoff = _as_timestamps(frame[cutoff_col], cutoff_col)
    records: list[pd.DataFrame] = []
    for feature, timestamp_col in feature_times.items():
        # NaT is explicitly recorded rather than rejected by the shared parser.
        try:
            available = pd.to_datetime(frame[timestamp_col], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"column {timestamp_col!r} is not valid datetime data") from exc
        bad = available.isna() | (available > cutoff)
        if bad.any():
            records.append(
                pd.DataFrame(
                    {
                        "row_index": frame.index[bad],
                        "feature": feature,
                        "feature_time_column": timestamp_col,
                        "feature_time": available.loc[bad].array,
                        "cutoff_time": cutoff.loc[bad].array,
                        "reason": ["missing_timestamp" if pd.isna(value) else "after_cutoff" for value in available.loc[bad]],
                    }
                )
            )
    violations = pd.concat(records, ignore_index=True) if records else pd.DataFrame(
        columns=["row_index", "feature", "feature_time_column", "feature_time", "cutoff_time", "reason"]
    )
    audit = CausalAudit(cutoff_col, tuple(feature_times.values()), violations)
    if raise_on_error:
        audit.raise_for_violations()
    return audit

def assert_causal_features(*args, **kwargs) -> None:
    """Raise if :func:`audit_causal_features` finds a violation."""
    kwargs["raise_on_error"] = True
    audit_causal_features(*args, **kwargs)

def _visibility_lags(issue_times: pd.Series, vis: pd.DataFrame) -> pd.DataFrame:
    """An hourly observation is available only after its hour has ended."""
    known = vis.loc[vis.category.notna(), ['time', 'category']].sort_values('time')
    known = known.rename(columns={'time': 'lag_time', 'category': 'lag_category'})
    known['lag_time'] = pd.to_datetime(known.lag_time).astype('datetime64[ns]')
    cutoff = pd.to_datetime(issue_times).astype('datetime64[ns]').reset_index(drop=True) - pd.Timedelta(hours=1)
    issues = pd.DataFrame({'cutoff': cutoff, '_row': range(len(issue_times))})
    lag = pd.merge_asof(issues.sort_values('cutoff'), known, left_on='cutoff', right_on='lag_time',
                        direction='backward', tolerance=pd.Timedelta(hours=2))
    lag = lag.sort_values('_row').reset_index(drop=True)
    lag['lag_category'] = lag.lag_category.astype('Int64')
    assert (lag.loc[lag.lag_time.notna(), 'lag_time'] <= lag.loc[lag.lag_time.notna(), 'cutoff']).all()
    return lag[['lag_category', 'lag_time']]


def add_neighbor_lags(labels_by_site: dict[str, pd.DataFrame],
                      vis_by_site: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    result = {}
    for site, labels in labels_by_site.items():
        neighbors = pd.DataFrame({
            other: _visibility_lags(labels.issue_time, vis).lag_category.astype(float)
            for other, vis in vis_by_site.items() if other != site
        }, index=range(len(labels)))
        frame = labels.copy()
        frame['nbr_min_cat'] = neighbors.min(axis=1).to_numpy(dtype=float)
        frame['nbr_mean_cat'] = neighbors.mean(axis=1).to_numpy(dtype=float)
        frame['nbr_fog_count'] = neighbors.le(FOG_MAX_CATEGORY).sum(axis=1).to_numpy(dtype=int)
        frame['nbr_reporting'] = neighbors.notna().sum(axis=1).to_numpy(dtype=int)
        result[site] = frame
    return result


NEAR_SAT_DPD_C = 4.2
NEAR_SAT_WIND_KMH = 7.2


def label_tables(weather,vis,site):
    known=vis.dropna(subset=['category']).copy()
    known['category']=known.category.astype(int)
    known['fog']=known.category.le(FOG_MAX_CATEGORY)
    now=known[known.time.isin(set(weather.time))].copy().rename(columns={'time':'issue_time'})
    now['label_time']=now.issue_time; now['row_key']=now.issue_time.dt.strftime('%Y-%m-%dT%H:%M:%S'); now['site']=site
    lead=make_lead_target(known,time_col='time',target_col='category',group_cols=(),output_target_col='category',require_complete_hours=True)
    lead['fog']=lead.category.le(FOG_MAX_CATEGORY)
    lead['label_time']=lead.verification_start
    lead=lead[lead.issue_time.isin(set(weather.time))].copy(); lead['row_key']=lead.issue_time.dt.strftime('%Y-%m-%dT%H:%M:%S');lead['site']=site
    lead['valid_target_hours']=10
    strata = weather.set_index('time').reindex(
        columns=['temperature_2m', 'dew_point_2m', 'wind_speed_10m'])
    strata = pd.DataFrame({'dpd_c': strata.temperature_2m - strata.dew_point_2m,
                           'wind_kmh': strata.wind_speed_10m}, index=strata.index)
    now[['dpd_c', 'wind_kmh']] = strata.reindex(now.issue_time).to_numpy(dtype=float)
    # min_periods=10 makes an absent hour or measurement invalidate the mean.
    verification = strata.reindex(pd.date_range(weather.time.min(), weather.time.max(), freq='h'))
    means = verification.rolling(10, min_periods=10).mean()
    lead[['dpd_c', 'wind_kmh']] = means.reindex(
        lead.verification_end - pd.Timedelta(hours=1)).to_numpy(dtype=float)
    # Explicit core causal audit: feature window ends at issue, never verification time.
    for z,task in ((now,'nowcast'),(lead,'lead')):
        z['_max_feature_time']=z.issue_time
        lag = _visibility_lags(z.issue_time, known)
        z['lag_category'] = pd.array(lag.lag_category, dtype='Int64')
        z['lag_time'] = lag.lag_time.to_numpy()
        assert (z.loc[z.lag_time.notna(), 'lag_time'] <=
                z.loc[z.lag_time.notna(), 'issue_time'] - pd.Timedelta(hours=1)).all()
        audit_causal_features(z,['_max_feature_time'],task=task).raise_for_violations()
    return now.sort_values('issue_time').reset_index(drop=True),lead.sort_values('issue_time').reset_index(drop=True)
