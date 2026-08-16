"""Leakage-resistant data contracts for the v2 fog experiments.

This module deliberately contains no model training.  It turns the experiment's
important assumptions into small, testable transformations: target creation,
purged chronological splits, matched AQI arms, lead-label creation, and feature
availability audits.  Functions never mutate caller-owned data frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

import pandas as pd


DEFAULT_VISIBILITY_THRESHOLD_METERS = 1_610.0
DEFAULT_INVALID_VISIBILITY_VALUES = (999_999.0,)


@dataclass(frozen=True)
class TemporalSplit:
    """The three partitions produced by :func:`chronological_split`."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def val(self) -> pd.DataFrame:
        """Short alias useful in experiment code."""
        return self.validation

    def __iter__(self) -> Iterator[pd.DataFrame]:
        yield self.train
        yield self.validation
        yield self.test


@dataclass(frozen=True)
class MatchedArms:
    """AQI and non-AQI frames containing the same ordered observational units."""

    with_aqi: pd.DataFrame
    without_aqi: pd.DataFrame
    key_columns: tuple[str, ...]

    def __iter__(self) -> Iterator[pd.DataFrame]:
        yield self.with_aqi
        yield self.without_aqi


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


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"missing required column(s): {missing}")


def _as_timestamps(values: pd.Series, column: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"column {column!r} is not valid datetime data") from exc
    if parsed.isna().any():
        raise ValueError(f"column {column!r} contains missing timestamps")
    return parsed


def parse_target(
    frame: pd.DataFrame,
    *,
    visibility_col: str = "visibility_meters",
    target_col: str = "fog",
    threshold_meters: float = DEFAULT_VISIBILITY_THRESHOLD_METERS,
    invalid_visibility_values: Sequence[float] = DEFAULT_INVALID_VISIBILITY_VALUES,
    drop_missing: bool = True,
) -> pd.DataFrame:
    """Drop invalid observations and create the binary fog target.

    Fog is defined strictly as visibility below 1,610 metres.  Consequently
    1,609 is positive and 1,610 is negative.  The source sentinel 999,999 is
    dropped rather than imputed; imputing it could manufacture labels and make
    the two experimental arms incomparable.
    """
    _require_columns(frame, [visibility_col])
    result = frame.copy()
    visibility = pd.to_numeric(result[visibility_col], errors="coerce")
    invalid = visibility.isin(tuple(invalid_visibility_values))
    if drop_missing:
        invalid |= visibility.isna()
    elif visibility.isna().any():
        raise ValueError(f"column {visibility_col!r} contains non-numeric/missing values")
    result = result.loc[~invalid].copy()
    numeric_visibility = visibility.loc[result.index].astype(float)
    result[visibility_col] = numeric_visibility
    result[target_col] = (numeric_visibility < threshold_meters).astype("int8")
    return result


def make_fog_target(*args, **kwargs) -> pd.DataFrame:
    """Descriptive alias for :func:`parse_target`."""
    return parse_target(*args, **kwargs)


def chronological_split(
    frame: pd.DataFrame,
    *,
    time_col: str = "issue_time",
    validation_start: object | None = None,
    test_start: object | None = None,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    purge: object = "24h",
) -> TemporalSplit:
    """Return stable, sorted train/validation/test partitions with purge gaps.

    Explicit ``validation_start`` and ``test_start`` are recommended for a
    reproducible experiment.  If both are omitted, boundaries are selected
    from sorted unique timestamps using the supplied fractions.  The interval
    immediately before each evaluation boundary is discarded: train rows are
    earlier than ``validation_start - purge`` and validation rows are earlier
    than ``test_start - purge``.  This prevents a 24-hour input window ending at
    an evaluation issue time from overlapping the preceding partition.
    """
    _require_columns(frame, [time_col])
    result = frame.copy()
    result[time_col] = _as_timestamps(result[time_col], time_col)
    result = result.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    if result.empty:
        raise ValueError("cannot split an empty frame")

    explicit = validation_start is not None or test_start is not None
    if explicit and (validation_start is None or test_start is None):
        raise ValueError("validation_start and test_start must be provided together")
    if explicit:
        validation_boundary = pd.Timestamp(validation_start)
        test_boundary = pd.Timestamp(test_start)
    else:
        if not (0 < train_fraction < 1):
            raise ValueError("train_fraction must be between zero and one")
        if not (0 < validation_fraction < 1):
            raise ValueError("validation_fraction must be between zero and one")
        if train_fraction + validation_fraction >= 1:
            raise ValueError("train_fraction + validation_fraction must be below one")
        unique_times = result[time_col].drop_duplicates().reset_index(drop=True)
        validation_index = int(len(unique_times) * train_fraction)
        test_index = int(len(unique_times) * (train_fraction + validation_fraction))
        if validation_index == 0 or test_index >= len(unique_times) or validation_index >= test_index:
            raise ValueError("not enough unique timestamps for requested split fractions")
        validation_boundary = unique_times.iloc[validation_index]
        test_boundary = unique_times.iloc[test_index]

    try:
        purge_delta = pd.Timedelta(purge)
    except (TypeError, ValueError) as exc:
        raise ValueError("purge must be a pandas-compatible duration") from exc
    if purge_delta < pd.Timedelta(0):
        raise ValueError("purge cannot be negative")
    if not validation_boundary < test_boundary:
        raise ValueError("validation_start must be earlier than test_start")

    times = result[time_col]
    try:
        train_mask = times < validation_boundary - purge_delta
        validation_mask = (times >= validation_boundary) & (times < test_boundary - purge_delta)
        test_mask = times >= test_boundary
    except TypeError as exc:
        raise ValueError("split boundaries and data timestamps have incompatible timezones") from exc

    return TemporalSplit(
        train=result.loc[train_mask].reset_index(drop=True),
        validation=result.loc[validation_mask].reset_index(drop=True),
        test=result.loc[test_mask].reset_index(drop=True),
    )


def make_matched_arms(
    with_aqi: pd.DataFrame,
    without_aqi: pd.DataFrame,
    *,
    key_cols: Sequence[str] = ("location_id", "time"),
) -> MatchedArms:
    """Restrict two experimental arms to their common row keys.

    Keys must uniquely identify observations in each input.  Both outputs are
    sorted by the same key table, so equality of row populations is structural
    rather than an assumption based on row counts.
    """
    keys = tuple(key_cols)
    if not keys:
        raise ValueError("key_cols cannot be empty")
    _require_columns(with_aqi, keys)
    _require_columns(without_aqi, keys)
    for name, arm in (("with_aqi", with_aqi), ("without_aqi", without_aqi)):
        duplicates = arm.duplicated(list(keys), keep=False)
        if duplicates.any():
            examples = arm.loc[duplicates, list(keys)].head(3).to_dict("records")
            raise ValueError(f"{name} has duplicate row keys, for example {examples}")

    # An inner merge preserves exact key values and avoids string composite keys.
    common = with_aqi.loc[:, list(keys)].merge(
        without_aqi.loc[:, list(keys)], on=list(keys), how="inner", validate="one_to_one"
    )
    common = common.drop_duplicates().sort_values(list(keys), kind="mergesort").reset_index(drop=True)
    if common.empty:
        raise ValueError("the AQI and non-AQI arms have no row keys in common")

    def align(arm: pd.DataFrame) -> pd.DataFrame:
        return common.merge(arm, on=list(keys), how="left", validate="one_to_one", sort=False)

    return MatchedArms(align(with_aqi), align(without_aqi), keys)


def make_lead_target(
    observations: pd.DataFrame,
    *,
    time_col: str = "time",
    target_col: str = "fog",
    group_cols: Sequence[str] = ("location_id",),
    issue_hour: int = 18,
    verification_start_hour: int = 0,
    verification_end_hour: int = 9,
    output_target_col: str = "fog_next_00_09",
    require_complete_hours: bool = True,
) -> pd.DataFrame:
    """Build one next-morning lead label per location and verification day.

    An issue at 18:00 on day D predicts whether *any* target is fog on day
    D+1 from 00:00 through 09:59 (hours 0--9, inclusive). By default all ten
    distinct hourly targets are required; this prevents a missing fog hour from
    becoming a synthetic negative. Call :func:`parse_target` first so invalid
    visibility sentinels do not contribute labels. Set ``require_complete_hours``
    false only when explicitly estimating "any observed fog" instead.
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
    if not target.isin([0, 1]).all():
        raise ValueError(f"column {target_col!r} must be binary")
    work[target_col] = target.astype("int8")
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
    hourly = work.groupby([*aggregate_keys,"_verification_hour"],as_index=False,sort=True)[target_col].max()
    agg = hourly.groupby(aggregate_keys,as_index=False,sort=True)[target_col].agg(["max","size"]).reset_index()
    if require_complete_hours:
        expected = verification_end_hour-verification_start_hour+1
        agg = agg.loc[agg["size"] == expected].copy()
    labels = agg.rename(columns={"max":target_col}).drop(columns="size")
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
