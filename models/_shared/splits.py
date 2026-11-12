from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence
import pandas as pd
from .data import _require_columns, _as_timestamps

FOLD_SPECS = ((0.45, 0.55, 0.70), (0.55, 0.70, 0.85), (0.70, 0.85, 1.00))

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

def rolling_origin_folds(frame, time_col="issue_time", purge="24h", specs=FOLD_SPECS):
    """Three expanding-origin train/validation/test folds with disjoint tests.

    Fractions are predeclared, chronological, and never shifted to fog events.
    ``chronological_split`` from the leakage-safe core applies both purges.
    """
    z=frame.copy(); z[time_col]=pd.to_datetime(z[time_col]); z=z.sort_values(time_col,kind="mergesort").reset_index(drop=True)
    if len(z)<30: raise ValueError("at least 30 rows are required")
    times=z[time_col].drop_duplicates().reset_index(drop=True)
    out=[]
    for fi,(vfrac,tfrac,stopfrac) in enumerate(specs):
        vi=min(len(times)-2, int(len(times)*vfrac)); ti=min(len(times)-1,int(len(times)*tfrac))
        split=chronological_split(z,time_col=time_col,validation_start=times.iloc[vi],test_start=times.iloc[ti],purge=purge)
        test=split.test
        if stopfrac < 1:
            stop=times.iloc[min(len(times)-1,int(len(times)*stopfrac))]
            test=test.loc[test[time_col] < stop].reset_index(drop=True)
        fold={"train":split.train,"val":split.validation,"test":test,"fold":fi+1,
              "validation_start":times.iloc[vi],"test_start":times.iloc[ti]}
        for a,b in ((fold["train"],fold["val"]),(fold["val"],fold["test"])):
            if len(a) and len(b):
                assert b[time_col].min()-a[time_col].max() >= pd.Timedelta(purge)
        out.append(fold)
    test_keys=[]
    for f in out: test_keys.extend(f["test"][time_col].tolist())
    if len(test_keys)!=len(set(test_keys)): raise AssertionError("rolling test folds overlap")
    return out

def matched_cohort(with_aqi, without_aqi, key_cols=("site","issue_time")):
    return make_matched_arms(with_aqi,without_aqi,key_cols=key_cols)
