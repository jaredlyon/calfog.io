"""Weather-adjusted Nov–Mar fog trends (D-057–D-063, D-066).

Run from the repository root: python eda/long_term_trend/long_term_trend_v2.py
Uses realized daily weather, not an operational forecast. All dates are fixed PST.
No air-quality or year variables enter the diagnostic fog classifiers.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models._shared.data import SEED, SITES, WEATHER_COLS, load_sources
from models._shared.environment import capture_environment
from models._shared.families import ebm
from models._shared.visibility import FOG_MAX_CATEGORY
from models._shared.windows import fit_preprocess, tree_view

OUT_DIR = Path(__file__).resolve().parent
STEPS = {"location_7": "1995-09-01", "location_10": "1996-06-01",
         "location_8": "1990-09-18"}
WINTER_MONTHS = (11, 12, 1, 2, 3)


def winter_year(dates):
    dates = pd.DatetimeIndex(dates)
    return dates.year.to_numpy() - (dates.month.to_numpy() <= 3)


def winter_validity(daily):
    """Check *each* full calendar month, including unobserved month-end hours."""
    rows = []
    for (site, year), part in daily.groupby(["site", "winter"]):
        coverage = {}
        for month in WINTER_MONTHS:
            cal_year = int(year) + (month <= 3)
            denominator = 24 * calendar.monthrange(cal_year, month)[1]
            coverage[str(month)] = float(part.loc[part.date.dt.month == month, "known_hours"].sum() / denominator)
        rows.append({"site": site, "winter": int(year),
                     "valid": all(value >= 0.90 for value in coverage.values()),
                     "month_coverage": coverage})
    return pd.DataFrame(rows, columns=["site", "winter", "valid", "month_coverage"])


def daily_table(weather, visibility, site, *, start_year=1980, end_year=2026):
    """Return winter daily labels and aligned (day,24,weather-feature) arrays.

    Reindexing to whole days prevents partial days or absent hours from becoming
    100%-covered observations. Fog means ANY known hourly category <= 4.
    """
    first = pd.Timestamp(start_year, 11, 1)
    last = pd.Timestamp(end_year + 1, 4, 1)
    dates = pd.date_range(first, last, freq="D", inclusive="left")
    dates = dates[dates.month.isin(WINTER_MONTHS)]
    hours = (dates.to_numpy()[:, None] + np.arange(24)[None, :] * np.timedelta64(1, "h")).ravel()
    v = visibility.dropna(subset=["category"]).groupby("time").category.min().reindex(hours)
    categories = v.to_numpy(dtype=float, na_value=np.nan).reshape(-1, 24)
    hourly_weather = weather.set_index("time")[WEATHER_COLS].reindex(hours)
    X = hourly_weather.to_numpy(dtype=np.float32).reshape(-1, 24, len(WEATHER_COLS))
    daily = pd.DataFrame({"date": dates, "site": site, "winter": winter_year(dates),
                          "fog_day": np.any(categories <= FOG_MAX_CATEGORY, axis=1).astype(int),
                          "known_hours": np.isfinite(categories).sum(axis=1)})
    # Daily meteorological means use observed weather hours, never visibility.
    # Retain source precision for the strict Gray thresholds (4.2 and 7.2).
    dpd = (hourly_weather.temperature_2m - hourly_weather.dew_point_2m).to_numpy(dtype=float).reshape(-1, 24)
    wind = hourly_weather.wind_speed_10m.to_numpy(dtype=float).reshape(-1, 24)
    for name, values, threshold in (("dpd_c", dpd, 4.2), ("wind_kmh", wind, 7.2)):
        count = np.isfinite(values).sum(axis=1)
        # Centering preserves equality when all hours sit on a strict cutoff.
        daily[name] = threshold + np.divide(np.nansum(values - threshold, axis=1), count,
                                            out=np.full(len(daily), np.nan), where=count > 0)
    daily["optimal"] = (daily.dpd_c < 4.2) & (daily.wind_kmh < 7.2)
    valid = winter_validity(daily)
    daily = daily.merge(valid[["site", "winter", "valid"]], on=["site", "winter"], validate="many_to_one", sort=False)
    return daily, X


def fit_binary(family, X, y, *, seed):
    """Binary paths: the shared LR fit_model is ordinal, not binary-capable."""
    if family == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=seed).fit(X, y)
    if family == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier
        if len(y) > ebm.MAX_TRAIN_ROWS:
            keep = np.random.default_rng(seed).choice(len(y), ebm.MAX_TRAIN_ROWS, replace=False)
            X, y = X[keep], y[keep]
        return ExplainableBoostingClassifier(outer_bags=2, interactions=0,
                                            random_state=seed, n_jobs=-1).fit(X, y)
    raise ValueError(f"unknown diagnostic family: {family}")


def diagnostic_predictions(daily, X, family, *, seed=SEED, fitter=fit_binary, preprocess=fit_preprocess):
    """Pooled leave-one-winter-out predictions; the held-out year is global.

    Unobserved target days and invalid site-winters cannot train or evaluate the
    model. Missing training classes are reported, never replaced with in-sample
    predictions or an invented constant model.
    """
    eligible = daily.valid.to_numpy() & (daily.known_hours.to_numpy() >= 18)
    sites = sorted(daily.site.unique())
    onehot = (daily.site.to_numpy()[:, None] == np.array(sites)[None, :]).astype(np.float32)
    phase = 2 * np.pi * (daily.date.dt.dayofyear.to_numpy() - 1) / 365.25
    calendar_features = np.column_stack([np.sin(phase), np.cos(phase), onehot]).astype(np.float32)
    predictions = np.full(len(daily), np.nan)
    folds = []
    years = daily.winter.to_numpy()
    for year in np.unique(years[eligible]):
        train = np.flatnonzero(eligible & (years != year))
        test = np.flatnonzero(eligible & (years == year))
        fold = {"winter": int(year), "n_train": len(train), "n_test": len(test)}
        if len(train) == 0 or daily.fog_day.iloc[train].nunique() < 2:
            folds.append({**fold, "status": "not_estimable", "reason": "training needs two fog classes"})
            continue
        # SimpleImputer otherwise drops all-empty channels and changes the shape.
        channels = np.isfinite(X[train]).any(axis=(0, 1))
        if not channels.any():
            folds.append({**fold, "status": "not_available", "reason": "no training weather"})
            continue
        (tr, te), _, _ = preprocess(X[train][:, :, channels], X[test][:, :, channels])
        tr = np.column_stack([tree_view(tr), calendar_features[train]])
        te = np.column_stack([tree_view(te), calendar_features[test]])
        model = fitter(family, tr, daily.fog_day.iloc[train].to_numpy(), seed=seed)
        positive = int(np.flatnonzero(np.asarray(model.classes_) == 1)[0])
        predictions[test] = model.predict_proba(te)[:, positive]
        folds.append({**fold, "status": "ok", "weather_columns": np.array(WEATHER_COLS)[channels].tolist()})
    return predictions, folds


def winter_panel(daily, predictions):
    data = daily.copy()
    data["probability"] = predictions
    rows = []
    for (site, year), part in data.loc[data.valid].groupby(["site", "winter"]):
        eligible = part.loc[part.known_hours >= 18]
        # A partial prediction sum must not masquerade as a complete winter.
        if eligible.empty or eligible.probability.isna().any():
            continue
        observed = int(eligible.fog_day.sum())
        expected = float(eligible.probability.sum())
        rows.append({"site": site, "winter": int(year), "year": int(year),
                     "observed_all_days": int(part.fog_day.sum()), "observed": observed,
                     "expected": expected, "n_days": len(eligible),
                     "dpd_c": float(part.dpd_c.mean()),
                     "fog_day_residual": observed - expected,
                     "y": float(np.log((observed + 0.5) / (expected + 0.5)))})
    return pd.DataFrame(rows, columns=["site", "winter", "year", "observed_all_days", "observed", "expected", "n_days", "dpd_c", "fog_day_residual", "y"])


def winter_bootstrap_indices(years, rng):
    """One draw selects whole winters, jointly for every airport in that winter."""
    years = np.asarray(years)
    unique = np.unique(years)
    return np.concatenate([np.flatnonzero(years == year) for year in rng.choice(unique, len(unique), replace=True)])


def panel_design(panel, predictors=("year",), *, steps=STEPS):
    sites = sorted(panel.site.unique())
    columns = [(panel.site.to_numpy() == site).astype(float) for site in sites]
    names = [f"intercept_{site}" for site in sites]
    for predictor in predictors:
        values = panel[predictor].to_numpy(dtype=float)
        # Centering only changes intercepts and avoids poor year conditioning.
        columns.append(values - values.mean())
        names.append(predictor)
    omitted = []
    start = pd.to_datetime(panel.winter.astype(str) + "-11-01")
    for site, date in steps.items():
        at_site = panel.site.to_numpy() == site
        values = (at_site & (start >= pd.Timestamp(date)).to_numpy()).astype(float)
        # Constant site steps are absorbed by its intercept (post-ASOS check).
        if not at_site.any() or np.unique(values[at_site]).size < 2:
            omitted.append(f"step_{site}")
        else:
            columns.append(values)
            names.append(f"step_{site}")
    return np.column_stack(columns), names, omitted


def _ols(X, y):
    if len(y) <= X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ coef
    sse = float(residual @ residual)
    sst = float(np.square(y - y.mean()).sum())
    # Gaussian maximum likelihood includes the fitted variance parameter in AIC.
    variance = max(sse / len(y), np.finfo(float).tiny)
    loglike = -len(y) / 2 * (np.log(2 * np.pi * variance) + 1)
    return coef, (1 - sse / sst if sst > 0 else None), float(2 * (X.shape[1] + 1) - 2 * loglike)


def _centered_summary(estimate, draws, *, reason=None):
    draws = np.asarray(draws, dtype=float)
    valid = int(np.isfinite(draws).sum())
    if reason is None and (not len(draws) or valid != len(draws)):
        reason = "incomplete_bootstrap_support"
    if reason:
        return {"ci95": None, "p": None, "p_two_sided": None,
                "valid_draws": valid, "inference_status": "not_available" if reason == "unit_leverage" else "not_estimable",
                "inference_reason": reason}
    deviation = np.abs(draws - estimate)
    radius = float(np.quantile(deviation, 0.95))
    p = float((1 + np.count_nonzero(deviation >= abs(estimate))) / (len(draws) + 1))
    return {"ci95": [float(estimate - radius), float(estimate + radius)],
            "p": p, "p_two_sided": p, "valid_draws": valid,
            "inference_status": "ok", "inference_reason": None}


def _calendar_lengths(years):
    span = int(np.max(years) - np.min(years) + 1)
    primary = int(np.ceil(span ** (1 / 3)))
    return span, primary, sorted({1, 3, 5, primary})


def _winter_multipliers(years, length, rng):
    years = np.asarray(years, dtype=int)
    span = int(years.max() - years.min() + 1)
    normals = rng.standard_normal(span + length - 1)
    weights = np.convolve(normals, np.ones(length), mode="valid") / np.sqrt(length)
    return weights[years - years.min()]


def fit_panel(panel, predictors=("year",), *, response="y", n_boot=10000, seed=SEED, steps=STEPS):
    """Fixed-design dependent wild inference; winter-pairs is secondary."""
    columns = list(predictors) + [response]
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=columns)
    base = {"n_rows": len(panel), "n_winters": int(panel.winter.nunique()),
            "method": "dependent_wild_bootstrap", "n_boot": n_boot, "seed": seed,
            "block_length": _calendar_lengths(panel.winter)[1] if len(panel) else None,
            "dependence_sensitivity": {}, "pairs_winter_full_rank_draws": 0}
    if panel.empty:
        return {**base, "status": "not_available", "reason": "no complete winters"}
    X, names, omitted = panel_design(panel, predictors, steps=steps)
    y = panel[response].to_numpy(dtype=float)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return {**base, "status": "not_estimable", "reason": "collinear"}
    fitted = _ols(X, y)
    if fitted is None:
        return {**base, "status": "not_estimable", "reason": "no residual degrees of freedom"}
    coef, r2, aic = fitted
    years = panel.winter.to_numpy(dtype=int)
    span, primary, lengths = _calendar_lengths(years)
    P = np.linalg.pinv(X)
    leverage = np.einsum("ij,ji->i", X, P)
    unit_leverage = bool(np.any(leverage >= 1 - 1e-10))
    sensitivity = {}
    for length in lengths:
        draws = np.full((n_boot, len(names)), np.nan)
        if not unit_leverage:
            residual = (y - X @ coef) / np.sqrt(1 - leverage)
            rng = np.random.default_rng(seed)
            for draw in range(n_boot):
                draws[draw] = coef + P @ (residual * _winter_multipliers(years, length, rng))
        sensitivity[str(length)] = {
            "method": "dependent_wild_bootstrap", "block_length": length,
            "calendar_span": span, "seed": seed, "n_boot": n_boot,
            "coefficients": {name: _centered_summary(coef[j], draws[:, j],
                              reason="unit_leverage" if unit_leverage else None)
                             for j, name in enumerate(names)}}
    pairs = np.full((n_boot, len(names)), np.nan)
    full_rank = 0
    rng = np.random.default_rng(seed)
    for draw in range(n_boot):
        index = winter_bootstrap_indices(years, rng)
        A, target = X[index], y[index]
        if len(target) <= X.shape[1]:
            continue
        U, singular, Vt = np.linalg.svd(A, full_matrices=False)
        rank = int(np.count_nonzero(singular > singular[0] * max(A.shape) * np.finfo(float).eps))
        estimate = Vt[:rank].T @ ((U[:, :rank].T @ target) / singular[:rank])
        supported = np.linalg.norm(Vt[rank:], axis=0) <= 1e-8
        pairs[draw, supported] = estimate[supported]
        full_rank += int(rank == X.shape[1])
    coefficients = {
        name: {"estimate": float(coef[j]),
               **sensitivity[str(primary)]["coefficients"][name],
               "pairs_winter": _centered_summary(coef[j], pairs[:, j])}
        for j, name in enumerate(names)}
    return {**base, "status": "ok", "coefficients": coefficients, "r_squared": r2, "aic": aic,
            "block_length": primary, "calendar_span": span, "dependence_sensitivity": sensitivity,
            "pairs_winter_full_rank_draws": full_rank, "omitted_absorbed_steps": omitted}


def monotone_trend(years, values, *, n_boot=10000, seed=SEED):
    years, values = np.asarray(years, float), np.asarray(values, float)
    keep = np.isfinite(years) & np.isfinite(values)
    years, values = years[keep], values[keep]
    if len(values) < 3 or len(np.unique(years)) < 2:
        return {"status": "not_estimable", "n": len(values)}
    if len(np.unique(years)) != len(years):
        raise ValueError("rank trends require one value per calendar winter")
    slope = float(stats.theilslopes(values, years).slope)
    tau = 0.0 if np.ptp(values) == 0 else float(stats.kendalltau(years, values).statistic)
    span, primary, lengths = _calendar_lengths(years)
    grid = np.arange(int(years.min()), int(years.max()) + 1)
    residual = np.full(span, np.nan)
    residual[(years - years.min()).astype(int)] = values - np.median(values - slope * years) - slope * years
    sensitivity = {}
    for length in lengths:
        rng = np.random.default_rng(seed)
        slopes, taus = [], []
        # A block longer than the observed calendar span cannot be sampled.
        if length <= span:
            for _ in range(n_boot):
                starts = rng.integers(0, span - length + 1, size=int(np.ceil(span / length)))
                sample = np.concatenate([residual[start:start + length] for start in starts])[:span]
                finite = np.isfinite(sample)
                if finite.sum() < 3:
                    break
                slopes.append(float(stats.theilslopes(sample[finite], grid[finite]).slope))
                taus.append(0.0 if np.ptp(sample[finite]) == 0 else float(stats.kendalltau(grid[finite], sample[finite]).statistic))
        available = len(slopes) == n_boot
        radius = float(np.quantile(np.abs(slopes), 0.95)) if available else None
        sensitivity[str(length)] = {
            "block_length": length, "n_boot": n_boot, "seed": seed,
            "inference_status": "ok" if available else "not_available",
            "inference_reason": None if available else "insufficient_resampled_observations",
            "ci95": [slope - radius, slope + radius] if available else None,
            "mann_kendall_p": float((1 + np.count_nonzero(np.abs(taus) >= abs(tau))) / (n_boot + 1)) if available else None}
    return {"status": "ok", "n": len(values), "slope": slope, "kendall_tau": tau,
            "method": "calendar_residual_block_bootstrap", "calendar_span": span,
            "block_length": primary, "n_boot": n_boot, "seed": seed,
            **sensitivity[str(primary)], "dependence_sensitivity": sensitivity}


def site_trends(frame, value):
    return {site: monotone_trend(part.winter, part[value],
                                n_boot=frame.attrs.get("n_boot", 10000),
                                seed=frame.attrs.get("seed", SEED))
            for site, part in frame.groupby("site")}


def measured_nox(site, *, root=ROOT):
    """Monthly 65% rule uses the complete month, not just rows in the file."""
    path = Path(root) / "datasets" / "aq_measured" / f"{site}_aq_measured.csv"
    if not path.exists():
        return pd.DataFrame(columns=["site", "winter", "nox_ppb"]), {"status": "not_available", "reason": "measured NOx file not available"}
    hourly = pd.read_csv(path, usecols=["time", "nox_ppb"], parse_dates=["time"])
    if hourly.time.duplicated().any() or hourly.time.dt.tz is not None:
        raise ValueError(f"{path}: expected unique fixed-PST naive hours")
    hourly = hourly.set_index("time").sort_index()
    rows = []
    for year in np.unique(winter_year(hourly.index[hourly.index.month.isin(WINTER_MONTHS)])):
        months, total, count = [], 0.0, 0
        for month in WINTER_MONTHS:
            cal_year = int(year) + (month <= 3)
            start = pd.Timestamp(cal_year, month, 1)
            end = start + pd.offsets.MonthBegin(1)
            values = hourly.loc[(hourly.index >= start) & (hourly.index < end), "nox_ppb"].dropna()
            required = 0.65 * calendar.monthrange(cal_year, month)[1] * 24
            months.append(len(values) >= required)
            total += float(values.sum())
            count += len(values)
        if all(months):
            rows.append({"site": site, "winter": int(year), "nox_ppb": total / count})
    return pd.DataFrame(rows, columns=["site", "winter", "nox_ppb"]), {"status": "ok" if rows else "not_available", "valid_winters": len(rows), "reason": None if rows else "no winters pass monthly 65% NOx coverage"}


def attribution(panel, nox, *, n_boot=10000, seed=SEED):
    if nox.empty or panel.empty:
        return {"status": "not_available", "reason": "measured NOx winters not available"}
    common = panel.merge(nox, on=["site", "winter"], how="inner", validate="one_to_one").dropna(subset=["nox_ppb"])
    if common.empty:
        return {"status": "not_available", "reason": "measured NOx winters not available"}
    models = {name: fit_panel(common, predictors, n_boot=n_boot, seed=seed)
              for name, predictors in (("year_only", ("year",)), ("nox_only", ("nox_ppb",)), ("year_plus_nox", ("year", "nox_ppb")))}
    year, both = models["year_only"], models["year_plus_nox"]
    change = None
    if year["status"] == both["status"] == "ok":
        change = both["coefficients"]["year"]["estimate"] - year["coefficients"]["year"]["estimate"]
    gray = {}
    for name, rows in (("all", common), ("near_saturation", common.loc[common.dpd_c < 4.2])):
        fit = fit_panel(rows, ("nox_ppb",), response="fog_day_residual", n_boot=n_boot, seed=seed)
        if fit["status"] == "ok":
            coef = fit["coefficients"]["nox_ppb"]
            scaled = {**coef, "estimate": 10 * coef["estimate"],
                      "ci95": [10 * x for x in coef["ci95"]] if coef["ci95"] else None}
            scaled["pairs_winter"] = {**coef["pairs_winter"],
                "ci95": [10 * x for x in coef["pairs_winter"]["ci95"]] if coef["pairs_winter"]["ci95"] else None}
            fit["fog_days_per_10ppb"] = scaled
        gray[name] = fit
    return {"status": "ok", "common_rows": common[["site", "winter", "nox_ppb"]].to_dict("records"),
            "models": models, "year_slope_change_with_nox": change, "gray_comparison": gray}


def gray_replication(daily):
    rows = []
    for (site, year), part in daily.loc[daily.valid].groupby(["site", "winter"]):
        fog = int(part.fog_day.sum())
        optimal = int(part.optimal.sum())
        both = int((part.fog_day.astype(bool) & part.optimal).sum())
        rows.append({"site": site, "winter": int(year), "n_days": len(part),
                     "n_fog": fog, "n_optimal": optimal, "n_both": both,
                     "fog_share_of_optimal": both / optimal if optimal else None,
                     "optimal_share_of_fog": both / fog if fog else None})
    table = pd.DataFrame(rows, columns=["site", "winter", "n_days", "n_fog", "n_optimal", "n_both", "fog_share_of_optimal", "optimal_share_of_fog"])
    table.attrs.update(daily.attrs)
    trends = {value: site_trends(table, value)
              for value in ("fog_share_of_optimal", "optimal_share_of_fog")}
    return {"winters": table.to_dict("records"), "trends": trends}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _coefficient_lines(value, prefix=""):
    lines = []
    if not isinstance(value, dict):
        return lines
    if "coefficients" in value:
        lines.append(f"- {prefix}: {value['method']}, block length {value['block_length']}.")
        for name, coef in value["coefficients"].items():
            interval, p, label = coef["ci95"], coef["p"], "dependent wild bootstrap"
            if interval is None:
                secondary = coef["pairs_winter"]
                interval, p = secondary["ci95"], secondary["p"]
                label = "exploratory; assumes independent winters"
            uncertainty = f"95% CI {interval}, p={p:.6g} ({label})" if interval is not None else "uncertainty unavailable"
            lines.append(f"  - {name}: {coef['estimate']:.8g}; {uncertainty}.")
        return lines
    for key, child in value.items():
        if key not in ("dependence_sensitivity", "winters", "folds"):
            lines.extend(_coefficient_lines(child, f"{prefix}.{key}" if prefix else key))
    return lines


def write_outputs(result, out_dir):
    """Persist machine-readable results, complete Markdown tables and figures."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "long_term_trend_v2"
    result = _json_safe(result)
    for analysis in result["diagnostics"].values():
        analysis["uncertainty_scope"] = "fixed_diagnostic_predictions"
        analysis["first_stage_refit"] = False
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Long-term weather-adjusted fog study", "", "Fixed PST; Nov–Mar winter labeled by November year. Fog = any known hour in categories 0–4.",
             "Every month needs 90% known hourly labels. Diagnostic LR/EBM use realized 00–23 weather, calendar phase and site only; leave one whole winter out across all sites.",
             "Expected and observed use the same days with ≥18 known hours. y = log((observed+0.5)/(expected+0.5)).",
             "Inference: dependent wild bootstrap over calendar winters, default B=10,000; primary block length ceil(T^(1/3)), sensitivities 1, 3, 5. Centered 95% CIs and matching two-sided p-values. Bandwidths are audit-derived conventions, not validated optimal choices.",
             "Winter-pairs intervals use coefficient-specific support and are secondary; incomplete support gives null uncertainty. Where primary uncertainty is unavailable (unit_leverage), the pairs_winter interval is exploratory; assumes independent winters.",
             "For diagnostics.* only, intervals and p-values hold the learned expected-fog predictions fixed. They exclude first-stage estimation uncertainty and do not establish unconditional coverage for the complete weather-adjustment procedure.",
             "NOx requires 65% hourly coverage in each of five months; attribution models share rows. Gaussian OLS AIC counts the variance parameter.",
             "These are diagnostic associations, not causal attribution or operational forecast skill. Gray comparison uses observed−expected fog days per 10 ppb.", ""]
    lines.extend(["## Input provenance", "", "```json", json.dumps(
        {key: result.get(key) for key in ("weather_provenance", "visibility_mode", "analysis_environment")}, indent=2), "```", ""])
    for family, analysis in result["diagnostics"].items():
        lines.extend([f"## {family}", "", "### Panel and attribution", "", "```json", json.dumps({key: value for key, value in analysis.items() if key not in ("winters", "folds")}, indent=2), "```", "",
                      "### Winter observations and expectations", "", "| Site | Winter | Observed (all days) | Observed (≥18 h) | Expected | Days |", "|---|---:|---:|---:|---:|---:|"])
        panel = pd.DataFrame(analysis["winters"])
        for row in analysis["winters"]:
            lines.append(f"| {row['site']} | {row['winter']} | {row['observed_all_days']} | {row['observed']} | {row['expected']:.3f} | {row['n_days']} |")
        lines.extend(["", "### Diagnostic fold availability", "", "```json", json.dumps(analysis["folds"], indent=2), "```", ""])
        lines.extend(_coefficient_lines(analysis))
        lines.append("")
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        if len(panel):
            for site, part in panel.groupby("site"):
                axes[0].plot(part.winter, part.observed, label=f"{SITES.get(site, site)} observed")
                axes[0].plot(part.winter, part.expected, linestyle="--", label=f"{SITES.get(site, site)} expected")
                axes[1].plot(part.winter, part.y, marker=".", label=SITES.get(site, site))
            axes[0].legend(fontsize=7, ncol=2)
            axes[1].legend(fontsize=8)
        axes[0].set_ylabel("Fog days")
        axes[1].set_ylabel("log((observed+.5)/(expected+.5))")
        axes[1].set_xlabel("Winter (November year)")
        axes[1].axhline(0, color="gray", linewidth=0.5)
        fig.suptitle(f"{family}: leave-one-winter-out weather adjustment\n"
                     f"{result.get('weather_provenance', {}).get('source', 'source not recorded')}; "
                     f"visibility: {result.get('visibility_mode', 'minimum')}")
        fig.tight_layout()
        fig.savefig(out_dir / f"long_term_trend_v2_{family}.png", dpi=150)
        plt.close(fig)
    lines.extend(["## Coverage and measured NOx availability", "", "```json", json.dumps({"coverage": result["coverage"], "nox": result["nox_availability"]}, indent=2), "```", "",
                  "## ≤1-SM extension (not a replication of Gray's dense-fog analysis)", "", "Optimal: daily mean DPD <4.2 °C and daily mean wind <7.2 km/h. Undefined shares are null, not zero. Trends are reported separately by airport; no regional pooled rank test is reported.", "", "```json", json.dumps(result["gray_replication"], indent=2), "```", ""])
    if "cadence_sensitivity" in result:
        lines.extend(["## Reporting-cadence sensitivity", "",
                      "Eligible-cohort and paired-control analyses refit both diagnostics. Paired controls use only hours with known categories at both endpoints, with eligibility recomputed. Differences are descriptive, not paired-effect intervals. Routine-only support is the actual supported period, not a filled historical record.",
                      "", "```json", json.dumps(result["cadence_sensitivity"], indent=2), "```", ""])
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    gray = pd.DataFrame(result["gray_replication"]["winters"])
    if len(gray):
        for site, part in gray.groupby("site"):
            for axis, value in zip(axes, ("fog_share_of_optimal", "optimal_share_of_fog")):
                axis.plot(part.winter, part[value], marker=".", label=SITES.get(site, site))
        axes[0].legend(fontsize=8)
    axes[0].set_ylabel("Share of optimal days with fog")
    axes[1].set_ylabel("Share of fog days optimal")
    axes[1].set_xlabel("Winter (November year)")
    fig.suptitle(f"≤1-SM extension; {result.get('weather_provenance', {}).get('source', 'source not recorded')}; "
                 f"visibility: {result.get('visibility_mode', 'minimum')}")
    fig.tight_layout()
    fig.savefig(out_dir / "long_term_trend_v2_gray.png", dpi=150)
    plt.close(fig)


def _visibility(site, mode, *, root=ROOT):
    suffix = "" if mode == "minimum" else f"_{mode}"
    path = Path(root) / "datasets" / "visibility" / f"{site}_visibility_hourly{suffix}.csv"
    frame = pd.read_csv(path, parse_dates=["time"])
    if frame.time.isna().any() or frame.time.duplicated().any() or frame.time.dt.tz is not None:
        raise ValueError(f"{path}: expected unique fixed-PST naive hours")
    return frame, path


def _label_days(visibility, site, start_year, end_year):
    dates = pd.date_range(pd.Timestamp(start_year, 11, 1), pd.Timestamp(end_year + 1, 4, 1),
                          freq="D", inclusive="left")
    dates = dates[dates.month.isin(WINTER_MONTHS)]
    hours = (dates.to_numpy()[:, None] + np.arange(24)[None, :] * np.timedelta64(1, "h")).ravel()
    values = visibility.set_index("time").category.reindex(hours).to_numpy(dtype=float, na_value=np.nan).reshape(-1, 24)
    return pd.DataFrame({"site": site, "date": dates, "winter": winter_year(dates),
                         "known_hours": np.isfinite(values).sum(axis=1)})


def _weather(site, source, valid_winters, *, root=ROOT):
    if source == "archive_best_match":
        weather, _ = load_sources(site)
        return weather, {"source": source, "product": "historical unpinned archive"}
    path = Path(root) / "datasets" / "weather_era5_trend" / f"{site}_weather_era5_trend.csv"
    weather = pd.read_csv(path, parse_dates=["time"])
    if list(weather.columns) != ["time", *WEATHER_COLS]:
        raise ValueError(f"{path}: ERA5 weather columns do not match the required channels")
    if weather.time.isna().any() or weather.time.duplicated().any() or weather.time.dt.tz is not None or not weather.time.is_monotonic_increasing:
        raise ValueError(f"{path}: expected sorted unique fixed-PST naive hours")
    indexed = weather.set_index("time")
    for year in valid_winters:
        # Internal tables use November year; K14 files/requests use January year.
        hours = pd.date_range(pd.Timestamp(int(year), 11, 1), pd.Timestamp(int(year) + 1, 4, 1),
                              freq="h", inclusive="left")
        if not np.isfinite(indexed.reindex(hours)[WEATHER_COLS].to_numpy(dtype=float)).all():
            raise ValueError(f"{path}: missing or nonfinite ERA5 winter {int(year) + 1}")
    return weather, {"source": source, "model": "era5", "timezone": "Etc/GMT+8",
                     "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "request_winters": [int(year) + 1 for year in valid_winters]}


def _panel_analysis(panel, nox, *, n_boot, seed=SEED):
    panel.attrs.update(n_boot=n_boot, seed=seed)
    return {
        "panel": fit_panel(panel, n_boot=n_boot, seed=seed),
        "post_asos_1996": fit_panel(panel.loc[panel.winter >= 1996], n_boot=n_boot, seed=seed),
        "kvis_1992_step": fit_panel(panel, n_boot=n_boot, seed=seed, steps={**STEPS, "location_8": "1992-07-01"}),
        "per_site_trends": site_trends(panel, "y"),
        "attribution": attribution(panel, nox, n_boot=n_boot, seed=seed),
        "uncertainty_scope": "fixed_diagnostic_predictions", "first_stage_refit": False}


def _daily_analysis(daily, X, nox, *, n_boot, seed=SEED):
    daily.attrs.update(n_boot=n_boot, seed=seed)
    result = {"coverage": winter_validity(daily).to_dict("records"),
              "gray_replication": gray_replication(daily), "diagnostics": {}}
    for family in ("logistic_regression", "ebm"):
        probability, folds = diagnostic_predictions(daily, X, family, seed=seed)
        panel = winter_panel(daily, probability)
        result["diagnostics"][family] = {
            "folds": folds, "winters": panel.to_dict("records"),
            **_panel_analysis(panel, nox, n_boot=n_boot, seed=seed)}
    return result


def _refresh_inference(result, *, n_boot, seed):
    for analysis in result["diagnostics"].values():
        panel = pd.DataFrame(analysis["winters"])
        if panel.empty:
            panel = pd.DataFrame(columns=["site", "winter", "year", "y", "dpd_c", "fog_day_residual"])
        nox = pd.DataFrame(analysis.get("attribution", {}).get("common_rows", []),
                           columns=["site", "winter", "nox_ppb"])
        analysis.update(_panel_analysis(panel, nox, n_boot=n_boot, seed=seed))
    gray = result["gray_replication"]
    frame = pd.DataFrame(gray["winters"], columns=["site", "winter", "fog_share_of_optimal", "optimal_share_of_fog"])
    frame.attrs.update(n_boot=n_boot, seed=seed)
    gray["trends"] = {value: site_trends(frame, value)
                      for value in ("fog_share_of_optimal", "optimal_share_of_fog")}
    if "cadence_sensitivity" in result:
        sensitivity = result["cadence_sensitivity"]
        for endpoint in ("minimum", "selected"):
            _refresh_inference(sensitivity["paired_control"][endpoint], n_boot=n_boot, seed=seed)
        sensitivity["differences"] = _cadence_differences(sensitivity["paired_control"])


def _cadence_differences(paired):
    def compare(base, selected):
        if not isinstance(base, dict) or not isinstance(selected, dict):
            return None
        for key in ("estimate", "slope"):
            if key in base and key in selected:
                a, b = base[key], selected[key]
                return {"minimum": base, "selected": selected,
                        "delta_estimate": b - a if a is not None and b is not None else None}
        result = {}
        for key in sorted(base.keys() & selected.keys()):
            if key not in ("dependence_sensitivity", "folds", "winters"):
                difference = compare(base[key], selected[key])
                if difference:
                    result[key] = difference
        return result
    return compare(paired["minimum"], paired["selected"])


def _cadence_audit(frame, site):
    known = frame.loc[frame.category.notna()].copy()
    known["year"] = known.time.dt.year
    rows = []
    for year, part in known.groupby("year"):
        rows.append({"site": site, "year": int(year), "known_hours": len(part),
                     "report_type": part.report_type.fillna("").value_counts().to_dict() if "report_type" in part else {},
                     "obs_minute": part.obs_minute.value_counts().to_dict() if "obs_minute" in part else {}})
    return {"site_year_distributions": rows,
            "first_known_hour": str(known.time.min()) if len(known) else None,
            "last_known_hour": str(known.time.max()) if len(known) else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--weather-source", choices=("era5", "archive_best_match"), default="era5")
    parser.add_argument("--visibility-mode", choices=("minimum", "one_per_hour", "routine_only"), default="minimum")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--write-valid-winters", type=Path)
    args = parser.parse_args(argv)
    if args.n_boot < 1 or args.start_year > args.end_year:
        parser.error("positive --n-boot and start-year <= end-year required")
    if args.visibility_mode != "minimum" and (args.out_dir is None or args.out_dir.resolve() == OUT_DIR.resolve()):
        parser.error("nondefault --visibility-mode requires a separate explicit --out-dir")
    out_dir = args.out_dir or OUT_DIR
    if args.write_valid_winters:
        rows = []
        for site in SITES:
            visibility, _ = _visibility(site, args.visibility_mode)
            valid = winter_validity(_label_days(visibility, site, args.start_year, args.end_year))
            rows.extend({"site": site, "winter": int(year) + 1} for year in valid.loc[valid.valid, "winter"])
        args.write_valid_winters.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["site", "winter"]).to_csv(args.write_valid_winters, index=False)
        return
    environment = capture_environment()
    if args.inference_only:
        result = json.loads((out_dir / "long_term_trend_v2.json").read_text(encoding="utf-8"))
        _refresh_inference(result, n_boot=args.n_boot, seed=result.get("seed", SEED))
        result.update(n_boot=args.n_boot, analysis_environment=environment)
        write_outputs(result, out_dir)
        return
    tables, arrays, nox_tables, availability, provenance = [], [], [], {}, {}
    paired_tables = {"minimum": [], "selected": []}
    cadence = {}
    for site in SITES:
        visibility, _ = _visibility(site, args.visibility_mode)
        valid = winter_validity(_label_days(visibility, site, args.start_year, args.end_year))
        weather, provenance[site] = _weather(site, args.weather_source, valid.loc[valid.valid, "winter"])
        table, X = daily_table(weather, visibility, site, start_year=args.start_year, end_year=args.end_year)
        tables.append(table)
        arrays.append(X)
        nox, availability[site] = measured_nox(site)
        nox_tables.append(nox)
        if args.visibility_mode != "minimum":
            minimum, _ = _visibility(site, "minimum")
            paired = minimum[["time", "category"]].merge(visibility[["time", "category"]], on="time", suffixes=("_minimum", "_selected"))
            paired = paired.dropna(subset=["category_minimum", "category_selected"])
            for endpoint in paired_tables:
                labels = paired[["time", f"category_{endpoint}"]].rename(columns={f"category_{endpoint}": "category"})
                daily, _ = daily_table(weather, labels, site, start_year=args.start_year, end_year=args.end_year)
                paired_tables[endpoint].append(daily)
            cadence[site] = _cadence_audit(visibility, site)
    daily = pd.concat(tables, ignore_index=True)
    X = np.concatenate(arrays)
    nox = pd.concat(nox_tables, ignore_index=True)
    result = {"start_year": args.start_year, "end_year": args.end_year, "n_boot": args.n_boot, "seed": SEED,
              "analysis_environment": environment, "visibility_mode": args.visibility_mode,
              "weather_provenance": {"source": args.weather_source, "sites": provenance},
              "nox_availability": availability, **_daily_analysis(daily, X, nox, n_boot=args.n_boot)}
    if args.visibility_mode != "minimum":
        paired = {endpoint: _daily_analysis(pd.concat(parts, ignore_index=True), X, nox, n_boot=args.n_boot)
                  for endpoint, parts in paired_tables.items()}
        result["cadence_sensitivity"] = {"observation_process": cadence, "paired_control": paired,
                                          "differences": _cadence_differences(paired)}
    write_outputs(result, out_dir)


if __name__ == "__main__":
    main()
