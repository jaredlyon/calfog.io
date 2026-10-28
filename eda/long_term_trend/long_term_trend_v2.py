"""Weather-adjusted Nov–Mar fog trends (D-057–D-063, D-066).

Run from the repository root: python eda/long_term_trend/long_term_trend_v2.py
Uses realized daily weather, not an operational forecast. All dates are fixed PST.
No air-quality or year variables enter the diagnostic fog classifiers.
"""
from __future__ import annotations

import argparse
import calendar
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


def fit_panel(panel, predictors=("year",), *, response="y", n_boot=2000, seed=SEED, steps=STEPS):
    """OLS with airport intercepts/steps and paired winter-block inference."""
    columns = list(predictors) + [response]
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=columns)
    base = {"n_rows": len(panel), "n_winters": int(panel.winter.nunique())}
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
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        index = winter_bootstrap_indices(panel.winter.to_numpy(), rng)
        draw = _ols(X[index], y[index])
        if draw is not None:
            draws.append(draw[0])
    draws = np.asarray(draws).reshape(-1, len(names))
    coefficients = {}
    for index, name in enumerate(names):
        sample = draws[:, index]
        coefficients[name] = {"estimate": float(coef[index]),
                              "ci95": np.quantile(sample, [0.025, 0.975]).tolist() if len(sample) else None,
                              "p_two_sided": float((1 + np.count_nonzero(np.abs(sample - coef[index]) >= abs(coef[index]))) / (len(sample) + 1)) if len(sample) else None}
    return {**base, "status": "ok", "coefficients": coefficients, "r_squared": r2, "aic": aic,
            "valid_draws": len(draws), "n_boot": n_boot, "omitted_absorbed_steps": omitted}


def monotone_trend(years, values):
    years, values = np.asarray(years, float), np.asarray(values, float)
    keep = np.isfinite(years) & np.isfinite(values)
    years, values = years[keep], values[keep]
    if len(values) < 3 or len(np.unique(years)) < 2:
        return {"status": "not_estimable", "n": len(values)}
    slope = stats.theilslopes(values, years, alpha=0.95)
    tau = stats.kendalltau(years, values)
    return {"status": "ok", "n": len(values), "slope": float(slope.slope),
            "ci95": [float(slope.low_slope), float(slope.high_slope)],
            "kendall_tau": float(tau.statistic), "mann_kendall_p": float(tau.pvalue)}


def site_trends(frame, value, *, pooled=False):
    result = {site: monotone_trend(part.winter, part[value]) for site, part in frame.groupby("site")}
    if pooled and len(frame):
        subset = frame.dropna(subset=[value]).copy()
        # Within-site centering of both axes handles unequal airport records.
        years = subset.winter - subset.groupby("site").winter.transform("mean")
        values = subset[value] - subset.groupby("site")[value].transform("mean")
        result["pooled_site_demeaned"] = monotone_trend(years, values)
    return result


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


def attribution(panel, nox, *, n_boot=2000, seed=SEED):
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
            fit["fog_days_per_10ppb"] = {"estimate": 10 * coef["estimate"],
                                           "ci95": [10 * x for x in coef["ci95"]] if coef["ci95"] else None,
                                           "p_two_sided": coef["p_two_sided"]}
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
    trends = {value: site_trends(table, value, pooled=True)
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


def write_outputs(result, out_dir):
    """Persist machine-readable results, complete Markdown tables and figures."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "long_term_trend_v2"
    result = _json_safe(result)
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Long-term weather-adjusted fog study", "", "Fixed PST; Nov–Mar winter labeled by November year. Fog = any known hour in categories 0–4.",
             "Every month needs 90% known hourly labels. Diagnostic LR/EBM use realized 00–23 weather, calendar phase and site only; leave one whole winter out across all sites.",
             "Expected and observed use the same days with ≥18 known hours. y = log((observed+0.5)/(expected+0.5)).",
             "Inference: 2,000 joint winter-block draws by default; percentile 95% CIs and centered two-sided bootstrap p. Singular resamples are excluded and valid draw counts are reported.",
             "NOx requires 65% hourly coverage in each of five months; attribution models share rows. Gaussian OLS AIC counts the variance parameter.",
             "These are diagnostic associations, not causal attribution or operational forecast skill. Gray comparison uses observed−expected fog days per 10 ppb.", ""]
    for family, analysis in result["diagnostics"].items():
        lines.extend([f"## {family}", "", "### Panel and attribution", "", "```json", json.dumps({key: value for key, value in analysis.items() if key not in ("winters", "folds")}, indent=2), "```", "",
                      "### Winter observations and expectations", "", "| Site | Winter | Observed (all days) | Observed (≥18 h) | Expected | Days |", "|---|---:|---:|---:|---:|---:|"])
        panel = pd.DataFrame(analysis["winters"])
        for row in analysis["winters"]:
            lines.append(f"| {row['site']} | {row['winter']} | {row['observed_all_days']} | {row['observed']} | {row['expected']:.3f} | {row['n_days']} |")
        lines.extend(["", "### Diagnostic fold availability", "", "```json", json.dumps(analysis["folds"], indent=2), "```", ""])
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
        fig.suptitle(f"{family}: leave-one-winter-out weather adjustment")
        fig.tight_layout()
        fig.savefig(out_dir / f"long_term_trend_v2_{family}.png", dpi=150)
        plt.close(fig)
    lines.extend(["## Coverage and measured NOx availability", "", "```json", json.dumps({"coverage": result["coverage"], "nox": result["nox_availability"]}, indent=2), "```", "",
                  "## Gray optimal-day replication (1980–2026 by default)", "", "Optimal: daily mean DPD <4.2 °C and daily mean wind <7.2 km/h. Undefined shares are null, not zero. Pooled trends demean both axes within site.", "", "```json", json.dumps(result["gray_replication"], indent=2), "```", ""])
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
    fig.tight_layout()
    fig.savefig(out_dir / "long_term_trend_v2_gray.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    if args.n_boot < 1 or args.start_year > args.end_year:
        parser.error("positive --n-boot and start-year <= end-year required")
    tables, arrays, nox_tables, availability = [], [], [], {}
    for site in SITES:
        weather, visibility = load_sources(site)
        table, X = daily_table(weather, visibility, site, start_year=args.start_year, end_year=args.end_year)
        tables.append(table)
        arrays.append(X)
        nox, availability[site] = measured_nox(site)
        nox_tables.append(nox)
    daily = pd.concat(tables, ignore_index=True)
    X = np.concatenate(arrays)
    nox = pd.concat(nox_tables, ignore_index=True)
    result = {"start_year": args.start_year, "end_year": args.end_year, "n_boot": args.n_boot, "seed": SEED,
              "coverage": winter_validity(daily).to_dict("records"),
              "nox_availability": availability, "gray_replication": gray_replication(daily), "diagnostics": {}}
    for family in ("logistic_regression", "ebm"):
        probability, folds = diagnostic_predictions(daily, X, family)
        panel = winter_panel(daily, probability)
        result["diagnostics"][family] = {
            "folds": folds, "winters": panel.to_dict("records"),
            "panel": fit_panel(panel, n_boot=args.n_boot),
            "post_asos_1996": fit_panel(panel.loc[panel.winter >= 1996], n_boot=args.n_boot),
            "kvis_1992_step": fit_panel(panel, n_boot=args.n_boot, steps={**STEPS, "location_8": "1992-07-01"}),
            "per_site_trends": site_trends(panel, "y"),
            "attribution": attribution(panel, nox, n_boot=args.n_boot),
        }
    write_outputs(result, args.out_dir)


if __name__ == "__main__":
    main()
