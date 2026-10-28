"""Paired, equal-airport V2 analyses on saved fixed-PST predictions (D-054–067)."""
from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT, SEED, PRIMARY_FAMILIES
from .evaluation import _weighted_boot_metrics, pmf_to_fog, rps, safe_ap
from .labels import NEAR_SAT_DPD_C, NEAR_SAT_WIND_KMH
from .visibility import FOG_MAX_CATEGORY, N_CATEGORIES

TASKS = ("nowcast", "lead_time")
COLUMNS = ["site", "task", "issue_time", "fold", "c", "pmf", "dpd_c", "wind_kmh"]
CI_CONVENTION = (
    "Equal-weight mean of per-airport paired deltas; fixed-PST calendar days "
    "resampled jointly across airports. CI = estimate ± quantile(|draw − estimate|) "
    "at 0.95 (Bonferroni: 1 − 0.05/6). Two-sided p = "
    "(1 + count(|draw − estimate| ≥ |estimate|))/(valid draws + 1). "
    "AP gains are positive; RPS gains are negative. A pooled draw is valid only "
    "if every included airport has rows (and, for AP, positive labels); invalid "
    "draws are discarded jointly, never by dropping an airport from the mean."
)


def _family_dirs():
    from .study import FAMILY_DIRS
    return FAMILY_DIRS


def load_predictions(family, arm, replicate=0) -> pd.DataFrame:
    """Read compressed, non-pickled prediction arrays across sites and tasks (replicate > 0: D-081 seed refits)."""
    directory = _family_dirs()[family] / f"{family}_predictions_v2"
    if replicate:
        directory = directory / "replicates" / f"r{replicate}"
    frames = []
    for path in sorted(directory.glob(f"*__*__{arm}.npz")):
        site, task, saved_arm = path.stem.split("__")
        if task not in TASKS or saved_arm != arm:
            raise ValueError(f"unexpected prediction filename: {path}")
        with np.load(path, allow_pickle=False) as saved:
            c = saved["c"]
            pmf = saved["pmf"].astype(float)
            if pmf.shape != (len(c), N_CATEGORIES):
                raise ValueError(f"{path}: expected an n×{N_CATEGORIES} PMF")
            frames.append(pd.DataFrame({
                "site": site, "task": task,
                "issue_time": pd.to_datetime(saved["issue_time"], unit="ns"),
                "fold": saved["fold"], "c": c, "pmf": list(pmf),
                "dpd_c": saved["dpd_c"], "wind_kmh": saved["wind_kmh"],
            }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def near_saturation(frame):
    return (frame["dpd_c"] < NEAR_SAT_DPD_C) & (frame["wind_kmh"] < NEAR_SAT_WIND_KMH)


def _pmf(frame):
    values = (np.stack(frame["pmf"].to_numpy()) if "pmf" in frame
              else frame[[f"p{k}" for k in range(N_CATEGORIES)]].to_numpy())
    values = np.asarray(values, dtype=float)
    if (values.shape != (len(frame), N_CATEGORIES) or not np.isfinite(values).all()
            or (values < 0).any() or not np.allclose(values.sum(axis=1), 1., atol=.005, rtol=0)):
        raise ValueError("predictions must contain finite nonnegative 18-category PMFs")
    return values


def _align(a, b, mask=None):
    a, b = a.copy(), b.copy()
    for frame in (a, b):
        if "task" in frame and frame["task"].nunique() > 1:
            raise ValueError("analyze one task at a time")
        frame["issue_time"] = pd.to_datetime(frame["issue_time"])
        if frame["issue_time"].dt.tz is not None:
            raise ValueError("issue_time must use naive fixed PST")
        if frame[["site", "issue_time"]].isna().any().any():
            raise ValueError("pairing keys must not be missing")
        if frame.duplicated(["site", "issue_time"]).any():
            raise ValueError("duplicate (site, issue_time) prediction")
    if mask is not None:
        selected = mask(a) if callable(mask) else mask
        selected = np.asarray(selected)
        if selected.shape != (len(a),) or selected.dtype != bool:
            raise ValueError("mask must be a boolean per input-a row")
        a = a.loc[selected]
    a = a.set_index(["site", "issue_time"]).sort_index()
    b = b.set_index(["site", "issue_time"]).sort_index()
    common = a.index.intersection(b.index)
    if not len(common):
        raise ValueError("no common prediction rows")
    a, b = a.loc[common].reset_index(), b.loc[common].reset_index()
    for column in ("c", "fold", "task"):
        if column in a and column in b and not np.array_equal(a[column], b[column]):
            raise ValueError(f"paired predictions disagree on {column}")
    categories = a["c"].to_numpy()
    if not np.isin(categories, np.arange(N_CATEGORIES)).all():
        raise ValueError("category outside the ordinal visibility scale")
    return a, b


def _joint_day_counts(frame, n_boot, seed):
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    days, codes = np.unique(frame["issue_time"].dt.normalize(), return_inverse=True)
    counts = np.random.default_rng(seed).multinomial(
        len(days), np.full(len(days), 1 / len(days)), size=n_boot).astype(np.int32)
    return codes, counts


def _summary(estimate, draws):
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return {"estimate": float(estimate), "ci95": None, "ci_bonf": None,
                "p_two_sided": None, "valid_draws": 0}
    distance = np.abs(draws - estimate)
    def interval(level):
        radius = float(np.quantile(distance, level))
        return [float(estimate - radius), float(estimate + radius)]
    return {"estimate": float(estimate), "ci95": interval(.95),
            "ci_bonf": interval(1 - .05 / 6),
            "p_two_sided": float((1 + np.count_nonzero(distance >= abs(estimate))) / (len(draws) + 1)),
            "valid_draws": len(draws)}


def _loss_draws(loss, codes, counts):
    """Mean loss on each whole-day draw, without materializing repeated rows."""
    sizes = np.bincount(codes, minlength=counts.shape[1])
    totals = np.bincount(codes, weights=loss, minlength=counts.shape[1])
    denominator = counts @ sizes
    return np.divide(counts @ totals, denominator,
                     out=np.full(len(counts), np.nan), where=denominator > 0)


def pooled_delta(a: pd.DataFrame, b: pd.DataFrame, metric: str, n_boot: int,
                 seed: int, mask=None) -> dict:
    """Paired a−b: equal-airport estimand, not a row-pooled metric difference."""
    if metric not in {"ap", "rps"}:
        raise ValueError("metric must be ap or rps")
    input_sites = set(a["site"]) | set(b["site"])
    a, b = _align(a, b, mask)
    pa, pb = _pmf(a), _pmf(b)
    codes, counts = _joint_day_counts(a, n_boot, seed)
    per_airport, estimates, draws = {}, [], []
    for site, group in a.groupby("site", sort=True):
        ix = group.index.to_numpy()
        c, site_codes = group["c"].to_numpy(dtype=int), codes[ix]
        if metric == "ap":
            y = c <= FOG_MAX_CATEGORY
            estimate = safe_ap(y, pmf_to_fog(pa[ix])) - safe_ap(y, pmf_to_fog(pb[ix]))
            # The existing AP helper omits zero-positive draws; filter explicitly
            # so their original joint-draw positions remain paired across sites.
            positives = np.bincount(site_codes, weights=y, minlength=counts.shape[1])
            valid = counts @ positives > 0
            delta = np.full(n_boot, np.nan)
            if valid.any():
                aa, _ = _weighted_boot_metrics(y, pmf_to_fog(pa[ix]), site_codes, counts[valid])
                bb, _ = _weighted_boot_metrics(y, pmf_to_fog(pb[ix]), site_codes, counts[valid])
                delta[valid] = np.asarray(aa) - np.asarray(bb)
        else:
            loss = threshold_brier(c, pa[ix]).mean(axis=1) - threshold_brier(c, pb[ix]).mean(axis=1)
            estimate, delta = float(loss.mean()), _loss_draws(loss, site_codes, counts)
        summary = _summary(estimate, delta)
        per_airport[str(site)] = {"estimate": summary["estimate"], "ci95": summary["ci95"]}
        estimates.append(estimate)
        draws.append(delta)
    joint = np.asarray(draws)
    valid = np.isfinite(joint).all(axis=0)
    result = _summary(float(np.mean(estimates)), joint[:, valid].mean(axis=0))
    result.update(per_airport=per_airport, n_rows=len(a), n_days=counts.shape[1],
                  excluded_sites=sorted(input_sites - set(per_airport)))
    return result


def holm_adjust(p_values):
    """Holm's step-down adjusted p values, returned in original test order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p values must be finite and in [0, 1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted.tolist()


def rules_out_gain_0_02(result):
    return result["ci_bonf"] is not None and result["ci_bonf"][1] < .02


def threshold_brier(c, pmf):
    """The 17 threshold contributions whose row-mean is ordinal RPS."""
    pmf = np.asarray(pmf, dtype=float)
    scores = (np.asarray(c, dtype=int)[:, None] <= np.arange(N_CATEGORIES - 1))
    scores = (scores - pmf.cumsum(axis=1)[:, :-1]) ** 2
    assert np.isclose(scores.mean(), rps(c, pmf), rtol=0, atol=1e-12)
    return scores


def haze_decomposition(a, b, n_boot, seed):
    a, b = _align(a, b)
    loss = threshold_brier(a["c"], _pmf(a)) - threshold_brier(b["c"], _pmf(b))
    codes, counts = _joint_day_counts(a, n_boot, seed)
    estimates, draws = [], []
    for _, group in a.groupby("site", sort=True):
        ix = group.index.to_numpy()
        estimates.append(loss[ix].mean(axis=0))
        draws.append(np.column_stack([_loss_draws(loss[ix, k], codes[ix], counts)
                                      for k in range(N_CATEGORIES - 1)]))
    point = np.mean(estimates, axis=0)
    draws = np.asarray(draws)
    valid = np.isfinite(draws).all(axis=(0, 2))
    draws = draws[:, valid, :].mean(axis=0)
    rps_delta = pooled_delta(a, b, "rps", n_boot, seed)
    assert np.isclose(point.mean(), rps_delta["estimate"], rtol=0, atol=1e-12)
    return {"thresholds": [{"k": k, **_summary(point[k], draws[:, k])}
                           for k in range(N_CATEGORIES - 1)],
            "fog_0_4": _summary(point[:5].mean(), draws[:, :5].mean(axis=1)),
            "haze_5_15": _summary(point[5:16].mean(), draws[:, 5:16].mean(axis=1)),
            "mean_threshold_delta": float(point.mean()), "rps_delta": rps_delta["estimate"]}


def _adjust_tests(tests, key):
    # Missing/non-estimable tests retain their place in the correction family.
    adjusted = holm_adjust([row[key]["p_two_sided"] if row.get(key)
                            and row[key]["p_two_sided"] is not None else 1. for row in tests])
    for row, p in zip(tests, adjusted):
        if row.get(key):
            row[key]["p_holm"] = p if row[key]["p_two_sided"] is not None else None


def _compare(a, b, n_boot, seed, mask=None):
    if a.empty or b.empty:
        return {"status": "not_estimable", "reason": "paired prediction arm missing"}
    try:
        result = {metric: pooled_delta(a, b, metric, n_boot, seed, mask) for metric in ("ap", "rps")}
    except ValueError as exc:
        if str(exc) != "no common prediction rows":
            raise
        return {"status": "not_estimable", "reason": str(exc)}
    result["ap"]["rules_out_gain_0.02"] = rules_out_gain_0_02(result["ap"])
    return {"status": "ok", **result}


def _replicate_ids(family):
    root = _family_dirs()[family] / f"{family}_predictions_v2" / "replicates"
    return sorted(int(p.name[1:]) for p in root.glob("r*") if p.name[1:].isdigit() and int(p.name[1:]) > 0)


def _seed_summary(results, deterministic):
    """Spread of one primary AP test over replicates (replicate 0 = the pre-registered main run)."""
    if any(row["status"] != "ok" for row in results):
        return {"status": "not_estimable_in_some_replicates"}
    scores = [row["ap"] for row in results]
    estimates = np.array([score["estimate"] for score in scores])
    significant = [score["p_holm"] < .05 for score in scores]
    rules_out = [score["rules_out_gain_0.02"] for score in scores]
    return {"status": "ok", "training_seed": "deterministic fit" if deterministic else f"{len(scores)} seeds",
            "replicates": len(scores), "estimate_r0": scores[0]["estimate"], "ci_bonf_r0": scores[0]["ci_bonf"],
            "seed_mean": float(estimates.mean()), "seed_sd": float(estimates.std(ddof=1)),
            "seed_min": float(estimates.min()), "seed_max": float(estimates.max()),
            "holm_significant_r0": significant[0], "holm_agree": sum(s == significant[0] for s in significant),
            "rules_out_r0": rules_out[0], "rules_out_agree": sum(r == rules_out[0] for r in rules_out)}


def seed_sensitivity(n_boot=1000, seed=SEED):
    """D-081: the six primary NOx AP tests (all rows; near-saturation) recomputed for every training seed.

    Every replicate reruns the same Holm family on the same day resamples, so only the model seed differs.
    Families without replicate predictions (logistic regression: no training randomness) keep their single fit.
    """
    extra = {family: _replicate_ids(family) for family in PRIMARY_FAMILIES}
    ids = sorted(set().union(*extra.values()))
    if not ids:
        return None
    for family, found in extra.items():
        if found and found != ids:
            raise ValueError(f"{family}: replicates {found} differ from {ids}")
    cache, per_replicate = {}, []
    for replicate in [0] + ids:
        rows = {"all": [], "near_saturation": []}
        for family in PRIMARY_FAMILIES:
            key = (family, replicate if extra[family] else 0)
            if key not in cache:
                a, b = (load_predictions(family, f"nox_{arm}", key[1]) for arm in ("with", "without"))
                cache[key] = [(_compare(a[a.task == t], b[b.task == t], n_boot, seed),
                               _compare(a[a.task == t], b[b.task == t], n_boot, seed, near_saturation))
                              for t in TASKS]
            for full, near in cache[key]:
                # copies: Holm p depends on the replicate's other five tests
                rows["all"].append({**full, "ap": dict(full["ap"])} if full["status"] == "ok" else full)
                rows["near_saturation"].append({**near, "ap": dict(near["ap"])} if near["status"] == "ok" else near)
        for stratum in rows.values():
            _adjust_tests(stratum, "ap")
        per_replicate.append(rows)
    labels = [(family, task) for family in PRIMARY_FAMILIES for task in TASKS]
    tests = [{"family": family, "task": task, "stratum": stratum,
              **_seed_summary([rows[stratum][i] for rows in per_replicate], not extra[family])}
             for i, (family, task) in enumerate(labels) for stratum in ("all", "near_saturation")]
    return {"replicate_ids": [0] + ids, "verdict_rule": "robust if both verdicts agree in ≥90% of replicates",
            "tests": tests}


def analyze_aqi(n_boot=1000, seed=SEED):
    tests = []
    for family in _family_dirs():
        if family == "xgboost_pooled":
            continue
        for aq_set in ("nox", "pm25", "cams", "nox_monitor"):
            if aq_set == "nox_monitor" and family not in PRIMARY_FAMILIES:
                continue
            a = load_predictions(family, f"{aq_set}_with")
            b = load_predictions(family, f"{aq_set}_without")
            for task in TASKS:
                aa, bb = a[a.task == task], b[b.task == task]
                primary = aq_set == "nox" and family in PRIMARY_FAMILIES
                row = {"family": family, "set": aq_set, "task": task, "primary": primary,
                       **_compare(aa, bb, n_boot, seed)}
                if primary:
                    row["near_saturation"] = _compare(aa, bb, n_boot, seed, near_saturation)
                    if row["status"] == "ok":
                        row["haze"] = haze_decomposition(aa, bb, n_boot, seed)
                tests.append(row)
    primary = [row for row in tests if row["primary"]]
    for metric in ("ap", "rps"):
        _adjust_tests(primary, metric)
        _adjust_tests([row["near_saturation"] for row in primary], metric)
    return {"n_boot": n_boot, "seed": seed, "convention": CI_CONVENTION,
            "holm_family": "six primary NOx tests; near-saturation six tested separately; RPS secondary",
            "tests": tests, "seed_sensitivity": seed_sensitivity(n_boot, seed)}


def analyze_families(n_boot=1000, seed=SEED):
    families = list(_family_dirs()) + ["climatology"]
    tests = []
    for left, right in combinations(families, 2):
        a = load_predictions(left, "main")
        b = (load_predictions("logistic_regression", "climatology") if right == "climatology"
             else load_predictions(right, "main"))
        if a.empty or b.empty:
            raise FileNotFoundError(f"missing main predictions for {left} or {right}; run all V2 families first")
        for task in TASKS:
            tests.append({"family_a": left, "family_b": right, "task": task,
                          **_compare(a[a.task == task], b[b.task == task], n_boot, seed)})
    for task in TASKS:
        rows = [row for row in tests if row["task"] == task]
        for metric in ("ap", "rps"):
            _adjust_tests(rows, metric)
    return {"n_boot": n_boot, "seed": seed, "convention": CI_CONVENTION,
            "holm_family": "all family pairs within each task, separately for AP and secondary RPS",
            "climatology_source": "logistic_regression climatology arm", "tests": tests}


def _fmt(value):
    return "—" if value is None else f"{value:.4g}"


def _interval(value):
    return "—" if value is None else f"[{_fmt(value[0])}, {_fmt(value[1])}]"


def _table_row(label, row, metric, equivalence=False):
    score = row.get(metric)
    if not score:
        return f"| {label} | {metric.upper()} | not estimable | — | — | — | — | — |"
    interval = "ci_bonf" if equivalence else "ci95"
    verdict = str(score["rules_out_gain_0.02"]).lower() if equivalence and metric == "ap" else "—"
    return (f"| {label} | {metric.upper()} | {_fmt(score['estimate'])} | {_interval(score[interval])} | "
            f"{_fmt(score['p_two_sided'])} | {_fmt(score.get('p_holm'))} | "
            f"{score['n_rows']}/{score['n_days']}/{score['valid_draws']} | {verdict} |")


def aqi_markdown(result):
    lines = ["# Air-quality paired tests", "", result["convention"], "",
             "Primary/near-saturation intervals use α/6; all others use 95%. "
             "Rules-out = upper simultaneous AP bound < 0.02. RPS has no equivalence margin. "
             "Holm is applied to six primary tests, separately in each stratum and metric.", "",
             "| Family / set / task / stratum | Metric | Δ with−without | Interval | p | p Holm | Rows/days/draws | Rules-out |",
             "|---|---|---:|---|---:|---:|---:|---|"]
    for row in result["tests"]:
        label = f"{row['family']} / {row['set']} / {row['task']}"
        for metric in ("ap", "rps"):
            lines.append(_table_row(label + " / all", row, metric, row["primary"]))
            if row["primary"]:
                lines.append(_table_row(label + " / near-saturation", row["near_saturation"], metric, True))
    lines += ["", "## Primary NOx threshold-Brier decomposition", "",
              "Thresholds 0–4 are fog; 5–15 are haze. The mean of all 17 thresholds equals ΔRPS. "
              "Threshold and range intervals below are pointwise 95% (exploratory).", "",
              "| Family / task | Threshold or range | Δ Brier | 95% CI |", "|---|---|---:|---|"]
    for row in result["tests"]:
        if "haze" not in row:
            continue
        haze = row["haze"]
        scores = [(str(score["k"]), score) for score in haze["thresholds"]]
        scores += [(name, haze[name]) for name in ("fog_0_4", "haze_5_15")]
        for name, score in scores:
            lines.append(f"| {row['family']} / {row['task']} | {name} | {_fmt(score['estimate'])} | {_interval(score['ci95'])} |")
    lines += ["", "## Secondary per-airport primary-NOx estimates", "",
              "| Family / task | Airport | Metric | Δ | 95% CI |", "|---|---|---|---:|---|"]
    for row in result["tests"]:
        if row["primary"] and row["status"] == "ok":
            for metric in ("ap", "rps"):
                for site, score in row[metric]["per_airport"].items():
                    lines.append(f"| {row['family']} / {row['task']} | {site} | {metric.upper()} | {_fmt(score['estimate'])} | {_interval(score['ci95'])} |")
    seeds = result.get("seed_sensitivity")
    if seeds:
        lines += ["", "## Training-seed sensitivity (D-081)", "",
                  f"Primary NOx AP tests refit under {len(seeds['replicate_ids'])} model seeds (replicate 0 = the "
                  "result above), each with the same day resamples and its own Holm family. "
                  f"Agreement counts replicates matching replicate 0's verdict; {seeds['verdict_rule']}.", "",
                  "| Family / task / stratum | Training | ΔAP (r0) | Bonferroni CI (r0) | Seed mean ± SD | Seed range "
                  "| Holm-significant (r0): agree | Rules out +0.02 (r0): agree |",
                  "|---|---|---:|---|---:|---|---|---|"]
        for row in seeds["tests"]:
            label = f"{row['family']} / {row['task']} / {row['stratum']}"
            if row["status"] != "ok":
                lines.append(f"| {label} | {row['status']} | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {label} | {row['training_seed']} | {_fmt(row['estimate_r0'])} | {_interval(row['ci_bonf_r0'])} | "
                f"{_fmt(row['seed_mean'])} ± {_fmt(row['seed_sd'])} | {_interval([row['seed_min'], row['seed_max']])} | "
                f"{str(row['holm_significant_r0']).lower()}: {row['holm_agree']}/{row['replicates']} | "
                f"{str(row['rules_out_r0']).lower()}: {row['rules_out_agree']}/{row['replicates']} |")
    return "\n".join(lines) + "\n"


def family_markdown(result):
    lines = ["# Paired family significance", "", result["convention"], "",
             "Rows are aligned separately for each pair. Holm covers all pairs within each task, "
             "separately for AP and secondary RPS. Climatology uses the logistic-regression saved baseline.", "",
             "| Family A − family B / task | Metric | Δ | 95% CI | p | p Holm | Rows/days/draws | Rules-out |",
             "|---|---|---:|---|---:|---:|---:|---|"]
    for row in result["tests"]:
        for metric in ("ap", "rps"):
            lines.append(_table_row(f"{row['family_a']} − {row['family_b']} / {row['task']}", row, metric))
    return "\n".join(lines) + "\n"


def write_analysis(n_boot=1000, seed=SEED):
    directory = ROOT / "models" / "comparison_v2"
    directory.mkdir(parents=True, exist_ok=True)
    for name, analyze, markdown in (("aqi_tests_v2", analyze_aqi, aqi_markdown),
                                     ("family_significance_v2", analyze_families, family_markdown)):
        result = analyze(n_boot, seed)
        (directory / f"{name}.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        (directory / f"{name}.md").write_text(markdown(result), encoding="utf-8")
