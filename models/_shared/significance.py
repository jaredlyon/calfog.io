"""Paired, equal-airport V2 analyses on saved fixed-PST predictions (D-054–067)."""
from __future__ import annotations

from itertools import combinations
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import binomtest, quantile_test

from .data import ROOT, SEED, PRIMARY_FAMILIES, SITES, N_BOOT
from .artifacts import canonical_hash, load_completion, read_verified
from .environment import capture_environment
from .evaluation import (_weighted_boot_metrics, _fog_year_draw_counts, pmf_to_fog, rps, safe_ap,
                         block_bootstrap, ordinal_bootstrap, paired_delta_bootstrap,
                         paired_rps_delta_bootstrap)
from .labels import NEAR_SAT_DPD_C, NEAR_SAT_WIND_KMH
from .visibility import FOG_MAX_CATEGORY, N_CATEGORIES

TASKS = ("nowcast", "lead_time")
COLUMNS = ["site", "task", "issue_time", "fold", "c", "pmf", "dpd_c", "wind_kmh"]
DETERMINISTIC_FAMILIES = frozenset({"logistic_regression"})
CORE_ARMS = frozenset({"main", "climatology"} | {
    f"{name}_{arm}" for name in ("nox", "pm25", "cams", "nox_monitor")
    for arm in ("with", "without")})
MC_CONFIDENCE = 1 - .05 / 12
BENCHMARK_TEXT = (
    'The upper end of each simultaneous interval is the largest gain the data do not exclude. '
    'No practical-significance margin is claimed. "Bound < +0.02 benchmark" compares that upper end '
    'with +0.02 AP, the benchmark registered before the first analysis (D-052); it is owner-selected, '
    'has no literature or cost basis (D-102), and was kept after superseded results had been seen. '
    'RPS has no benchmark.'
)
CI_CONVENTION = (
    "Equal-weight mean of per-airport paired deltas; fixed-PST calendar days "
    "resampled jointly across airports. CI = estimate ± quantile(|draw − estimate|) "
    "at 0.95 (Bonferroni: 1 − 0.05/6). Two-sided p = "
    "(1 + count(|draw − estimate| ≥ |estimate|))/(valid draws + 1). "
    "AP gains are positive; RPS gains are negative. A pooled draw is valid only "
    "if every included airport has rows (and, for AP, positive labels); invalid "
    "draws are discarded jointly, never by dropping an airport from the mean."
)

def _convention(block):
    return (CI_CONVENTION if block == "day" else
            CI_CONVENTION.replace("calendar days", "July–June fog years") +
            " Exploratory: few seasonal clusters, unequal/partial records; nominal coverage is not established.")


def _mc_decision(point, band, threshold):
    if point is None or band is None:
        return "unavailable"
    if band[0] < threshold <= band[1]:
        return "MC-unresolved"
    return "yes" if point < threshold else "no"


def _benchmark(score):
    if score.get("ci_bonf") is None:
        return "unavailable"
    if score.get("ci_bonf_upper_mc_interval") is None:
        return "MC-unresolved"
    return _mc_decision(score["ci_bonf"][1], score["ci_bonf_upper_mc_interval"], .02)


def _family_dirs():
    from .study import FAMILY_DIRS
    return FAMILY_DIRS


def _core_inventory(family):
    directory = _family_dirs()[family]
    record = load_completion(directory, family)
    if not record or record.get("family") != family or record.get("sites") != sorted(SITES):
        raise ValueError(f"{family}: missing completion record or five-site roster")
    metrics_path = directory / f"{family}_metrics_v2.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if canonical_hash(metrics) != record.get("metrics_sha256"):
        raise ValueError(f"{family}: metrics sha256 mismatch")
    if metrics.get("family") != family or set(metrics.get("per_site", {})) != set(SITES):
        raise ValueError(f"{family}: metrics family or roster mismatch")
    expected = set()
    for site, sd in metrics["per_site"].items():
        for task in TASKS:
            for name, arm in ((family, "main"), ("climatology", "climatology")):
                result = sd["tasks"][task][name]
                status = result.get("status", "ok")
                if status == "ok":
                    expected.add(f"{family}_predictions_v2/{site}__{task}__{arm}.npz")
                elif status not in {"not_estimable", "not_applicable"} or not result.get("reason"):
                    raise ValueError(f"{family}: invalid main status {status}")
            for name in ("nox", "pm25", "cams") + (("nox_monitor",) if family in PRIMARY_FAMILIES else ()):
                result = sd["aqi_ablation"][name][task][family]
                status = result.get("status", "ok")
                if status == "ok":
                    expected.update(f"{family}_predictions_v2/{site}__{task}__{name}_{arm}.npz"
                                    for arm in ("with", "without"))
                elif status not in {"not_estimable", "not_applicable"} or not result.get("reason"):
                    raise ValueError(f"{family}: invalid AQ status {status}")
    inventory = record.get("prediction_sha256", {})
    actual = {p.relative_to(directory).as_posix()
              for p in (directory / f"{family}_predictions_v2").glob("*__*__*.npz")
              if p.stem.split("__")[-1] in CORE_ARMS}
    if set(inventory) != expected or actual != expected:
        raise ValueError(f"{family}: core prediction inventory mismatch")
    return inventory


def load_predictions(family, arm, replicate=0) -> pd.DataFrame:
    """Read compressed, non-pickled prediction arrays across sites and tasks (replicate > 0: D-081 seed refits)."""
    directory = _family_dirs()[family] / f"{family}_predictions_v2"
    if replicate:
        directory = directory / "replicates" / f"r{replicate}"
    frames = []
    inventory = _core_inventory(family) if not replicate and arm in CORE_ARMS else None
    for path in sorted(directory.glob(f"*__*__{arm}.npz")):
        site, task, saved_arm = path.stem.split("__")
        if task not in TASKS or saved_arm != arm:
            raise ValueError(f"unexpected prediction filename: {path}")
        relative = path.relative_to(_family_dirs()[family]).as_posix()
        source = BytesIO(read_verified(path, inventory[relative])) if inventory is not None else path
        with np.load(source, allow_pickle=False) as saved:
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

def _joint_counts(frame, n_boot, seed, block):
    if block == "fog_year":
        return _fog_year_draw_counts(frame, n_boot, seed)
    if block != "day":
        raise ValueError("block must be day or fog_year")
    return _joint_day_counts(frame, n_boot, seed)



def _summary(estimate, draws):
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    result = {"estimate": float(estimate), "ci95": None, "ci_bonf": None,
              "p_two_sided": None, "valid_draws": len(draws),
              "p_mcse": None, "p_mc_interval": None, "ci_bonf_upper_mc_interval": None,
              "mc_confidence": MC_CONFIDENCE,
              "inference_status": "ok" if len(draws) else "not_available"}
    if not len(draws):
        return result
    distance = np.abs(draws - estimate)
    def interval(level):
        radius = float(np.quantile(distance, level))
        return [float(estimate - radius), float(estimate + radius)]
    n = len(draws)
    k = int(np.count_nonzero(distance >= abs(estimate)))
    rate = k / n
    p_ci = binomtest(k, n).proportion_ci(MC_CONFIDENCE, method="exact")
    upper_ci = quantile_test(distance, p=1 - .05 / 6).confidence_interval(MC_CONFIDENCE)
    result.update(ci95=interval(.95), ci_bonf=interval(1 - .05 / 6),
                  p_two_sided=float((1 + k) / (n + 1)),
                  p_mcse=float(np.sqrt(n * rate * (1-rate)) / (n+1)),
                  p_mc_interval=[float(p_ci.low), float(p_ci.high)])
    endpoints = np.asarray([upper_ci.low, upper_ci.high], dtype=float) + estimate
    if np.isfinite(endpoints).all():
        result["ci_bonf_upper_mc_interval"] = endpoints.tolist()
    return result


def _loss_draws(loss, codes, counts):
    """Mean loss on each whole-day draw, without materializing repeated rows."""
    sizes = np.bincount(codes, minlength=counts.shape[1])
    totals = np.bincount(codes, weights=loss, minlength=counts.shape[1])
    denominator = counts @ sizes
    return np.divide(counts @ totals, denominator,
                     out=np.full(len(counts), np.nan), where=denominator > 0)


def pooled_delta(a: pd.DataFrame, b: pd.DataFrame, metric: str, n_boot: int,
                 seed: int, mask=None, *, block="day") -> dict:
    """Paired a−b: equal-airport estimand, not a row-pooled metric difference."""
    if metric not in {"ap", "rps"}:
        raise ValueError("metric must be ap or rps")
    input_sites = set(a["site"]) | set(b["site"])
    a, b = _align(a, b, mask)
    pa, pb = _pmf(a), _pmf(b)
    codes, counts = _joint_counts(a, n_boot, seed, block)
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
        clusters = int(np.unique(site_codes).size)
        summary = _summary(estimate, [] if block == "fog_year" and clusters < 2 else delta)
        per_airport[str(site)] = {"estimate": summary["estimate"], "ci95": summary["ci95"],
                                  "n_clusters": clusters, "inference_status": summary["inference_status"]}
        if block == "fog_year" and clusters < 2:
            delta[:] = np.nan
        estimates.append(estimate)
        draws.append(delta)
    joint = np.asarray(draws)
    valid = np.isfinite(joint).all(axis=0)
    result = _summary(float(np.mean(estimates)), joint[:, valid].mean(axis=0))
    result.update(per_airport=per_airport, n_rows=len(a),
                  n_days=int(a.issue_time.dt.normalize().nunique()),
                  n_clusters=counts.shape[1], block=block,
                  clusters={site: row["n_clusters"] for site, row in per_airport.items()},
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


def haze_decomposition(a, b, n_boot, seed, *, block="day"):
    a, b = _align(a, b)
    loss = threshold_brier(a["c"], _pmf(a)) - threshold_brier(b["c"], _pmf(b))
    codes, counts = _joint_counts(a, n_boot, seed, block)
    estimates, draws = [], []
    for _, group in a.groupby("site", sort=True):
        ix = group.index.to_numpy()
        estimates.append(loss[ix].mean(axis=0))
        site_draws = np.column_stack([_loss_draws(loss[ix, k], codes[ix], counts)
                                     for k in range(N_CATEGORIES - 1)])
        if block == "fog_year" and np.unique(codes[ix]).size < 2:
            site_draws[:] = np.nan
        draws.append(site_draws)
    point = np.mean(estimates, axis=0)
    draws = np.asarray(draws)
    valid = np.isfinite(draws).all(axis=(0, 2))
    draws = draws[:, valid, :].mean(axis=0)
    rps_delta = pooled_delta(a, b, "rps", n_boot, seed, block=block)
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
    bounds = []
    for endpoint in (0, 1):
        bounds.append(holm_adjust([
            row[key]["p_mc_interval"][endpoint]
            if row.get(key) and row[key].get("p_mc_interval") is not None else 1.
            for row in tests]))
    for i, row in enumerate(tests):
        if row.get(key):
            row[key]["p_holm_mc_interval"] = (
                [bounds[0][i], bounds[1][i]] if row[key].get("p_mc_interval") is not None else None)


def _compare(a, b, n_boot, seed, mask=None, *, block="day"):
    if a.empty or b.empty:
        return {"status": "not_estimable", "reason": "paired prediction arm missing"}
    try:
        result = {metric: pooled_delta(a, b, metric, n_boot, seed, mask, block=block) for metric in ("ap", "rps")}
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
    significant = [score["p_holm"] < .05 if score["p_holm"] is not None else None for score in scores]
    rules_out = [score["rules_out_gain_0.02"] for score in scores]
    return {"status": "ok", "training_seed": "deterministic fit" if deterministic else f"{len(scores)} seeds",
            "replicates": len(scores), "estimate_r0": scores[0]["estimate"], "ci_bonf_r0": scores[0]["ci_bonf"],
            "seed_mean": float(estimates.mean()), "seed_sd": float(estimates.std(ddof=1)),
            "seed_min": float(estimates.min()), "seed_max": float(estimates.max()),
            "holm_significant_r0": significant[0], "holm_agree": sum(s == significant[0] for s in significant),
            "rules_out_r0": rules_out[0], "rules_out_agree": sum(r == rules_out[0] for r in rules_out),
            "holm_mc_unresolved": sum(_mc_decision(s.get("p_holm"), s.get("p_holm_mc_interval"), .05) == "MC-unresolved" for s in scores),
            "benchmark_mc_unresolved": sum(_benchmark(s) == "MC-unresolved" for s in scores)}


def seed_sensitivity(n_boot=50000, seed=SEED, *, block="day"):
    """D-081: the six primary NOx AP tests (all rows; near-saturation) recomputed for every training seed.

    Every replicate uses the same block resamples and its own Holm family.
    Only the explicitly deterministic LR family reuses its main fit.
    """
    extra = {family: _replicate_ids(family) for family in PRIMARY_FAMILIES
             if family not in DETERMINISTIC_FAMILIES}
    ids = sorted(set().union(*extra.values()))
    if not ids:
        return None
    for family, found in extra.items():
        if found != ids:
            raise ValueError(f"{family}: replicates {found} differ from {ids}")
    cache, per_replicate = {}, []
    for replicate in [0] + ids:
        rows = {"all": [], "near_saturation": []}
        for family in PRIMARY_FAMILIES:
            key = (family, 0 if family in DETERMINISTIC_FAMILIES else replicate)
            if key not in cache:
                a, b = (load_predictions(family, f"nox_{arm}", key[1]) for arm in ("with", "without"))
                cache[key] = [(_compare(a[a.task == t], b[b.task == t], n_boot, seed, block=block),
                               _compare(a[a.task == t], b[b.task == t], n_boot, seed, near_saturation, block=block))
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
              **_seed_summary([rows[stratum][i] for rows in per_replicate], family in DETERMINISTIC_FAMILIES)}
             for i, (family, task) in enumerate(labels) for stratum in ("all", "near_saturation")]
    return {"replicate_ids": [0] + ids, "verdict_rule": "robust if both verdicts agree in ≥90% of replicates",
            "tests": tests}


def analyze_aqi(n_boot=50000, seed=SEED, *, block="day"):
    tests = []
    for family in _family_dirs():
        if family == "xgboost_pooled":
            continue
        for aq_set in ("nox", "pm25", "cams", "nox_monitor", "nox_lag"):
            if aq_set in ("nox_monitor", "nox_lag") and family not in PRIMARY_FAMILIES:
                continue
            a = load_predictions(family, f"{aq_set}_with")
            b = load_predictions(family, f"{aq_set}_without")
            for task in TASKS:
                aa, bb = a[a.task == task], b[b.task == task]
                primary = aq_set == "nox" and family in PRIMARY_FAMILIES
                # D-097: lag-aware NOx tests mirror the primary design as their own Holm family.
                lag_aware = aq_set == "nox_lag"
                row = {"family": family, "set": aq_set, "task": task, "primary": primary,
                       "lag_aware": lag_aware, **_compare(aa, bb, n_boot, seed, block=block)}
                if primary or lag_aware:
                    row["near_saturation"] = _compare(aa, bb, n_boot, seed, near_saturation, block=block)
                if primary and row["status"] == "ok":
                    row["haze"] = haze_decomposition(aa, bb, n_boot, seed, block=block)
                tests.append(row)
    for group in ("primary", "lag_aware"):
        rows = [row for row in tests if row[group]]
        for metric in ("ap", "rps"):
            _adjust_tests(rows, metric)
            _adjust_tests([row["near_saturation"] for row in rows], metric)
    return {"n_boot": n_boot, "seed": seed, "block": block, "convention": _convention(block),
            "holm_family": "six primary NOx tests; near-saturation six tested separately; the six lag-aware "
                           "NOx tests (nox_lag) and their near-saturation six as two further families; RPS secondary",
            "tests": tests, "seed_sensitivity": seed_sensitivity(n_boot, seed, block=block)}


def analyze_families(n_boot=10000, seed=SEED, *, block="day"):
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
                          **_compare(a[a.task == task], b[b.task == task], n_boot, seed, block=block)})
    for task in TASKS:
        rows = [row for row in tests if row["task"] == task]
        for metric in ("ap", "rps"):
            _adjust_tests(rows, metric)
    return {"n_boot": n_boot, "seed": seed, "block": block, "convention": _convention(block),
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
    verdict = _benchmark(score) if equivalence and metric == "ap" else "—"
    if score.get("block_sensitive"):
        label += " (block-sensitive)"
    return (f"| {label} | {metric.upper()} | {_fmt(score['estimate'])} | {_interval(score[interval])} | "
            f"{_fmt(score['p_two_sided'])} (MC {_interval(score.get('p_mc_interval'))}) | "
            f"{_fmt(score.get('p_holm'))} (MC {_interval(score.get('p_holm_mc_interval'))}; "
            f"{_mc_decision(score.get('p_holm'), score.get('p_holm_mc_interval'), .05)}) | "
            f"{score['n_rows']}/{score['n_days']}/{score['valid_draws']} | {verdict} "
            f"(upper MC {_interval(score.get('ci_bonf_upper_mc_interval'))}) |")


def aqi_markdown(result):
    lines = ["# Air-quality paired tests", "", result["convention"], "",
             "Primary, lag-aware (nox_lag) and near-saturation intervals use α/6; all others use 95%. "
             + BENCHMARK_TEXT + " "
             "Holm is applied to six primary tests, separately in each stratum and metric; the six lag-aware "
             "tests (models also given the last visibility report, D-097) form their own Holm families.", "",
             "NOx conditional on monitor ID (sensitivity): both arms contain identical monitor controls; "
             "only the with-arm adds NOx history.", "",
             "| Family / set / task / stratum | Metric | Δ with−without | Interval | p (MC band) | p Holm (MC band) | Rows/days/draws | Bound < +0.02 benchmark |",
             "|---|---|---:|---|---:|---:|---:|---|"]
    for row in result["tests"]:
        label = f"{row['family']} / {row['set']} / {row['task']}"
        equivalence = row["primary"] or row.get("lag_aware", False)
        for metric in ("ap", "rps"):
            lines.append(_table_row(label + " / all", row, metric, equivalence))
            if equivalence:
                lines.append(_table_row(label + " / near-saturation", row["near_saturation"], metric, True))
    lines += ["", "## Primary NOx threshold-Brier decomposition", "",
              "Each k scores the cumulative event C≤k on all paired cases. k=4 is reported visibility ≤1 SM "
              "(the study fog event). Range rows are arithmetic means of threshold-Brier changes, not "
              "disjoint observation strata or additive ΔRPS contributions. All intervals below are "
              "exploratory, pointwise 95%, not simultaneous.", "",
              "| Family / task | Threshold or range | Δ Brier | 95% CI |", "|---|---|---:|---|"]
    for row in result["tests"]:
        if "haze" not in row:
            continue
        haze = row["haze"]
        scores = [(f"C≤{score['k']}" + (" (≤1 SM; fog event)" if score["k"] == 4 else ""), score)
                  for score in haze["thresholds"]]
        scores += [("mean over k=0–4", haze["fog_0_4"]), ("mean over k=5–15", haze["haze_5_15"])]
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
                  "result above), each with the same block resamples and its own Holm family. "
                  f"Nominal agreement counts replicates matching replicate 0's verdict; {seeds['verdict_rule']}. "
                  "MC-unresolved replicate counts are reported separately.", "",
                  "| Family / task / stratum | Training | ΔAP (r0) | Bonferroni CI (r0) | Seed mean ± SD | Seed range "
                  "| Holm-significant (r0): nominal agree; MC-unresolved | Bound < +0.02 benchmark (r0): agree; MC-unresolved |",
                  "|---|---|---:|---|---:|---|---|---|"]
        for row in seeds["tests"]:
            label = f"{row['family']} / {row['task']} / {row['stratum']}"
            if row["status"] != "ok":
                lines.append(f"| {label} | {row['status']} | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {label} | {row['training_seed']} | {_fmt(row['estimate_r0'])} | {_interval(row['ci_bonf_r0'])} | "
                f"{_fmt(row['seed_mean'])} ± {_fmt(row['seed_sd'])} | {_interval([row['seed_min'], row['seed_max']])} | "
                f"{str(row['holm_significant_r0']).lower()}: {row['holm_agree']}/{row['replicates']}; "
                f"MC-unresolved {row['holm_mc_unresolved']} | "
                f"{str(row['rules_out_r0']).lower()}: {row['rules_out_agree']}/{row['replicates']}; "
                f"MC-unresolved {row['benchmark_mc_unresolved']} |")
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


def _pm25_manifests():
    manifests = {}
    expected = [(family, site, check) for family in PRIMARY_FAMILIES for site in SITES
                for check in ("pm25_lag24", "pm25_fem")]
    for family, site, check in expected:
        path = _family_dirs()[family] / f"{family}_predictions_v2" / f"{site}__{check}.environment.json"
        if path.exists():
            manifests[family, site, check] = json.loads(path.read_text(encoding="utf-8"))
    if manifests and len(manifests) != len(expected):
        raise RuntimeError("pm25 robustness predictions incomplete")
    return manifests


def analyze_pm25_robustness(n_boot=10000, seed=SEED, *, block="day"):
    manifests = _pm25_manifests()
    if not manifests:
        return None
    tests = []
    for family in PRIMARY_FAMILIES:
        for check in ("pm25_lag24", "pm25_fem"):
            arms = {arm: load_predictions(family, f"{check}_{arm}")
                    for arm in ("with", "reference", "without")}
            for task in TASKS:
                expected_sites = set()
                for site in SITES:
                    manifest = manifests[family, site, check]
                    entry = manifest["tasks"][task]
                    status = entry["status"]
                    filenames = {f"{site}__{task}__{check}_{arm}.npz" for arm in arms}
                    listed = set(manifest["prediction_files"])
                    if status == "ok":
                        if not filenames <= listed:
                            raise RuntimeError("pm25 robustness predictions incomplete")
                        expected_sites.add(site)
                    elif status != "not_estimable" or not entry.get("reason") or filenames & listed:
                        raise RuntimeError("pm25 robustness predictions incomplete")
                selected = {arm: frame[frame.task == task] for arm, frame in arms.items()}
                keys = None
                for frame in selected.values():
                    if set(frame.site) != expected_sites:
                        raise RuntimeError("pm25 robustness predictions incomplete")
                    current = frame.sort_values(["site", "issue_time"])[["site", "issue_time", "fold", "c"]].reset_index(drop=True)
                    if keys is not None and not current.equals(keys):
                        raise ValueError("pm25 robustness arms disagree on paired keys")
                    keys = current
                for contrast, left, right in (
                        ("candidate_minus_baseline", "with", "without"),
                        ("reference_minus_baseline", "reference", "without"),
                        ("candidate_minus_reference", "with", "reference")):
                    result = _compare(selected[left], selected[right], n_boot, seed, block=block)
                    if result.get("ap"):
                        result["ap"].pop("rules_out_gain_0.02")
                    tests.append({"family": family, "check": check, "task": task,
                                  "contrast": contrast, **result})
    return {"n_boot": n_boot, "seed": seed, "block": block, "tests": tests,
            "convention": _convention(block) + " Exploratory pointwise 95% intervals; no primary Holm or benchmark verdict."}


def pm25_markdown(result):
    lines = ["# PM2.5 robustness (exploratory)", "", result["convention"], "",
             "| Family / check / task / contrast | Metric | Δ | 95% CI | p | p Holm | Rows/days/draws | Benchmark |",
             "|---|---|---:|---|---:|---:|---:|---|"]
    for row in result["tests"]:
        for metric in ("ap", "rps"):
            lines.append(_table_row(f"{row['family']} / {row['check']} / {row['task']} / {row['contrast']}", row, metric))
    return "\n".join(lines) + "\n"


def _per_site_intervals(seed, *, block="fog_year"):
    rows = []
    for family in _family_dirs():
        frames = {}
        for arm in sorted(CORE_ARMS):
            frame = load_predictions(family, arm)
            frames[arm] = frame
            for (site, task), group in frame.groupby(["site", "task"], sort=True):
                pmf = _pmf(group)
                y = group.c.to_numpy() <= FOG_MAX_CATEGORY
                binary = block_bootstrap(y, pmf_to_fog(pmf), group, N_BOOT, seed, block)
                ordinal = ordinal_bootstrap(group.c, pmf, group, N_BOOT, seed, block)
                rows.append({"family": family, "arm": arm, "site": site, "task": task,
                             **binary, **ordinal, "n_boot": N_BOOT})
        for name in ("nox", "pm25", "cams", "nox_monitor"):
            a, b = frames[f"{name}_with"], frames[f"{name}_without"]
            for (site, task), group in a.groupby(["site", "task"], sort=True):
                other = b[(b.site == site) & (b.task == task)]
                aa, bb = _align(group, other)
                pa, pb = _pmf(aa), _pmf(bb)
                y = aa.c.to_numpy() <= FOG_MAX_CATEGORY
                ap, ap_ci, ap_valid = paired_delta_bootstrap(y, pmf_to_fog(pa), pmf_to_fog(pb), aa, N_BOOT, seed, block)
                rp, rp_ci, rp_valid = paired_rps_delta_bootstrap(aa.c, pa, pb, aa, N_BOOT, seed, block)
                clusters = len(np.unique(_fog_year_draw_counts(aa, 1, seed)[0]))
                rows.append({"family": family, "set": name, "site": site, "task": task,
                             "delta_ap": ap, "delta_ap_ci": ap_ci, "valid_ap_resamples": ap_valid,
                             "delta_rps": rp, "delta_rps_ci": rp_ci, "valid_rps_resamples": rp_valid,
                             "n_boot": N_BOOT, "n_clusters": clusters, "bootstrap_unit": block,
                             "inference_status": "ok" if clusters >= 2 else "not_available"})
    return rows


def _mark_block_sensitive(annual, daily, keys):
    previous = {tuple(row[k] for k in keys): row for row in daily["tests"]}
    for row in annual["tests"]:
        reference = previous[tuple(row[k] for k in keys)]
        for current, old in ((row, reference), (row.get("near_saturation"), reference.get("near_saturation"))):
            if not current or not old:
                continue
            for metric in ("ap", "rps"):
                if not current.get(metric) or not old.get(metric):
                    continue
                score, baseline = current[metric], old[metric]
                hp, dp = score.get("p_holm"), baseline.get("p_holm")
                changed = hp is not None and dp is not None and (hp < .05) != (dp < .05)
                if metric == "ap" and (row.get("primary", False) or row.get("lag_aware", False)):
                    changed |= rules_out_gain_0_02(score) != rules_out_gain_0_02(baseline)
                score["block_sensitive"] = bool(changed)


def block_markdown(result):
    return ("# Fog-year block robustness\n\nExploratory July–June clusters; few clusters and partial records "
            "limit inference. Block-sensitive marks nominal Holm or benchmark disagreement with calendar days.\n\n"
            + aqi_markdown(result["aqi"]) + "\n" + family_markdown(result["family"])
            + "\n## Per-site saved-prediction intervals\n\n```json\n"
            + json.dumps(result["per_site"], indent=2, allow_nan=False) + "\n```\n")


def write_analysis(n_boot=50000, seed=SEED, family_n_boot=10000, include_families=True, *,
                   block_robustness=False, robustness_n_boot=10000, pm25_n_boot=10000):
    environment = capture_environment()
    directory = ROOT / "models" / "comparison_v2"
    directory.mkdir(parents=True, exist_ok=True)
    if block_robustness:
        aqi = analyze_aqi(robustness_n_boot, seed, block="fog_year")
        family = analyze_families(robustness_n_boot, seed, block="fog_year")
        for name, result, keys in (("aqi_tests_v2", aqi, ("family", "set", "task")),
                                   ("family_significance_v2", family, ("family_a", "family_b", "task"))):
            daily = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
            _mark_block_sensitive(result, daily, keys)
            result["analysis_environment"] = environment
        per_site = _per_site_intervals(seed)
        clusters = {}
        for row in per_site:
            clusters[row["site"]] = max(clusters.get(row["site"], 0), row["n_clusters"])
        result = {"block": "fog_year", "n_boot": robustness_n_boot, "seed": seed,
                  "clusters": clusters, "aqi": aqi, "family": family, "per_site": per_site,
                  "analysis_environment": environment}
        outputs = [("block_robustness_v2", result, block_markdown)]
    else:
        # Compute every requested table before writing any output.
        pm25 = analyze_pm25_robustness(pm25_n_boot, seed) if include_families else None
        aqi = analyze_aqi(n_boot, seed)
        if include_families and pm25 is None:
            aqi["pm25_robustness"] = "not_run"
        outputs = [("aqi_tests_v2", aqi, aqi_markdown)]
        if include_families:
            outputs.append(("family_significance_v2", analyze_families(family_n_boot or 10000, seed), family_markdown))
            if pm25 is not None:
                outputs.append(("pm25_robustness_v2", pm25, pm25_markdown))
    for name, result, markdown in outputs:
        result["analysis_environment"] = environment
        (directory / f"{name}.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        (directory / f"{name}.md").write_text(markdown(result), encoding="utf-8")
