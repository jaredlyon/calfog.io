"""Independent annual-cluster oracles; no data acquisition or fitting."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from models._shared import evaluation as ev, significance as sig


def _annual_predictions():
    rows = []
    for site, hours in (("a", 2), ("b", 6)):
        for year in (2018, 2020, 2023):
            for hour in range(hours):
                p = (.8 if hour % 2 == 0 else .2) if year != 2020 else .5
                rows.append({"site": site, "task": "nowcast", "issue_time": pd.Timestamp(year, 11, 1, hour),
                             "fold": 1 if site == "a" else 3, "c": 0 if hour % 2 == 0 else 17,
                             "pmf": np.r_[p, np.zeros(16), 1-p], "dpd_c": 3., "wind_kmh": 5.})
    a = pd.DataFrame(rows)
    b = a.copy(); b["pmf"] = [np.r_[.5, np.zeros(16), .5] for _ in rows]
    return a, b


def test_fog_year_boundaries_and_calendar_gaps():
    meta = pd.DataFrame({"issue_time": pd.to_datetime(["2020-06-30 23:00", "2020-07-01 00:00", "2020-12-31 00:00",
                                                      "2021-01-01 00:00", "2024-02-29 00:00"]), "fold": [1,2,2,3,3]})
    codes, counts = ev._fog_year_draw_counts(meta, 50, 7)
    np.testing.assert_array_equal(codes, [0,1,1,1,2])
    expected = np.random.default_rng(7).multinomial(3, [1/3]*3, size=50)
    np.testing.assert_array_equal(counts, expected)
    np.testing.assert_array_equal(ev._block_draw_counts(meta,50,7,"fog_year")[1], expected)


def test_fog_year_joint_airport_multiplicities(monkeypatch):
    a, b = _annual_predictions()
    seen = []
    original = sig._weighted_boot_metrics
    def spy(y, p, codes, counts):
        seen.append((codes.copy(), counts.copy()))
        return original(y,p,codes,counts)
    monkeypatch.setattr(sig, "_weighted_boot_metrics", spy)
    result = sig.pooled_delta(a,b,"ap",40,31,block="fog_year")
    assert len(seen) == 4
    for _, counts in seen:
        np.testing.assert_array_equal(counts, seen[0][1])
    assert result["n_days"] == 3 and result["n_clusters"] == 3
    assert result["clusters"] == {"a": 3, "b": 3}
    assert result["estimate"] == sig.pooled_delta(a,b,"ap",40,31)["estimate"]


@pytest.mark.parametrize("metric", ["ap", "rps"])
def test_annual_weighting_equals_physical_row_repetition(metric):
    a,b = _annual_predictions()
    codes, counts = ev._fog_year_draw_counts(a,35,9)
    draws = []
    for count in counts:
        deltas = []
        for site in ("a","b"):
            ix = np.flatnonzero(a.site.to_numpy() == site)
            selected = np.repeat(ix, count[codes[ix]])
            aa,bb = a.iloc[selected],b.iloc[selected]
            if metric == "ap":
                y = aa.c.to_numpy() <= 4
                score = lambda f: average_precision_score(y, np.stack(f.pmf)[:,:5].sum(axis=1,dtype=np.float64))
                deltas.append(score(aa)-score(bb))
            else:
                observed = aa.c.to_numpy()[:,None] <= np.arange(17)
                loss = lambda f: np.mean((np.stack(f.pmf).cumsum(axis=1)[:,:17]-observed)**2)
                deltas.append(loss(aa)-loss(bb))
        draws.append(np.mean(deltas))
    result = sig.pooled_delta(a,b,metric,35,9,block="fog_year")
    radius = np.quantile(abs(np.asarray(draws)-result["estimate"]),.95)
    np.testing.assert_allclose(result["ci95"], [result["estimate"]-radius,result["estimate"]+radius],rtol=0,atol=1e-12)
    decomposition = sig.haze_decomposition(a,b,35,9,block="fog_year")
    assert decomposition["mean_threshold_delta"] == pytest.approx(sig.pooled_delta(a,b,"rps",35,9,block="fog_year")["estimate"])


def test_one_cluster_is_unavailable_in_all_interval_helpers():
    a,b = _annual_predictions()
    a,b = a[a.issue_time.dt.year == 2018],b[b.issue_time.dt.year == 2018]
    for metric in ("ap","rps"):
        result = sig.pooled_delta(a,b,metric,20,7,block="fog_year")
        assert result["ci95"] is None and result["ci_bonf"] is None
        assert result["inference_status"] == "not_available"
    pmf = np.stack(a.pmf); other = np.stack(b.pmf); y = a.c.to_numpy() <= 4
    result = ev.block_bootstrap(y,ev.pmf_to_fog(pmf),a,20,7,"fog_year")
    assert result["ap_ci"] is result["roc_auc_ci"] is None
    result = ev.ordinal_bootstrap(a.c,pmf,a,20,7,"fog_year")
    assert result["rps_ci"] is None
    assert ev.paired_delta_bootstrap(y,ev.pmf_to_fog(pmf),ev.pmf_to_fog(other),a,20,7,"fog_year")[1] is None
    assert ev.paired_rps_delta_bootstrap(a.c,pmf,other,a,20,7,"fog_year")[1] is None


def test_pmf_fog_sum_preserves_float32_origin_ranks():
    pmf = np.zeros((2,18),np.float32)
    pmf[:,:2] = [[.25,.25],[.25,np.nextafter(np.float32(.25),np.float32(np.inf))]]
    pmf[:,17] = 1-pmf[:,:2].sum(axis=1)
    scores = ev.pmf_to_fog(pmf)
    assert scores.dtype == np.float64 and scores[1] > scores[0]
    np.testing.assert_array_equal(scores,ev.pmf_to_fog(pmf.astype(np.float64)))
    assert average_precision_score([0,1],scores) == 1


def test_robustness_dispatch_writes_only_annual_output(tmp_path, monkeypatch):
    directory = tmp_path / "models" / "comparison_v2"; directory.mkdir(parents=True)
    daily = {"n_boot": 9, "seed": 3, "convention": "day", "tests": []}
    sentinels = {}
    for name in ("aqi_tests_v2","family_significance_v2"):
        for suffix in ("json","md"):
            path = directory / f"{name}.{suffix}"
            content = json.dumps(daily) if suffix == "json" else "unchanged"
            path.write_text(content); sentinels[path] = path.read_bytes()
    calls = []
    def analyze(n,seed,*,block):
        calls.append((n,seed,block)); return {**daily,"tests":[],"n_boot":n,"block":block}
    monkeypatch.setattr(sig,"ROOT",tmp_path)
    monkeypatch.setattr(sig,"capture_environment",lambda: {"sentinel":"environment"})
    monkeypatch.setattr(sig,"analyze_aqi",analyze)
    monkeypatch.setattr(sig,"analyze_families",analyze)
    monkeypatch.setattr(sig,"_per_site_intervals",lambda seed: [{"site":"a","n_clusters":3}])
    sig.write_analysis(block_robustness=True, robustness_n_boot=37)
    assert calls == [(37,sig.SEED,"fog_year")]*2
    assert all(path.read_bytes() == content for path,content in sentinels.items())
    result = json.loads((directory / "block_robustness_v2.json").read_text())
    assert set(result) == {"block","n_boot","seed","clusters","aqi","family","per_site","analysis_environment"}
    assert result["analysis_environment"] == {"sentinel":"environment"}
    assert result["clusters"] == {"a":3}


def test_block_sensitive_flags_nominal_disagreement():
    def row(p,upper):
        return {"family":"lr","task":"nowcast","set":"nox","primary":True,
                "ap":{"p_holm":p,"ci_bonf":[-.01,upper]},"rps":{"p_holm":.5}}
    daily,annual = {"tests":[row(.04,.019)]},{"tests":[row(.06,.021)]}
    sig._mark_block_sensitive(annual,daily,("family","set","task"))
    assert annual["tests"][0]["ap"]["block_sensitive"] is True
    assert annual["tests"][0]["rps"]["block_sensitive"] is False


def test_saved_prediction_interval_pass_preserves_scores_and_calibration_free_inputs(monkeypatch):
    a,b = _annual_predictions()
    original = [p.copy() for p in a.pmf]
    monkeypatch.setattr(sig, "_family_dirs", lambda: {"example": None})
    monkeypatch.setattr(sig, "N_BOOT", 20)
    def load(family,arm):
        frame = a if arm in {"main","nox_with"} else b
        if arm not in {"main","climatology","nox_with","nox_without"}:
            return pd.DataFrame(columns=sig.COLUMNS)
        return frame.copy()
    monkeypatch.setattr(sig,"load_predictions",load)
    result = sig._per_site_intervals(7)
    assert len(result) == 10  # four marginal arms × two sites, two paired site deltas
    main = next(row for row in result if row.get("arm") == "main" and row["site"] == "a")
    frame = a[a.site == "a"]
    assert main["ap"] == average_precision_score(frame.c <= 4,ev.pmf_to_fog(np.stack(frame.pmf)))
    assert main["rps"] == ev.rps(frame.c,np.stack(frame.pmf))
    assert main["n_boot"] == 20 and main["n_clusters"] == 3
    paired = next(row for row in result if row.get("set") == "nox")
    assert paired["delta_ap_ci"] is not None and paired["delta_rps_ci"] is not None
    for before,after in zip(original,a.pmf):
        np.testing.assert_array_equal(before,after)


def test_sparse_airport_with_one_fog_year_invalidates_joint_inference():
    a,b = _annual_predictions()
    keep = (a.site == "a") | (a.issue_time.dt.year == 2018)
    result = sig.pooled_delta(a[keep],b[keep],"rps",30,7,block="fog_year")
    assert result["clusters"] == {"a":3,"b":1}
    assert result["ci95"] is None and result["valid_draws"] == 0
