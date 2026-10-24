"""Checks for the preserved elbow rule, matched cohorts and descriptive reports."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

spec = importlib.util.spec_from_file_location(
    "kmeans_aqi_v2", Path(__file__).with_name("kmeans_aqi_sensitivity_analysis_v2.py")
)
kmeans = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kmeans)


def test_select_k_preserves_v1_offset():
    # Decreases = [20%, 50%, 10%]; second differences = [-30, 40].
    # V1 selects k_values[argmax(...) + 2] = k_values[3] = 6, not the bend at 5.
    assert kmeans.select_k([3, 4, 5, 6], [100.0, 80.0, 40.0, 36.0]) == 6


def test_matched_cohorts_have_identical_hour_indices_after_cleaning():
    times = pd.date_range("2024-07-01", periods=8, freq="h")
    weather = pd.DataFrame({"time": times})
    for column in kmeans.WEATHER_COLS + kmeans.AQI_COLS:
        weather[column] = np.arange(8, dtype=float)
    weather.loc[1, kmeans.WEATHER_COLS[0]] = np.nan
    weather.loc[2, kmeans.AQI_COLS[0]] = np.nan
    weather.loc[3, kmeans.AQI_COLS[-1]] = np.nan
    weather.loc[4, kmeans.AQI_COLS[1]] = np.inf
    cohorts = kmeans.build_cohorts(weather.sample(frac=1, random_state=42))
    assert cohorts["without_aqi"].index.equals(pd.DatetimeIndex(times.delete(1), name="time"))
    expected = pd.DatetimeIndex(times[[0, 5, 6, 7]], name="time")
    assert cohorts["with_aqi"].index.equals(expected)
    assert cohorts["without_aqi_reduced"].index.equals(expected)
    pd.testing.assert_frame_equal(
        cohorts["without_aqi_reduced"], cohorts["with_aqi"][kmeans.WEATHER_COLS]
    )
    assert list(cohorts["without_aqi"].columns) == kmeans.WEATHER_COLS


def test_fog_is_descriptive_and_report_retains_v1_layout(tmp_path):
    values, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.2, random_state=42)
    times = pd.date_range("2024-01-01", periods=len(values), freq="h", name="time")
    features = pd.DataFrame(values, index=times, columns=["weather_1", "weather_2"])
    fog = pd.Series(np.arange(len(values)) % 2, index=times, dtype=float)
    fog.iloc[0] = np.nan
    result = kmeans.analyze_dataset_kmeans(features, fog, 3)
    changed_labels = kmeans.analyze_dataset_kmeans(features, 1 - fog, 3)
    assert result["total_records"] == 120
    assert result["fog_rates"].records.sum() == 120
    assert result["fog_rates"].labeled_records.sum() == 119
    for metric in kmeans.METRICS:
        assert np.isclose(result[metric], changed_labels[metric], rtol=1e-12)
    pd.testing.assert_frame_equal(result["feature_variance"], changed_labels["feature_variance"])
    assert np.isclose(result["feature_variance"].variance_pct.sum(), 100)
    with_aqi = dict(result, inertia=result["inertia"] + 2)
    comparison = kmeans.compare_results(result, with_aqi)
    assert np.isclose(comparison["inertia"], 2)
    report = tmp_path / "kmeans_v2.txt"
    kmeans.write_report([{
        "location": "10", "without_aqi": result, "with_aqi": with_aqi,
        "without_aqi_reduced": result, "comparison": comparison,
    }], 3, report)
    text = report.read_text(encoding="utf-8")
    assert "Selected k: 3" in text
    assert "including its +2 offset" in text
    assert "LOCATION 10\n" + "-" * 80 in text
    for title in kmeans.COHORTS.values():
        assert f"{title}:" in text
        assert f"{title} - Average across 1 locations:" in text
    assert "AVERAGE METRICS" in text
    assert "MATCHED METRIC DELTAS (with-AQI minus without-AQI reduced):" in text
    assert "  Inertia: +2.0000" in text
    assert "Descriptive cluster fog rates" in text

def test_main_uses_shared_hourly_fog_labels(monkeypatch):
    times=pd.date_range("2024-01-01",periods=6,freq="h")
    weather=pd.DataFrame({"time":times,**{c:np.arange(6,dtype=float) for c in kmeans.WEATHER_COLS+kmeans.AQI_COLS}})
    visibility=pd.DataFrame({"time":times[[1,3]],"category":[4,5],"fog":[True,False]})
    loaded=[];scored=[];reports=[]
    monkeypatch.setattr(kmeans,"load_sources",lambda site:(loaded.append(site) or weather,visibility))
    monkeypatch.setattr(kmeans,"find_optimal_clusters",lambda X:(3,[],[],[]))
    def analyze(features,fog,k):
        pd.testing.assert_series_equal(fog,visibility.set_index("time")["fog"])
        assert k==3
        scored.append(features)
        return {"total_records":len(features),**{key:1. for key in kmeans.METRICS}}
    monkeypatch.setattr(kmeans,"analyze_dataset_kmeans",analyze)
    monkeypatch.setattr(kmeans,"write_report",lambda results,k:reports.append((results,k)))
    kmeans.main(["--sites","location_10"])
    assert loaded==["location_10"] and len(scored)==3
    assert reports[0][1]==3
