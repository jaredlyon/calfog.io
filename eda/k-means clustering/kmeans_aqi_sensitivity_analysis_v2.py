"""Unsupervised hourly AQI comparison; GHCNh ordinal-label fog rates never fit clusters."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from models._shared.data import AQI_COLS, SITES, WEATHER_COLS, load_sources

COHORTS = {
    "without_aqi": "WITHOUT AQI (all rows, no AQI features)",
    "with_aqi": "WITH AQI (filtered rows, with AQI features)",
    "without_aqi_reduced": "WITHOUT AQI REDUCED (filtered rows, no AQI features)",
}
METRICS = {
    "inertia": "Inertia",
    "silhouette_score": "Silhouette Score",
    "davies_bouldin_index": "Davies-Bouldin Index",
}
OUTPUT = Path(__file__).with_suffix(".txt")


def select_k(k_values, inertias):
    """Apply V1's rule verbatim, including its +2 index offset."""
    rate_of_change = [
        (previous - current) / previous * 100 if previous else 0.0
        for previous, current in zip(inertias, inertias[1:])
    ]
    if len(rate_of_change) > 1:
        elbow_idx = int(np.argmax(np.subtract(rate_of_change[:-1], rate_of_change[1:]))) + 2
    else:
        elbow_idx = len(k_values) // 2
    return k_values[elbow_idx]


def find_optimal_clusters(X_scaled, k_range=range(3, 21)):
    k_values = list(k_range)
    if not k_values or len(X_scaled) <= max(k_values):
        raise ValueError("Elbow analysis requires more rows than the largest candidate k")
    inertias = [
        KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_
        for k in k_values
    ]
    rate_of_change = [
        (previous - current) / previous * 100 if previous else 0.0
        for previous, current in zip(inertias, inertias[1:])
    ]
    return select_k(k_values, inertias), inertias, k_values, rate_of_change


def build_cohorts(weather):
    """Select complete features once, then remove AQI without changing rows."""
    hourly = weather.set_index("time", verify_integrity=True).sort_index()
    full = hourly[WEATHER_COLS].replace([np.inf, -np.inf], np.nan).dropna()
    with_aqi = hourly.loc[full.index, WEATHER_COLS + AQI_COLS]
    with_aqi = with_aqi.replace([np.inf, -np.inf], np.nan).dropna()
    reduced = with_aqi[WEATHER_COLS].dropna()
    assert reduced.index.equals(with_aqi.index), "Matched AQI cohorts must have identical indices"
    return {"without_aqi": full, "with_aqi": with_aqi, "without_aqi_reduced": reduced}


def analyze_dataset_kmeans(features, fog, n_clusters):
    """Fit only standardized features; use observed fog solely for the note table."""
    if len(features) <= n_clusters:
        raise ValueError(f"Need more than {n_clusters} complete rows for clustering metrics")
    scaled = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(scaled)
    if len(np.unique(clusters)) < 2:
        raise ValueError("Clustering metrics require at least two distinct clusters")
    standardized = pd.DataFrame(scaled, index=features.index, columns=features.columns)
    cluster_means = standardized.groupby(clusters).mean()
    variances = cluster_means.var(ddof=0).rename("variance_across_clusters")
    feature_variance = variances.rename_axis("feature").reset_index()
    total = variances.sum()
    feature_variance["variance_pct"] = feature_variance.variance_across_clusters / total * 100 if total else 0.0
    feature_variance = feature_variance.sort_values("variance_across_clusters", ascending=False)
    notes = pd.DataFrame({"cluster": clusters, "fog": fog.reindex(features.index)})
    fog_rates = notes.groupby("cluster").agg(
        records=("fog", "size"), labeled_records=("fog", "count"), fog_rate=("fog", "mean")
    )
    return {
        "total_records": len(features),
        "feature_count": len(features.columns),
        "features": list(features.columns),
        "n_clusters": n_clusters,
        "inertia": model.inertia_,
        "silhouette_score": silhouette_score(
            scaled, clusters, sample_size=min(5000, len(scaled)), random_state=42
        ),
        "davies_bouldin_index": davies_bouldin_score(scaled, clusters),
        "feature_variance": feature_variance,
        "fog_rates": fog_rates,
    }


def compare_results(without_aqi_results, with_aqi_results):
    """Matched-row changes: with AQI minus without AQI (reduced)."""
    return {key: with_aqi_results[key] - without_aqi_results[key] for key in METRICS}


def write_report(all_results, selected_k, output_file=OUTPUT):
    """Keep V1's location/cohort/average layout, adding k, matched deltas and notes."""
    with Path(output_file).open("w", encoding="utf-8") as stream:
        stream.write(f"Selected k: {selected_k} (V1 elbow rule; first site's full weather cohort)\n\n")
        stream.write("V1 elbow rule is kept as-is, including its +2 offset.\n\n")
        for result in all_results:
            stream.write(f"LOCATION {result['location']}\n" + "-" * 80 + "\n")
            for cohort, title in COHORTS.items():
                values = result[cohort]
                stream.write(f"{title}:\n  Records: {values['total_records']}\n")
                for key, label in METRICS.items():
                    digits = 2 if key == "inertia" else 4
                    stream.write(f"  {label}: {values[key]:.{digits}f}\n")
                variance = values["feature_variance"]
                stream.write(f"\n  Feature Variance Across Clusters (sorted, {len(variance)} features):\n")
                for row in variance.itertuples(index=False):
                    stream.write(f"    {row.feature}: {row.variance_pct:.2f}%\n")
                stream.write("\n  Descriptive cluster fog rates (hourly GHCNh category <=4; not fitting inputs):\n")
                stream.write("    Cluster  Records  Labeled records  Fog rate\n")
                for cluster, row in values["fog_rates"].iterrows():
                    rate = f"{row.fog_rate:.4f}" if pd.notna(row.fog_rate) else "NA"
                    stream.write(f"    {cluster}  {int(row.records)}  {int(row.labeled_records)}  {rate}\n")
                stream.write("\n")
            stream.write("MATCHED METRIC DELTAS (with-AQI minus without-AQI reduced):\n")
            for key, label in METRICS.items():
                stream.write(f"  {label}: {result['comparison'][key]:+.4f}\n")
            stream.write("\n" + "=" * 80 + "\n\n")
        stream.write("AVERAGE METRICS\n" + "-" * 80 + "\n")
        for cohort, title in COHORTS.items():
            stream.write(f"{title} - Average across {len(all_results)} locations:\n")
            for key, label in METRICS.items():
                digits = 2 if key == "inertia" else 4
                average = np.mean([result[cohort][key] for result in all_results])
                stream.write(f"  {label}: {average:.{digits}f}\n")
            name = title.split(" (")[0]
            stream.write(f"\nAVERAGE FEATURE VARIANCE {name} (across {len(all_results)} locations):\n")
            variance = pd.concat([result[cohort]["feature_variance"] for result in all_results])
            average = variance.groupby("feature").variance_pct.mean().sort_values(ascending=False)
            for feature, value in average.items():
                stream.write(f"  {feature}: {value:.2f}%\n")
            stream.write("\n")
        stream.write("AVERAGE MATCHED METRIC DELTAS (with-AQI minus without-AQI reduced):\n")
        for key, label in METRICS.items():
            average = np.mean([result["comparison"][key] for result in all_results])
            stream.write(f"  {label}: {average:+.4f}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", choices=tuple(SITES), default=sorted(SITES))
    args = parser.parse_args(argv)
    all_results = []
    selected_k = None
    for site in args.sites:
        weather, visibility = load_sources(site)
        cohorts = build_cohorts(weather)
        if selected_k is None:
            scaled = StandardScaler().fit_transform(cohorts["without_aqi"])
            selected_k, inertias, k_values, rates = find_optimal_clusters(scaled)
            print(f"Elbow method on {site} full weather cohort:")
            print(f"{'K':<6}{'Inertia':<18}{'% Decrease':<15}")
            for i, k in enumerate(k_values):
                decrease = "-" if i == 0 else f"{rates[i - 1]:.2f}"
                print(f"{k:<6}{inertias[i]:<18.2f}{decrease:<15}")
            print(f"Using k={selected_k} clusters for all analyses")
        fog = visibility.set_index("time", verify_integrity=True)["fog"]
        result = {"location": site.removeprefix("location_")}
        print(f"\n{site}")
        for cohort, features in cohorts.items():
            values = analyze_dataset_kmeans(features, fog, selected_k)
            result[cohort] = values
            print(f"  {COHORTS[cohort]}: {values['total_records']} records")
            for key, label in METRICS.items():
                print(f"    {label}: {values[key]:.4f}")
        result["comparison"] = compare_results(result["without_aqi_reduced"], result["with_aqi"])
        all_results.append(result)
    write_report(all_results, selected_k)
    print(f"Results saved to {OUTPUT}")


if __name__ == "__main__":
    main()
