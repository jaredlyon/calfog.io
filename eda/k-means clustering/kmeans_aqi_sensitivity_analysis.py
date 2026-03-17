import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from datetime import datetime

def find_optimal_clusters(X_scaled, k_range=range(3, 21)):
    """Use elbow method to find optimal number of clusters"""
    inertias = []
    silhouette_scores = []
    k_values = list(k_range)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette score (skip if too many samples for efficiency)
        if len(X_scaled) < 10000:
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled))))
        else:
            silhouette_scores.append(None)
    
    # Calculate rate of change in inertia
    rate_of_change = []
    for i in range(1, len(inertias)):
        pct_decrease = (inertias[i-1] - inertias[i]) / inertias[i-1] * 100
        rate_of_change.append(pct_decrease)
    
    # Find elbow using second derivative approach
    if len(rate_of_change) > 1:
        second_derivative = []
        for i in range(1, len(rate_of_change)):
            second_derivative.append(rate_of_change[i-1] - rate_of_change[i])
        
        # Find the point with maximum second derivative (sharpest change in rate)
        elbow_idx = np.argmax(second_derivative) + 2  # +2 because we started at k=3 and lost 2 indices
    else:
        elbow_idx = len(k_values) // 2  # Default to middle if can't determine
    
    optimal_k = k_values[elbow_idx]
    
    return optimal_k, inertias, k_values, rate_of_change

def analyze_dataset_kmeans(file_path, n_clusters):
    """Analyze a single dataset with K-means clustering"""
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None
    
    print(f"\nAnalyzing: {file_path.name}")
    df = pd.read_csv(file_path)
    
    # Filter out invalid visibility readings (>999000 indicates no data)
    df_clean = df[df['visibility_meters'] <= 999000].copy()
    print(f"  {len(df)} initial records")
    print(f"  {len(df) - len(df_clean)} records dropped (visibility > 999000)")
    print(f"  {len(df_clean)} records after filtering")
    
    if len(df_clean) == 0:
        print(f"  No valid data after filtering")
        return None
    
    # Separate features from target and time
    feature_cols = [col for col in df_clean.columns if col not in ['time', 'visibility_meters']]
    X = df_clean[feature_cols].copy()
    y = df_clean['visibility_meters'].copy()
    
    print(f"  Features: {len(feature_cols)}")
    
    # Drop rows with NaN values
    original_len = len(X)
    X = X.dropna()
    y = y.loc[X.index]  # Keep corresponding y values
    df_clean = df_clean.loc[X.index]  # Filter df_clean to match
    
    if len(X) < original_len:
        print(f"  {original_len - len(X)} records dropped (NaN values)")
        print(f"  {len(X)} records for clustering")
    
    if len(X) == 0:
        print(f"  No valid data after dropping NaN")
        return None
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    df_clean['cluster'] = cluster_labels
    
    # Calculate clustering metrics
    inertia = kmeans.inertia_
    
    # Calculate silhouette score (with sampling for large datasets)
    if len(X_scaled) < 10000:
        silhouette = silhouette_score(X_scaled, cluster_labels, sample_size=min(5000, len(X_scaled)))
    else:
        silhouette = silhouette_score(X_scaled, cluster_labels, sample_size=5000)
    
    # Calculate Davies-Bouldin Index (lower is better)
    davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
    
    # Calculate feature variance across clusters to identify most discriminative features
    feature_variance = []
    for i, feature in enumerate(feature_cols):
        # Get mean value of this feature in each cluster
        cluster_means = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if cluster_mask.sum() > 0:
                cluster_mean = X_scaled[cluster_mask, i].mean()
                cluster_means.append(cluster_mean)
        
        # Calculate variance of cluster means (how much this feature varies across clusters)
        if len(cluster_means) > 0:
            variance = np.var(cluster_means)
            feature_variance.append({
                'feature': feature,
                'variance_across_clusters': variance
            })
    
    # Sort features by variance across clusters (descending)
    feature_variance_df = pd.DataFrame(feature_variance)
    feature_variance_df = feature_variance_df.sort_values('variance_across_clusters', ascending=False)
    
    # Normalize to percentages
    total_variance = feature_variance_df['variance_across_clusters'].sum()
    if total_variance > 0:
        feature_variance_df['variance_pct'] = (feature_variance_df['variance_across_clusters'] / total_variance) * 100
    else:
        feature_variance_df['variance_pct'] = 0.0
    
    print(f"  Clusters: {n_clusters}")
    print(f"  Inertia: {inertia:.2f}")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")
    
    results = {
        'file_name': file_path.name,
        'total_records': len(df_clean),
        'feature_count': len(feature_cols),
        'features': feature_cols,
        'n_clusters': n_clusters,
        'inertia': inertia,
        'silhouette_score': silhouette,
        'davies_bouldin_index': davies_bouldin,
        'feature_variance': feature_variance_df
    }
    
    return results

def compare_results(without_aqi_results, with_aqi_results):
    """Compare clustering results with and without AQI features"""
    comparison = {}
    
    # Compare clustering quality metrics
    comparison['inertia_difference'] = with_aqi_results['inertia'] - without_aqi_results['inertia']
    comparison['silhouette_difference'] = with_aqi_results['silhouette_score'] - without_aqi_results['silhouette_score']
    comparison['davies_bouldin_difference'] = with_aqi_results['davies_bouldin_index'] - without_aqi_results['davies_bouldin_index']
    
    return comparison

def main():
    datasets_dir = Path("eda/datasets")
    output_dir = Path("eda/k-means clustering")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"kmeans_aqi_sensitivity_analysis.txt"
    without_aqi_files = sorted(datasets_dir.glob("location_*_without_aqi.csv"))
    
    if len(without_aqi_files) == 0:
        print("No location_*_without_aqi.csv files found!")
        return
    
    print(f"Found {len(without_aqi_files)} locations to analyze")
    
    
    first_file = without_aqi_files[0]
    df_first = pd.read_csv(first_file)
    df_first_clean = df_first[df_first['visibility_meters'] <= 999000].copy()
    feature_cols_first = [col for col in df_first_clean.columns if col not in ['time', 'visibility_meters']]
    X_first = df_first_clean[feature_cols_first].dropna()
    
    scaler_first = StandardScaler()
    X_first_scaled = scaler_first.fit_transform(X_first)
    
    print(f"\nRunning elbow method on {first_file.name}...")
    print(f"Sample size: {len(X_first_scaled)} records")
    
    optimal_k, inertias, k_values, rate_of_change = find_optimal_clusters(X_first_scaled, k_range=range(3, 21))
    
    print("\nElbow Method Results:")
    print(f"{'K':<6}{'Inertia':<18}{'% Decrease':<15}")
    print("-" * 39)
    for i, k in enumerate(k_values):
        if i == 0:
            print(f"{k:<6}{inertias[i]:<18.2f}{'-':<15}")
        else:
            print(f"{k:<6}{inertias[i]:<18.2f}{rate_of_change[i-1]:<15.2f}")
    
    print(f"\n>>> Using k={optimal_k} clusters for all analyses <<<")
    
    all_results = []
    
    for without_aqi_file in without_aqi_files:
        # Extract location number
        location_num = without_aqi_file.stem.replace('location_', '').replace('_without_aqi', '')
        
        with_aqi_file = datasets_dir / f"location_{location_num}_with_aqi.csv"
        without_aqi_reduced_file = datasets_dir / f"location_{location_num}_without_aqi_reduced.csv"
        
        if not with_aqi_file.exists() or not without_aqi_reduced_file.exists():
            print(f"\nWarning: Missing files for location {location_num}, skipping")
            continue
        
        print(f"\n{'='*80}")
        print(f"Location {location_num}")
        print(f"{'='*80}")
        
        without_results = analyze_dataset_kmeans(without_aqi_file, optimal_k)
        with_results = analyze_dataset_kmeans(with_aqi_file, optimal_k)
        without_reduced_results = analyze_dataset_kmeans(without_aqi_reduced_file, optimal_k)
        
        if without_results and with_results and without_reduced_results:
            comparison = compare_results(without_results, with_results)
            
            all_results.append({
                'location': location_num,
                'without_aqi': without_results,
                'with_aqi': with_results,
                'without_aqi_reduced': without_reduced_results,
                'comparison': comparison
            })
        else:
            print(f"  Skipping location {location_num} due to insufficient data")
    
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            location = result['location']
            without = result['without_aqi']
            with_aqi = result['with_aqi']
            without_reduced = result['without_aqi_reduced']
            comparison = result['comparison']
            
            f.write(f"LOCATION {location}\n")
            f.write("-"*80 + "\n")
            
            f.write(f"WITHOUT AQI (all rows, no AQI features):\n")
            f.write(f"  Records: {without['total_records']}\n")
            f.write(f"  Inertia: {without['inertia']:.2f}\n")
            f.write(f"  Silhouette Score: {without['silhouette_score']:.4f}\n")
            f.write(f"  Davies-Bouldin Index: {without['davies_bouldin_index']:.4f}\n")
            f.write(f"\n  Feature Variance Across Clusters (sorted, {len(without['feature_variance'])} features):\n")
            for idx, row in without['feature_variance'].iterrows():
                f.write(f"    {row['feature']}: {row['variance_pct']:.2f}%\n")
            f.write("\n")
            
            f.write(f"WITH AQI (filtered rows, with AQI features):\n")
            f.write(f"  Records: {with_aqi['total_records']}\n")
            f.write(f"  Inertia: {with_aqi['inertia']:.2f}\n")
            f.write(f"  Silhouette Score: {with_aqi['silhouette_score']:.4f}\n")
            f.write(f"  Davies-Bouldin Index: {with_aqi['davies_bouldin_index']:.4f}\n")
            f.write(f"\n  Feature Variance Across Clusters (sorted, {len(with_aqi['feature_variance'])} features):\n")
            for idx, row in with_aqi['feature_variance'].iterrows():
                f.write(f"    {row['feature']}: {row['variance_pct']:.2f}%\n")
            f.write("\n")
            
            f.write(f"WITHOUT AQI REDUCED (filtered rows, no AQI features):\n")
            f.write(f"  Records: {without_reduced['total_records']}\n")
            f.write(f"  Inertia: {without_reduced['inertia']:.2f}\n")
            f.write(f"  Silhouette Score: {without_reduced['silhouette_score']:.4f}\n")
            f.write(f"  Davies-Bouldin Index: {without_reduced['davies_bouldin_index']:.4f}\n")
            f.write(f"\n  Feature Variance Across Clusters (sorted, {len(without_reduced['feature_variance'])} features):\n")
            for idx, row in without_reduced['feature_variance'].iterrows():
                f.write(f"    {row['feature']}: {row['variance_pct']:.2f}%\n")
            f.write("\n" + "="*80 + "\n\n")
        
        # Averages
        f.write("AVERAGE METRICS\n")
        f.write("-"*80 + "\n")
        
        f.write(f"WITHOUT AQI (all rows, no AQI features) - Average across {len(all_results)} locations:\n")
        f.write(f"  Inertia: {np.mean([r['without_aqi']['inertia'] for r in all_results]):.2f}\n")
        f.write(f"  Silhouette Score: {np.mean([r['without_aqi']['silhouette_score'] for r in all_results]):.4f}\n")
        f.write(f"  Davies-Bouldin Index: {np.mean([r['without_aqi']['davies_bouldin_index'] for r in all_results]):.4f}\n")
        f.write("\n")
        
        # Average feature variance across all locations
        f.write(f"AVERAGE FEATURE VARIANCE WITHOUT AQI (across {len(all_results)} locations):\n")
        all_features_without = {}
        for r in all_results:
            for idx, row in r['without_aqi']['feature_variance'].iterrows():
                feature = row['feature']
                variance = row['variance_pct']
                if feature not in all_features_without:
                    all_features_without[feature] = []
                all_features_without[feature].append(variance)
        avg_variance_without = {k: np.mean(v) for k, v in all_features_without.items()}
        sorted_features_without = sorted(avg_variance_without.items(), key=lambda x: x[1], reverse=True)
        for feature, variance in sorted_features_without:
            f.write(f"  {feature}: {variance:.2f}%\n")
        f.write("\n")
        
        f.write(f"WITH AQI (filtered rows, with AQI features) - Average across {len(all_results)} locations:\n")
        f.write(f"  Inertia: {np.mean([r['with_aqi']['inertia'] for r in all_results]):.2f}\n")
        f.write(f"  Silhouette Score: {np.mean([r['with_aqi']['silhouette_score'] for r in all_results]):.4f}\n")
        f.write(f"  Davies-Bouldin Index: {np.mean([r['with_aqi']['davies_bouldin_index'] for r in all_results]):.4f}\n")
        f.write("\n")
        
        # Average feature variance across all locations
        f.write(f"AVERAGE FEATURE VARIANCE WITH AQI (across {len(all_results)} locations):\n")
        all_features_with = {}
        for r in all_results:
            for idx, row in r['with_aqi']['feature_variance'].iterrows():
                feature = row['feature']
                variance = row['variance_pct']
                if feature not in all_features_with:
                    all_features_with[feature] = []
                all_features_with[feature].append(variance)
        avg_variance_with = {k: np.mean(v) for k, v in all_features_with.items()}
        sorted_features_with = sorted(avg_variance_with.items(), key=lambda x: x[1], reverse=True)
        for feature, variance in sorted_features_with:
            f.write(f"  {feature}: {variance:.2f}%\n")
        f.write("\n")
        
        f.write(f"WITHOUT AQI REDUCED (filtered rows, no AQI features) - Average across {len(all_results)} locations:\n")
        f.write(f"  Inertia: {np.mean([r['without_aqi_reduced']['inertia'] for r in all_results]):.2f}\n")
        f.write(f"  Silhouette Score: {np.mean([r['without_aqi_reduced']['silhouette_score'] for r in all_results]):.4f}\n")
        f.write(f"  Davies-Bouldin Index: {np.mean([r['without_aqi_reduced']['davies_bouldin_index'] for r in all_results]):.4f}\n")
        f.write("\n")
        
        # Average feature variance across all locations
        f.write(f"AVERAGE FEATURE VARIANCE WITHOUT AQI REDUCED (across {len(all_results)} locations):\n")
        all_features_reduced = {}
        for r in all_results:
            for idx, row in r['without_aqi_reduced']['feature_variance'].iterrows():
                feature = row['feature']
                variance = row['variance_pct']
                if feature not in all_features_reduced:
                    all_features_reduced[feature] = []
                all_features_reduced[feature].append(variance)
        avg_variance_reduced = {k: np.mean(v) for k, v in all_features_reduced.items()}
        sorted_features_reduced = sorted(avg_variance_reduced.items(), key=lambda x: x[1], reverse=True)
        for feature, variance in sorted_features_reduced:
            f.write(f"  {feature}: {variance:.2f}%\n")
    
    print(f"\nAnalysis complete! Results saved to:")
    print(f"  {output_file}")
    print(f"\nProcessed {len(all_results)} locations")

if __name__ == "__main__":
    main()
