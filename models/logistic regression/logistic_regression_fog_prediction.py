import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def temporal_train_test_split(df, test_years=2):
    df['year'] = pd.to_datetime(df['time']).dt.year
    
    train_df = df[df['year'] != 2024].copy()
    test_df = df[df['year'] == 2024].copy()
    
    train_df = train_df.drop('year', axis=1)
    test_df = test_df.drop('year', axis=1)
    
    return train_df, test_df

def analyze_dataset(file_path, fog_threshold=1610):
    if not file_path.exists():
        return None
    
    df = pd.read_csv(file_path)
    
    invalid_count = (df['visibility_meters'] > 999000).sum()
    if invalid_count > 0:
        valid_visibility = df[df['visibility_meters'] <= 999000]['visibility_meters']
        median_visibility = valid_visibility.median()
        df.loc[df['visibility_meters'] > 999000, 'visibility_meters'] = median_visibility
    
    df['fog'] = (df['visibility_meters'] < fog_threshold).astype(int)
    
    feature_cols = [col for col in df.columns if col not in ['time', 'visibility_meters', 'fog']]
    for col in feature_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    train_df, test_df = temporal_train_test_split(df.copy())
    
    if len(train_df) < 100 or len(test_df) < 10:
        return None
    
    X_train = train_df[feature_cols]
    y_train = train_df['fog']
    X_test = test_df[feature_cols]
    y_test = test_df['fog']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(
        max_iter=1000, 
        random_state=42,
        class_weight='balanced',
        solver='lbfgs'
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    y_prob_train = model.predict_proba(X_train_scaled)[:, 1]
    y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_precision = precision_score(y_train, y_pred_train, zero_division=0)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    train_recall = recall_score(y_train, y_pred_train, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    train_f1 = f1_score(y_train, y_pred_train, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    train_auc = roc_auc_score(y_train, y_prob_train)
    test_auc = roc_auc_score(y_test, y_prob_test)
    
    importance_values = np.abs(model.coef_[0])
    feature_importance_list = []
    for feature, importance in zip(feature_cols, importance_values):
        feature_importance_list.append({'feature': feature, 'importance': importance})
    feature_importance = pd.DataFrame(feature_importance_list)
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    total_importance = feature_importance['importance'].sum()
    if total_importance > 0:
        feature_importance['importance_pct'] = (feature_importance['importance'] / total_importance) * 100
    else:
        feature_importance['importance_pct'] = 0.0
    
    train_fog_pct = (y_train.sum() / len(y_train)) * 100
    test_fog_pct = (y_test.sum() / len(y_test)) * 100
    
    results = {
        'file_name': file_path.name,
        'total_records': len(df),
        'train_records': len(train_df),
        'test_records': len(test_df),
        'train_fog_pct': train_fog_pct,
        'test_fog_pct': test_fog_pct,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'feature_importance': feature_importance
    }
    
    return results

def main():
    datasets_dir = Path("eda/datasets")
    output_dir = Path("models/logistic regression")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"logistic_regression_results.txt"
    without_aqi_files = sorted(datasets_dir.glob("location_*_without_aqi.csv"))
    
    if len(without_aqi_files) == 0:
        print("No datasets found!")
        return
    
    print(f"Found {len(without_aqi_files)} locations")
    
    all_results = []
    
    for without_aqi_file in without_aqi_files:
        location_num = without_aqi_file.stem.replace('location_', '').replace('_without_aqi', '')
        with_aqi_file = datasets_dir / f"location_{location_num}_with_aqi.csv"
        without_aqi_reduced_file = datasets_dir / f"location_{location_num}_without_aqi_reduced.csv"
        
        if not with_aqi_file.exists() or not without_aqi_reduced_file.exists():
            continue
        
        print(f"Processing location {location_num}...")
        print(f"  Dataset: WITHOUT AQI (all rows, no AQI features)")
        without_results = analyze_dataset(without_aqi_file)
        print(f"  Dataset: WITH AQI (filtered rows, with AQI features)")
        with_results = analyze_dataset(with_aqi_file)
        print(f"  Dataset: WITHOUT AQI REDUCED (filtered rows, no AQI features)")
        without_reduced_results = analyze_dataset(without_aqi_reduced_file)
        
        if without_results and with_results and without_reduced_results:
            all_results.append({
                'location': location_num,
                'without_aqi': without_results,
                'with_aqi': with_results,
                'without_aqi_reduced': without_reduced_results
            })
    
    # Write results
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            location = result['location']
            without = result['without_aqi']
            with_aqi = result['with_aqi']
            without_reduced = result['without_aqi_reduced']
            
            f.write(f"LOCATION {location}\n")
            f.write("-"*80 + "\n")
            
            f.write(f"WITHOUT AQI (all rows, no AQI features):\n")
            f.write(f"  Records: {without['total_records']} (Train: {without['train_records']}, Test: {without['test_records']})\n")
            f.write(f"  Fog %: Train={without['train_fog_pct']:.2f}%, Test={without['test_fog_pct']:.2f}%\n")
            f.write(f"  Accuracy: Train={without['train_accuracy']:.4f}, Test={without['test_accuracy']:.4f}\n")
            f.write(f"  Precision: Train={without['train_precision']:.4f}, Test={without['test_precision']:.4f}\n")
            f.write(f"  Recall: Train={without['train_recall']:.4f}, Test={without['test_recall']:.4f}\n")
            f.write(f"  F1: Train={without['train_f1']:.4f}, Test={without['test_f1']:.4f}\n")
            f.write(f"  AUC: Train={without['train_auc']:.4f}, Test={without['test_auc']:.4f}\n")
            f.write(f"\n  Feature Importance (sorted by coefficient magnitude, {len(without['feature_importance'])} features):\n")
            for idx, row in without['feature_importance'].iterrows():
                f.write(f"    {row['feature']}: {row['importance_pct']:.2f}%\n")
            f.write("\n")
            
            f.write(f"WITH AQI (filtered rows, with AQI features):\n")
            f.write(f"  Records: {with_aqi['total_records']} (Train: {with_aqi['train_records']}, Test: {with_aqi['test_records']})\n")
            f.write(f"  Fog %: Train={with_aqi['train_fog_pct']:.2f}%, Test={with_aqi['test_fog_pct']:.2f}%\n")
            f.write(f"  Accuracy: Train={with_aqi['train_accuracy']:.4f}, Test={with_aqi['test_accuracy']:.4f}\n")
            f.write(f"  Precision: Train={with_aqi['train_precision']:.4f}, Test={with_aqi['test_precision']:.4f}\n")
            f.write(f"  Recall: Train={with_aqi['train_recall']:.4f}, Test={with_aqi['test_recall']:.4f}\n")
            f.write(f"  F1: Train={with_aqi['train_f1']:.4f}, Test={with_aqi['test_f1']:.4f}\n")
            f.write(f"  AUC: Train={with_aqi['train_auc']:.4f}, Test={with_aqi['test_auc']:.4f}\n")
            f.write(f"\n  Feature Importance (sorted by coefficient magnitude, {len(with_aqi['feature_importance'])} features):\n")
            for idx, row in with_aqi['feature_importance'].iterrows():
                f.write(f"    {row['feature']}: {row['importance_pct']:.2f}%\n")
            f.write("\n")
            
            f.write(f"WITHOUT AQI REDUCED (filtered rows, no AQI features):\n")
            f.write(f"  Records: {without_reduced['total_records']} (Train: {without_reduced['train_records']}, Test: {without_reduced['test_records']})\n")
            f.write(f"  Fog %: Train={without_reduced['train_fog_pct']:.2f}%, Test={without_reduced['test_fog_pct']:.2f}%\n")
            f.write(f"  Accuracy: Train={without_reduced['train_accuracy']:.4f}, Test={without_reduced['test_accuracy']:.4f}\n")
            f.write(f"  Precision: Train={without_reduced['train_precision']:.4f}, Test={without_reduced['test_precision']:.4f}\n")
            f.write(f"  Recall: Train={without_reduced['train_recall']:.4f}, Test={without_reduced['test_recall']:.4f}\n")
            f.write(f"  F1: Train={without_reduced['train_f1']:.4f}, Test={without_reduced['test_f1']:.4f}\n")
            f.write(f"  AUC: Train={without_reduced['train_auc']:.4f}, Test={without_reduced['test_auc']:.4f}\n")
            f.write(f"\n  Feature Importance (sorted by coefficient magnitude, {len(without_reduced['feature_importance'])} features):\n")
            for idx, row in without_reduced['feature_importance'].iterrows():
                f.write(f"    {row['feature']}: {row['importance_pct']:.2f}%\n")
            f.write("\n" + "="*80 + "\n\n")
        
        # Averages
        f.write("AVERAGE METRICS\n")
        f.write("-"*80 + "\n")
        
        without_metrics = {
            'test_accuracy': np.mean([r['without_aqi']['test_accuracy'] for r in all_results]),
            'test_precision': np.mean([r['without_aqi']['test_precision'] for r in all_results]),
            'test_recall': np.mean([r['without_aqi']['test_recall'] for r in all_results]),
            'test_f1': np.mean([r['without_aqi']['test_f1'] for r in all_results]),
            'test_auc': np.mean([r['without_aqi']['test_auc'] for r in all_results])
        }
        
        with_metrics = {
            'test_accuracy': np.mean([r['with_aqi']['test_accuracy'] for r in all_results]),
            'test_precision': np.mean([r['with_aqi']['test_precision'] for r in all_results]),
            'test_recall': np.mean([r['with_aqi']['test_recall'] for r in all_results]),
            'test_f1': np.mean([r['with_aqi']['test_f1'] for r in all_results]),
            'test_auc': np.mean([r['with_aqi']['test_auc'] for r in all_results])
        }
        
        without_reduced_metrics = {
            'test_accuracy': np.mean([r['without_aqi_reduced']['test_accuracy'] for r in all_results]),
            'test_precision': np.mean([r['without_aqi_reduced']['test_precision'] for r in all_results]),
            'test_recall': np.mean([r['without_aqi_reduced']['test_recall'] for r in all_results]),
            'test_f1': np.mean([r['without_aqi_reduced']['test_f1'] for r in all_results]),
            'test_auc': np.mean([r['without_aqi_reduced']['test_auc'] for r in all_results])
        }
        
        f.write(f"WITHOUT AQI (all rows, no AQI features) - Average across {len(all_results)} locations:\n")
        f.write(f"  Test Accuracy: {without_metrics['test_accuracy']:.4f}\n")
        f.write(f"  Test Precision: {without_metrics['test_precision']:.4f}\n")
        f.write(f"  Test Recall: {without_metrics['test_recall']:.4f}\n")
        f.write(f"  Test F1: {without_metrics['test_f1']:.4f}\n")
        f.write(f"  Test AUC: {without_metrics['test_auc']:.4f}\n")
        f.write("\n")
        
        f.write(f"WITH AQI (filtered rows, with AQI features) - Average across {len(all_results)} locations:\n")
        f.write(f"  Test Accuracy: {with_metrics['test_accuracy']:.4f}\n")
        f.write(f"  Test Precision: {with_metrics['test_precision']:.4f}\n")
        f.write(f"  Test Recall: {with_metrics['test_recall']:.4f}\n")
        f.write(f"  Test F1: {with_metrics['test_f1']:.4f}\n")
        f.write(f"  Test AUC: {with_metrics['test_auc']:.4f}\n")
        f.write("\n")
        
        f.write(f"WITHOUT AQI REDUCED (filtered rows, no AQI features) - Average across {len(all_results)} locations:\n")
        f.write(f"  Test Accuracy: {without_reduced_metrics['test_accuracy']:.4f}\n")
        f.write(f"  Test Precision: {without_reduced_metrics['test_precision']:.4f}\n")
        f.write(f"  Test Recall: {without_reduced_metrics['test_recall']:.4f}\n")
        f.write(f"  Test F1: {without_reduced_metrics['test_f1']:.4f}\n")
        f.write(f"  Test AUC: {without_reduced_metrics['test_auc']:.4f}\n")
        f.write("\n")
        
        f.write(f"AVERAGE FEATURE IMPORTANCE WITHOUT AQI (across {len(all_results)} locations):\n")
        all_features_without = {}
        for r in all_results:
            for idx, row in r['without_aqi']['feature_importance'].iterrows():
                feature = row['feature']
                importance = row['importance_pct']
                if feature not in all_features_without:
                    all_features_without[feature] = []
                all_features_without[feature].append(importance)
        avg_importance_without = {k: np.mean(v) for k, v in all_features_without.items()}
        sorted_features_without = sorted(avg_importance_without.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features_without:
            f.write(f"  {feature}: {importance:.2f}%\n")
        f.write("\n")
        
        f.write(f"AVERAGE FEATURE IMPORTANCE WITH AQI (across {len(all_results)} locations):\n")
        all_features_with = {}
        for r in all_results:
            for idx, row in r['with_aqi']['feature_importance'].iterrows():
                feature = row['feature']
                importance = row['importance_pct']
                if feature not in all_features_with:
                    all_features_with[feature] = []
                all_features_with[feature].append(importance)
        avg_importance_with = {k: np.mean(v) for k, v in all_features_with.items()}
        sorted_features_with = sorted(avg_importance_with.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features_with:
            f.write(f"  {feature}: {importance:.2f}%\n")

if __name__ == "__main__":
    main()
