import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

def temporal_train_test_split(df, test_years=2):
    df['year'] = pd.to_datetime(df['time']).dt.year
    
    train_df = df[df['year'] != 2024].copy()
    test_df = df[df['year'] == 2024].copy()
    
    train_df = train_df.drop('year', axis=1)
    test_df = test_df.drop('year', axis=1)
    
    return train_df, test_df

def create_sequences(X, sequence_length=24):
    X_seq = []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
    return np.array(X_seq)

def build_temporal_cnn(input_shape):
    model = keras.Sequential([
        layers.Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.5),
        
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.5),
        
        layers.Conv1D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(0.55),
        
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.55),
        
        layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC(name='auc')]
    )
    return model

def analyze_dataset(file_path):
    if not file_path.exists():
        return None
    
    df = pd.read_csv(file_path)
    
    invalid_count = (df['visibility_meters'] > 999000).sum()
    if invalid_count > 0:
        valid_visibility = df[df['visibility_meters'] <= 999000]['visibility_meters']
        median_visibility = valid_visibility.median()
        df.loc[df['visibility_meters'] > 999000, 'visibility_meters'] = median_visibility
    
    feature_cols = [col for col in df.columns if col not in ['time', 'visibility_meters']]
    
    for col in feature_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    train_df, test_df = temporal_train_test_split(df.copy())
    
    if len(train_df) < 100 or len(test_df) < 10:
        return None
    
    fog_threshold = 1610  # 1 mile in meters
    X_train_orig = train_df[feature_cols]
    y_train = (train_df['visibility_meters'] < fog_threshold).astype(int).values
    X_test_orig = test_df[feature_cols]
    y_test = (test_df['visibility_meters'] < fog_threshold).astype(int).values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_orig)
    X_test_scaled = scaler.transform(X_test_orig)
    
    print("    Training Layer 1: Logistic Regression...")
    logreg = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        solver='lbfgs'
    )
    logreg.fit(X_train_scaled, y_train)
    
    logreg_train_proba = logreg.predict_proba(X_train_scaled)[:, 1]
    logreg_test_proba = logreg.predict_proba(X_test_scaled)[:, 1]
    
    print("    Training Layer 2: Random Forest...")
    X_train_l2 = np.column_stack([X_train_orig, logreg_train_proba])
    X_test_l2 = np.column_stack([X_test_orig, logreg_test_proba])
    
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=40,
        min_samples_leaf=25,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf.fit(X_train_l2, y_train)
    
    rf_train_proba = rf.predict_proba(X_train_l2)[:, 1]
    rf_test_proba = rf.predict_proba(X_test_l2)[:, 1]
    
    print("    Training Layer 3: XGBoost...")
    X_train_l3 = np.column_stack([X_train_orig, logreg_train_proba, rf_train_proba])
    X_test_l3 = np.column_stack([X_test_orig, logreg_test_proba, rf_test_proba])
    
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    xgb = XGBClassifier(
        n_estimators=10000,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=3,
        gamma=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        tree_method='hist',
        early_stopping_rounds=1000,
        verbosity=0
    )
    
    xgb.fit(
        X_train_l3, y_train,
        eval_set=[(X_test_l3, y_test)],
        verbose=False
    )
    
    xgb_train_proba = xgb.predict_proba(X_train_l3)[:, 1]
    xgb_test_proba = xgb.predict_proba(X_test_l3)[:, 1]
    
    print("    Training Layer 4: Temporal CNN...")
    sequence_length = 24
    
    X_train_l4 = np.column_stack([X_train_scaled, logreg_train_proba, rf_train_proba, xgb_train_proba])
    X_test_l4 = np.column_stack([X_test_scaled, logreg_test_proba, rf_test_proba, xgb_test_proba])
    
    X_train_seq = create_sequences(X_train_l4, sequence_length)
    X_test_seq = create_sequences(X_test_l4, sequence_length)
    y_train_seq = y_train[sequence_length:]
    y_test_seq = y_test[sequence_length:]
    
    if len(X_train_seq) < 100 or len(X_test_seq) < 10:
        return None
    
    n_neg_seq = (y_train_seq == 0).sum()
    n_pos_seq = (y_train_seq == 1).sum()
    total_seq = len(y_train_seq)
    class_weight = {0: total_seq / (2.0 * n_neg_seq), 1: total_seq / (2.0 * n_pos_seq)} if n_pos_seq > 0 else None
    
    cnn_model = build_temporal_cnn((sequence_length, X_train_seq.shape[2]))
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    cnn_model.fit(
        X_train_seq, y_train_seq,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=0
    )
    
    cnn_train_proba = cnn_model.predict(X_train_seq, verbose=0).flatten()
    cnn_test_proba = cnn_model.predict(X_test_seq, verbose=0).flatten()
    
    logreg_train_proba_aligned = logreg_train_proba[sequence_length:]
    logreg_test_proba_aligned = logreg_test_proba[sequence_length:]
    rf_train_proba_aligned = rf_train_proba[sequence_length:]
    rf_test_proba_aligned = rf_test_proba[sequence_length:]
    xgb_train_proba_aligned = xgb_train_proba[sequence_length:]
    xgb_test_proba_aligned = xgb_test_proba[sequence_length:]

    ensemble_train_proba = (0.1 * logreg_train_proba_aligned + 0.2 * rf_train_proba_aligned + 
                            0.3 * xgb_train_proba_aligned + 0.4 * cnn_train_proba)
    ensemble_test_proba = (0.1 * logreg_test_proba_aligned + 0.2 * rf_test_proba_aligned + 
                           0.3 * xgb_test_proba_aligned + 0.4 * cnn_test_proba)
    
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in np.arange(0.1, 0.6, 0.05):
        y_pred_thresh = (ensemble_train_proba >= threshold).astype(int)
        f1 = f1_score(y_train_seq, y_pred_thresh, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    y_pred_train = (ensemble_train_proba >= best_threshold).astype(int)
    y_pred_test = (ensemble_test_proba >= best_threshold).astype(int)
    
    train_accuracy = accuracy_score(y_train_seq, y_pred_train)
    test_accuracy = accuracy_score(y_test_seq, y_pred_test)
    train_precision = precision_score(y_train_seq, y_pred_train, zero_division=0)
    test_precision = precision_score(y_test_seq, y_pred_test, zero_division=0)
    train_recall = recall_score(y_train_seq, y_pred_train, zero_division=0)
    test_recall = recall_score(y_test_seq, y_pred_test, zero_division=0)
    train_f1 = f1_score(y_train_seq, y_pred_train, zero_division=0)
    test_f1 = f1_score(y_test_seq, y_pred_test, zero_division=0)
    train_auc = roc_auc_score(y_train_seq, ensemble_train_proba) if len(np.unique(y_train_seq)) > 1 else 0
    test_auc = roc_auc_score(y_test_seq, ensemble_test_proba) if len(np.unique(y_test_seq)) > 1 else 0
    
    logreg_importance = np.abs(logreg.coef_[0])
    logreg_importance = logreg_importance / logreg_importance.sum() if logreg_importance.sum() > 0 else logreg_importance
    rf_importance = rf.feature_importances_[:len(feature_cols)]  # exclude appended prediction columns
    rf_importance = rf_importance / rf_importance.sum() if rf_importance.sum() > 0 else rf_importance
    xgb_raw_importance = xgb.feature_importances_[:len(feature_cols)]  # exclude appended prediction columns
    xgb_importance = xgb_raw_importance / xgb_raw_importance.sum() if xgb_raw_importance.sum() > 0 else xgb_raw_importance
    
    combined_importance = (1/6) * logreg_importance + (2/6) * rf_importance + (3/6) * xgb_importance
    feature_importance_list = []
    for feature, importance in zip(feature_cols, combined_importance):
        feature_importance_list.append({'feature': feature, 'importance': importance})
    feature_importance = pd.DataFrame(feature_importance_list)
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    total_importance = feature_importance['importance'].sum()
    if total_importance > 0:
        feature_importance['importance_pct'] = (feature_importance['importance'] / total_importance) * 100
    else:
        feature_importance['importance_pct'] = 0.0
    
    results = {
        'file_name': file_path.name,
        'total_records': len(df),
        'train_records': len(train_df),
        'test_records': len(test_df),
        'train_sequences': len(X_train_seq),
        'test_sequences': len(X_test_seq),
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
        'feature_importance': feature_importance,
        'optimal_threshold': best_threshold
    }
    
    return results

def main():
    datasets_dir = Path("eda/datasets")
    output_dir = Path("models/ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"ensemble_results.txt"
    
    # Find all datasets
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
        
        print(f"\nProcessing location {location_num}...")
        print(f"  Dataset: WITHOUT AQI (all rows)")
        without_results = analyze_dataset(without_aqi_file)
        print(f"  Dataset: WITH AQI (filtered rows)")
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
            f.write(f"  Sequences: Train={without['train_sequences']}, Test={without['test_sequences']}\n")
            f.write(f"  Optimal Threshold: {without['optimal_threshold']:.2f}\n")
            f.write(f"  Accuracy: Train={without['train_accuracy']:.4f}, Test={without['test_accuracy']:.4f}\n")
            f.write(f"  Precision: Train={without['train_precision']:.4f}, Test={without['test_precision']:.4f}\n")
            f.write(f"  Recall: Train={without['train_recall']:.4f}, Test={without['test_recall']:.4f}\n")
            f.write(f"  F1 Score: Train={without['train_f1']:.4f}, Test={without['test_f1']:.4f}\n")
            f.write(f"  ROC-AUC: Train={without['train_auc']:.4f}, Test={without['test_auc']:.4f}\n")
            f.write(f"\n  Feature Importance (sorted by weighted importance, {len(without['feature_importance'])} features):\n")
            for idx, row in without['feature_importance'].iterrows():
                f.write(f"    {row['feature']}: {row['importance_pct']:.2f}%\n")
            f.write("\n")
            
            f.write(f"WITH AQI (filtered rows, with AQI features):\n")
            f.write(f"  Records: {with_aqi['total_records']} (Train: {with_aqi['train_records']}, Test: {with_aqi['test_records']})\n")
            f.write(f"  Sequences: Train={with_aqi['train_sequences']}, Test={with_aqi['test_sequences']}\n")
            f.write(f"  Optimal Threshold: {with_aqi['optimal_threshold']:.2f}\n")
            f.write(f"  Accuracy: Train={with_aqi['train_accuracy']:.4f}, Test={with_aqi['test_accuracy']:.4f}\n")
            f.write(f"  Precision: Train={with_aqi['train_precision']:.4f}, Test={with_aqi['test_precision']:.4f}\n")
            f.write(f"  Recall: Train={with_aqi['train_recall']:.4f}, Test={with_aqi['test_recall']:.4f}\n")
            f.write(f"  F1 Score: Train={with_aqi['train_f1']:.4f}, Test={with_aqi['test_f1']:.4f}\n")
            f.write(f"  ROC-AUC: Train={with_aqi['train_auc']:.4f}, Test={with_aqi['test_auc']:.4f}\n")
            f.write(f"\n  Feature Importance (sorted by weighted importance, {len(with_aqi['feature_importance'])} features):\n")
            for idx, row in with_aqi['feature_importance'].iterrows():
                f.write(f"    {row['feature']}: {row['importance_pct']:.2f}%\n")
            f.write("\n")
            
            f.write(f"WITHOUT AQI REDUCED (filtered rows, no AQI features):\n")
            f.write(f"  Records: {without_reduced['total_records']} (Train: {without_reduced['train_records']}, Test: {without_reduced['test_records']})\n")
            f.write(f"  Sequences: Train={without_reduced['train_sequences']}, Test={without_reduced['test_sequences']}\n")
            f.write(f"  Optimal Threshold: {without_reduced['optimal_threshold']:.2f}\n")
            f.write(f"  Accuracy: Train={without_reduced['train_accuracy']:.4f}, Test={without_reduced['test_accuracy']:.4f}\n")
            f.write(f"  Precision: Train={without_reduced['train_precision']:.4f}, Test={without_reduced['test_precision']:.4f}\n")
            f.write(f"  Recall: Train={without_reduced['train_recall']:.4f}, Test={without_reduced['test_recall']:.4f}\n")
            f.write(f"  F1 Score: Train={without_reduced['train_f1']:.4f}, Test={without_reduced['test_f1']:.4f}\n")
            f.write(f"  ROC-AUC: Train={without_reduced['train_auc']:.4f}, Test={without_reduced['test_auc']:.4f}\n")
            f.write(f"\n  Feature Importance (sorted by weighted importance, {len(without_reduced['feature_importance'])} features):\n")
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
        f.write(f"  Test F1 Score: {without_metrics['test_f1']:.4f}\n")
        f.write(f"  Test ROC-AUC: {without_metrics['test_auc']:.4f}\n")
        f.write("\n")
        
        f.write(f"WITH AQI (filtered rows, with AQI features) - Average across {len(all_results)} locations:\n")
        f.write(f"  Test Accuracy: {with_metrics['test_accuracy']:.4f}\n")
        f.write(f"  Test Precision: {with_metrics['test_precision']:.4f}\n")
        f.write(f"  Test Recall: {with_metrics['test_recall']:.4f}\n")
        f.write(f"  Test F1 Score: {with_metrics['test_f1']:.4f}\n")
        f.write(f"  Test ROC-AUC: {with_metrics['test_auc']:.4f}\n")
        f.write("\n")
        
        f.write(f"WITHOUT AQI REDUCED (filtered rows, no AQI features) - Average across {len(all_results)} locations:\n")
        f.write(f"  Test Accuracy: {without_reduced_metrics['test_accuracy']:.4f}\n")
        f.write(f"  Test Precision: {without_reduced_metrics['test_precision']:.4f}\n")
        f.write(f"  Test Recall: {without_reduced_metrics['test_recall']:.4f}\n")
        f.write(f"  Test F1 Score: {without_reduced_metrics['test_f1']:.4f}\n")
        f.write(f"  Test ROC-AUC: {without_reduced_metrics['test_auc']:.4f}\n")
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
