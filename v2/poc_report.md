# CalFog V2: Bakersfield leakage-free proof of concept

## Scope and question
This run is restricted to `location_10`. It evaluates an hourly **nowcast** and the operational deliverable: an 18:00 local issue predicting whether *any hourly visibility* is below 1,610 m during next-day 00:00–09:00. Times remain naive local clocks as supplied. Seed: 20250827. Full valid aligned target rows: 396,416; fog rate 1.9855%. AQI 24-hour-window-complete nowcast rows: 26,786; their different fog rate (0.6011%) is reported, never used as the full-vs-AQI effect. Lead labels require ten of ten valid verification hours; 15,505 complete mornings remain.

## Leakage safeguards
`VIS` is parsed from its first comma-delimited token. Missing and 999999 are discarded **before** forming `visibility < 1610`; visibility is never imputed. The 1609/1610 boundary is unit tested. Original suspect `cooling_rate_*` and `previous_night_low` columns are excluded rather than trusted. Inputs are timestamped hourly weather/AQI observations plus calendar terms computed at their observation time. Each CNN sample has exactly 24 consecutive clock hours and ends at the issue time. Thus nowcast feature time is at most its label time; lead feature time is at most 18:00 D while labels occupy D+1 00:00–09:00.

Rows are sorted, then label/issue keys are split chronologically 70/15/15. A 24-hour purge precedes validation and test ({'nowcast': {'train_val_boundary': '2012-01-23 03:00:00', 'val_test_boundary': '2018-11-05 23:00:00', 'purge_hours': 24}, 'lead_time': {'train_val_boundary': '2012-11-13 18:00:00', 'val_test_boundary': '2019-03-30 18:00:00', 'purge_hours': 24}}). Windows are materialized only afterward from verified hourly site data. Median imputation and standardization are fitted on training timesteps only and applied unchanged. XGBoost tree count, CNN epochs/configuration, and probability thresholds use validation only. The final test probabilities are computed only after those choices. AP (PR-AUC) is primary because fog is rare; ROC-AUC, Brier/ECE calibration, threshold metrics and TN/FP/FN/TP are preserved in `metrics.json`.

## Main held-out results
### Nowcast
| model | AP | ROC-AUC | precision | recall | F1 | threshold | test n (+) |
|---|---:|---:|---:|---:|---:|---:|---:|
| climatology | 0.0076 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 59463 (449) |
| xgboost | 0.0636 | 0.8938 | 0.0835 | 0.3252 | 0.1328 | 0.7192 | 59463 (449) |
| random_forest | 0.0773 | 0.8987 | 0.1102 | 0.3296 | 0.1652 | 0.5830 | 59463 (449) |
| temporal_cnn | 0.0831 | 0.9025 | 0.1197 | 0.3987 | 0.1842 | 0.8535 | 59463 (449) |

### Lead time: issue 18:00 -> next-day 00:00–09:00
| model | AP | ROC-AUC | precision | recall | F1 | threshold | test n (+) |
|---|---:|---:|---:|---:|---:|---:|---:|
| climatology | 0.0112 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 2326 (26) |
| xgboost | 0.0453 | 0.8105 | 0.0437 | 0.2692 | 0.0753 | 0.6319 | 2326 (26) |
| random_forest | 0.0488 | 0.7960 | 0.0671 | 0.3846 | 0.1143 | 0.2547 | 2326 (26) |
| temporal_cnn | 0.0642 | 0.8447 | 0.0612 | 0.1154 | 0.0800 | 0.7980 | 2326 (26) |

The climatology emits the training base rate and makes the most-frequent (no-fog) decision at 0.5. It therefore anchors discrimination and operational threshold metrics. The highest lead test AP here is 0.0642 (temporal_cnn). These are single chronological holdout estimates, not uncertainty-adjusted claims.

## Matched AQI sensitivity
The matched cohort requires all five AQI variables at every one of the 24 feature hours. For each task, with-AQI and no-AQI arms use byte-for-byte identical issue keys, labels, split boundaries, seeds and model protocols; only the five AQI channels are removed. This answers a conditional recent-era feature question and does not compare the ~2022+ cohort with 1980–2025 history.

### Nowcast ablation
| model | with AQI AP | matched no-AQI AP | ΔAP | with AUC | no-AQI AUC | ΔAUC |
|---|---:|---:|---:|---:|---:|---:|
| xgboost | 0.0490 | 0.0417 | +0.0073 | 0.9305 | 0.9296 | +0.0009 |
| temporal_cnn | 0.0449 | 0.0703 | -0.0255 | 0.9382 | 0.9510 | -0.0129 |

### Lead-time ablation
| model | with AQI AP | matched no-AQI AP | ΔAP | with AUC | no-AQI AUC | ΔAUC |
|---|---:|---:|---:|---:|---:|---:|
| xgboost | 0.0566 | 0.0566 | +0.0000 | 0.9149 | 0.9149 | +0.0000 |
| temporal_cnn | 0.3564 | 0.0359 | +0.3205 | 0.8439 | 0.6636 | +0.1803 |

The signed deltas are reported without selecting a favorable model or direction. Sensitivity uses two **predeclared expanding-origin** folds: 55% train/15% validation/15% forward test, then 70% train/15% validation/final 15% test, with a 24-hour purge at every boundary. The two disjoint test blocks cover the latest 30% exactly once and their out-of-fold probabilities are pooled for AP/AUC. This is necessary because the strictly latest 15% alone has zero fog positives; moving one final boundary to a convenient fog date would be p-hacking. Fold counts and positives are stored in `aqi_rolling_fold_audit`. The pooled nowcast sensitivity tests contain 43 fog positives; the lead sensitivity tests contain only 3. Thus the large lead CNN delta is extremely unstable, is not corroborated by XGBoost, and must not be interpreted as evidence. Deltas are descriptive and do not establish a causal aerosol effect.

## GPU training and Ray Tune
PyTorch reports `NVIDIA GeForce RTX 5070` and all CNN fitting used CUDA. Ray Tune executed 8 sequential GPU trials on the lead validation split; the search was {"channels": [16, 32, 48, 64], "kernel": [3, 5], "dropout": [0.15, 0.3, 0.45], "lr": "loguniform(2e-4,2e-3)", "weight_decay": [1e-05, 0.0001, 0.001], "batch_size": [256, 512]}. Best configuration: `{"channels": 48, "kernel": 3, "dropout": 0.3, "lr": 0.001736390235799647, "weight_decay": 1e-05, "batch_size": 256, "max_neg": 40000}`; best sweep validation AP 0.1142. The selected lead model then trained with validation-only early stopping for 14 epochs and its real `torch.save` state is `artifacts/temporal_cnn_leadtime.pt`. Seeds for Python, NumPy, Torch, XGBoost and the run are recorded.

## Limitations and recommendation
METAR coverage and reporting cadence vary. To prevent an unobserved fog hour from becoming a negative, lead labels require all ten valid hourly targets from 00:00 through 09:00; 1,164 incomplete mornings are excluded, which can itself induce coverage selection. Hourly reanalysis weather is treated as available at timestamp t; a real deployment must replace it with latency-controlled observations/forecast products. The AQI era is short and contains very few positive lead days. This POC has one airport, one chronological test, no uncertainty intervals, no probability recalibration, and possible regime shifts. Thresholds maximizing validation F1 may be inappropriate for real asymmetric costs.

**Recommendation: NO-GO for blindly scaling a claimed performant model to all five airports.** The corrected pipeline itself is a **GO for cautious replication**: run it unchanged per airport, preserve site-local chronology, add latency metadata and repeated rolling-origin evaluation, and decide operational viability only after confidence intervals and cost-based thresholds. This separates engineering readiness from evidence of predictive generalization.

## Reproduction
From the repository root: `bash v2/run_poc.sh` (equivalently `/home/pa/work/venv/bin/python v2/run_poc.py`). Tests: `/home/pa/work/venv/bin/python -m pytest v2/tests -q`. All generated output remains under `v2/`.
