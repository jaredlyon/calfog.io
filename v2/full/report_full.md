# CalFog V2 — cautious five-airport replication

## Methodology
Each airport is modeled independently with seed 20250901. Fog is visibility <1,610 m after invalid targets are removed. Nowcasts use 24 hourly observations ending at t; lead issues at 18:00 D use only observations through issue time and require all ten target hours on D+1 00:00–09:00. Three predeclared expanding-origin folds use a 24 h purge at both validation and test boundaries. Test predictions are pooled exactly once across disjoint origins.

The primary long-history comparison excludes AQI so all sites retain their historical cohort; AQI is evaluated only in the separate matched recent-era with/without experiment. Uncertainty uses 1000 paired whole-day block resamples stratified by fold and a point-centered 95% absolute-deviation interval; no hourly IID bootstrap is used. AP is primary and ROC-AUC secondary. Isotonic calibration is fitted separately on each fold's validation data. Cost thresholds are validation-selected over the predeclared threshold grid 0..1 by 0.005 for every FN:FP ratio {1,3,5,10,20}; none is designated as the true deployment cost.

## Feature availability and latency metadata
NOAA visibility timestamps are interpreted as UTC and converted to `America/Los_Angeles` (including DST) before alignment with the locally requested Open-Meteo weather/AQI clock. Weather and AQI values stamped h are assumed available by the end of h. Calendar variables are deterministic at issue time. Every 24-hour window ends at the issue timestamp; lead verification outcomes never enter features. **Deployment warning:** supplied hourly weather is reanalysis, not a latency-controlled feed. It must be replaced by timestamped operational observations and/or forecasts whose publication latency is enforced. AQI channels carry the same optimistic timestamp-availability assumption and require a production latency audit. Visibility is label-only. Suspect supplied rolling/night fields are excluded and calendar terms are recomputed.

## location_6 — Madera
Cohort: {"nowcast_rows": 171532, "nowcast_positives": 3921, "nowcast_fog_rate": 0.02285870857915724, "lead_complete_10h_issues": 7053, "lead_positives": 681, "aqi_matched_nowcast_rows": 26621, "aqi_matched_lead_issues": 1083, "aqi_channels": ["pm10", "pm2_5", "aerosol_optical_depth", "dust", "nitrogen_dioxide"]}
### Nowcast
| model | AP (95% block CI) | ROC-AUC (95% CI) | positives / n | Brier → calibrated | ECE → calibrated |
|---|---:|---:|---:|---:|---:|
| climatology | 0.0255 [0.0206, 0.0304] | 0.5575 [0.5164, 0.5985] | 1630 / 77190 | 0.0207 → 0.0207 | 0.0020 → 0.0020 |
| xgboost | 0.3017 [0.2508, 0.3526] | 0.9466 [0.9379, 0.9554] | 1630 / 77190 | 0.0874 → 0.0185 | 0.1493 → 0.0070 |
| random_forest | 0.2637 [0.2153, 0.3120] | 0.9409 [0.9311, 0.9507] | 1630 / 77190 | 0.0485 → 0.0186 | 0.0695 → 0.0054 |
| temporal_cnn | 0.2339 [0.1927, 0.2751] | 0.9364 [0.9251, 0.9477] | 1630 / 77190 | 0.1046 → 0.0178 | 0.1648 → 0.0034 |

Cost operating points (validation-selected thresholds; all ratios reported):
| model | FN:FP | mean threshold | precision | recall | F1 | alert rate | cost/issue |
|---|---:|---:|---:|---:|---:|---:|---:|
| climatology | 1:1 | 0.025 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0211 |
| climatology | 3:1 | 0.025 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0634 |
| climatology | 5:1 | 0.025 | 0.000 | 0.000 | 0.000 | 0.000 | 0.1056 |
| climatology | 10:1 | 0.025 | 0.000 | 0.000 | 0.000 | 0.000 | 0.2112 |
| climatology | 20:1 | 0.025 | 0.000 | 0.000 | 0.000 | 0.000 | 0.4223 |
| xgboost | 1:1 | 0.432 | 0.422 | 0.055 | 0.097 | 0.003 | 0.0215 |
| xgboost | 3:1 | 0.218 | 0.207 | 0.292 | 0.243 | 0.030 | 0.0684 |
| xgboost | 5:1 | 0.143 | 0.216 | 0.547 | 0.310 | 0.053 | 0.0897 |
| xgboost | 10:1 | 0.077 | 0.187 | 0.648 | 0.291 | 0.073 | 0.1337 |
| xgboost | 20:1 | 0.038 | 0.141 | 0.867 | 0.242 | 0.130 | 0.1680 |
| random_forest | 1:1 | 0.420 | 0.385 | 0.058 | 0.100 | 0.003 | 0.0218 |
| random_forest | 3:1 | 0.240 | 0.224 | 0.291 | 0.253 | 0.027 | 0.0662 |
| random_forest | 5:1 | 0.162 | 0.177 | 0.371 | 0.240 | 0.044 | 0.1028 |
| random_forest | 10:1 | 0.080 | 0.164 | 0.660 | 0.263 | 0.085 | 0.1429 |
| random_forest | 20:1 | 0.035 | 0.146 | 0.862 | 0.250 | 0.125 | 0.1647 |
| temporal_cnn | 1:1 | 0.452 | 0.415 | 0.017 | 0.032 | 0.001 | 0.0213 |
| temporal_cnn | 3:1 | 0.230 | 0.273 | 0.320 | 0.294 | 0.025 | 0.0611 |
| temporal_cnn | 5:1 | 0.160 | 0.245 | 0.499 | 0.328 | 0.043 | 0.0854 |
| temporal_cnn | 10:1 | 0.067 | 0.171 | 0.725 | 0.277 | 0.089 | 0.1322 |
| temporal_cnn | 20:1 | 0.037 | 0.138 | 0.850 | 0.237 | 0.130 | 0.1757 |

[Reliability curve](artifacts/location_6/reliability_nowcast.png) · [full cost curves](artifacts/location_6/cost_curves_nowcast.png)
### Lead Time
| model | AP (95% block CI) | ROC-AUC (95% CI) | positives / n | Brier → calibrated | ECE → calibrated |
|---|---:|---:|---:|---:|---:|
| climatology | 0.1019 [0.0890, 0.1148] | 0.5358 [0.5037, 0.5679] | 298 / 3174 | 0.0850 → 0.0850 | 0.0018 → 0.0018 |
| xgboost | 0.4341 [0.3792, 0.4890] | 0.8823 [0.8636, 0.9011] | 298 / 3174 | 0.1309 → 0.0664 | 0.2042 → 0.0141 |
| random_forest | 0.4255 [0.3697, 0.4813] | 0.8950 [0.8796, 0.9103] | 298 / 3174 | 0.0818 → 0.0673 | 0.0744 → 0.0264 |
| temporal_cnn | 0.3798 [0.3260, 0.4335] | 0.8812 [0.8650, 0.8973] | 298 / 3174 | 0.1759 → 0.0666 | 0.2683 → 0.0218 |

Cost operating points (validation-selected thresholds; all ratios reported):
| model | FN:FP | mean threshold | precision | recall | F1 | alert rate | cost/issue |
|---|---:|---:|---:|---:|---:|---:|---:|
| climatology | 1:1 | 0.098 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0939 |
| climatology | 3:1 | 0.098 | 0.000 | 0.000 | 0.000 | 0.000 | 0.2817 |
| climatology | 5:1 | 0.098 | 0.000 | 0.000 | 0.000 | 0.000 | 0.4694 |
| climatology | 10:1 | 0.033 | 0.086 | 0.611 | 0.151 | 0.667 | 0.9748 |
| climatology | 20:1 | 0.000 | 0.094 | 1.000 | 0.172 | 1.000 | 0.9061 |
| xgboost | 1:1 | 0.452 | 0.443 | 0.235 | 0.307 | 0.050 | 0.0996 |
| xgboost | 3:1 | 0.188 | 0.358 | 0.678 | 0.469 | 0.178 | 0.2048 |
| xgboost | 5:1 | 0.130 | 0.327 | 0.822 | 0.468 | 0.236 | 0.2426 |
| xgboost | 10:1 | 0.060 | 0.256 | 0.919 | 0.400 | 0.338 | 0.3270 |
| xgboost | 20:1 | 0.028 | 0.239 | 0.930 | 0.380 | 0.366 | 0.4108 |
| random_forest | 1:1 | 0.455 | 0.455 | 0.205 | 0.282 | 0.042 | 0.0977 |
| random_forest | 3:1 | 0.207 | 0.389 | 0.658 | 0.489 | 0.159 | 0.1934 |
| random_forest | 5:1 | 0.145 | 0.296 | 0.846 | 0.439 | 0.268 | 0.2612 |
| random_forest | 10:1 | 0.067 | 0.262 | 0.903 | 0.406 | 0.324 | 0.3305 |
| random_forest | 20:1 | 0.028 | 0.233 | 0.919 | 0.372 | 0.371 | 0.4357 |
| temporal_cnn | 1:1 | 0.430 | 0.530 | 0.208 | 0.299 | 0.037 | 0.0917 |
| temporal_cnn | 3:1 | 0.162 | 0.349 | 0.752 | 0.477 | 0.202 | 0.2016 |
| temporal_cnn | 5:1 | 0.115 | 0.328 | 0.792 | 0.464 | 0.227 | 0.2498 |
| temporal_cnn | 10:1 | 0.083 | 0.298 | 0.866 | 0.443 | 0.273 | 0.3176 |
| temporal_cnn | 20:1 | 0.023 | 0.240 | 0.943 | 0.383 | 0.368 | 0.3869 |

[Reliability curve](artifacts/location_6/reliability_lead_time.png) · [full cost curves](artifacts/location_6/cost_curves_lead_time.png)

### Matched-cohort AQI ablation
| task/model | with AQI AP | no AQI AP | ΔAP (paired 95% day-block CI) | positives |
|---|---:|---:|---:|---:|
| nowcast/xgboost | 0.2138 | 0.2584 | -0.0446 [-0.1137, +0.0245] | 234 |
| nowcast/temporal_cnn | 0.1488 | 0.1490 | -0.0002 [-0.0329, +0.0326] | 234 |
| lead_time/xgboost | 0.3589 | 0.4524 | -0.0936 [-0.1852, -0.0019] | 46 |
| lead_time/temporal_cnn | 0.5517 | 0.2478 | +0.3039 [+0.1770, +0.4309] | 46 |

**Sparse-positive warning:** lead_time/xgboost (46 positives), lead_time/temporal_cnn (46 positives). These AQI delta intervals are potentially uninformative even when they exclude zero. Zero-positive test folds: nowcast=0/3, lead=0/3; pooled metrics rely on the other fixed folds.

**Operational decision:** NO-GO for unsupervised deployment; retrospective best AP nowcast=0.302, lead=0.434. Consider only latency-controlled shadow testing and locally chosen costs.

## location_7 — Fresno
Cohort: {"nowcast_rows": 399949, "nowcast_positives": 13061, "nowcast_fog_rate": 0.03265666372462489, "lead_complete_10h_issues": 16575, "lead_positives": 1630, "aqi_matched_nowcast_rows": 26833, "aqi_matched_lead_issues": 1113, "aqi_channels": ["pm10", "pm2_5", "aerosol_optical_depth", "dust", "nitrogen_dioxide"]}
### Nowcast
| model | AP (95% block CI) | ROC-AUC (95% CI) | positives / n | Brier → calibrated | ECE → calibrated |
|---|---:|---:|---:|---:|---:|
| climatology | 0.0203 [0.0175, 0.0230] | 0.5289 [0.4975, 0.5602] | 3302 / 179978 | 0.0187 → 0.0187 | 0.0255 → 0.0255 |
| xgboost | 0.2287 [0.1935, 0.2640] | 0.9396 [0.9327, 0.9466] | 3302 / 179978 | 0.0958 → 0.0155 | 0.1919 → 0.0030 |
| random_forest | 0.2531 [0.2104, 0.2957] | 0.9439 [0.9330, 0.9548] | 3302 / 179978 | 0.0379 → 0.0154 | 0.0613 → 0.0023 |
| temporal_cnn | 0.2591 [0.2271, 0.2910] | 0.9443 [0.9306, 0.9580] | 3302 / 179978 | 0.0828 → 0.0151 | 0.1273 → 0.0014 |

Cost operating points (validation-selected thresholds; all ratios reported):
| model | FN:FP | mean threshold | precision | recall | F1 | alert rate | cost/issue |
|---|---:|---:|---:|---:|---:|---:|---:|
| climatology | 1:1 | 0.045 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0183 |
| climatology | 3:1 | 0.045 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0550 |
| climatology | 5:1 | 0.045 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0917 |
| climatology | 10:1 | 0.045 | 0.000 | 0.000 | 0.000 | 0.000 | 0.1835 |
| climatology | 20:1 | 0.045 | 0.000 | 0.000 | 0.000 | 0.000 | 0.3669 |
| xgboost | 1:1 | 0.360 | 0.422 | 0.060 | 0.105 | 0.003 | 0.0188 |
| xgboost | 3:1 | 0.227 | 0.299 | 0.173 | 0.219 | 0.011 | 0.0529 |
| xgboost | 5:1 | 0.147 | 0.239 | 0.498 | 0.323 | 0.038 | 0.0752 |
| xgboost | 10:1 | 0.082 | 0.185 | 0.720 | 0.294 | 0.072 | 0.1098 |
| xgboost | 20:1 | 0.040 | 0.153 | 0.827 | 0.258 | 0.099 | 0.1477 |
| random_forest | 1:1 | 0.423 | 0.459 | 0.055 | 0.099 | 0.002 | 0.0185 |
| random_forest | 3:1 | 0.230 | 0.320 | 0.156 | 0.210 | 0.009 | 0.0525 |
| random_forest | 5:1 | 0.152 | 0.251 | 0.494 | 0.333 | 0.036 | 0.0734 |
| random_forest | 10:1 | 0.085 | 0.180 | 0.750 | 0.291 | 0.076 | 0.1085 |
| random_forest | 20:1 | 0.043 | 0.151 | 0.801 | 0.254 | 0.097 | 0.1556 |
| temporal_cnn | 1:1 | 0.450 | 0.436 | 0.048 | 0.086 | 0.002 | 0.0186 |
| temporal_cnn | 3:1 | 0.225 | 0.303 | 0.318 | 0.311 | 0.019 | 0.0509 |
| temporal_cnn | 5:1 | 0.143 | 0.244 | 0.612 | 0.348 | 0.046 | 0.0705 |
| temporal_cnn | 10:1 | 0.085 | 0.196 | 0.751 | 0.311 | 0.070 | 0.1022 |
| temporal_cnn | 20:1 | 0.045 | 0.165 | 0.841 | 0.275 | 0.094 | 0.1367 |

[Reliability curve](artifacts/location_7/reliability_nowcast.png) · [full cost curves](artifacts/location_7/cost_curves_nowcast.png)
### Lead Time
| model | AP (95% block CI) | ROC-AUC (95% CI) | positives / n | Brier → calibrated | ECE → calibrated |
|---|---:|---:|---:|---:|---:|
| climatology | 0.0728 [0.0650, 0.0807] | 0.5285 [0.5032, 0.5539] | 502 / 7459 | 0.0658 → 0.0658 | 0.0553 → 0.0553 |
| xgboost | 0.3806 [0.3382, 0.4231] | 0.9114 [0.9007, 0.9221] | 502 / 7459 | 0.1076 → 0.0487 | 0.1758 → 0.0110 |
| random_forest | 0.3975 [0.3552, 0.4399] | 0.9183 [0.9088, 0.9278] | 502 / 7459 | 0.0644 → 0.0487 | 0.0719 → 0.0098 |
| temporal_cnn | 0.4111 [0.3690, 0.4531] | 0.9242 [0.9151, 0.9332] | 502 / 7459 | 0.1138 → 0.0486 | 0.1588 → 0.0095 |

Cost operating points (validation-selected thresholds; all ratios reported):
| model | FN:FP | mean threshold | precision | recall | F1 | alert rate | cost/issue |
|---|---:|---:|---:|---:|---:|---:|---:|
| climatology | 1:1 | 0.123 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0673 |
| climatology | 3:1 | 0.123 | 0.000 | 0.000 | 0.000 | 0.000 | 0.2019 |
| climatology | 5:1 | 0.123 | 0.000 | 0.000 | 0.000 | 0.000 | 0.3365 |
| climatology | 10:1 | 0.080 | 0.080 | 0.396 | 0.133 | 0.333 | 0.7128 |
| climatology | 20:1 | 0.000 | 0.067 | 1.000 | 0.126 | 1.000 | 0.9327 |
| xgboost | 1:1 | 0.450 | 0.476 | 0.201 | 0.283 | 0.028 | 0.0686 |
| xgboost | 3:1 | 0.195 | 0.361 | 0.665 | 0.468 | 0.124 | 0.1469 |
| xgboost | 5:1 | 0.115 | 0.308 | 0.773 | 0.441 | 0.169 | 0.1931 |
| xgboost | 10:1 | 0.062 | 0.262 | 0.900 | 0.406 | 0.231 | 0.2377 |
| xgboost | 20:1 | 0.038 | 0.253 | 0.920 | 0.397 | 0.245 | 0.2900 |
| random_forest | 1:1 | 0.440 | 0.466 | 0.068 | 0.118 | 0.010 | 0.0680 |
| random_forest | 3:1 | 0.215 | 0.350 | 0.697 | 0.466 | 0.134 | 0.1483 |
| random_forest | 5:1 | 0.155 | 0.319 | 0.775 | 0.452 | 0.164 | 0.1872 |
| random_forest | 10:1 | 0.055 | 0.264 | 0.908 | 0.409 | 0.232 | 0.2322 |
| random_forest | 20:1 | 0.038 | 0.241 | 0.920 | 0.382 | 0.257 | 0.3022 |
| temporal_cnn | 1:1 | 0.450 | 0.482 | 0.245 | 0.325 | 0.034 | 0.0685 |
| temporal_cnn | 3:1 | 0.232 | 0.369 | 0.653 | 0.472 | 0.119 | 0.1451 |
| temporal_cnn | 5:1 | 0.125 | 0.332 | 0.787 | 0.467 | 0.159 | 0.1782 |
| temporal_cnn | 10:1 | 0.067 | 0.280 | 0.882 | 0.425 | 0.212 | 0.2321 |
| temporal_cnn | 20:1 | 0.032 | 0.240 | 0.912 | 0.380 | 0.256 | 0.3126 |

[Reliability curve](artifacts/location_7/reliability_lead_time.png) · [full cost curves](artifacts/location_7/cost_curves_lead_time.png)

### Matched-cohort AQI ablation
| task/model | with AQI AP | no AQI AP | ΔAP (paired 95% day-block CI) | positives |
|---|---:|---:|---:|---:|
| nowcast/xgboost | 0.1467 | 0.1552 | -0.0085 [-0.1127, +0.0956] | 186 |
| nowcast/temporal_cnn | 0.1741 | 0.1537 | +0.0204 [-0.0401, +0.0809] | 186 |
| lead_time/xgboost | 0.2898 | 0.2229 | +0.0669 [-0.0434, +0.1772] | 32 |
| lead_time/temporal_cnn | 0.2833 | 0.1927 | +0.0906 [-0.0254, +0.2066] | 32 |

**Sparse-positive warning:** lead_time/xgboost (32 positives), lead_time/temporal_cnn (32 positives). These AQI delta intervals are potentially uninformative even when they exclude zero. Zero-positive test folds: nowcast=0/3, lead=0/3; pooled metrics rely on the other fixed folds.

**Operational decision:** NO-GO for unsupervised deployment; retrospective best AP nowcast=0.259, lead=0.411. Consider only latency-controlled shadow testing and locally chosen costs.

## location_8 — Visalia
Cohort: {"nowcast_rows": 168203, "nowcast_positives": 5367, "nowcast_fog_rate": 0.031907873224615496, "lead_complete_10h_issues": 6738, "lead_positives": 827, "aqi_matched_nowcast_rows": 26378, "aqi_matched_lead_issues": 1077, "aqi_channels": ["pm10", "pm2_5", "aerosol_optical_depth", "dust", "nitrogen_dioxide"]}
### Nowcast
| model | AP (95% block CI) | ROC-AUC (95% CI) | positives / n | Brier → calibrated | ECE → calibrated |
|---|---:|---:|---:|---:|---:|
| climatology | 0.0297 [0.0251, 0.0344] | 0.5335 [0.4959, 0.5710] | 2047 / 75692 | 0.0264 → 0.0264 | 0.0073 → 0.0073 |
| xgboost | 0.3454 [0.3026, 0.3882] | 0.9529 [0.9474, 0.9584] | 2047 / 75692 | 0.0919 → 0.0207 | 0.1595 → 0.0063 |
| random_forest | 0.3421 [0.2946, 0.3895] | 0.9512 [0.9455, 0.9569] | 2047 / 75692 | 0.0493 → 0.0214 | 0.0731 → 0.0060 |
| temporal_cnn | 0.3405 [0.2952, 0.3859] | 0.9514 [0.9457, 0.9572] | 2047 / 75692 | 0.1031 → 0.0209 | 0.1528 → 0.0067 |

Cost operating points (validation-selected thresholds; all ratios reported):
| model | FN:FP | mean threshold | precision | recall | F1 | alert rate | cost/issue |
|---|---:|---:|---:|---:|---:|---:|---:|
| climatology | 1:1 | 0.037 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0270 |
| climatology | 3:1 | 0.037 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0811 |
| climatology | 5:1 | 0.037 | 0.000 | 0.000 | 0.000 | 0.000 | 0.1352 |
| climatology | 10:1 | 0.037 | 0.000 | 0.000 | 0.000 | 0.000 | 0.2704 |
| climatology | 20:1 | 0.037 | 0.000 | 0.000 | 0.000 | 0.000 | 0.5409 |
| xgboost | 1:1 | 0.455 | 0.482 | 0.222 | 0.304 | 0.012 | 0.0275 |
| xgboost | 3:1 | 0.235 | 0.320 | 0.523 | 0.397 | 0.044 | 0.0688 |
| xgboost | 5:1 | 0.157 | 0.269 | 0.754 | 0.397 | 0.076 | 0.0886 |
| xgboost | 10:1 | 0.077 | 0.217 | 0.849 | 0.346 | 0.106 | 0.1236 |
| xgboost | 20:1 | 0.043 | 0.172 | 0.906 | 0.288 | 0.143 | 0.1693 |
| random_forest | 1:1 | 0.440 | 0.463 | 0.147 | 0.223 | 0.009 | 0.0277 |
| random_forest | 3:1 | 0.235 | 0.305 | 0.536 | 0.389 | 0.048 | 0.0707 |
| random_forest | 5:1 | 0.155 | 0.249 | 0.718 | 0.369 | 0.078 | 0.0968 |
| random_forest | 10:1 | 0.083 | 0.214 | 0.820 | 0.339 | 0.104 | 0.1301 |
| random_forest | 20:1 | 0.045 | 0.182 | 0.880 | 0.302 | 0.131 | 0.1716 |
| temporal_cnn | 1:1 | 0.477 | 0.468 | 0.227 | 0.305 | 0.013 | 0.0279 |
| temporal_cnn | 3:1 | 0.217 | 0.346 | 0.565 | 0.429 | 0.044 | 0.0642 |
| temporal_cnn | 5:1 | 0.153 | 0.282 | 0.743 | 0.408 | 0.071 | 0.0860 |
| temporal_cnn | 10:1 | 0.088 | 0.228 | 0.824 | 0.357 | 0.098 | 0.1232 |
| temporal_cnn | 20:1 | 0.047 | 0.186 | 0.900 | 0.308 | 0.131 | 0.1608 |

[Reliability curve](artifacts/location_8/reliability_nowcast.png) · [full cost curves](artifacts/location_8/cost_curves_nowcast.png)
### Lead Time
| model | AP (95% block CI) | ROC-AUC (95% CI) | positives / n | Brier → calibrated | ECE → calibrated |
|---|---:|---:|---:|---:|---:|
| climatology | 0.1133 [0.0997, 0.1269] | 0.5045 [0.4735, 0.5354] | 337 / 3033 | 0.0991 → 0.0991 | 0.0181 → 0.0181 |
| xgboost | 0.4654 [0.4096, 0.5212] | 0.8872 [0.8695, 0.9050] | 337 / 3033 | 0.1161 → 0.0719 | 0.1539 → 0.0234 |
| random_forest | 0.4944 [0.4367, 0.5521] | 0.9006 [0.8851, 0.9161] | 337 / 3033 | 0.0830 → 0.0735 | 0.0651 → 0.0247 |
| temporal_cnn | 0.5400 [0.4824, 0.5976] | 0.9025 [0.8871, 0.9179] | 337 / 3033 | 0.1594 → 0.0717 | 0.2496 → 0.0344 |

Cost operating points (validation-selected thresholds; all ratios reported):
| model | FN:FP | mean threshold | precision | recall | F1 | alert rate | cost/issue |
|---|---:|---:|---:|---:|---:|---:|---:|
| climatology | 1:1 | 0.132 | 0.000 | 0.000 | 0.000 | 0.000 | 0.1111 |
| climatology | 3:1 | 0.132 | 0.000 | 0.000 | 0.000 | 0.000 | 0.3333 |
| climatology | 5:1 | 0.132 | 0.000 | 0.000 | 0.000 | 0.000 | 0.5556 |
| climatology | 10:1 | 0.000 | 0.111 | 1.000 | 0.200 | 1.000 | 0.8889 |
| climatology | 20:1 | 0.000 | 0.111 | 1.000 | 0.200 | 1.000 | 0.8889 |
| xgboost | 1:1 | 0.453 | 0.570 | 0.401 | 0.470 | 0.078 | 0.1002 |
| xgboost | 3:1 | 0.200 | 0.400 | 0.774 | 0.527 | 0.215 | 0.2044 |
| xgboost | 5:1 | 0.107 | 0.334 | 0.887 | 0.485 | 0.295 | 0.2591 |
| xgboost | 10:1 | 0.053 | 0.307 | 0.914 | 0.459 | 0.331 | 0.3251 |
| xgboost | 20:1 | 0.030 | 0.272 | 0.935 | 0.421 | 0.382 | 0.4237 |
| random_forest | 1:1 | 0.388 | 0.530 | 0.528 | 0.529 | 0.111 | 0.1045 |
| random_forest | 3:1 | 0.220 | 0.406 | 0.712 | 0.517 | 0.195 | 0.2117 |
| random_forest | 5:1 | 0.135 | 0.354 | 0.858 | 0.501 | 0.269 | 0.2532 |
| random_forest | 10:1 | 0.048 | 0.311 | 0.902 | 0.463 | 0.322 | 0.3304 |
| random_forest | 20:1 | 0.030 | 0.279 | 0.926 | 0.429 | 0.369 | 0.4309 |
| temporal_cnn | 1:1 | 0.428 | 0.564 | 0.588 | 0.576 | 0.116 | 0.0963 |
| temporal_cnn | 3:1 | 0.210 | 0.411 | 0.780 | 0.538 | 0.211 | 0.1975 |
| temporal_cnn | 5:1 | 0.153 | 0.399 | 0.804 | 0.533 | 0.224 | 0.2433 |
| temporal_cnn | 10:1 | 0.047 | 0.313 | 0.908 | 0.465 | 0.322 | 0.3238 |
| temporal_cnn | 20:1 | 0.037 | 0.302 | 0.914 | 0.454 | 0.337 | 0.4263 |

[Reliability curve](artifacts/location_8/reliability_lead_time.png) · [full cost curves](artifacts/location_8/cost_curves_lead_time.png)

### Matched-cohort AQI ablation
| task/model | with AQI AP | no AQI AP | ΔAP (paired 95% day-block CI) | positives |
|---|---:|---:|---:|---:|
| nowcast/xgboost | 0.2586 | 0.3063 | -0.0477 [-0.0975, +0.0022] | 292 |
| nowcast/temporal_cnn | 0.3234 | 0.3072 | +0.0162 [-0.0350, +0.0673] | 292 |
| lead_time/xgboost | 0.4813 | 0.5521 | -0.0708 [-0.1300, -0.0116] | 43 |
| lead_time/temporal_cnn | 0.5558 | 0.2273 | +0.3285 [+0.1860, +0.4710] | 43 |

**Sparse-positive warning:** lead_time/xgboost (43 positives), lead_time/temporal_cnn (43 positives). These AQI delta intervals are potentially uninformative even when they exclude zero. Zero-positive test folds: nowcast=0/3, lead=0/3; pooled metrics rely on the other fixed folds.

**Operational decision:** NO-GO for unsupervised deployment; retrospective best AP nowcast=0.345, lead=0.540. Consider only latency-controlled shadow testing and locally chosen costs.

## location_9 — Hanford
Cohort: {"nowcast_rows": 171108, "nowcast_positives": 7536, "nowcast_fog_rate": 0.044042359211725925, "lead_complete_10h_issues": 7014, "lead_positives": 1197, "aqi_matched_nowcast_rows": 26757, "aqi_matched_lead_issues": 1097, "aqi_channels": ["pm10", "pm2_5", "aerosol_optical_depth", "dust", "nitrogen_dioxide"]}
### Nowcast
| model | AP (95% block CI) | ROC-AUC (95% CI) | positives / n | Brier → calibrated | ECE → calibrated |
|---|---:|---:|---:|---:|---:|
| climatology | 0.0397 [0.0344, 0.0450] | 0.5121 [0.4811, 0.5431] | 2897 / 76999 | 0.0363 → 0.0363 | 0.0103 → 0.0103 |
| xgboost | 0.4597 [0.4195, 0.4998] | 0.9589 [0.9548, 0.9631] | 2897 / 76999 | 0.0915 → 0.0261 | 0.1369 → 0.0121 |
| random_forest | 0.4016 [0.3611, 0.4421] | 0.9529 [0.9481, 0.9577] | 2897 / 76999 | 0.0508 → 0.0272 | 0.0709 → 0.0114 |
| temporal_cnn | 0.4969 [0.4541, 0.5396] | 0.9610 [0.9567, 0.9654] | 2897 / 76999 | 0.1082 → 0.0253 | 0.1633 → 0.0126 |

Cost operating points (validation-selected thresholds; all ratios reported):
| model | FN:FP | mean threshold | precision | recall | F1 | alert rate | cost/issue |
|---|---:|---:|---:|---:|---:|---:|---:|
| climatology | 1:1 | 0.050 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0376 |
| climatology | 3:1 | 0.050 | 0.000 | 0.000 | 0.000 | 0.000 | 0.1129 |
| climatology | 5:1 | 0.050 | 0.000 | 0.000 | 0.000 | 0.000 | 0.1881 |
| climatology | 10:1 | 0.050 | 0.000 | 0.000 | 0.000 | 0.000 | 0.3762 |
| climatology | 20:1 | 0.033 | 0.040 | 0.353 | 0.072 | 0.333 | 0.8071 |
| xgboost | 1:1 | 0.477 | 0.552 | 0.352 | 0.430 | 0.024 | 0.0351 |
| xgboost | 3:1 | 0.232 | 0.342 | 0.736 | 0.467 | 0.081 | 0.0831 |
| xgboost | 5:1 | 0.158 | 0.321 | 0.808 | 0.459 | 0.095 | 0.1005 |
| xgboost | 10:1 | 0.087 | 0.260 | 0.912 | 0.405 | 0.132 | 0.1309 |
| xgboost | 20:1 | 0.043 | 0.209 | 0.959 | 0.343 | 0.173 | 0.1680 |
| random_forest | 1:1 | 0.488 | 0.511 | 0.312 | 0.387 | 0.023 | 0.0371 |
| random_forest | 3:1 | 0.220 | 0.338 | 0.697 | 0.455 | 0.078 | 0.0855 |
| random_forest | 5:1 | 0.155 | 0.292 | 0.818 | 0.430 | 0.105 | 0.1089 |
| random_forest | 10:1 | 0.083 | 0.242 | 0.900 | 0.382 | 0.140 | 0.1435 |
| random_forest | 20:1 | 0.043 | 0.207 | 0.953 | 0.340 | 0.173 | 0.1726 |
| temporal_cnn | 1:1 | 0.453 | 0.502 | 0.526 | 0.514 | 0.039 | 0.0375 |
| temporal_cnn | 3:1 | 0.223 | 0.376 | 0.784 | 0.508 | 0.078 | 0.0734 |
| temporal_cnn | 5:1 | 0.150 | 0.339 | 0.825 | 0.481 | 0.092 | 0.0933 |
| temporal_cnn | 10:1 | 0.082 | 0.264 | 0.914 | 0.410 | 0.130 | 0.1281 |
| temporal_cnn | 20:1 | 0.040 | 0.223 | 0.948 | 0.361 | 0.160 | 0.1638 |

[Reliability curve](artifacts/location_9/reliability_nowcast.png) · [full cost curves](artifacts/location_9/cost_curves_nowcast.png)
### Lead Time
| model | AP (95% block CI) | ROC-AUC (95% CI) | positives / n | Brier → calibrated | ECE → calibrated |
|---|---:|---:|---:|---:|---:|
| climatology | 0.1626 [0.1472, 0.1781] | 0.5433 [0.5173, 0.5693] | 472 / 3157 | 0.1282 → 0.1282 | 0.0339 → 0.0339 |
| xgboost | 0.5957 [0.5498, 0.6415] | 0.9127 [0.9003, 0.9251] | 472 / 3157 | 0.1273 → 0.0826 | 0.1715 → 0.0425 |
| random_forest | 0.5863 [0.5378, 0.6348] | 0.9148 [0.9028, 0.9268] | 472 / 3157 | 0.0914 → 0.0826 | 0.0793 → 0.0372 |
| temporal_cnn | 0.6301 [0.5816, 0.6786] | 0.9222 [0.9112, 0.9333] | 472 / 3157 | 0.1446 → 0.0777 | 0.1924 → 0.0406 |

Cost operating points (validation-selected thresholds; all ratios reported):
| model | FN:FP | mean threshold | precision | recall | F1 | alert rate | cost/issue |
|---|---:|---:|---:|---:|---:|---:|---:|
| climatology | 1:1 | 0.187 | 0.000 | 0.000 | 0.000 | 0.000 | 0.1495 |
| climatology | 3:1 | 0.187 | 0.000 | 0.000 | 0.000 | 0.000 | 0.4485 |
| climatology | 5:1 | 0.063 | 0.139 | 0.619 | 0.227 | 0.667 | 0.8594 |
| climatology | 10:1 | 0.000 | 0.150 | 1.000 | 0.260 | 1.000 | 0.8505 |
| climatology | 20:1 | 0.000 | 0.150 | 1.000 | 0.260 | 1.000 | 0.8505 |
| xgboost | 1:1 | 0.437 | 0.595 | 0.663 | 0.627 | 0.167 | 0.1178 |
| xgboost | 3:1 | 0.163 | 0.453 | 0.898 | 0.603 | 0.296 | 0.2075 |
| xgboost | 5:1 | 0.080 | 0.427 | 0.932 | 0.586 | 0.326 | 0.2376 |
| xgboost | 10:1 | 0.030 | 0.382 | 0.964 | 0.547 | 0.377 | 0.2870 |
| xgboost | 20:1 | 0.015 | 0.362 | 0.968 | 0.527 | 0.399 | 0.3497 |
| random_forest | 1:1 | 0.448 | 0.567 | 0.661 | 0.611 | 0.174 | 0.1261 |
| random_forest | 3:1 | 0.143 | 0.456 | 0.900 | 0.606 | 0.295 | 0.2049 |
| random_forest | 5:1 | 0.108 | 0.440 | 0.922 | 0.596 | 0.313 | 0.2338 |
| random_forest | 10:1 | 0.058 | 0.399 | 0.949 | 0.561 | 0.356 | 0.2901 |
| random_forest | 20:1 | 0.017 | 0.359 | 0.966 | 0.523 | 0.403 | 0.3595 |
| temporal_cnn | 1:1 | 0.453 | 0.629 | 0.703 | 0.664 | 0.167 | 0.1064 |
| temporal_cnn | 3:1 | 0.205 | 0.468 | 0.888 | 0.613 | 0.284 | 0.2015 |
| temporal_cnn | 5:1 | 0.082 | 0.415 | 0.939 | 0.575 | 0.338 | 0.2439 |
| temporal_cnn | 10:1 | 0.027 | 0.377 | 0.960 | 0.541 | 0.381 | 0.2978 |
| temporal_cnn | 20:1 | 0.012 | 0.365 | 0.975 | 0.531 | 0.400 | 0.3301 |

[Reliability curve](artifacts/location_9/reliability_lead_time.png) · [full cost curves](artifacts/location_9/cost_curves_lead_time.png)

### Matched-cohort AQI ablation
| task/model | with AQI AP | no AQI AP | ΔAP (paired 95% day-block CI) | positives |
|---|---:|---:|---:|---:|
| nowcast/xgboost | 0.4074 | 0.3770 | +0.0304 [+0.0071, +0.0537] | 338 |
| nowcast/temporal_cnn | 0.4645 | 0.5137 | -0.0492 [-0.0969, -0.0015] | 338 |
| lead_time/xgboost | 0.4677 | 0.5710 | -0.1033 [-0.1887, -0.0179] | 50 |
| lead_time/temporal_cnn | 0.5516 | 0.3874 | +0.1642 [+0.0583, +0.2702] | 50 |

**Operational decision:** NO-GO for unsupervised deployment; retrospective best AP nowcast=0.497, lead=0.630. Consider only latency-controlled shadow testing and locally chosen costs.

## location_10 — Bakersfield
Cohort: {"nowcast_rows": 396364, "nowcast_positives": 7871, "nowcast_fog_rate": 0.01985800930457862, "lead_complete_10h_issues": 15398, "lead_positives": 1047, "aqi_matched_nowcast_rows": 26776, "aqi_matched_lead_issues": 1106, "aqi_channels": ["pm10", "pm2_5", "aerosol_optical_depth", "dust", "nitrogen_dioxide"]}
### Nowcast
| model | AP (95% block CI) | ROC-AUC (95% CI) | positives / n | Brier → calibrated | ECE → calibrated |
|---|---:|---:|---:|---:|---:|
| climatology | 0.0100 [0.0081, 0.0120] | 0.5551 [0.5138, 0.5964] | 1502 / 178364 | 0.0088 → 0.0088 | 0.0201 → 0.0201 |
| xgboost | 0.1266 [0.0924, 0.1608] | 0.9248 [0.9130, 0.9367] | 1502 / 178364 | 0.1184 → 0.0078 | 0.2344 → 0.0018 |
| random_forest | 0.1116 [0.0798, 0.1434] | 0.9265 [0.9120, 0.9409] | 1502 / 178364 | 0.0306 → 0.0079 | 0.0595 → 0.0020 |
| temporal_cnn | 0.1281 [0.0983, 0.1578] | 0.9378 [0.9250, 0.9507] | 1502 / 178364 | 0.0817 → 0.0079 | 0.1363 → 0.0013 |

Cost operating points (validation-selected thresholds; all ratios reported):
| model | FN:FP | mean threshold | precision | recall | F1 | alert rate | cost/issue |
|---|---:|---:|---:|---:|---:|---:|---:|
| climatology | 1:1 | 0.032 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0084 |
| climatology | 3:1 | 0.032 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0253 |
| climatology | 5:1 | 0.032 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0421 |
| climatology | 10:1 | 0.032 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0842 |
| climatology | 20:1 | 0.032 | 0.000 | 0.000 | 0.000 | 0.000 | 0.1684 |
| xgboost | 1:1 | 0.308 | 0.225 | 0.015 | 0.029 | 0.001 | 0.0087 |
| xgboost | 3:1 | 0.193 | 0.231 | 0.085 | 0.124 | 0.003 | 0.0255 |
| xgboost | 5:1 | 0.138 | 0.220 | 0.192 | 0.205 | 0.007 | 0.0397 |
| xgboost | 10:1 | 0.067 | 0.113 | 0.524 | 0.186 | 0.039 | 0.0746 |
| xgboost | 20:1 | 0.040 | 0.092 | 0.662 | 0.161 | 0.061 | 0.1120 |
| random_forest | 1:1 | 0.275 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0085 |
| random_forest | 3:1 | 0.188 | 0.188 | 0.042 | 0.069 | 0.002 | 0.0257 |
| random_forest | 5:1 | 0.145 | 0.173 | 0.143 | 0.156 | 0.007 | 0.0419 |
| random_forest | 10:1 | 0.082 | 0.116 | 0.417 | 0.181 | 0.030 | 0.0760 |
| random_forest | 20:1 | 0.045 | 0.085 | 0.619 | 0.149 | 0.061 | 0.1205 |
| temporal_cnn | 1:1 | 0.333 | 0.225 | 0.021 | 0.039 | 0.001 | 0.0089 |
| temporal_cnn | 3:1 | 0.193 | 0.227 | 0.098 | 0.137 | 0.004 | 0.0256 |
| temporal_cnn | 5:1 | 0.147 | 0.200 | 0.202 | 0.201 | 0.009 | 0.0404 |
| temporal_cnn | 10:1 | 0.082 | 0.129 | 0.503 | 0.206 | 0.033 | 0.0704 |
| temporal_cnn | 20:1 | 0.040 | 0.104 | 0.644 | 0.179 | 0.052 | 0.1067 |

[Reliability curve](artifacts/location_10/reliability_nowcast.png) · [full cost curves](artifacts/location_10/cost_curves_nowcast.png)
### Lead Time
| model | AP (95% block CI) | ROC-AUC (95% CI) | positives / n | Brier → calibrated | ECE → calibrated |
|---|---:|---:|---:|---:|---:|
| climatology | 0.0415 [0.0357, 0.0472] | 0.5353 [0.5027, 0.5680] | 265 / 6930 | 0.0394 → 0.0394 | 0.0514 → 0.0514 |
| xgboost | 0.2312 [0.1919, 0.2705] | 0.9001 [0.8820, 0.9182] | 265 / 6930 | 0.1086 → 0.0324 | 0.2285 → 0.0083 |
| random_forest | 0.2522 [0.2052, 0.2991] | 0.9103 [0.8971, 0.9234] | 265 / 6930 | 0.0491 → 0.0327 | 0.0708 → 0.0082 |
| temporal_cnn | 0.2650 [0.2183, 0.3118] | 0.9140 [0.8999, 0.9282] | 265 / 6930 | 0.1082 → 0.0321 | 0.1651 → 0.0089 |

Cost operating points (validation-selected thresholds; all ratios reported):
| model | FN:FP | mean threshold | precision | recall | F1 | alert rate | cost/issue |
|---|---:|---:|---:|---:|---:|---:|---:|
| climatology | 1:1 | 0.092 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0382 |
| climatology | 3:1 | 0.092 | 0.000 | 0.000 | 0.000 | 0.000 | 0.1147 |
| climatology | 5:1 | 0.092 | 0.000 | 0.000 | 0.000 | 0.000 | 0.1912 |
| climatology | 10:1 | 0.092 | 0.000 | 0.000 | 0.000 | 0.000 | 0.3824 |
| climatology | 20:1 | 0.060 | 0.044 | 0.385 | 0.079 | 0.333 | 0.7890 |
| xgboost | 1:1 | 0.408 | 0.394 | 0.140 | 0.206 | 0.014 | 0.0411 |
| xgboost | 3:1 | 0.207 | 0.331 | 0.340 | 0.335 | 0.039 | 0.1020 |
| xgboost | 5:1 | 0.140 | 0.232 | 0.623 | 0.338 | 0.103 | 0.1509 |
| xgboost | 10:1 | 0.085 | 0.199 | 0.706 | 0.310 | 0.136 | 0.2212 |
| xgboost | 20:1 | 0.032 | 0.154 | 0.857 | 0.262 | 0.212 | 0.2892 |
| random_forest | 1:1 | 0.407 | 0.343 | 0.087 | 0.139 | 0.010 | 0.0413 |
| random_forest | 3:1 | 0.203 | 0.299 | 0.313 | 0.306 | 0.040 | 0.1069 |
| random_forest | 5:1 | 0.137 | 0.226 | 0.585 | 0.326 | 0.099 | 0.1560 |
| random_forest | 10:1 | 0.065 | 0.173 | 0.777 | 0.283 | 0.172 | 0.2274 |
| random_forest | 20:1 | 0.028 | 0.144 | 0.906 | 0.248 | 0.241 | 0.2781 |
| temporal_cnn | 1:1 | 0.433 | 0.267 | 0.015 | 0.029 | 0.002 | 0.0392 |
| temporal_cnn | 3:1 | 0.198 | 0.275 | 0.464 | 0.346 | 0.065 | 0.1082 |
| temporal_cnn | 5:1 | 0.123 | 0.240 | 0.672 | 0.354 | 0.107 | 0.1442 |
| temporal_cnn | 10:1 | 0.083 | 0.211 | 0.781 | 0.332 | 0.142 | 0.1955 |
| temporal_cnn | 20:1 | 0.042 | 0.168 | 0.857 | 0.281 | 0.195 | 0.2722 |

[Reliability curve](artifacts/location_10/reliability_lead_time.png) · [full cost curves](artifacts/location_10/cost_curves_lead_time.png)

### Matched-cohort AQI ablation
| task/model | with AQI AP | no AQI AP | ΔAP (paired 95% day-block CI) | positives |
|---|---:|---:|---:|---:|
| nowcast/xgboost | 0.0442 | 0.0395 | +0.0047 [-0.0020, +0.0115] | 43 |
| nowcast/temporal_cnn | 0.0524 | 0.0574 | -0.0049 [-0.0407, +0.0309] | 43 |
| lead_time/xgboost | 0.1322 | 0.1420 | -0.0098 [-0.0687, +0.0492] | 10 |
| lead_time/temporal_cnn | 0.1068 | 0.1312 | -0.0244 [-0.1035, +0.0548] | 10 |

**Sparse-positive warning:** nowcast/xgboost (43 positives), nowcast/temporal_cnn (43 positives), lead_time/xgboost (10 positives), lead_time/temporal_cnn (10 positives). These AQI delta intervals are potentially uninformative even when they exclude zero. Zero-positive test folds: nowcast=2/3, lead=2/3; pooled metrics rely on the other fixed folds.

**Operational decision:** NO-GO for unsupervised deployment; retrospective best AP nowcast=0.128, lead=0.265. Consider only latency-controlled shadow testing and locally chosen costs.

## Cross-airport summary
| site/task | climatology AP (CI) | XGBoost AP (CI) | RF AP (CI) | CNN AP (CI) |
|---|---:|---:|---:|---:|
| location_6/nowcast | 0.025 [0.021, 0.030] | 0.302 [0.251, 0.353] | 0.264 [0.215, 0.312] | 0.234 [0.193, 0.275] |
| location_6/lead_time | 0.102 [0.089, 0.115] | 0.434 [0.379, 0.489] | 0.426 [0.370, 0.481] | 0.380 [0.326, 0.433] |
| location_7/nowcast | 0.020 [0.018, 0.023] | 0.229 [0.193, 0.264] | 0.253 [0.210, 0.296] | 0.259 [0.227, 0.291] |
| location_7/lead_time | 0.073 [0.065, 0.081] | 0.381 [0.338, 0.423] | 0.398 [0.355, 0.440] | 0.411 [0.369, 0.453] |
| location_8/nowcast | 0.030 [0.025, 0.034] | 0.345 [0.303, 0.388] | 0.342 [0.295, 0.390] | 0.341 [0.295, 0.386] |
| location_8/lead_time | 0.113 [0.100, 0.127] | 0.465 [0.410, 0.521] | 0.494 [0.437, 0.552] | 0.540 [0.482, 0.598] |
| location_9/nowcast | 0.040 [0.034, 0.045] | 0.460 [0.420, 0.500] | 0.402 [0.361, 0.442] | 0.497 [0.454, 0.540] |
| location_9/lead_time | 0.163 [0.147, 0.178] | 0.596 [0.550, 0.642] | 0.586 [0.538, 0.635] | 0.630 [0.582, 0.679] |
| location_10/nowcast | 0.010 [0.008, 0.012] | 0.127 [0.092, 0.161] | 0.112 [0.080, 0.143] | 0.128 [0.098, 0.158] |
| location_10/lead_time | 0.041 [0.036, 0.047] | 0.231 [0.192, 0.270] | 0.252 [0.205, 0.299] | 0.265 [0.218, 0.312] |

| site/task/model | AQI ΔAP (paired CI) | positives |
|---|---:|---:|
| location_6/nowcast/xgboost | -0.045 [-0.114, +0.025] | 234 |
| location_6/nowcast/temporal_cnn | -0.000 [-0.033, +0.033] | 234 |
| location_6/lead_time/xgboost | -0.094 [-0.185, -0.002] | 46 |
| location_6/lead_time/temporal_cnn | +0.304 [+0.177, +0.431] | 46 |
| location_7/nowcast/xgboost | -0.009 [-0.113, +0.096] | 186 |
| location_7/nowcast/temporal_cnn | +0.020 [-0.040, +0.081] | 186 |
| location_7/lead_time/xgboost | +0.067 [-0.043, +0.177] | 32 |
| location_7/lead_time/temporal_cnn | +0.091 [-0.025, +0.207] | 32 |
| location_8/nowcast/xgboost | -0.048 [-0.097, +0.002] | 292 |
| location_8/nowcast/temporal_cnn | +0.016 [-0.035, +0.067] | 292 |
| location_8/lead_time/xgboost | -0.071 [-0.130, -0.012] | 43 |
| location_8/lead_time/temporal_cnn | +0.329 [+0.186, +0.471] | 43 |
| location_9/nowcast/xgboost | +0.030 [+0.007, +0.054] | 338 |
| location_9/nowcast/temporal_cnn | -0.049 [-0.097, -0.002] | 338 |
| location_9/lead_time/xgboost | -0.103 [-0.189, -0.018] | 50 |
| location_9/lead_time/temporal_cnn | +0.164 [+0.058, +0.270] | 50 |
| location_10/nowcast/xgboost | +0.005 [-0.002, +0.011] | 43 |
| location_10/nowcast/temporal_cnn | -0.005 [-0.041, +0.031] | 43 |
| location_10/lead_time/xgboost | -0.010 [-0.069, +0.049] | 10 |
| location_10/lead_time/temporal_cnn | -0.024 [-0.104, +0.055] | 10 |

## Aggregate verdict
location_6 nowcast: strongest xgboost AP=0.302 (interval-separated from climatology); location_6 lead_time: strongest xgboost AP=0.434 (interval-separated from climatology); location_7 nowcast: strongest temporal_cnn AP=0.259 (interval-separated from climatology); location_7 lead_time: strongest temporal_cnn AP=0.411 (interval-separated from climatology); location_8 nowcast: strongest xgboost AP=0.345 (interval-separated from climatology); location_8 lead_time: strongest temporal_cnn AP=0.540 (interval-separated from climatology); location_9 nowcast: strongest temporal_cnn AP=0.497 (interval-separated from climatology); location_9 lead_time: strongest temporal_cnn AP=0.630 (interval-separated from climatology); location_10 nowcast: strongest temporal_cnn AP=0.128 (interval-separated from climatology); location_10 lead_time: strongest temporal_cnn AP=0.265 (interval-separated from climatology). Results are heterogeneous and retrospective; no site is approved for operations without prospective latency-controlled shadow validation.

Isolated directionally resolved intervals occurred for: location_6/lead_time/xgboost (n+=46), location_6/lead_time/temporal_cnn (n+=46), location_8/lead_time/xgboost (n+=43), location_8/lead_time/temporal_cnn (n+=43), location_9/nowcast/xgboost (n+=338), location_9/nowcast/temporal_cnn (n+=338), location_9/lead_time/xgboost (n+=50), location_9/lead_time/temporal_cnn (n+=50). No site/task had same-direction interval exclusion for both model families; where both excluded zero their signs conflicted. Lead cohorts had only 10–50 positives and 20 comparisons were inspected. Therefore this run finds no robust, credible AQI contribution anywhere, while retaining the isolated signed results rather than hiding them.

## Limitations
These are retrospective associations, not proof of causal aerosol effects. Reanalysis availability is optimistic; METAR reporting and missing complete mornings create selection effects. NOAA UTC is converted to Pacific local time, but the Open-Meteo export DST representation and the intended clock contract still require independent production verification. METAR times are floored to the hour and the last report is retained, so sub-hour label provenance is another deployment limitation. Fold-block CIs condition on the fitted predictions and reflect test-calendar block sampling, but not training/tuning variability, airport sampling, multiple-comparison selection, label-definition uncertainty, or future climate/regime shift. Isotonic calibration can overfit small positive validation sets. Cost ratios were not supplied, so all five scenarios are decision-support only. A wide interval or few positive days is explicitly treated as inconclusive, not as evidence of no effect.

## Reproduction
From the repository root run `bash v2/full/run_full.sh`. It regenerates JSON, report, calibration/cost curve PNGs and their underlying JSON arrays, CNN weights and the training log.