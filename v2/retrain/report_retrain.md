# CalFog V2 — schema-compatible archive retrain

> **STATUS: SCHEMA-COMPATIBLE ONLY; NOT FAITHFUL; NOT PRODUCTION-READY.** The models were trained on ERA5 Archive reanalysis while the operational endpoint supplies forecast/analysis values. Matching names and non-null availability removes the null-channel blocker but does not resolve distribution shift. A forward-feed evaluation from the separately deployed shadow logger over a fog season is required before any production claim.

## Methodology
All five weather-only airport models use the existing `v2/pipeline.py` and `v2/full` harness with seed 20260228. We fetched the approved variables from the Open-Meteo ERA5 Archive from 1980-01-01 through 2025-08-27, rebuilt 24-hour causal windows, and evaluated nowcast and 18:00-to-next-morning lead tasks. Three expanding-origin folds have a 24-hour purge. Predictions are pooled out-of-fold. AP is primary and ROC-AUC secondary. Uncertainty is 1000 seeded whole-day block bootstrap resamples, never IID hourly sampling. Calibration is fold-local validation-fit isotonic regression. The complete validation-selected cost grid is FN:FP (1, 3, 5, 10, 20):1. CNN tuning uses 8 Ray Tune trials per site/task on the RTX GPU.

## Production availability contract
The live Forecast API was queried for seven recent past days plus one forecast day at every site, then queried again after artifact construction. Approval requires every returned value to be non-null at all five airports in both rounds. Calendar sine/cosine terms are deterministic derived inputs, not remote API fields.

**Approved, exact API order:** `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `precipitation`, `surface_pressure`, `vapour_pressure_deficit`, `wind_speed_10m`, `wind_speed_100m`, `wind_gusts_10m`, `weather_code`, `cloud_cover_low`

**Dropped:** `soil_temperature_0_to_7cm` (live Forecast field returned all NULL; archive-only layer is not servable), `soil_temperature_0cm` (live-served, but ERA5 Archive returned all NULL; excluded rather than substituted), `soil_temperature_6cm` (live-served, but ERA5 Archive returned all NULL; excluded rather than substituted)

Temperature, RH, dew point, precipitation, surface pressure, VPD, 10 m/100 m wind, gusts, weather code, and low cloud passed the all-site non-null and Archive-reproducibility rule. Forecast served the 0 cm and 6 cm soil fields, but ERA5 Archive returned them entirely NULL even in a dedicated feasibility probe. Because `train_source=era5_archive` is mandatory, both replacement soil fields were explicitly dropped rather than silently synthesized from a different depth layer. This is a narrower no-soil schema and an unavoidable deviation from the requested soil replacement; it preserves source honesty.

## Archive evaluation and skill cost versus v2/full
The table reports AP with paired-day-block confidence intervals from this retrain and the corresponding old-feature-set AP. Deltas are descriptive across two completed runs; they are not a paired refit uncertainty interval.

| site/task/model | retrain AP [95% CI] | v2/full AP | Δ AP |
|---|---:|---:|---:|
| location_6/nowcast/climatology | 0.0255 [0.0205, 0.0305] | 0.0255 | +0.0000 |
| location_6/nowcast/xgboost | 0.2617 [0.2168, 0.3066] | 0.3017 | -0.0400 |
| location_6/nowcast/random_forest | 0.2579 [0.2115, 0.3042] | 0.2637 | -0.0058 |
| location_6/nowcast/temporal_cnn | 0.2608 [0.2179, 0.3037] | 0.2339 | +0.0269 |
| location_6/lead_time/climatology | 0.1019 [0.0884, 0.1153] | 0.1019 | +0.0000 |
| location_6/lead_time/xgboost | 0.4092 [0.3525, 0.4659] | 0.4341 | -0.0249 |
| location_6/lead_time/random_forest | 0.4318 [0.3759, 0.4878] | 0.4255 | +0.0063 |
| location_6/lead_time/temporal_cnn | 0.4524 [0.3943, 0.5106] | 0.3798 | +0.0727 |
| location_7/nowcast/climatology | 0.0203 [0.0176, 0.0230] | 0.0203 | +0.0000 |
| location_7/nowcast/xgboost | 0.2283 [0.1892, 0.2673] | 0.2287 | -0.0005 |
| location_7/nowcast/random_forest | 0.2459 [0.2057, 0.2862] | 0.2531 | -0.0071 |
| location_7/nowcast/temporal_cnn | 0.2678 [0.2325, 0.3030] | 0.2591 | +0.0087 |
| location_7/lead_time/climatology | 0.0728 [0.0653, 0.0804] | 0.0728 | +0.0000 |
| location_7/lead_time/xgboost | 0.3985 [0.3538, 0.4431] | 0.3806 | +0.0178 |
| location_7/lead_time/random_forest | 0.3935 [0.3495, 0.4374] | 0.3975 | -0.0041 |
| location_7/lead_time/temporal_cnn | 0.3942 [0.3511, 0.4374] | 0.4111 | -0.0168 |
| location_8/nowcast/climatology | 0.0297 [0.0248, 0.0346] | 0.0297 | +0.0000 |
| location_8/nowcast/xgboost | 0.3670 [0.3194, 0.4146] | 0.3454 | +0.0216 |
| location_8/nowcast/random_forest | 0.3344 [0.2864, 0.3825] | 0.3421 | -0.0076 |
| location_8/nowcast/temporal_cnn | 0.3866 [0.3408, 0.4325] | 0.3405 | +0.0461 |
| location_8/lead_time/climatology | 0.1133 [0.0997, 0.1269] | 0.1133 | +0.0000 |
| location_8/lead_time/xgboost | 0.5420 [0.4849, 0.5992] | 0.4654 | +0.0767 |
| location_8/lead_time/random_forest | 0.4918 [0.4357, 0.5478] | 0.4944 | -0.0026 |
| location_8/lead_time/temporal_cnn | 0.4846 [0.4318, 0.5374] | 0.5400 | -0.0553 |
| location_9/nowcast/climatology | 0.0397 [0.0346, 0.0449] | 0.0397 | +0.0000 |
| location_9/nowcast/xgboost | 0.4432 [0.4048, 0.4815] | 0.4597 | -0.0165 |
| location_9/nowcast/random_forest | 0.3783 [0.3381, 0.4186] | 0.4016 | -0.0232 |
| location_9/nowcast/temporal_cnn | 0.5095 [0.4698, 0.5493] | 0.4969 | +0.0127 |
| location_9/lead_time/climatology | 0.1626 [0.1464, 0.1788] | 0.1626 | +0.0000 |
| location_9/lead_time/xgboost | 0.5470 [0.5016, 0.5924] | 0.5957 | -0.0487 |
| location_9/lead_time/random_forest | 0.5787 [0.5330, 0.6244] | 0.5863 | -0.0076 |
| location_9/lead_time/temporal_cnn | 0.6564 [0.6088, 0.7039] | 0.6301 | +0.0263 |
| location_10/nowcast/climatology | 0.0100 [0.0081, 0.0120] | 0.0100 | +0.0000 |
| location_10/nowcast/xgboost | 0.1358 [0.1030, 0.1686] | 0.1266 | +0.0092 |
| location_10/nowcast/random_forest | 0.1134 [0.0841, 0.1426] | 0.1116 | +0.0018 |
| location_10/nowcast/temporal_cnn | 0.1438 [0.1124, 0.1752] | 0.1281 | +0.0157 |
| location_10/lead_time/climatology | 0.0415 [0.0355, 0.0475] | 0.0415 | +0.0000 |
| location_10/lead_time/xgboost | 0.2718 [0.2207, 0.3229] | 0.2312 | +0.0406 |
| location_10/lead_time/random_forest | 0.2524 [0.2047, 0.3002] | 0.2522 | +0.0003 |
| location_10/lead_time/temporal_cnn | 0.2449 [0.2041, 0.2856] | 0.2650 | -0.0202 |

## Approximate distribution-shift cross-check
**This is an approximate, non-vintage estimate—not evidence of faithfulness.** Open-Meteo Historical Forecast supplies a best-match historical forecast/analysis series, not the exact vintage that would have been available at each issue time. Each 24-hour feature window ends at issue time, but model-run vintages are not preserved. Archival METAR labels are fetched from the Iowa Environmental Mesonet. No missing proxy variable is silently substituted; a missing channel makes that site cross-check not evaluated.

| site/task/model | archive OOF AP | HF-proxy AP | proxy − archive | proxy positives/n |
|---|---:|---:|---:|---:|
| location_6/nowcast/climatology | 0.0255 | 0.1452 | +0.1197 | 308/2121 |
| location_6/nowcast/xgboost | 0.2617 | 0.4510 | +0.1892 | 308/2121 |
| location_6/nowcast/random_forest | 0.2579 | 0.3323 | +0.0745 | 308/2121 |
| location_6/nowcast/temporal_cnn | 0.2608 | 0.3018 | +0.0410 | 308/2121 |
| location_6/lead_time/climatology | 0.1019 | 0.4943 | +0.3924 | 43/87 |
| location_6/lead_time/xgboost | 0.4092 | 0.7295 | +0.3203 | 43/87 |
| location_6/lead_time/random_forest | 0.4318 | 0.6778 | +0.2459 | 43/87 |
| location_6/lead_time/temporal_cnn | 0.4524 | 0.7410 | +0.2886 | 43/87 |
| location_7/nowcast/climatology | 0.0203 | 0.1259 | +0.1056 | 272/2160 |
| location_7/nowcast/xgboost | 0.2283 | 0.3146 | +0.0863 | 272/2160 |
| location_7/nowcast/random_forest | 0.2459 | 0.2638 | +0.0178 | 272/2160 |
| location_7/nowcast/temporal_cnn | 0.2678 | 0.2291 | -0.0387 | 272/2160 |
| location_7/lead_time/climatology | 0.0728 | 0.4222 | +0.3494 | 38/90 |
| location_7/lead_time/xgboost | 0.3985 | 0.7948 | +0.3964 | 38/90 |
| location_7/lead_time/random_forest | 0.3935 | 0.6699 | +0.2764 | 38/90 |
| location_7/lead_time/temporal_cnn | 0.3942 | 0.6940 | +0.2998 | 38/90 |
| location_8/nowcast/climatology | 0.0297 | 0.1843 | +0.1545 | 398/2160 |
| location_8/nowcast/xgboost | 0.3670 | 0.5340 | +0.1670 | 398/2160 |
| location_8/nowcast/random_forest | 0.3344 | 0.2616 | -0.0728 | 398/2160 |
| location_8/nowcast/temporal_cnn | 0.3866 | 0.3248 | -0.0618 | 398/2160 |
| location_8/lead_time/climatology | 0.1133 | 0.4778 | +0.3645 | 43/90 |
| location_8/lead_time/xgboost | 0.5420 | 0.7828 | +0.2407 | 43/90 |
| location_8/lead_time/random_forest | 0.4918 | 0.6755 | +0.1838 | 43/90 |
| location_8/lead_time/temporal_cnn | 0.4846 | 0.8012 | +0.3166 | 43/90 |
| location_9/nowcast/climatology | 0.0397 | 0.1897 | +0.1500 | 393/2072 |
| location_9/nowcast/xgboost | 0.4432 | 0.4494 | +0.0063 | 393/2072 |
| location_9/nowcast/random_forest | 0.3783 | 0.3781 | -0.0002 | 393/2072 |
| location_9/nowcast/temporal_cnn | 0.5095 | 0.3210 | -0.1885 | 393/2072 |
| location_9/lead_time/climatology | 0.1626 | 0.5116 | +0.3490 | 44/86 |
| location_9/lead_time/xgboost | 0.5470 | 0.8023 | +0.2553 | 44/86 |
| location_9/lead_time/random_forest | 0.5787 | 0.7271 | +0.1484 | 44/86 |
| location_9/lead_time/temporal_cnn | 0.6564 | 0.7354 | +0.0790 | 44/86 |
| location_10/nowcast/climatology | 0.0100 | 0.0935 | +0.0835 | 202/2160 |
| location_10/nowcast/xgboost | 0.1358 | 0.3258 | +0.1900 | 202/2160 |
| location_10/nowcast/random_forest | 0.1134 | 0.2480 | +0.1346 | 202/2160 |
| location_10/nowcast/temporal_cnn | 0.1438 | 0.3477 | +0.2039 | 202/2160 |
| location_10/lead_time/climatology | 0.0415 | 0.3444 | +0.3030 | 31/90 |
| location_10/lead_time/xgboost | 0.2718 | 0.5519 | +0.2800 | 31/90 |
| location_10/lead_time/random_forest | 0.2524 | 0.4783 | +0.2258 | 31/90 |
| location_10/lead_time/temporal_cnn | 0.2449 | 0.4749 | +0.2300 | 31/90 |

The proxy-minus-archive delta mixes distribution shift, a different calendar/season, label availability, and sampling uncertainty. It is only a present-day warning signal. It cannot prove equivalence or quantify the vintage forecast penalty cleanly.

## Serving artifacts and limitations
Each `serving/location_*` directory contains serialized XGBoost, RandomForest, temporal-CNN, preprocessing, and isotonic-calibrator artifacts for both tasks, plus a strictly ordered feature spec and metadata. Every metadata file sets `faithful=false`, `train_source="era5_archive"`, and `production_ready=false`. The re-probe proves only that required schema fields are currently populated.

Other limitations include METAR time flooring, airport/grid mismatch, rare positive events, fitted-prediction bootstrap intervals that omit training variability, possible calibration instability, and changing upstream forecast models. Thresholds under all cost ratios are decision-support scenarios, not chosen operational utilities.

## Required next step
Continue the separately deployed forward-feed shadow logger through a representative fog season, preserve issue-time vintages and publication latency, join prospective METAR outcomes, and compare drift, AP, calibration, and operating costs. Only that forward-feed evidence can establish faithfulness. Until then this build is schema-compatible only and must not be described as production-ready.

## Reproduction
Run `bash v2/retrain/run_retrain.sh` from the repository root. Network fetch manifests include parameters, resolved URLs, response status, row counts, and response SHA-256 hashes. All outputs remain under `v2/retrain/`.