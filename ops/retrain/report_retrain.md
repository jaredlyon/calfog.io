# CalFog V2 — schema-compatible archive retrain

> **STATUS: SCHEMA-COMPATIBLE ONLY; NOT FAITHFUL; NOT PRODUCTION-READY.** The models were trained on ERA5 Archive reanalysis while the operational endpoint supplies forecast/analysis values. Matching names and non-null availability removes the null-channel blocker but does not resolve distribution shift. A forward-feed evaluation from the separately deployed shadow logger over a fog season is required before any production claim.

## Methodology
All five weather-only airport models use the shared `models/_shared/` protocol with seed 20260228. We read the exact approved variables from the local Open-Meteo ERA5 Archive weather files, already aligned to fixed UTC-8 (Etc/GMT+8), retaining rows through 2025-08-27 23:00:00 rather than re-pulling the archive API. We rebuilt 24-hour causal windows and evaluated nowcast and 18:00-to-next-morning lead tasks. Three expanding-origin folds have a 24-hour purge. Predictions are pooled out-of-fold. AP is primary and ROC-AUC secondary. Uncertainty is 1000 seeded whole-day block bootstrap resamples, never IID hourly sampling. Calibration is fold-local validation-fit isotonic regression. The complete validation-selected cost grid is FN:FP (1, 3, 5, 10, 20):1. CNN tuning uses 8 Ray Tune trials per site/task on the RTX GPU.

The target is GHCNh airport visibility decoded by one date-free rule into 18 ordinal categories, with <1/4 SM and 10+ SM censored endpoints. Each family predicts a category PMF; fog probability sums categories 0–4 (reported visibility <=1 SM). Serving calibrates only this derived fog probability with isotonic regression. Proxy RPS scores the uncalibrated category PMF; IEM labels use the same statute-mile decoder and hourly category minimum.

## Production availability contract
The live Forecast API was queried for seven recent past days plus one forecast day at every site, then queried again after artifact construction. Approval requires every returned value to be non-null at all five airports in both rounds. Calendar sine/cosine terms are deterministic derived inputs, not remote API fields.

**Approved, exact API order:** `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `precipitation`, `surface_pressure`, `vapour_pressure_deficit`, `wind_speed_10m`, `wind_speed_100m`, `wind_gusts_10m`, `weather_code`, `cloud_cover_low`

**Dropped:** `soil_temperature_0_to_7cm` (live Forecast field returned all NULL; archive-only layer is not servable), `soil_temperature_0cm` (live-served, but ERA5 Archive returned all NULL; excluded rather than substituted), `soil_temperature_6cm` (live-served, but ERA5 Archive returned all NULL; excluded rather than substituted)

Temperature, RH, dew point, precipitation, surface pressure, VPD, 10 m/100 m wind, gusts, weather code, and low cloud passed the all-site non-null and Archive-reproducibility rule. Forecast served the 0 cm and 6 cm soil fields, but ERA5 Archive returned them entirely NULL even in a dedicated feasibility probe. Because `train_source=era5_archive` is mandatory, both replacement soil fields were explicitly dropped rather than silently synthesized from a different depth layer. This is a narrower no-soil schema and an unavoidable deviation from the requested soil replacement; it preserves source honesty.

## Archive evaluation and skill cost versus family studies
The table reports AP with marginal per-model day-block confidence intervals from this retrain and the corresponding original-feature-set AP. Deltas are descriptive across two completed runs; they are not a paired refit uncertainty interval.

| site/task/model | retrain AP [95% CI] | family study AP | Δ AP |
|---|---:|---:|---:|
| location_6/nowcast/climatology | 0.0254 [0.0225, 0.0282] | 0.0286 | -0.0032 |
| location_6/nowcast/xgboost | 0.3152 [0.2790, 0.3515] | 0.3166 | -0.0014 |
| location_6/nowcast/random_forest | 0.3150 [0.2764, 0.3536] | 0.3269 | -0.0119 |
| location_6/nowcast/temporal_cnn | 0.3178 [0.2774, 0.3581] | 0.3511 | -0.0333 |
| location_6/lead_time/climatology | 0.1162 [0.1046, 0.1278] | 0.1231 | -0.0069 |
| location_6/lead_time/xgboost | 0.5091 [0.4588, 0.5594] | 0.5233 | -0.0143 |
| location_6/lead_time/random_forest | 0.4851 [0.4372, 0.5330] | 0.5132 | -0.0281 |
| location_6/lead_time/temporal_cnn | 0.5319 [0.4823, 0.5815] | 0.5194 | +0.0125 |
| location_7/nowcast/climatology | 0.0218 [0.0191, 0.0245] | 0.0224 | -0.0007 |
| location_7/nowcast/xgboost | 0.2709 [0.2360, 0.3058] | 0.2849 | -0.0140 |
| location_7/nowcast/random_forest | 0.2640 [0.2274, 0.3005] | 0.2736 | -0.0096 |
| location_7/nowcast/temporal_cnn | 0.2591 [0.2228, 0.2955] | 0.2620 | -0.0029 |
| location_7/lead_time/climatology | 0.0729 [0.0651, 0.0808] | 0.0662 | +0.0067 |
| location_7/lead_time/xgboost | 0.3662 [0.3234, 0.4089] | 0.3833 | -0.0172 |
| location_7/lead_time/random_forest | 0.3597 [0.3177, 0.4017] | 0.3684 | -0.0087 |
| location_7/lead_time/temporal_cnn | 0.3818 [0.3391, 0.4245] | 0.3716 | +0.0102 |
| location_8/nowcast/climatology | 0.0378 [0.0343, 0.0412] | 0.0430 | -0.0052 |
| location_8/nowcast/xgboost | 0.4347 [0.4010, 0.4685] | 0.4494 | -0.0147 |
| location_8/nowcast/random_forest | 0.4190 [0.3844, 0.4537] | 0.4281 | -0.0090 |
| location_8/nowcast/temporal_cnn | 0.4213 [0.3833, 0.4592] | 0.4474 | -0.0261 |
| location_8/lead_time/climatology | 0.1357 [0.1241, 0.1472] | 0.1461 | -0.0104 |
| location_8/lead_time/xgboost | 0.5652 [0.5216, 0.6089] | 0.5753 | -0.0100 |
| location_8/lead_time/random_forest | 0.5594 [0.5159, 0.6028] | 0.5654 | -0.0061 |
| location_8/lead_time/temporal_cnn | 0.5803 [0.5355, 0.6252] | 0.5797 | +0.0006 |
| location_9/nowcast/climatology | 0.0517 [0.0468, 0.0566] | 0.0453 | +0.0064 |
| location_9/nowcast/xgboost | 0.5082 [0.4754, 0.5411] | 0.5279 | -0.0196 |
| location_9/nowcast/random_forest | 0.5077 [0.4739, 0.5414] | 0.5356 | -0.0280 |
| location_9/nowcast/temporal_cnn | 0.4777 [0.4421, 0.5133] | 0.5368 | -0.0591 |
| location_9/lead_time/climatology | 0.1675 [0.1540, 0.1810] | 0.1759 | -0.0084 |
| location_9/lead_time/xgboost | 0.6652 [0.6271, 0.7032] | 0.6862 | -0.0211 |
| location_9/lead_time/random_forest | 0.6634 [0.6265, 0.7003] | 0.6589 | +0.0046 |
| location_9/lead_time/temporal_cnn | 0.6432 [0.6041, 0.6823] | 0.6832 | -0.0400 |
| location_10/nowcast/climatology | 0.0118 [0.0099, 0.0138] | 0.0120 | -0.0001 |
| location_10/nowcast/xgboost | 0.1670 [0.1292, 0.2048] | 0.1838 | -0.0168 |
| location_10/nowcast/random_forest | 0.1686 [0.1306, 0.2066] | 0.1794 | -0.0108 |
| location_10/nowcast/temporal_cnn | 0.1378 [0.1059, 0.1698] | 0.1165 | +0.0213 |
| location_10/lead_time/climatology | 0.0426 [0.0363, 0.0488] | 0.0397 | +0.0029 |
| location_10/lead_time/xgboost | 0.2699 [0.2186, 0.3211] | 0.2908 | -0.0210 |
| location_10/lead_time/random_forest | 0.2324 [0.1879, 0.2769] | 0.2559 | -0.0235 |
| location_10/lead_time/temporal_cnn | 0.2343 [0.1928, 0.2759] | 0.2505 | -0.0162 |

## Approximate distribution-shift cross-check
**This is an approximate, non-vintage estimate—not evidence of faithfulness.** Open-Meteo Historical Forecast supplies a best-match historical forecast/analysis series, not the exact vintage that would have been available at each issue time. Each 24-hour feature window ends at issue time, but model-run vintages are not preserved. Archival METAR labels are fetched from the Iowa Environmental Mesonet. No missing proxy variable is silently substituted; a missing channel makes that site cross-check not evaluated.

| site/task/model | archive OOF AP | HF-proxy AP | proxy − archive | HF-proxy RPS | proxy positives/n |
|---|---:|---:|---:|---:|---:|
| location_6/nowcast/climatology | 0.0254 | 0.0930 | +0.0677 | 0.1749 | 139/1494 |
| location_6/nowcast/xgboost | 0.3152 | 0.2939 | -0.0213 | 0.1049 | 139/1494 |
| location_6/nowcast/temporal_cnn | 0.3178 | 0.1630 | -0.1548 | 0.1375 | 139/1494 |
| location_6/nowcast/random_forest | 0.3150 | 0.3225 | +0.0075 | 0.1728 | 139/1494 |
| location_6/lead_time/climatology | 0.1162 | 0.1429 | +0.0267 | 0.1372 | 2/14 |
| location_6/lead_time/xgboost | 0.5091 | 0.8333 | +0.3243 | 0.1799 | 2/14 |
| location_6/lead_time/temporal_cnn | 0.5319 | 0.8333 | +0.3014 | 0.2650 | 2/14 |
| location_6/lead_time/random_forest | 0.4851 | 1.0000 | +0.5149 | 0.3089 | 2/14 |
| location_7/nowcast/climatology | 0.0218 | 0.1708 | +0.1490 | 0.2538 | 308/1803 |
| location_7/nowcast/xgboost | 0.2709 | 0.5360 | +0.2651 | 0.1173 | 308/1803 |
| location_7/nowcast/temporal_cnn | 0.2591 | 0.2971 | +0.0380 | 0.1461 | 308/1803 |
| location_7/nowcast/random_forest | 0.2640 | 0.5000 | +0.2360 | 0.1678 | 308/1803 |
| location_7/lead_time/climatology | 0.0729 | 0.4419 | +0.3689 | 0.3386 | 19/43 |
| location_7/lead_time/xgboost | 0.3662 | 0.7599 | +0.3937 | 0.1600 | 19/43 |
| location_7/lead_time/temporal_cnn | 0.3818 | 0.6320 | +0.2502 | 0.1726 | 19/43 |
| location_7/lead_time/random_forest | 0.3597 | 0.8689 | +0.5092 | 0.1909 | 19/43 |
| location_8/nowcast/climatology | 0.0378 | 0.1161 | +0.0784 | 0.1885 | 172/1481 |
| location_8/nowcast/xgboost | 0.4347 | 0.2849 | -0.1498 | 0.1434 | 172/1481 |
| location_8/nowcast/temporal_cnn | 0.4213 | 0.2087 | -0.2125 | 0.1446 | 172/1481 |
| location_8/nowcast/random_forest | 0.4190 | 0.2493 | -0.1697 | 0.2217 | 172/1481 |
| location_8/lead_time/climatology | 0.1357 | 0.2308 | +0.0951 | 0.1812 | 3/13 |
| location_8/lead_time/xgboost | 0.5652 | 0.5769 | +0.0117 | 0.2288 | 3/13 |
| location_8/lead_time/temporal_cnn | 0.5803 | 0.5769 | -0.0034 | 0.3071 | 3/13 |
| location_8/lead_time/random_forest | 0.5594 | 0.4658 | -0.0935 | 0.3076 | 3/13 |
| location_9/nowcast/climatology | 0.0517 | 0.1051 | +0.0534 | 0.1796 | 143/1361 |
| location_9/nowcast/xgboost | 0.5082 | 0.2730 | -0.2353 | 0.1275 | 143/1361 |
| location_9/nowcast/temporal_cnn | 0.4777 | 0.2512 | -0.2265 | 0.1357 | 143/1361 |
| location_9/nowcast/random_forest | 0.5077 | 0.2953 | -0.2124 | 0.1915 | 143/1361 |
| location_9/lead_time/climatology | 0.1675 | 0.2500 | +0.0825 | 0.1759 | 1/4 |
| location_9/lead_time/xgboost | 0.6652 | 0.3333 | -0.3318 | 0.1380 | 1/4 |
| location_9/lead_time/temporal_cnn | 0.6432 | 1.0000 | +0.3568 | 0.1109 | 1/4 |
| location_9/lead_time/random_forest | 0.6634 | 1.0000 | +0.3366 | 0.1409 | 1/4 |
| location_10/nowcast/climatology | 0.0118 | 0.0621 | +0.0502 | 0.1777 | 97/1563 |
| location_10/nowcast/xgboost | 0.1670 | 0.1801 | +0.0131 | 0.1050 | 97/1563 |
| location_10/nowcast/temporal_cnn | 0.1378 | 0.1603 | +0.0225 | 0.1524 | 97/1563 |
| location_10/nowcast/random_forest | 0.1686 | 0.1926 | +0.0240 | 0.1733 | 97/1563 |
| location_10/lead_time/climatology | 0.0426 | 0.0500 | +0.0074 | 0.1543 | 1/20 |
| location_10/lead_time/xgboost | 0.2699 | 0.1667 | -0.1032 | 0.1358 | 1/20 |
| location_10/lead_time/temporal_cnn | 0.2343 | 0.1250 | -0.1093 | 0.1801 | 1/20 |
| location_10/lead_time/random_forest | 0.2324 | 0.5000 | +0.2676 | 0.2601 | 1/20 |

The proxy-minus-archive delta mixes distribution shift, a different calendar/season, label availability, and sampling uncertainty. It is only a present-day warning signal. It cannot prove equivalence or quantify the vintage forecast penalty cleanly.

## Serving artifacts and limitations
Each committed `serving/location_*` directory contains serialized XGBoost, temporal-CNN, preprocessing, and isotonic-calibrator artifacts for both tasks, plus a strictly ordered feature spec and metadata. RandomForest pickles are optional, generated locally by a full retrain, and excluded from Git; resumed proxy checks score RF only when its pickle is present. Published RF scores describe the original complete local run, not the RF-free committed bundle. Every metadata file sets `faithful=false`, `train_source="era5_archive"`, and `production_ready=false`. The re-probe proves only that required schema fields are currently populated.

Hourly METAR labels record any valid fog report (minimum valid visibility), not the last report; nowcasts are hourly interval targets and mornings require ten represented hours. Other limitations include METAR time flooring, airport/grid mismatch, rare positive events, fitted-prediction bootstrap intervals that omit training variability, possible calibration instability, and changing upstream forecast models. Thresholds under all cost ratios are decision-support scenarios, not chosen operational utilities.

## Required next step
Continue the separately deployed forward-feed shadow logger through a representative fog season, preserve issue-time vintages and publication latency, join prospective METAR outcomes, and compare drift, AP, calibration, and operating costs. Only that forward-feed evidence can establish faithfulness. Until then this build is schema-compatible only and must not be described as production-ready.

## Reproduction
Run `bash ops/retrain/run_retrain.sh` from the repository root. Network fetch manifests include parameters, resolved URLs, response status, row counts, and response SHA-256 hashes. All outputs remain under `ops/retrain/`.