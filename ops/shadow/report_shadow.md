# CalFog V2 private shadow-mode report

**Decision: NO-GO for a limited soft launch at every site. Historical replay only; not production-ready.**

This is a historical replay with **non-vintage forecasts**, not forward/prospective evidence. Every scored group has `vintage_faithful=false` and `faithful=false`. Keep probabilities private and accumulate prospective evidence before reconsidering deployment.

## What was measured—and what was not

The sole numerical source for this report is [`shadow_metrics_historical.json`](shadow_metrics_historical.json), generated at `2026-10-24T21:02:10.373325+00:00`. Its half-open UTC issue cohort is **[2025-12-01T08:00:00+00:00, 2026-03-01T08:00:00+00:00)**, with fixed-PST issue days (`Etc/GMT+8`), not a daylight-saving clock. The feed is `Open-Meteo Historical-Forecast best-match (non-vintage approximate proxy)`. Predictions are deduplicated by `first eligible append per site/task/issue/model/feed` and scored only where an eligible outcome joins.

The target is GHCNh visibility in **18 ordinal categories** (`ordinal-18`), with **fog = reported visibility ≤ 1 statute mile**, equivalently category ≤4. This replaces the old metric-threshold label definition. The replay verification observations come from **IEM ASOS/METAR**, decoded onto that ordinal contract; they are not a new GHCNh verification extract. Hourly labels use the minimum decoded category bounds. The lead target is the minimum over all ten complete fixed-PST hours 00–09 of the following morning (any fog in that window). Unknown binary outcomes are excluded; a known fog outcome can still have an unresolved ordinal category.

The metrics record the exact label contract as:

> decoded ordinal hourly minimum; fog category <=4; lead all ten fixed-PST hours 00-09; forward finalization lag 1h

Serving models are XGBoost and temporal CNN, built by `build_serving.py`. They predict an ordinal probability mass function (PMF); the serving pipeline derives and calibrates the binary fog probability from it. The current builder fits preprocessing and base models before a purged chronological calibration holdout, learns isotonic fog calibration on that holdout, and does not perform an all-history refit. Scoring the historical replay does not itself establish that every replay issue is an independent out-of-time evaluation of those serving models.

Open-Meteo Historical-Forecast best-match is a **non-vintage approximate proxy**: it does not establish which forecast initialization was available at the historical issue time. A feature timestamp no later than its issue time is necessary but does not prove historical availability. Raw responses and retrieval provenance are retained separately in `raw_snapshots/` and `fetch_manifest.json`.

## Immediate replay results

There are **11,154 joined site/task issues per model family**, of which **10,590 have a resolved ordinal category**. These counts sum `n_issues` and `n_ordinal` across site/task groups for a single family, avoiding double-counting the same verification outcomes across model families; they are not totals of all emitted predictions.

The JSON field `live_ap` is labeled **replay AP** here: its name is not evidence of live operation. AP intervals use 1,000 `Etc/GMT+8 issue day` resamples with the recorded method, `shared whole-day block bootstrap; point-centered 95% absolute-deviation interval`. `retrospective_ap` is the saved model evaluation reference; ΔAP is the JSON's `delta_ap`, not a paired feed-effect estimate. Different cohorts, label provenance, and fog prevalence preclude interpreting a positive difference as forecast-feed improvement. Values are rounded for display.

| Site | Task | Model | n | Fog positives | Retro AP | Replay AP (95% day-block CI) | ΔAP | Brier | ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| location_6 | nowcast | xgboost | 2150 | 375 | 0.317 | 0.582 [0.471, 0.693] | +0.265 | 0.107 | 0.056 |
| location_6 | nowcast | temporal_cnn | 2150 | 375 | 0.351 | 0.415 [0.308, 0.522] | +0.064 | 0.152 | 0.136 |
| location_6 | lead_time | xgboost | 89 | 46 | 0.523 | 0.849 [0.758, 0.941] | +0.326 | 0.171 | 0.168 |
| location_6 | lead_time | temporal_cnn | 89 | 46 | 0.519 | 0.693 [0.570, 0.816] | +0.174 | 0.212 | 0.189 |
| location_7 | nowcast | xgboost | 2160 | 347 | 0.285 | 0.546 [0.444, 0.648] | +0.261 | 0.105 | 0.061 |
| location_7 | nowcast | temporal_cnn | 2160 | 347 | 0.262 | 0.358 [0.258, 0.459] | +0.096 | 0.141 | 0.123 |
| location_7 | lead_time | xgboost | 90 | 39 | 0.383 | 0.703 [0.552, 0.854] | +0.320 | 0.189 | 0.149 |
| location_7 | lead_time | temporal_cnn | 90 | 39 | 0.372 | 0.661 [0.515, 0.808] | +0.290 | 0.186 | 0.113 |
| location_8 | nowcast | xgboost | 2160 | 442 | 0.449 | 0.585 [0.469, 0.700] | +0.135 | 0.117 | 0.027 |
| location_8 | nowcast | temporal_cnn | 2160 | 442 | 0.447 | 0.298 [0.212, 0.383] | -0.150 | 0.171 | 0.118 |
| location_8 | lead_time | xgboost | 90 | 49 | 0.575 | 0.792 [0.698, 0.885] | +0.216 | 0.198 | 0.101 |
| location_8 | lead_time | temporal_cnn | 90 | 49 | 0.580 | 0.657 [0.535, 0.780] | +0.078 | 0.212 | 0.070 |
| location_9 | nowcast | xgboost | 2080 | 439 | 0.528 | 0.616 [0.510, 0.721] | +0.088 | 0.129 | 0.083 |
| location_9 | nowcast | temporal_cnn | 2080 | 439 | 0.537 | 0.410 [0.313, 0.507] | -0.127 | 0.175 | 0.154 |
| location_9 | lead_time | xgboost | 87 | 45 | 0.686 | 0.755 [0.639, 0.871] | +0.069 | 0.195 | 0.104 |
| location_9 | lead_time | temporal_cnn | 87 | 45 | 0.683 | 0.766 [0.662, 0.870] | +0.082 | 0.197 | 0.122 |
| location_10 | nowcast | xgboost | 2158 | 246 | 0.184 | 0.360 [0.247, 0.474] | +0.177 | 0.088 | 0.048 |
| location_10 | nowcast | temporal_cnn | 2158 | 246 | 0.117 | 0.376 [0.259, 0.493] | +0.259 | 0.096 | 0.078 |
| location_10 | lead_time | xgboost | 90 | 34 | 0.291 | 0.627 [0.476, 0.778] | +0.336 | 0.217 | 0.209 |
| location_10 | lead_time | temporal_cnn | 90 | 34 | 0.251 | 0.511 [0.362, 0.659] | +0.260 | 0.235 | 0.178 |

### Ordinal recovery and calibration

RPS (ranked probability score) evaluates the ordinal PMF only on resolved categories; Brier and ECE above evaluate the calibrated binary fog probability on the larger binary cohort. Lower RPS, Brier, and ECE are better. RPS is not a calibration score for the separately calibrated fog probability, and these differently sized cohorts must not be conflated.

| Site | Task | Model | Resolved ordinal n | RPS |
|---|---|---|---:|---:|
| location_6 | nowcast | xgboost | 2033 | 0.109 |
| location_6 | nowcast | temporal_cnn | 2033 | 0.121 |
| location_6 | lead_time | xgboost | 66 | 0.153 |
| location_6 | lead_time | temporal_cnn | 66 | 0.178 |
| location_7 | nowcast | xgboost | 2159 | 0.122 |
| location_7 | nowcast | temporal_cnn | 2159 | 0.149 |
| location_7 | lead_time | xgboost | 90 | 0.176 |
| location_7 | lead_time | temporal_cnn | 90 | 0.178 |
| location_8 | nowcast | xgboost | 2032 | 0.133 |
| location_8 | nowcast | temporal_cnn | 2032 | 0.166 |
| location_8 | lead_time | xgboost | 68 | 0.177 |
| location_8 | lead_time | temporal_cnn | 68 | 0.196 |
| location_9 | nowcast | xgboost | 1926 | 0.122 |
| location_9 | nowcast | temporal_cnn | 1926 | 0.148 |
| location_9 | lead_time | xgboost | 61 | 0.157 |
| location_9 | lead_time | temporal_cnn | 61 | 0.169 |
| location_10 | nowcast | xgboost | 2083 | 0.142 |
| location_10 | nowcast | temporal_cnn | 2083 | 0.111 |
| location_10 | lead_time | xgboost | 72 | 0.185 |
| location_10 | lead_time | temporal_cnn | 72 | 0.167 |

Across site/model groups, nowcast AP spans **0.298–0.616**, and lead AP spans **0.511–0.849**. The largest ECE is **0.209**. Favorable ranking at some sites does not remove the provenance, feature-completeness, or calibration limitations.

## Replay cost-grid operating points

Decisions were saved using the serving bundles' frozen retrospective cost thresholds, not selected by optimizing this replay's outcomes. Each entry below is `FN:FP = precision / recall / alert-rate / expected-cost-per-issue`, copied from `cost_operating_points`. Expected cost uses the stated false-negative weight and unit false-positive cost. A displayed zero precision can mean there were no alerts; it is not proof of reliable negative predictions.

- **location_6 / nowcast / xgboost:** 1:1=0.706/0.237/0.059/0.150; 3:1=0.567/0.563/0.173/0.304; 5:1=0.476/0.787/0.288/0.337; 10:1=0.351/0.941/0.468/0.407; 20:1=0.308/0.979/0.555/0.459
- **location_6 / nowcast / temporal_cnn:** 1:1=0.000/0.000/0.000/0.174; 3:1=0.000/0.000/0.000/0.523; 5:1=1.000/0.024/0.004/0.851; 10:1=0.410/0.480/0.204/1.027; 20:1=0.378/0.829/0.382/0.833
- **location_6 / lead_time / xgboost:** 1:1=0.867/0.565/0.337/0.270; 3:1=0.667/1.000/0.775/0.258; 5:1=0.605/1.000/0.854/0.337; 10:1=0.529/1.000/0.978/0.461; 20:1=0.523/1.000/0.989/0.472
- **location_6 / lead_time / temporal_cnn:** 1:1=0.681/0.696/0.528/0.326; 3:1=0.584/0.978/0.865/0.393; 5:1=0.590/1.000/0.876/0.360; 10:1=0.575/1.000/0.899/0.382; 20:1=0.541/1.000/0.955/0.438
- **location_7 / nowcast / xgboost:** 1:1=0.789/0.173/0.035/0.140; 3:1=0.527/0.499/0.152/0.313; 5:1=0.455/0.738/0.261/0.353; 10:1=0.397/0.821/0.332/0.487; 20:1=0.327/0.937/0.461/0.514
- **location_7 / nowcast / temporal_cnn:** 1:1=0.333/0.017/0.008/0.163; 3:1=0.423/0.086/0.033/0.459; 5:1=0.360/0.130/0.058/0.736; 10:1=0.301/0.144/0.077/1.429; 20:1=0.387/0.680/0.282/1.201
- **location_7 / lead_time / xgboost:** 1:1=0.750/0.769/0.444/0.211; 3:1=0.688/0.846/0.533/0.367; 5:1=0.621/0.923/0.644/0.411; 10:1=0.551/0.974/0.767/0.456; 20:1=0.443/1.000/0.978/0.544
- **location_7 / lead_time / temporal_cnn:** 1:1=0.697/0.590/0.367/0.289; 3:1=0.667/0.872/0.567/0.356; 5:1=0.551/0.974/0.767/0.400; 10:1=0.500/1.000/0.867/0.433; 20:1=0.476/1.000/0.911/0.478
- **location_8 / nowcast / xgboost:** 1:1=0.574/0.590/0.211/0.174; 3:1=0.407/0.853/0.429/0.345; 5:1=0.386/0.869/0.461/0.417; 10:1=0.338/0.939/0.568/0.500; 20:1=0.290/0.984/0.695/0.558
- **location_8 / nowcast / temporal_cnn:** 1:1=0.275/0.050/0.037/0.221; 3:1=0.305/0.283/0.190/0.572; 5:1=0.312/0.448/0.294/0.767; 10:1=0.319/0.758/0.486/0.826; 20:1=0.297/0.837/0.577/1.072
- **location_8 / lead_time / xgboost:** 1:1=0.662/0.878/0.722/0.311; 3:1=0.600/0.980/0.889/0.389; 5:1=0.571/0.980/0.933/0.456; 10:1=0.544/1.000/1.000/0.456; 20:1=0.544/1.000/1.000/0.456
- **location_8 / lead_time / temporal_cnn:** 1:1=0.677/0.857/0.689/0.300; 3:1=0.608/0.980/0.878/0.378; 5:1=0.571/0.980/0.933/0.456; 10:1=0.565/0.980/0.944/0.522; 20:1=0.563/1.000/0.967/0.422
- **location_9 / nowcast / xgboost:** 1:1=0.757/0.128/0.036/0.193; 3:1=0.607/0.636/0.221/0.318; 5:1=0.454/0.868/0.404/0.360; 10:1=0.390/0.945/0.512/0.427; 20:1=0.348/0.979/0.594/0.474
- **location_9 / nowcast / temporal_cnn:** 1:1=0.333/0.005/0.003/0.212; 3:1=0.508/0.068/0.028/0.604; 5:1=0.486/0.276/0.120/0.826; 10:1=0.422/0.542/0.271/1.123; 20:1=0.384/0.749/0.412/1.311
- **location_9 / lead_time / xgboost:** 1:1=0.700/0.778/0.575/0.287; 3:1=0.577/1.000/0.897/0.379; 5:1=0.577/1.000/0.897/0.379; 10:1=0.523/1.000/0.989/0.471; 20:1=0.523/1.000/0.989/0.471
- **location_9 / lead_time / temporal_cnn:** 1:1=0.625/0.667/0.552/0.379; 3:1=0.577/1.000/0.897/0.379; 5:1=0.577/1.000/0.897/0.379; 10:1=0.556/1.000/0.931/0.414; 20:1=0.517/1.000/1.000/0.483
- **location_10 / nowcast / xgboost:** 1:1=0.000/0.000/0.000/0.114; 3:1=0.424/0.305/0.082/0.285; 5:1=0.353/0.634/0.205/0.341; 10:1=0.284/0.846/0.340/0.419; 20:1=0.245/0.894/0.416/0.555
- **location_10 / nowcast / temporal_cnn:** 1:1=0.000/0.000/0.000/0.114; 3:1=0.496/0.252/0.058/0.285; 5:1=0.496/0.252/0.058/0.456; 10:1=0.409/0.520/0.145/0.633; 20:1=0.362/0.626/0.197/0.978
- **location_10 / lead_time / xgboost:** 1:1=0.706/0.353/0.189/0.300; 3:1=0.540/0.794/0.556/0.489; 5:1=0.540/0.794/0.556/0.644; 10:1=0.486/1.000/0.778/0.400; 20:1=0.415/1.000/0.911/0.533
- **location_10 / lead_time / temporal_cnn:** 1:1=0.500/0.441/0.333/0.378; 3:1=0.512/0.647/0.478/0.633; 5:1=0.533/0.706/0.500/0.789; 10:1=0.466/1.000/0.811/0.433; 20:1=0.453/1.000/0.833/0.456

No operational threshold is approved. This grid describes consequences under hypothetical costs, not a deployment decision or a locally agreed cost preference.

## GO/NO-GO by site

- **location_6: NO-GO.** Non-vintage historical replay and non-faithful inputs; retain private shadow operation.
- **location_7: NO-GO.** Non-vintage historical replay and non-faithful inputs; retain private shadow operation.
- **location_8: NO-GO.** Non-vintage historical replay and non-faithful inputs; retain private shadow operation.
- **location_9: NO-GO.** Non-vintage historical replay and non-faithful inputs; retain private shadow operation.
- **location_10: NO-GO.** Non-vintage historical replay and non-faithful inputs; retain private shadow operation.

A limited soft launch requires independently auditable prospective evidence, acceptable feature completeness and calibration, and a predeclared operational cost choice. This replay does not meet those requirements.

## Limitations

- **Unavailable input:** every scored group lists `soil_temperature_0_to_7cm` under `nonfaithful_features`. Serving preprocessing imputes missing values; a similar soil-depth variable is not an interchangeable trained feature. The metrics are therefore **NON-FAITHFUL**, despite the corrected ordinal label contract.
- **Non-vintage forecasts:** a later Historical-Forecast retrieval cannot prove what a real historical user would have received. These are approximate historical replay results, not forward/prospective evidence.
- **Verification-source and label uncertainty:** GHCNh defines the training target; this replay uses IEM reports decoded onto that contract. Resolved ordinal counts can be smaller than binary counts. Missing or ambiguous labels are not silently converted into clear conditions.
- **Cohort limits:** the selected winter cohort and its fog prevalence do not establish year-round skill. Retrospective reference AP and replay AP are not paired estimates. Day-block intervals address within-cohort sampling variation, not feed vintage, source mismatch, training overlap, or missing-feature bias.
- **Calibration and fitting:** isotonic fog calibration and ordinal PMF scoring are distinct. The metrics do not establish prospective calibration stability or a time-disjoint serving-model backtest; a favorable AP or RPS is not production-readiness evidence.

## Forward operation and labels

Prospective evidence must be accumulated separately with `forward_logger.py` and `fetch_forward_labels.py`, preserving actual retrieval/issuance timestamps and finalized outcomes. Use the fixed-PST issue clock and require all complete lead-window hours. The scorer enforces the reporting-finalization lag recorded in the label contract and excludes legacy provisional outcomes lacking completion evidence.

`shadow_score.py` requires an explicit feed and half-open UTC issue interval. Historical scoring reads `outcomes.jsonl` and writes `shadow_metrics_historical.json`; forward scoring reads `outcomes_forward.jsonl` and writes `shadow_metrics_forward.json`. Neither feed's metrics should be presented as evidence for the other. This report makes no claim that a forward collection season or deployment was completed.

## Reproduction

From the repository root, in the project's configured environment:

```bash
python ops/shadow/build_serving.py
python ops/shadow/replay_hindcast.py --reset
python ops/shadow/shadow_score.py --feed historical --start 2025-12-01T08:00:00Z --end 2026-03-01T08:00:00Z
```

The builder regenerates serving bundles. `--reset` replaces the local replay prediction/outcome logs and fetch manifest; preserve evidence before rerunning. The replay fetches public weather and verification data, so later retrievals need not reproduce identical provider responses. Retain the raw snapshots and manifest with the metrics for an auditable run.

## Audit inventory

The authoritative scored artifact is `shadow_metrics_historical.json`. Supporting evidence is in `fetch_manifest.json`, `raw_snapshots/`, `predictions.jsonl`, `predictions.sqlite`, `outcomes.jsonl`, and the serving bundle metadata. Counts in this report come from scored JSON groups, not from unjoined log sizes or an older generic metrics artifact.

**Final recommendation:** remain in private shadow mode. Address the unavailable trained feature through an explicitly approved serving-model change, then gather and assess genuinely prospective evidence before considering a limited soft launch. This historical replay is **not production-ready**.
