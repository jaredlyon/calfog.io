# CalFog V2 private shadow-mode report

**Decision: NO-GO for a limited soft launch at every site. Continue private forward accumulation.**

This harness is SHADOW ONLY. It writes private local JSONL/SQLite records; it has no web server, route, dashboard, public file export, or user-facing probability. No production mesh, secrets, Spark service, Ollama, Minecraft, or other production component was contacted. All network evidence here came from public Open-Meteo and Iowa Environmental Mesonet APIs.

## What was measured—and what was not

The immediate replay covers issue times from 2025-12-01 through 2026-02-28 in `America/Los_Angeles`. Each nowcast uses exactly 24 hourly rows t-23..t and labels the METAR hour at t. Each daily lead issue is 18:00 D and labels any fog in all ten complete local hours D+1 00:00–09:00. Fog is strictly visibility <1610 m. Multiple METARs in an hour are reduced to the minimum numeric visibility; an all-missing hour is not made negative.

Open-Meteo Historical-Forecast `best_match` is only a **non-vintage approximate proxy**. It does not identify the forecast initialization/run that was actually available at a historical issue time. Causality is still guarded—the selected feature timestamp never exceeds issue time—but this replay must not be called vintage-faithful or true live evidence. Its raw responses, retrieval times, URLs/parameters, statuses, row counts, and SHA-256 hashes are retained under `raw_snapshots/` and `fetch_manifest.json`.

More seriously, `soil_temperature_0_to_7cm`, one of the 12 trained weather channels, was returned entirely null with unit `undefined` by both tested Historical-Forecast best-match and current Forecast products. The similar `soil_temperature_0cm`/`soil_temperature_6cm` variables were deliberately **not substituted**. Inference used the serving pipeline's learned training-median missing-value behavior, and every prediction and metric is explicitly `faithful=false`, `nonfaithful_features=[soil_temperature_0_to_7cm]`. Therefore the immediate metrics are **NON-FAITHFUL**.

The prompt names dewpoint depression, 6/12-hour cooling, and previous-night low as trained engineered inputs. Inspection of every approved `v2/full` checkpoint and `WEATHER_COLS` found that none of those four is in the weights. The harness reconstructs them as diagnostics but does not silently change the 16-channel input order. The exact trained order is temperature, RH, dew point, precipitation, surface pressure, VPD, 10/100-m wind, gust, 0–7-cm soil temperature, weather code, low cloud, then four calendar sin/cos channels.

Serving bundles contain an XGBoost model, temporal CNN, median imputer, scaler, isotonic calibrator, cost thresholds, feature spec, and metadata for both tasks at all five sites. Auxiliary models trained before a chronological final 15% slice generated held-out calibration scores; after calibrator fitting, fixed base models and preprocessing were refit on all labeled history. This makes the base models all-history refits while preserving out-of-sample data for calibrator learning, though refitting can itself shift the score distribution and forward calibration remains the decisive check.

## Immediate replay results

`retrospective_ap` is copied from the approved `v2/full` expanding-origin ERA5/NOAA evaluation. `live_ap` below is the approximate best-match/IEM replay with 1,000 whole-local-day bootstrap resamples and a point-centered 95% interval. `delta` is simply live AP minus retrospective AP. It is **descriptive, not a paired causal feed-effect estimate**: the retrospective number used other historical folds and NOAA label sampling, while this replay deliberately selects one fog season and uses IEM hourly minima. The much higher fog prevalence in this fog-season cohort can increase AP even without better ranking.

| Site | Task | Model | Retro AP | Approx shadow AP (95% day-block CI) | ΔAP | n | positives | Brier | ECE | Faithful |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| location_6 | nowcast | xgboost | 0.302 | 0.536 [0.421, 0.650] | +0.234 | 2150 | 375 | 0.144 | 0.137 | **NO — soil_temperature_0_to_7cm** |
| location_6 | nowcast | temporal_cnn | 0.234 | 0.425 [0.310, 0.540] | +0.191 | 2150 | 375 | 0.142 | 0.119 | **NO — soil_temperature_0_to_7cm** |
| location_6 | lead_time | xgboost | 0.434 | 0.755 [0.628, 0.882] | +0.321 | 89 | 46 | 0.274 | 0.288 | **NO — soil_temperature_0_to_7cm** |
| location_6 | lead_time | temporal_cnn | 0.380 | 0.703 [0.588, 0.817] | +0.323 | 89 | 46 | 0.190 | 0.081 | **NO — soil_temperature_0_to_7cm** |
| location_7 | nowcast | xgboost | 0.229 | 0.487 [0.373, 0.601] | +0.258 | 2160 | 347 | 0.114 | 0.086 | **NO — soil_temperature_0_to_7cm** |
| location_7 | nowcast | temporal_cnn | 0.259 | 0.249 [0.171, 0.326] | -0.010 | 2160 | 347 | 0.157 | 0.151 | **NO — soil_temperature_0_to_7cm** |
| location_7 | lead_time | xgboost | 0.381 | 0.669 [0.519, 0.819] | +0.288 | 90 | 39 | 0.224 | 0.202 | **NO — soil_temperature_0_to_7cm** |
| location_7 | lead_time | temporal_cnn | 0.411 | 0.641 [0.491, 0.790] | +0.230 | 90 | 39 | 0.271 | 0.246 | **NO — soil_temperature_0_to_7cm** |
| location_8 | nowcast | xgboost | 0.345 | 0.455 [0.348, 0.561] | +0.109 | 2160 | 442 | 0.159 | 0.133 | **NO — soil_temperature_0_to_7cm** |
| location_8 | nowcast | temporal_cnn | 0.341 | 0.291 [0.212, 0.370] | -0.049 | 2160 | 442 | 0.185 | 0.161 | **NO — soil_temperature_0_to_7cm** |
| location_8 | lead_time | xgboost | 0.465 | 0.787 [0.688, 0.886] | +0.322 | 90 | 49 | 0.248 | 0.228 | **NO — soil_temperature_0_to_7cm** |
| location_8 | lead_time | temporal_cnn | 0.540 | 0.749 [0.631, 0.866] | +0.209 | 90 | 49 | 0.200 | 0.079 | **NO — soil_temperature_0_to_7cm** |
| location_9 | nowcast | xgboost | 0.460 | 0.607 [0.488, 0.725] | +0.147 | 2080 | 439 | 0.148 | 0.132 | **NO — soil_temperature_0_to_7cm** |
| location_9 | nowcast | temporal_cnn | 0.497 | 0.283 [0.207, 0.359] | -0.214 | 2080 | 439 | 0.202 | 0.190 | **NO — soil_temperature_0_to_7cm** |
| location_9 | lead_time | xgboost | 0.596 | 0.771 [0.657, 0.884] | +0.175 | 87 | 45 | 0.215 | 0.168 | **NO — soil_temperature_0_to_7cm** |
| location_9 | lead_time | temporal_cnn | 0.630 | 0.785 [0.689, 0.881] | +0.155 | 87 | 45 | 0.229 | 0.218 | **NO — soil_temperature_0_to_7cm** |
| location_10 | nowcast | xgboost | 0.127 | 0.397 [0.280, 0.514] | +0.270 | 2158 | 246 | 0.098 | 0.081 | **NO — soil_temperature_0_to_7cm** |
| location_10 | nowcast | temporal_cnn | 0.128 | 0.349 [0.210, 0.489] | +0.221 | 2158 | 246 | 0.103 | 0.093 | **NO — soil_temperature_0_to_7cm** |
| location_10 | lead_time | xgboost | 0.231 | 0.495 [0.352, 0.638] | +0.264 | 90 | 34 | 0.276 | 0.249 | **NO — soil_temperature_0_to_7cm** |
| location_10 | lead_time | temporal_cnn | 0.265 | 0.562 [0.424, 0.700] | +0.297 | 90 | 34 | 0.322 | 0.315 | **NO — soil_temperature_0_to_7cm** |

The temporal-CNN nowcast AP is below its retrospective AP at Fresno, Visalia, and Hanford; the Hanford change is especially large. Other entries rise, but those rises are not evidence that the feed improved because AP is prevalence-sensitive and the comparison is not paired on the same ERA5/IEM issues. Calibration is not uniformly good: several ECE values exceed 0.15 and Bakersfield lead CNN ECE exceeds 0.30. These are additional NO-GO signals.

## Live-feed cost-grid operating points

Thresholds were frozen from the approved retrospective validation cost grid; they were not retuned on replay labels. Values below are `FN:FP = precision/recall/alert-rate/expected-cost-per-issue` on joined replay outcomes.

- **location_6 / nowcast / xgboost:** 1:1=0.00/0.00/0.00/0.174; 3:1=0.89/0.06/0.01/0.491; 5:1=0.63/0.26/0.07/0.673; 10:1=0.56/0.57/0.18/0.837; 20:1=0.45/0.80/0.31/0.879
- **location_6 / nowcast / temporal_cnn:** 1:1=0.00/0.00/0.00/0.174; 3:1=0.51/0.18/0.06/0.460; 5:1=0.51/0.18/0.06/0.747; 10:1=0.38/0.73/0.33/0.675; 20:1=0.36/0.85/0.41/0.806
- **location_6 / lead_time / xgboost:** 1:1=0.86/0.13/0.08/0.461; 3:1=0.69/0.89/0.66/0.371; 5:1=0.61/1.00/0.84/0.326; 10:1=0.61/1.00/0.85/0.337; 20:1=0.57/1.00/0.91/0.393
- **location_6 / lead_time / temporal_cnn:** 1:1=0.70/0.85/0.63/0.270; 3:1=0.58/0.98/0.87/0.393; 5:1=0.56/1.00/0.92/0.404; 10:1=0.55/1.00/0.93/0.416; 20:1=0.52/1.00/1.00/0.483
- **location_7 / nowcast / xgboost:** 1:1=0.55/0.06/0.02/0.159; 3:1=0.54/0.36/0.11/0.359; 5:1=0.51/0.56/0.18/0.440; 10:1=0.41/0.83/0.32/0.464; 20:1=0.38/0.86/0.36/0.659
- **location_7 / nowcast / temporal_cnn:** 1:1=0.00/0.00/0.00/0.161; 3:1=0.00/0.00/0.00/0.482; 5:1=0.00/0.00/0.00/0.803; 10:1=0.38/0.02/0.01/1.583; 20:1=0.12/0.04/0.05/3.136
- **location_7 / lead_time / xgboost:** 1:1=0.73/0.41/0.24/0.322; 3:1=0.71/0.74/0.46/0.467; 5:1=0.64/0.87/0.59/0.489; 10:1=0.65/0.90/0.60/0.656; 20:1=0.56/0.95/0.73/0.767
- **location_7 / lead_time / temporal_cnn:** 1:1=0.00/0.00/0.00/0.433; 3:1=0.61/0.44/0.31/0.856; 5:1=0.58/0.85/0.63/0.600; 10:1=0.53/0.97/0.80/0.489; 20:1=0.51/0.97/0.82/0.622
- **location_8 / nowcast / xgboost:** 1:1=0.00/0.00/0.00/0.205; 3:1=0.55/0.12/0.04/0.562; 5:1=0.55/0.43/0.16/0.658; 10:1=0.38/0.67/0.36/0.893; 20:1=0.38/0.68/0.37/1.554
- **location_8 / nowcast / temporal_cnn:** 1:1=0.00/0.00/0.00/0.205; 3:1=0.39/0.03/0.02/0.605; 5:1=0.32/0.05/0.03/0.993; 10:1=0.22/0.14/0.13/1.859; 20:1=0.32/0.54/0.35/2.127
- **location_8 / lead_time / xgboost:** 1:1=0.91/0.43/0.26/0.333; 3:1=0.69/0.73/0.58/0.611; 5:1=0.59/0.98/0.90/0.422; 10:1=0.59/0.98/0.90/0.478; 20:1=0.55/1.00/0.99/0.444
- **location_8 / lead_time / temporal_cnn:** 1:1=0.73/0.82/0.61/0.267; 3:1=0.67/0.88/0.71/0.433; 5:1=0.64/0.94/0.80/0.456; 10:1=0.57/1.00/0.96/0.411; 20:1=0.57/1.00/0.96/0.411
- **location_9 / nowcast / xgboost:** 1:1=0.44/0.01/0.00/0.212; 3:1=0.74/0.25/0.07/0.494; 5:1=0.65/0.55/0.18/0.535; 10:1=0.49/0.83/0.36/0.538; 20:1=0.44/0.89/0.43/0.711
- **location_9 / nowcast / temporal_cnn:** 1:1=0.00/0.00/0.00/0.211; 3:1=0.00/0.00/0.00/0.633; 5:1=0.19/0.02/0.02/1.052; 10:1=0.18/0.07/0.08/2.032; 20:1=0.21/0.13/0.14/3.763
- **location_9 / lead_time / xgboost:** 1:1=0.77/0.53/0.36/0.322; 3:1=0.62/0.98/0.82/0.345; 5:1=0.56/1.00/0.93/0.414; 10:1=0.56/1.00/0.93/0.414; 20:1=0.56/1.00/0.93/0.414
- **location_9 / lead_time / temporal_cnn:** 1:1=0.88/0.33/0.20/0.368; 3:1=0.66/0.91/0.71/0.379; 5:1=0.60/1.00/0.86/0.345; 10:1=0.54/1.00/0.97/0.448; 20:1=0.54/1.00/0.97/0.448
- **location_10 / nowcast / xgboost:** 1:1=0.00/0.00/0.00/0.114; 3:1=0.00/0.00/0.00/0.342; 5:1=0.00/0.00/0.00/0.570; 10:1=0.40/0.64/0.19/0.520; 20:1=0.31/0.79/0.29/0.679
- **location_10 / nowcast / temporal_cnn:** 1:1=0.00/0.00/0.00/0.114; 3:1=0.00/0.00/0.00/0.342; 5:1=0.00/0.00/0.00/0.570; 10:1=0.41/0.39/0.11/0.764; 20:1=0.39/0.64/0.19/0.930
- **location_10 / lead_time / xgboost:** 1:1=0.00/0.00/0.00/0.378; 3:1=0.45/0.15/0.12/1.033; 5:1=0.51/0.71/0.52/0.811; 10:1=0.52/0.88/0.64/0.756; 20:1=0.45/1.00/0.83/0.456
- **location_10 / lead_time / temporal_cnn:** 1:1=0.00/0.00/0.00/0.378; 3:1=0.25/0.03/0.04/1.133; 5:1=0.25/0.03/0.04/1.867; 10:1=0.61/0.50/0.31/2.011; 20:1=0.55/0.91/0.62/0.944

No operational threshold is approved. A local cost preference was never supplied, and the inputs are non-faithful. The grid is retained only so a future reviewer can see the consequences rather than reverse-engineer a favorable threshold after outcomes.

## GO/NO-GO by site

- **location_6: NO-GO.** Immediate hindcast is non-vintage and non-faithful; keep probabilities private and accumulate forward snapshots and labels.
- **location_7: NO-GO.** Immediate hindcast is non-vintage and non-faithful; keep probabilities private and accumulate forward snapshots and labels.
- **location_8: NO-GO.** Immediate hindcast is non-vintage and non-faithful; keep probabilities private and accumulate forward snapshots and labels.
- **location_9: NO-GO.** Immediate hindcast is non-vintage and non-faithful; keep probabilities private and accumulate forward snapshots and labels.
- **location_10: NO-GO.** Immediate hindcast is non-vintage and non-faithful; keep probabilities private and accumulate forward snapshots and labels.

A limited soft launch requires prospective, replayable forward evidence from an acceptably complete analysis feed, stable calibration, and a predeclared cost choice. Because the immediate hindcast is non-faithful, the real evidence must come from forward accumulation over the coming fog season. Even a favorable approximate AP does not override that requirement.

## Forward operation and labels

`forward_logger.py` is a one-shot program intended for an external hourly timer. It calls only the public Open-Meteo Forecast API, snapshots the complete raw response before feature construction, logs the retrieval timestamp, takes no row later than issue time, and emits the lead record only at 18:00 local. `fetch_forward_labels.py` is a separate one-shot recent-label updater using the public AviationWeather Data API (bounded to its recent archive); it snapshots METAR JSON and never fabricates an all-missing label. Run `shadow_score.py` after labels arrive. The provider response lacks explicit per-hour analysis/run provenance, so forward reports must retain this ambiguity as well as the soil-temperature failure.

Example manual validation commands (always the mandated interpreter):

```bash
cd /opt/calfog-shadow-v2
/home/pa/work/venv/bin/python v2/shadow/forward_logger.py
/home/pa/work/venv/bin/python v2/shadow/fetch_forward_labels.py --hours 48
/home/pa/work/venv/bin/python v2/shadow/shadow_score.py
```

## Post-validation DGX Spark installation plan (documented only; not executed here)

Install in its own directory and service account. Do not place it inside, depend on, restart, stop, or modify Ollama or Minecraft directories/units. The following are the exact proposed units; deployment remains a separate authorized action after gate review.

```bash
sudo useradd --system --home /opt/calfog-shadow-v2 --shell /usr/sbin/nologin calfog-shadow
sudo install -d -o calfog-shadow -g calfog-shadow /opt/calfog-shadow-v2
sudo rsync -a --delete ./v2 /opt/calfog-shadow-v2/
sudo rsync -a --delete ./datasets /opt/calfog-shadow-v2/
sudo chown -R calfog-shadow:calfog-shadow /opt/calfog-shadow-v2
# Build an isolated interpreter in the harness directory (run only during the authorized DGX install):
sudo -u calfog-shadow uv venv /opt/calfog-shadow-v2/venv --python 3.12
sudo -u calfog-shadow uv pip install --python /opt/calfog-shadow-v2/venv/bin/python -r /opt/calfog-shadow-v2/v2/shadow/requirements-shadow.txt
sudo -u calfog-shadow uv pip install --python /opt/calfog-shadow-v2/venv/bin/python --index https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128
sudo install -m 0644 /opt/calfog-shadow-v2/v2/shadow/systemd/calfog-shadow.service /etc/systemd/system/
sudo install -m 0644 /opt/calfog-shadow-v2/v2/shadow/systemd/calfog-shadow.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now calfog-shadow.timer
systemctl list-timers calfog-shadow.timer
journalctl -u calfog-shadow.service -n 100 --no-pager
```

`calfog-shadow.service`:

```ini
[Unit]
Description=CalFog private forward shadow logger (no serving)
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
User=calfog-shadow
Group=calfog-shadow
WorkingDirectory=/opt/calfog-shadow-v2
ExecStart=/opt/calfog-shadow-v2/venv/bin/python /opt/calfog-shadow-v2/v2/shadow/forward_logger.py
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/calfog-shadow-v2/v2/shadow
Nice=10
```

`calfog-shadow.timer` (hourly; the program itself adds lead only at 18:00 local):

```ini
[Unit]
Description=Hourly CalFog private shadow collection

[Timer]
OnCalendar=*-*-* *:15:00 America/Los_Angeles
Persistent=true
AccuracySec=30s
Unit=calfog-shadow.service

[Install]
WantedBy=timers.target
```

A second label timer may call `fetch_forward_labels.py --hours 48` daily, followed by the scorer, but it should only be enabled after confirming AviationWeather response fields against saved snapshots. No scheduler was started in this sandbox.

## Audit inventory

The completed sandbox replay produced 22,500 prediction records, more than 11,000 joined outcome records, and 65 successful public network fetches (plus retained failed/retried attempts). There are serving artifacts for five sites × two tasks × two model families. `fetch_manifest.json`, raw payloads, `predictions.jsonl`, `predictions.sqlite`, `outcomes.jsonl`, and `shadow_metrics.json` are local auditable evidence. Fast tests exercise the latency guard, log schema/probability bounds, identity-based scoring join/AP, and actual monotone calibrator application.

**Final recommendation:** remain in shadow mode. Repair or retrain away the unavailable soil-temperature channel (as a new approved model, not an inference-time substitution), then gather a full forward fog season with saved vintage responses and AviationWeather outcomes before reconsidering an experimental soft launch.
