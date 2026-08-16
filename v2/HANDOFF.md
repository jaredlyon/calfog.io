# CalFog V2 — Handoff (2026-08-16)

Pick-up notes for the V2 rebuild of the tule-fog predictor. All V2 work lives under `v2/`.
Branch: **`calfog-v2-shadow`** (off `develop`).

## TL;DR of what happened
V1 (the class project) was leaky (test-set early stopping, global preprocessing before split, non-chronological
splits, in-sample ensemble stacking) and its "AQI hurts the signal" finding was a **confound** (the AQI cohort
was ~26k rows vs ~400k, different era/prevalence). V2 rebuilt everything leakage-free, per-airport, with proper
uncertainty — and then took the first safe step toward deployment ("shadow mode").

## Four phases (each self-contained under v2/)
1. **`v2/` (PoC)** — single airport (Bakersfield), proved the leakage-free pipeline + GPU + Ray Tune. `poc_report.md`.
2. **`v2/full/` (5-airport study)** — the real analysis. `report_full.md`, `metrics_full.json`.
   - **Result:** nowcast + lead-time both beat a climatology baseline with separated 95% CIs at **all five airports**
     (prevalence-normalized skill ~14–16x climatology nowcast / ~7–10x lead-time). Models calibrate well
     (isotonic ECE ~0.003) and are cost-tunable.
   - **AQI verdict:** *no robust, credible benefit anywhere* (matched-cohort, paired day-block bootstrap). The
     occasional big +ΔAP sits on 10–50 positive days and isn't corroborated across model families.
   - **Operational verdict:** NO-GO for **autonomous public alerting** (unattended) — not "no signal", but the
     precision at useful recall is low (false-alarm fatigue) and fog is safety-critical.
3. **`v2/shadow/` (shadow-mode harness)** — `report_shadow.md`. Ran the frozen models on a production-style feed.
   - **Key finding:** a trained input channel, `soil_temperature_0_to_7cm`, is **NULL from the live Open-Meteo
     Forecast feed** → the V2/full models can't be driven faithfully in production. Every immediate-hindcast
     metric was honestly flagged `faithful=false` (no silent substitution). NO-GO for soft launch, correctly.
4. **`v2/retrain/` (schema-compatible retrain)** — `report_retrain.md`, `production_feature_contract.json`.
   - Retrained all 5 sites on an **11-variable set the live feed actually serves** (all soil-temp layers dropped:
     `_0_to_7cm` null live; `_0cm`/`_6cm` null in the ERA5 archive → dropped, not substituted).
   - **~No skill lost** vs v2/full. BUT this is **SCHEMA-COMPATIBLE ONLY, `faithful=false`, NOT production-ready**:
     trained on ERA5 **archive** reanalysis while production serves **forecast/analysis** values (distribution
     shift unresolved). The HF-proxy cross-check is approximate and inconclusive.

## What is RUNNING right now (the only live thing)
**A passive data recorder on the DGX Spark — NOT a model, NOT learning.** It just saves real inputs+outcomes so a
future faithful evaluation is possible.
- Host: `spark` (DGX, aarch64). Dir: `~/calfog-shadow/`. Script: `calfog_snapshot.py` (system python3 + requests;
  no torch, no model, no predictions). Snapshots a *superset* of Open-Meteo Forecast hourly vars + AviationWeather
  METAR outcomes for all 5 airports → `snapshots.sqlite` + dated raw files (URL/sha256/retrieval-timestamp).
- User systemd timers (linger enabled → survive reboot, no sudo):
  - `calfog-snapshot.timer` — hourly (randomized), general evidence.
  - `calfog-issue1800.timer` — fixed **18:00 America/Los_Angeles** (DST-pinned, no random delay), tags rows
    `issue_label=lead_1800` for the strict lead-time contract.
- Copies of the script + units are committed here under `v2/shadow/spark_deploy/`.
- **Check it:** `ssh spark 'export XDG_RUNTIME_DIR=/run/user/$(id -u); systemctl --user list-timers "calfog-*"'`
  and `ssh spark 'sqlite3 ~/calfog-shadow/snapshots.sqlite "SELECT issue_label,COUNT(*) FROM snapshots GROUP BY 1"'`.
- **Stop it (if ever needed):** `ssh spark 'export XDG_RUNTIME_DIR=/run/user/$(id -u); systemctl --user disable --now calfog-snapshot.timer calfog-issue1800.timer'`.
- It runs entirely in its own dir and never touches the Spark's Ollama or Minecraft services.

## NEXT STEPS (in order)
1. **[TOP / from review] Make the Spark logger predict contemporaneously, don't just snapshot for later replay.**
   Run the *frozen* `v2/retrain` serving pipeline on each snapshot **at issue time** and store, immutably, the
   probability + model/artifact SHA-256 + `production_feature_contract` version + `faithful=false` flag alongside
   the raw payload. (Raw snapshots still allow replay, but writing predictions *contemporaneously* prevents future
   code/model edits from contaminating the prospective evaluation.) Note: the Spark is aarch64 — either install an
   arm64 torch/xgboost env there, or run XGBoost-only (CPU) which avoids torch entirely.
2. **Accumulate a fog season (Nov–Mar).** Tule fog is winter-only; nothing to evaluate until then.
3. **Faithful evaluation.** Join accumulated predictions ↔ realized METAR (`fetch_forward_labels.py` /
   `shadow_score.py`) → live AP + calibration + cost operating points on the *real* feed. THIS is the verdict that
   was missing (the immediate hindcast was non-faithful).
4. **Only if faithful metrics hold:** a *limited* soft launch — calibrated probabilities with an "experimental"
   label, per-airport, at a deliberately chosen cost threshold. Never jump straight to autonomous public alerts.
5. (Optional) revisit whether training on Historical-Forecast-proxy data (short history, ~2022+) closes the
   distribution shift better than archive training.

## How to reproduce / re-run (in the Prime Agent sandbox; see skill `drive-prime-agent`)
Sandbox lives at `~/work/calfog.io` in the hardened `Ubuntu-24.04` WSL distro on the waffle-house tower; GPU venv
at `~/work/venv` (torch cu128 on the RTX 5070). Each phase: `bash v2/<phase>/run_*.sh` regenerates its outputs.
Datasets are git-ignored (`*.csv`) and only on D:/the sandbox.

## Honesty caveats (do not overstate)
- Publication-**oriented**, not publication-**ready**. Features use reanalysis-as-available; single fold family;
  retrospective association, not causal; no external/held-out-site validation yet.
- "It works in production" is **unproven** until step 3's forward-feed evidence exists.

## What's committed vs excluded (see .gitignore)
Committed: all code (`.py`/`.sh`), reports (`.md`), metrics/contracts/manifests (`.json`), tests, small serving
models (temporal-CNN `.pt`, XGBoost, isotonic calibrators, feature specs), Spark deploy files.
Excluded (bulky/regenerable): re-fetched `*.csv` archive data, RandomForest `.joblib` pickles (tens of MB),
Ray Tune `tune/`/`ray_results/`/`ray_tmp/`, `*.npz`, `*.sqlite`, raw snapshot dumps, `predictions.jsonl`.
