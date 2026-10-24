# Spark snapshot logger (proposed corrected copy)

The original version was deployed on DGX Spark to bank production-feed data. The corrected files here are
**not deployed**; changing the live logger requires a separately authorized action. Passive recorder only:
no model, predictions, or learning.

Repository source now lives under `ops/shadow/spark_deploy/`. This relocation does not
change any deployed Spark file or unit.

- `calfog_snapshot.py` → `~/calfog-shadow/calfog_snapshot.py` on the Spark. Uses system `python3` + `requests`
  (no torch). Fetches an Open-Meteo Forecast superset + AviationWeather METAR for all 5 airports → `snapshots.sqlite`
  + dated raw files (URL, sha256, retrieval-timestamp, `issue_label`).
- `systemd/calfog-snapshot.{service,timer}` → hourly (randomized) general evidence.
- `systemd/calfog-issue1800.{service,timer}` → 01:59 UTC daily, before the fixed 18:00 PST
  (02:00 UTC) cutoff, no random delay and no persistent catch-up. Only captures completed in the last
  minute through that cutoff retain `issue_label=lead_1800`; late/manual captures become `lead_1800_late`.
  Replaying snapshots requires rechecking retrieval timestamps, including legacy rows whose label was
  unconditional. An on-time snapshot is not itself an on-time model prediction.

Installed as **user** units (linger enabled; no sudo) under `~/.config/systemd/user/`.

Manage:
```
export XDG_RUNTIME_DIR=/run/user/$(id -u)
systemctl --user list-timers "calfog-*"
systemctl --user disable --now calfog-snapshot.timer calfog-issue1800.timer   # stop
```
It runs only in `~/calfog-shadow`; it never touches the Spark's Ollama or Minecraft services.
