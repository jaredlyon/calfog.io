# Spark snapshot logger (deployed copy)

These are the exact files deployed on the DGX Spark (`spark`, aarch64) to bank real production-feed data for the
CalFog V2 shadow evaluation. **Passive recorder only** — no model, no predictions, no learning.

- `calfog_snapshot.py` → `~/calfog-shadow/calfog_snapshot.py` on the Spark. Uses system `python3` + `requests`
  (no torch). Fetches an Open-Meteo Forecast superset + AviationWeather METAR for all 5 airports → `snapshots.sqlite`
  + dated raw files (URL, sha256, retrieval-timestamp, `issue_label`).
- `systemd/calfog-snapshot.{service,timer}` → hourly (randomized) general evidence.
- `systemd/calfog-issue1800.{service,timer}` → fixed 18:00 America/Los_Angeles (DST-pinned, no random delay),
  tags rows `issue_label=lead_1800` for the strict lead-time issue.

Installed as **user** units (linger enabled; no sudo) under `~/.config/systemd/user/`.

Manage:
```
export XDG_RUNTIME_DIR=/run/user/$(id -u)
systemctl --user list-timers "calfog-*"
systemctl --user disable --now calfog-snapshot.timer calfog-issue1800.timer   # stop
```
It runs only in `~/calfog-shadow`; it never touches the Spark's Ollama or Minecraft services.
