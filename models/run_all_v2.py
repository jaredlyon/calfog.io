#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
from models._shared.data import SITES, AQ_SETS, PRIMARY_FAMILIES
from models._shared.study import FAMILY_DIRS, run_family
from models._shared.report import write_comparison


def stale_reason(family, out_dir, metrics):
    """Why saved metrics cannot be reused (None if reusable)."""
    if any('ordinal' not in result for sd in metrics['per_site'].values()
           for results in sd['tasks'].values() for result in results.values()):
        return 'binary-only metrics'
    expected_sets = set(AQ_SETS) | ({'nox_monitor'} if family in PRIMARY_FAMILIES else set())
    if not metrics['per_site'] or any(
        set(sd.get('aqi_ablation', {})) != expected_sets or
        any(task not in sd['aqi_ablation'][aq_set] or family not in sd['aqi_ablation'][aq_set][task]
            for aq_set in expected_sets for task in ('nowcast', 'lead_time'))
        for sd in metrics['per_site'].values()
    ):
        return 'old AQI nesting'
    if not (out_dir / f"{family}_predictions_v2").is_dir():
        return 'metrics without predictions'
    return None


def main():
    parser = argparse.ArgumentParser(description="Run every ordinal-visibility V2 family with shared per-task CNN tuning")
    parser.add_argument("--sites", nargs="+", choices=tuple(SITES), default=tuple(SITES))
    parser.add_argument("--reuse", action="store_true",
                        help="reuse a family's saved metrics+predictions when current; stale families are retrained")
    args = parser.parse_args()
    cache = {}
    runs = {}
    for family, out_dir in FAMILY_DIRS.items():
        saved = out_dir / f"{family}_metrics_v2.json"
        reason = 'no saved metrics'
        if args.reuse and saved.exists():
            metrics = json.loads(saved.read_text(encoding="utf-8"))
            reason = stale_reason(family, out_dir, metrics)
            if reason is None:
                runs[family] = metrics
                # reuse the tuned CNN configs so later CNN-bearing families (ensemble) don't re-tune
                for site, sd in metrics['per_site'].items():
                    for task, cfg in sd.get('cnn_cfg', {}).items():
                        if cfg is not None:
                            cache.setdefault((site, task), cfg)
                continue
        if args.reuse:
            print(f"{family}: retraining ({reason})", flush=True)
        runs[family] = run_family(family, sites=args.sites, cnn_cache=cache)
    write_comparison(runs, ROOT / "models" / "comparison_v2")


if __name__ == "__main__":
    main()
