#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
from models._shared.data import SITES
from models._shared.study import FAMILY_DIRS, run_family
from models._shared.report import write_comparison


def main():
    parser = argparse.ArgumentParser(description="Run every ordinal-visibility V2 family with shared per-task CNN tuning")
    parser.add_argument("--sites", nargs="+", choices=tuple(SITES), default=tuple(SITES))
    parser.add_argument("--reuse", action="store_true",
                        help="load an existing <family>_metrics_v2.json instead of retraining that family")
    args = parser.parse_args()
    cache = {}
    runs = {}
    for family, out_dir in FAMILY_DIRS.items():
        saved = out_dir / f"{family}_metrics_v2.json"
        if args.reuse and saved.exists():
            metrics = json.loads(saved.read_text(encoding="utf-8"))
            if any('ordinal' not in result for sd in metrics['per_site'].values()
                   for results in sd['tasks'].values() for result in results.values()):
                raise ValueError(f"{saved}: binary-only metrics cannot be reused; rerun without --reuse")
            runs[family] = metrics
        else:
            runs[family] = run_family(family, sites=args.sites, cnn_cache=cache)
    write_comparison(runs, ROOT / "models" / "comparison_v2")


if __name__ == "__main__":
    main()
