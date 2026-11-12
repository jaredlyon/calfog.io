#!/usr/bin/env python3
"""Write short-term AQI and paired-family significance tables from saved V2 predictions."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models._shared.data import SEED
from models._shared.significance import write_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=50000)
    parser.add_argument("--family-n-boot", type=int, default=10000,
                        help="draws for the family-pair table (default: 10000)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--aqi-only", action="store_true",
                        help="rewrite only aqi_tests_v2 (the family table does not use AQ arms)")
    parser.add_argument("--block-robustness", action="store_true",
                        help="write only the saved-prediction fog-year robustness table")
    parser.add_argument("--robustness-n-boot", type=int, default=10000)
    parser.add_argument("--pm25-n-boot", type=int, default=10000)
    args = parser.parse_args()
    if min(args.n_boot, args.family_n_boot, args.robustness_n_boot, args.pm25_n_boot) < 1:
        parser.error("bootstrap draw counts must be positive")
    if args.aqi_only and args.block_robustness:
        parser.error("--aqi-only and --block-robustness are mutually exclusive")
    write_analysis(args.n_boot, args.seed, args.family_n_boot, include_families=not args.aqi_only,
                   block_robustness=args.block_robustness, robustness_n_boot=args.robustness_n_boot,
                   pm25_n_boot=args.pm25_n_boot)


if __name__ == "__main__":
    main()
