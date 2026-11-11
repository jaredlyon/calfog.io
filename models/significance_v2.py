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
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--family-n-boot", type=int, default=None,
                        help="draws for the family-pair table (default: --n-boot)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--aqi-only", action="store_true",
                        help="rewrite only aqi_tests_v2 (the family table does not use AQ arms)")
    args = parser.parse_args()
    if args.n_boot < 1 or (args.family_n_boot is not None and args.family_n_boot < 1):
        parser.error("--n-boot and --family-n-boot must be positive")
    write_analysis(args.n_boot, args.seed, args.family_n_boot, include_families=not args.aqi_only)


if __name__ == "__main__":
    main()
