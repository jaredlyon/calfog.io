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
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.n_boot < 1:
        parser.error("--n-boot must be positive")
    write_analysis(args.n_boot, args.seed)


if __name__ == "__main__":
    main()
