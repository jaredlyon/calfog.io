#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
from models._shared.data import SITES
from models._shared.study import run_family

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leakage-free random_forest fog study")
    parser.add_argument("--sites", nargs="+", choices=tuple(SITES), default=tuple(SITES))
    args = parser.parse_args()
    run_family("random_forest", sites=args.sites)
