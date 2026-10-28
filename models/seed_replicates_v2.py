#!/usr/bin/env python3
"""Training-seed replicates of the primary NOx ablation (D-080/D-081) for significance_v2.

Replicate r > 0 refits the with/without-NOx pair under model seed seed_for(family, site, task, fold, 'replicate', r);
folds, matched rows, preprocessing and both arms' shared seed are otherwise those of the main run (replicate 0).
Predictions go to <family>_predictions_v2/replicates/r<r>/ with the main run's file names. Complete
(family, site, replicate) units are skipped, so the script is resumable. Logistic regression is excluded:
its fit uses no randomness, so every seed gives the same model.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from models._shared.data import AQ_MIN_HOURS, AQ_SETS, SEED, SITES, WEATHER_COLS, load_sources
from models._shared.families import seed_for
from models._shared.labels import label_tables
from models._shared.study import FAMILY_DIRS, evaluate_ablation
from models._shared.windows import build_windows

REPLICATE_FAMILIES = ('xgboost', 'ebm')
N_REPLICATES = 10  # including replicate 0, the main run
TASKS = ('nowcast', 'lead_time')


def replicate_dir(family, replicate):
    return FAMILY_DIRS[family] / f'{family}_predictions_v2' / 'replicates' / f'r{replicate}'


def complete(family, site, replicate):
    return all((replicate_dir(family, replicate) / f'{site}__{task}__nox_{arm}.npz').exists()
               for task in TASKS for arm in ('with', 'without'))


def run_unit(family, site, replicate):
    weather, vis = load_sources(site, weather_cols=WEATHER_COLS)
    for task, labels in zip(TASKS, label_tables(weather, vis, site)):
        _, _, matched, _ = build_windows(weather, labels, aq_cols=AQ_SETS['nox'],
                                         aq_min_hours=AQ_MIN_HOURS, weather_cols=WEATHER_COLS)
        evaluate_ablation(weather, matched, site, task, family, cnn_cfg=None, weather_cols=WEATHER_COLS,
                          seed=seed_for(SEED, family, site, task, 'nox', 'aqi_bootstrap'),
                          aq_cols=AQ_SETS['nox'], prediction_dir=replicate_dir(family, replicate),
                          aq_set='nox', replicate=replicate)
    return family, site, replicate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--replicates', type=int, nargs='+', default=list(range(1, N_REPLICATES)))
    parser.add_argument('--families', nargs='+', choices=REPLICATE_FAMILIES, default=REPLICATE_FAMILIES)
    parser.add_argument('--sites', nargs='+', choices=tuple(SITES), default=tuple(SITES))
    args = parser.parse_args()
    units = [(f, s, r) for r in args.replicates for f in args.families for s in args.sites
             if not complete(f, s, r)]
    print(f'{len(units)} replicate units to run', flush=True)
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
        for family, site, replicate in pool.map(run_unit, *zip(*units)) if units else ():
            print(f'{family} {site} r{replicate} done', flush=True)


if __name__ == '__main__':
    main()
