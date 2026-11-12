#!/usr/bin/env python3
"""Lag-aware NOx ablation (D-097) for significance_v2: does NOx help models that see the last visibility report?

The primary families (logistic regression, XGBoost, EBM) are refit with one extra hourly input,
`vis_lag_category` (labels.hourly_visibility_lag): its latest value is the last report an issue may see,
the rest of the 24-hour window is the causal visibility history. Both arms carry it; only nox_ppb differs.
Rows, folds, model seeds and the NOx eligibility rule are the main NOx run's (D-030). Predictions go to
<family>_predictions_v2/<site>__<task>__nox_lag_{with,without}.npz. Complete (family, site) units are
skipped, so the script is resumable; saved family metrics are not touched (no --reuse retraining).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from models._shared.data import AQ_MIN_HOURS, AQ_SETS, PRIMARY_FAMILIES, SEED, SITES, WEATHER_COLS, load_sources
from models._shared.families import seed_for
from models._shared.labels import hourly_visibility_lag, label_tables
from models._shared.study import FAMILY_DIRS, evaluate_ablation
from models._shared.windows import build_windows
from models._shared.environment import capture_environment
from models._shared.artifacts import atomic_write_json

AQ_SET = 'nox_lag'
LAG_WEATHER_COLS = [*WEATHER_COLS, 'vis_lag_category']
TASKS = ('nowcast', 'lead_time')


def prediction_dir(family):
    return FAMILY_DIRS[family] / f'{family}_predictions_v2'


def complete(family, site):
    return all((prediction_dir(family) / f'{site}__{task}__{AQ_SET}_{arm}.npz').exists()
               for task in TASKS for arm in ('with', 'without'))


def run_unit(family, site):
    environment = capture_environment()
    weather, vis = load_sources(site, weather_cols=WEATHER_COLS)
    weather['vis_lag_category'] = hourly_visibility_lag(weather, vis)
    for task, labels in zip(TASKS, label_tables(weather, vis, site)):
        _, _, matched, _ = build_windows(weather, labels, aq_cols=AQ_SETS['nox'],
                                         aq_min_hours=AQ_MIN_HOURS, weather_cols=LAG_WEATHER_COLS)
        result = evaluate_ablation(weather, matched, site, task, family, cnn_cfg=None, weather_cols=LAG_WEATHER_COLS,
                          seed=seed_for(SEED, family, site, task, AQ_SET, 'aqi_bootstrap'),
                          aq_cols=AQ_SETS['nox'], prediction_dir=prediction_dir(family), aq_set=AQ_SET)
        if family == 'logistic_regression':
            print(json.dumps({'family': family, 'site': site, 'task': task, 'folds': result['info']},
                             allow_nan=False), flush=True)
    atomic_write_json(prediction_dir(family) / f'{site}__{AQ_SET}.environment.json', {
        'environment': environment, 'prediction_files': [
            f'{site}__{task}__{AQ_SET}_{arm}.npz' for task in TASKS for arm in ('with', 'without')]})
    return family, site


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--families', nargs='+', choices=PRIMARY_FAMILIES, default=PRIMARY_FAMILIES)
    parser.add_argument('--sites', nargs='+', choices=tuple(SITES), default=tuple(SITES))
    args = parser.parse_args()
    # EBM units are the critical path (hours per site); start them first.
    units = sorted(((f, s) for f in args.families for s in args.sites if not complete(f, s)),
                   key=lambda unit: unit[0] != 'ebm')
    print(f'{len(units)} lag-aware units to run', flush=True)
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
        for family, site in pool.map(run_unit, *zip(*units)) if units else ():
            print(f'{family} {site} done', flush=True)


if __name__ == '__main__':
    main()
