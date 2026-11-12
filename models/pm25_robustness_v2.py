#!/usr/bin/env python3
"""Exploratory PM2.5 lag/FEM checks on jointly eligible, fixed-fold three-arm cohorts."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from models._shared.data import AQ_MIN_HOURS, PRIMARY_FAMILIES, SITES, WEATHER_COLS, load_sources
from models._shared.families import FAMILIES, seed_for
from models._shared.labels import label_tables
from models._shared.splits import rolling_origin_folds
from models._shared.study import FAMILY_DIRS, NotEstimable, _record, save_predictions
from models._shared.windows import build_windows, prepare
from models._shared.environment import capture_environment
from models._shared.artifacts import atomic_write_json

CHECKS = ('pm25_lag24', 'pm25_fem')
TASKS = ('nowcast', 'lead_time')
ARMS = ('with', 'reference', 'without')


def prediction_dir(family):
    return FAMILY_DIRS[family] / f'{family}_predictions_v2'


def complete(family, site, check):
    return (prediction_dir(family) / f'{site}__{check}.environment.json').exists()


def candidate_weather(weather, check):
    """Input time already denotes completion (K1); do not shift the reference again."""
    candidate = weather.copy()
    if check == 'pm25_lag24':
        candidate['pm25_ugm3'] = weather.pm25_ugm3.shift(24)
    elif check == 'pm25_fem':
        if (weather.pm25_ugm3.notna() & weather.pm25_method.isna()).any():
            raise ValueError('pm25_fem: unclassified PM2.5 method')
        keep = weather.pm25_parameter.eq(88101) & weather.pm25_method.isin((170, 204, 209))
        candidate['pm25_ugm3'] = weather.pm25_ugm3.where(keep.fillna(False))
    else:
        raise ValueError(f'unknown PM2.5 check: {check}')
    return candidate


def matched_labels(weather, candidate, labels):
    joint = weather.assign(pm25_candidate=candidate.pm25_ugm3)
    return build_windows(joint, labels, aq_cols=['pm25_ugm3', 'pm25_candidate'],
                         aq_min_hours=AQ_MIN_HOURS, weather_cols=WEATHER_COLS)[2]


def evaluate_check(weather, candidate, labels, site, task, family, check, directory):
    matched = matched_labels(weather, candidate, labels)
    if len(matched) < 30:
        raise NotEstimable('at least 30 matched rows are required')
    records = {arm: [] for arm in ARMS}
    for fi, fold in enumerate(rolling_origin_folds(matched), 1):
        parts = {key: fold[key] for key in ('train', 'val', 'test')}
        preps = {arm: prepare(candidate if arm == 'with' else weather, parts,
                              aq_cols=['pm25_ugm3'] if arm != 'without' else (),
                              aq_min_hours=AQ_MIN_HOURS if arm != 'without' else None,
                              weather_cols=WEATHER_COLS) for arm in ARMS}
        for split in parts:
            keys = preps['without'].metas[split].row_key.tolist()
            categories = getattr(preps['without'], {'train': 'ctr', 'val': 'cv', 'test': 'ct'}[split])
            for prep in preps.values():
                assert prep.metas[split].row_key.tolist() == keys
                assert np.array_equal(getattr(prep, {'train': 'ctr', 'val': 'cv', 'test': 'ct'}[split]), categories)
        for arm, prep in preps.items():
            if len(np.unique(prep.yv)) != 2:
                raise NotEstimable(f'fold {fi}, arm {arm}: validation requires both classes; keep fixed folds')
        for arm, prep in preps.items():
            output = FAMILIES[family].fit_predict(prep, cnn_cfg=None,
                                                 seed=seed_for(family, site, task, fi))
            records[arm].append(_record(prep, output, fi, family))
    names = []
    for arm, rows in records.items():
        name = f'{site}__{task}__{check}_{arm}.npz'
        save_predictions(rows, directory / name)
        names.append(name)
    return names


def run_unit(family, site, check):
    environment = capture_environment()
    directory = prediction_dir(family)
    manifest = directory / f'{site}__{check}.environment.json'
    manifest.unlink(missing_ok=True)
    weather, vis = load_sources(site, weather_cols=WEATHER_COLS)
    candidate = candidate_weather(weather, check)
    tasks, files = {}, []
    for task, labels in zip(TASKS, label_tables(weather, vis, site)):
        try:
            files.extend(evaluate_check(weather, candidate, labels, site, task, family, check, directory))
        except NotEstimable as error:
            tasks[task] = {'status': 'not_estimable', 'reason': str(error)}
            for arm in ARMS:
                (directory / f'{site}__{task}__{check}_{arm}.npz').unlink(missing_ok=True)
        else:
            tasks[task] = {'status': 'ok', 'reason': None}
    atomic_write_json(manifest, {'environment': environment, 'tasks': tasks, 'prediction_files': files})
    return family, site, check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--families', nargs='+', choices=PRIMARY_FAMILIES, default=PRIMARY_FAMILIES)
    parser.add_argument('--sites', nargs='+', choices=tuple(SITES), default=tuple(SITES))
    parser.add_argument('--checks', nargs='+', choices=CHECKS, default=CHECKS)
    args = parser.parse_args()
    units = sorted(((f, s, c) for f in args.families for s in args.sites for c in args.checks
                    if not complete(f, s, c)), key=lambda unit: unit[0] != 'ebm')
    print(f'{len(units)} PM2.5 robustness units to run', flush=True)
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
        for family, site, check in pool.map(run_unit, *zip(*units)) if units else ():
            print(f'{family} {site} {check} done', flush=True)


if __name__ == '__main__':
    main()
