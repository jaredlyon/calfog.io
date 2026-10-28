"""One-family expanding-origin experiments with a separately evaluated baseline."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import ROOT, SITES, WEATHER_COLS, AQ_SETS, AQI_COLS, AQ_MIN_HOURS, PRIMARY_FAMILIES, SEED, N_BOOT, load_sources
from .labels import label_tables, add_neighbor_lags
from .splits import rolling_origin_folds
from .windows import prepare, prepare_pooled, build_windows
from .evaluation import (fit_calibrator, calibrate, finalize_model, paired_delta_bootstrap, safe_ap,
                         pmf_to_fog, ordinal_bootstrap, paired_rps_delta_bootstrap)
from .families import FAMILIES, seed_for
from .tuning import tune_cnn

FAMILY_DIRS: dict[str, Path] = {family: ROOT / 'models' / folder for family, folder in {
    'logistic_regression': 'logistic regression', 'xgboost': 'xgboost', 'ebm': 'ebm',
    'random_forest': 'random forest', 'xgboost_focal_loss': 'xgboost',
    'temporal_cnn': 'temporal_cnn', 'ensemble': 'ensemble',
    'persistence': 'persistence', 'analog': 'analog', 'xgboost_ordinal': 'xgboost',
    'xgboost_aft': 'xgboost', 'xgboost_pooled': 'pooled'}.items()}

PREDICTOR_SETS: dict[str, str] = {family: 'weather' for family in FAMILIES}
PREDICTOR_SETS.update({
    'persistence': 'observed visibility history',
    'xgboost_pooled': 'weather + observed visibility (own + 4 neighbors), pooled sites',
})


class NotEstimable(ValueError):
    """A fixed-fold comparison lacks the validation classes it needs."""


def tuned_cnn_cfg(prep, site, task, *, art_dir, cache: dict) -> dict:
    key = (site, task)
    if key not in cache:
        cache[key] = tune_cnn(prep, site, task, art_dir=Path(art_dir), seed=seed_for('cnn_tune', site, task))
    return cache[key]


def _audit(prep, fold):
    return {'fold': fold, 'train': len(prep.ytr), 'val': len(prep.yv), 'test': len(prep.yt),
            'pos_train': int(prep.ytr.sum()), 'pos_val': int(prep.yv.sum()), 'pos_test': int(prep.yt.sum()),
            'train_end': str(prep.metas['train'].issue_time.max()),
            'val_start': str(prep.metas['val'].issue_time.min()),
            'test_start': str(prep.metas['test'].issue_time.min())}


def _record(prep, output, fold, family):
    pv, pt = pmf_to_fog(output.pmf_val), pmf_to_fog(output.pmf_test)
    cal = fit_calibrator(prep.yv, pv)
    meta = prep.metas['test'].reindex(columns=['issue_time', 'row_key', 'dpd_c', 'wind_kmh']).copy()
    meta['fold'] = fold
    return {'c': prep.ct.copy(), 'pmf': output.pmf_test, 'y': prep.yt.copy(),
            'p': pt, 'pc': calibrate(cal, pt), 'meta': meta,
            'val': {'fold': fold, 'y': prep.yv.copy(), family: calibrate(cal, pv)},
            'importance': output.importance, 'info': output.info}


def _finalize(records, family, seed, n_boot):
    y, p, pc = (np.concatenate([row[key] for row in records]) for key in ('y', 'p', 'pc'))
    meta = pd.concat([row['meta'] for row in records], ignore_index=True)
    result = finalize_model(y, p, pc, meta, [row['val'] for row in records], family, seed, n_boot=n_boot)
    c, pmf = (np.concatenate([row[key] for row in records]) for key in ('c', 'pmf'))
    result['ordinal'] = ordinal_bootstrap(c, pmf, meta, n_boot, seed, 'day')
    names = set().union(*(row['importance'] for row in records))
    if names:
        # Report the equally weighted fold importance, normalized over the same feature roster.
        importance = {name: float(np.mean([row['importance'].get(name, 0.) for row in records])) for name in sorted(names)}
        total = sum(importance.values())
        result['importance'] = {name: 100 * value / total for name, value in importance.items()} if total else {}
    return result


def save_predictions(records, path):
    """Persist uncalibrated ordinal predictions and evaluation-only strata."""
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = pd.concat([row['meta'] for row in records], ignore_index=True)
    np.savez_compressed(path,
        issue_time=meta.issue_time.to_numpy(dtype='datetime64[ns]').astype(np.int64),
        fold=meta.fold.to_numpy(dtype=np.int8),
        c=np.concatenate([row['c'] for row in records]).astype(np.int8),
        pmf=np.concatenate([row['pmf'] for row in records]).astype(np.float16),
        dpd_c=meta.dpd_c.to_numpy(dtype=np.float32),
        wind_kmh=meta.wind_kmh.to_numpy(dtype=np.float32))


def evaluate_family(weather, labels, site, task, family, *, cnn_cfg, weather_cols, seed,
                    save_dir: Path | None, n_boot: int = N_BOOT, prediction_path=None) -> tuple[dict, list, dict]:
    records, audits, info = [], [], []
    folds = rolling_origin_folds(labels)
    for fi, fold in enumerate(folds, 1):
        prep = prepare(weather, {key: fold[key] for key in ('train', 'val', 'test')},
                       weather_cols=weather_cols)
        output = FAMILIES[family].fit_predict(prep, cnn_cfg=cnn_cfg, seed=seed_for(family, site, task, fi))
        records.append(_record(prep, output, fi, family))
        audits.append(_audit(prep, fi))
        info.append(output.info)
        if fi == 3 and save_dir is not None and output.state is not None:
            import torch
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(output.state, Path(save_dir) / f'cnn_{task}.pt')
    save_predictions(records, prediction_path)
    return _finalize(records, family, seed, n_boot), audits, {'folds': info}


def evaluate_pooled(sources, labels_by_site, target_site, task, family, *, weather_cols, seed,
                    n_boot: int = N_BOOT, prediction_path=None) -> tuple[dict, list, dict]:
    records, audits, info = [], [], []
    for fi, fold in enumerate(rolling_origin_folds(labels_by_site[target_site]), 1):
        V, T = fold['val'].issue_time.min(), fold['test'].issue_time.min()
        parts = {}
        for site, labels in labels_by_site.items():
            parts[site] = {
                'train': labels.loc[labels.issue_time < V - pd.Timedelta(hours=24)],
                'val': labels.loc[(labels.issue_time >= V) & (labels.issue_time < T - pd.Timedelta(hours=24))],
                'test': fold['test'] if site == target_site else labels.iloc[:0],
            }
        prep = prepare_pooled(sources, parts, weather_cols=weather_cols)
        output = FAMILIES[family].fit_predict(
            prep, cnn_cfg=None, seed=seed_for(family, target_site, task, fi))
        records.append(_record(prep, output, fi, family))
        audit = _audit(prep, fi)
        audit['pooled_sites'] = {site: int(prep.metas['train'].site.eq(site).sum())
                                 for site in labels_by_site}
        audits.append(audit)
        info.append(output.info)
    save_predictions(records, prediction_path)
    return _finalize(records, family, seed, n_boot), audits, {'folds': info}


def evaluate_ablation(weather, matched, site, task, family, *, cnn_cfg, weather_cols, seed,
                      aq_cols=AQI_COLS, n_boot: int = N_BOOT, prediction_dir=None, aq_set='cams') -> dict:
    records = {'with': [], 'without': []}
    audits, info = [], []
    try:
        folds = rolling_origin_folds(matched)
    except ValueError:
        if len(matched) < 30:
            raise NotEstimable('at least 30 matched rows are required') from None
        raise
    for fi, fold in enumerate(folds, 1):
        parts = {key: fold[key] for key in ('train', 'val', 'test')}
        yes = prepare(weather, parts, aq_cols=aq_cols, aq_min_hours=AQ_MIN_HOURS, weather_cols=weather_cols)
        matched_parts = {key: yes.metas[key] for key in ('train', 'val', 'test')}
        no = prepare(weather, matched_parts, weather_cols=weather_cols)
        for split in ('train', 'val', 'test'):
            assert no.metas[split].row_key.tolist() == yes.metas[split].row_key.tolist()
        # FULL-4: do not move folds, fabricate a calibrator or begin any training.
        for arm, prep in (('without', no), ('with', yes)):
            if len(np.unique(prep.yv)) != 2:
                raise NotEstimable(
                    f'fold {fi}, arm {arm}: validation requires both classes; '
                    f'validation positives={int(np.sum(prep.yv))}/{len(prep.yv)}; keep fixed folds')
        for arm, prep in (('without', no), ('with', yes)):
            output = FAMILIES[family].fit_predict(prep, cnn_cfg=cnn_cfg,
                                                seed=seed_for(family, site, task, fi))
            record = _record(prep, output, fi, family)
            baseline = FAMILIES['climatology'].fit_predict(prep, cnn_cfg=None, seed=seed)
            record['baseline'] = pmf_to_fog(baseline.pmf_test)
            records[arm].append(record)
            info.append({'fold': fi, 'arm': arm, **output.info})
        audits.append({**_audit(no, fi), 'identical_issue_keys': True})
    aw, an = records['with'], records['without']
    y = np.concatenate([row['y'] for row in aw])
    pw, pn = (np.concatenate([row['p'] for row in records[arm]]) for arm in ('with', 'without'))
    meta = pd.concat([row['meta'] for row in aw], ignore_index=True)
    assert np.array_equal(y, np.concatenate([row['y'] for row in an]))
    assert meta.row_key.tolist() == pd.concat([row['meta'] for row in an], ignore_index=True).row_key.tolist()
    delta, ci, valid = paired_delta_bootstrap(y, pw, pn, meta, n_boot, seed, 'day')
    c = np.concatenate([row['c'] for row in aw])
    assert np.array_equal(c, np.concatenate([row['c'] for row in an]))
    pmfw, pmfn = (np.concatenate([row['pmf'] for row in records[arm]]) for arm in ('with', 'without'))
    delta_rps, rps_ci, rps_valid = paired_rps_delta_bootstrap(c, pmfw, pmfn, meta, n_boot, seed, 'day')
    result = {'with_ap': safe_ap(y, pw), 'matched_no_aqi_ap': safe_ap(y, pn), 'delta_ap': delta,
              'delta_ap_ci': ci, 'pos_count': int(y.sum()), 'n_test': len(y), 'paired_block': 'day',
              'bootstrap_resamples': n_boot, 'valid_resamples': valid, 'identical_issue_keys': True,
              'delta_rps': delta_rps, 'delta_rps_ci': rps_ci, 'valid_rps_resamples': rps_valid,
              'fold_audit': audits, 'info': info}
    for arm in records:
        result[arm] = _finalize(records[arm], family, seed, n_boot)
        result[arm]['climatology_ap'] = safe_ap(y, np.concatenate([row['baseline'] for row in records[arm]]))
        if prediction_dir is not None:
            save_predictions(records[arm], Path(prediction_dir) / f'{site}__{task}__{aq_set}_{arm}.npz')
    if y.sum() < 50:
        result['ci_warning'] = 'Fewer than 50 positives: AQI delta interval is sparse and potentially uninformative.'
    return result


def run_family(family: str, *, sites=tuple(SITES), weather_cols=WEATHER_COLS, cnn_cache: dict | None = None,
               out_dir: Path | None = None, n_boot=N_BOOT, write=True) -> dict:
    """Run one family; write=False returns per-site results only (for parallel shards, merged by write_family)."""

    module = FAMILIES[family]
    out_dir = Path(out_dir) if out_dir is not None else FAMILY_DIRS[family]
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = out_dir / f'{family}_artifacts_v2'
    prediction_dir = out_dir / f'{family}_predictions_v2'
    cache = cnn_cache if cnn_cache is not None else {}
    metrics = {'family': family, 'predictor_set': PREDICTOR_SETS.get(family, 'weather'),
               'seed': SEED, 'n_boot': n_boot, 'weather_cols': list(weather_cols), 'per_site': {}}
    pooled = getattr(module, 'POOLED', False)
    if pooled:
        loaded = {site: load_sources(site, weather_cols=weather_cols) for site in SITES}
        sources = {site: source[0] for site, source in loaded.items()}
        visibility = {site: source[1] for site, source in loaded.items()}
        tables = {site: label_tables(sources[site], visibility[site], site) for site in SITES}
        pooled_labels = {
            task: add_neighbor_lags({site: table[index] for site, table in tables.items()}, visibility)
            for index, task in enumerate(('nowcast', 'lead_time'))
        }
    for site in sites:
        if pooled:
            weather = sources[site]
            now, lead = (pooled_labels[task][site] for task in ('nowcast', 'lead_time'))
        else:
            weather, vis = load_sources(site, weather_cols=weather_cols)
            now, lead = label_tables(weather, vis, site)
        sd = {'cohort': {'nowcast_rows': len(now), 'nowcast_positives': int(now.fog.sum()),
                         'nowcast_fog_rate': float(now.fog.mean()), 'lead_complete_10h_issues': len(lead),
                         'lead_positives': int(lead.fog.sum()), 'aqi_channels': AQ_SETS},
              'cnn_cfg': {}, 'tasks': {}, 'aqi_ablation': {}, 'fold_audit': {}, 'info': {}}
        for task, labels in (('nowcast', now), ('lead_time', lead)):
            cfg = None
            if module.NEEDS_CNN_CFG:
                first = rolling_origin_folds(labels)[0]
                prep = prepare(weather, {key: first[key] for key in ('train', 'val', 'test')}, weather_cols=weather_cols)
                cfg = tuned_cnn_cfg(prep, site, task, art_dir=art_dir, cache=cache)
            sd['cnn_cfg'][task] = cfg
            if pooled:
                result, audit, info = evaluate_pooled(sources, pooled_labels[task], site, task, family,
                    weather_cols=weather_cols, seed=seed_for(SEED, family, site, task, 'bootstrap'),
                    n_boot=n_boot, prediction_path=prediction_dir / f'{site}__{task}__main.npz')
            else:
                result, audit, info = evaluate_family(weather, labels, site, task, family, cnn_cfg=cfg,
                    weather_cols=weather_cols, seed=seed_for(SEED, family, site, task, 'bootstrap'),
                    save_dir=art_dir / site, n_boot=n_boot,
                    prediction_path=prediction_dir / f'{site}__{task}__main.npz')
            baseline, _, _ = evaluate_family(weather, labels, site, task, 'climatology', cnn_cfg=None,
                weather_cols=weather_cols, seed=seed_for(SEED, 'climatology', site, task, 'bootstrap'),
                save_dir=None, n_boot=n_boot,
                prediction_path=prediction_dir / f'{site}__{task}__climatology.npz')
            sd['tasks'][task] = {family: result, 'climatology': baseline}
            sd['fold_audit'][task] = audit
            sd['info'][task] = info
            sets = dict(AQ_SETS)
            if family in PRIMARY_FAMILIES:
                sets['nox_monitor'] = ['nox_ppb'] + sorted(
                    col for col in weather.columns if col.startswith('nox_mon_'))
            for aq_set, aq_cols in sets.items():
                results = sd['aqi_ablation'].setdefault(aq_set, {})
                if pooled:
                    results[task] = {family: {
                        'status': 'not_applicable',
                        'reason': 'pooled multi-site model; AQI matched cohort is per site'}}
                    continue
                _, _, matched, _ = build_windows(weather, labels, aq_cols=aq_cols,
                    aq_min_hours=AQ_MIN_HOURS, weather_cols=weather_cols)
                sd['cohort'][f'{aq_set}_matched_{task}_rows'] = len(matched)
                try:
                    ablation = evaluate_ablation(weather, matched, site, task, family, cnn_cfg=cfg,
                        weather_cols=weather_cols, aq_cols=aq_cols,
                        seed=seed_for(SEED, family, site, task, aq_set, 'aqi_bootstrap'),
                        n_boot=n_boot, prediction_dir=prediction_dir, aq_set=aq_set)
                except NotEstimable as error:
                    results[task] = {family: {'status': 'not_estimable', 'reason': str(error)}}
                else:
                    results[task] = {family: ablation}
                    sd['fold_audit'][f'{task}_{aq_set}'] = ablation['fold_audit']
        metrics['per_site'][site] = sd
    return write_family(metrics, out_dir) if write else metrics


def write_family(metrics: dict, out_dir: Path | None = None) -> dict:
    """Aggregate a family's per-site results and write its metrics JSON, report and plots."""
    from .report import aggregate, write_family_report, write_plots

    family = metrics['family']
    out_dir = Path(out_dir) if out_dir is not None else FAMILY_DIRS[family]
    metrics['aggregate'] = aggregate(metrics)
    (out_dir / f'{family}_metrics_v2.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    write_family_report(metrics, out_dir / f'{family}_results_v2.txt')
    write_plots(metrics, art_dir=out_dir / f'{family}_artifacts_v2')
    return metrics
