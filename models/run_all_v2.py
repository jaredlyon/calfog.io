#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import multiprocessing
import time
import math
from datetime import datetime, timedelta
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from models._shared.data import SITES, AQ_SETS, PRIMARY_FAMILIES
from models._shared.families import FAMILIES
from models._shared.study import (FAMILY_DIRS, run_family, write_family, input_identity,
                                 prediction_inventory, core_prediction_files)
from models._shared.artifacts import canonical_hash, file_sha256, load_completion, completion_path
from models._shared.report import write_comparison

TASKS = ('nowcast', 'lead_time')


def stale_reason(family, out_dir, metrics, *, sites, expected_inputs):
    """Why saved metrics cannot be reused (None only for one complete generation)."""
    try:
        if metrics.get('family') != family or set(metrics['per_site']) != set(sites):
            return 'family or site roster differs'
        expected_sets = set(AQ_SETS) | ({'nox_monitor'} if family in PRIMARY_FAMILIES else set())
        record = load_completion(out_dir, family)
        for sd in metrics['per_site'].values():
            if set(sd['tasks']) != set(TASKS):
                return 'task roster differs'
            for task in TASKS:
                for name in (family, 'climatology'):
                    result = sd['tasks'][task][name]
                    if result.get('status') != 'not_estimable' and 'ordinal' not in result:
                        return 'binary-only metrics'
            if set(sd.get('aqi_ablation', {})) != expected_sets or any(
                task not in sd['aqi_ablation'][name] or family not in sd['aqi_ablation'][name][task]
                for name in expected_sets for task in TASKS):
                return 'old AQI nesting'
            if family in ('logistic_regression', 'ensemble') and not _converged(sd, family):
                return 'ordered-logit convergence diagnostics missing or failed'
            if record is None:
                return 'completion record missing or malformed'
            if FAMILIES[family].NEEDS_CNN_CFG and any(
                sd['tasks'][task][family].get('status') != 'not_estimable' and
                not _tuning_context_valid(sd, task)
                for task in TASKS):
                return 'cnn tuning context missing'
            if family in PRIMARY_FAMILIES and any(
                sd['aqi_ablation']['nox_monitor'][task][family].get('contrast') != 'nox_given_monitor'
                for task in TASKS):
                return 'monitor contrast missing'
        if record['family'] != family or record['sites'] != sorted(sites):
            return 'completion roster differs'
        if record['inputs'] != expected_inputs or metrics.get('inputs') != expected_inputs:
            return 'producing inputs differ'
        if record['metrics_sha256'] != canonical_hash(metrics):
            return 'metrics hash mismatch'
        expected = prediction_inventory(metrics)
        if set(record['prediction_sha256']) != expected or core_prediction_files(out_dir, family) != expected:
            return 'prediction inventory differs'
        if any(file_sha256(out_dir / name) != digest for name, digest in record['prediction_sha256'].items()):
            return 'prediction hash mismatch'
    except (KeyError, TypeError, ValueError, OSError, AttributeError):
        return 'malformed or incomplete saved artifacts'
    return None


def _tuning_context_valid(sd, task):
    context = sd.get('cnn_tuning', {}).get(task, {})
    keys = {'policy', 'first_test_start', 'cutoff', 'feature_columns', 'train_sha256',
            'valid_sha256', 'n_train', 'n_valid', 'train_end', 'valid_end'}
    if set(context) != keys or context['policy'] != 'pre_all_tests_v1':
        return False
    cutoff = datetime.fromisoformat(context['cutoff'])
    return (isinstance(sd.get('cnn_cfg', {}).get(task), dict) and
            isinstance(context['feature_columns'], list) and
            cutoff == datetime.fromisoformat(context['first_test_start']) - timedelta(hours=24) and
            all(context['n_' + name] > 0 and
                datetime.fromisoformat(context[name + '_end']) < cutoff and
                isinstance(context[name + '_sha256'], str) and len(context[name + '_sha256']) == 64
                for name in ('train', 'valid')))


def _converged(sd, family):
    def accepted(info):
        diagnostics = ([info.get('optimization')] if family == 'logistic_regression' else
                       [row.get('optimization') for row in info.get('lr_optimization', [])])
        if family == 'ensemble':
            if [(row.get('phase'), row.get('block')) for row in info.get('lr_optimization', [])] != [
                    ('oof', 2), ('oof', 3), ('oof', 4), ('oof', 5), ('refit', None)]:
                return False
        keys = {'method', 'maxiter', 'maxfun', 'maxls', 'maxcor', 'gtol', 'ftol', 'nit',
                'nfev', 'objective', 'grad_inf_norm', 'projected_grad_inf_norm', 'status', 'message', 'converged'}
        return bool(diagnostics) and all(isinstance(d, dict) and set(d) == keys and
            d['converged'] is True and d['method'] == 'L-BFGS-B' and
            0 <= d['projected_grad_inf_norm'] <= 1e-7 and
            all(math.isfinite(d[key]) for key in ('objective', 'grad_inf_norm', 'projected_grad_inf_norm'))
            for d in diagnostics)
    for task in TASKS:
        if sd['tasks'][task][family].get('status') != 'not_estimable':
            records = sd.get('info', {}).get(task, {}).get('folds', [])
            if len(records) != 3 or not all(accepted(row) for row in records):
                return False
        for tasks in sd['aqi_ablation'].values():
            result = tasks[task][family]
            if result.get('status') not in ('not_estimable', 'not_applicable'):
                records = result.get('info', [])
                if len(records) != 6 or not all(accepted(row) for row in records):
                    return False
    return True


def _shard(family, site, cache):
    start = time.time()
    metrics = run_family(family, sites=[site], cnn_cache=cache, write=False)
    return metrics, time.time() - start


def run_parallel(families, sites, cache, jobs):
    """Run every (family, site) shard in `jobs` worker processes and write each family once all its sites finish.

    Shards use the same per-site seeds, CNN configs and site-namespaced files as the serial path, so results match
    `--jobs 1`. The first CNN family tunes each site; later CNN families wait for that site's configs. A failed
    family is reported after the others finish, so a resumed `--reuse` run only repeats the failures.
    """
    cnn = [f for f in families if FAMILIES[f].NEEDS_CNN_CFG]
    waiting = {(f, s) for f in cnn[1:] for s in sites if any((s, t) not in cache for t in TASKS)}
    done_sites, headers, runs, failed = {f: {} for f in families}, {}, {}, {}
    inputs = {f: {} for f in families}
    for family in families:
        completion_path(FAMILY_DIRS[family], family).unlink(missing_ok=True)
    order = cnn[:1] + [f for f in families if f not in cnn[:1]]  # tuner first so dependents unblock early
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {pool.submit(_shard, f, s, dict(cache)): (f, s)
                   for f in order for s in sites if (f, s) not in waiting}
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                family, site = pending.pop(future)
                if family in failed:
                    continue
                try:
                    metrics, seconds = future.result()
                except Exception as error:  # keep other families running; surfaced below
                    failed[family] = f'{site}: {error!r}'
                    print(f'{family} {site} FAILED: {error!r}', flush=True)
                    continue
                print(f'{family} {site} done in {seconds:.0f}s', flush=True)
                headers[family], done_sites[family][site] = metrics, metrics['per_site'][site]
                inputs[family].update(metrics['inputs'])
                if cnn and family == cnn[0]:
                    for task, cfg in metrics['per_site'][site]['cnn_cfg'].items():
                        context = metrics['per_site'][site].get('cnn_tuning', {}).get(task)
                        if context is not None:
                            cache[(site, task)] = {'cfg': cfg, 'context': context}
                    for dependent in cnn[1:]:
                        if (dependent, site) in waiting:
                            waiting.discard((dependent, site))
                            pending[pool.submit(_shard, dependent, site, dict(cache))] = (dependent, site)
                if len(done_sites[family]) == len(sites):
                    runs[family] = write_family({**headers[family], 'inputs': inputs[family],
                        'per_site': {s: done_sites[family][s] for s in sites}})
    failed.update({f: f'waits on failed {cnn[0]}' for f, _ in waiting if f not in failed})
    if failed:
        raise RuntimeError(f'families failed: {failed}')
    return runs


def main():
    parser = argparse.ArgumentParser(description="Run every ordinal-visibility V2 family with shared per-task CNN tuning")
    parser.add_argument("--sites", nargs="+", choices=tuple(SITES), default=tuple(SITES))
    parser.add_argument("--reuse", action="store_true",
                        help="reuse a family's saved metrics+predictions when current; stale families are retrained")
    parser.add_argument("--jobs", type=int, default=1,
                        help="(family, site) shards run in parallel worker processes; results match --jobs 1")
    args = parser.parse_args()
    cache = {}
    runs = {}
    todo = []
    for family, out_dir in FAMILY_DIRS.items():
        saved = out_dir / f"{family}_metrics_v2.json"
        reason = 'no saved metrics'
        if args.reuse and saved.exists():
            try:
                metrics = json.loads(saved.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                metrics = {}
            expected_inputs = {site: input_identity(family, (site,)) for site in args.sites}
            reason = stale_reason(family, out_dir, metrics, sites=args.sites, expected_inputs=expected_inputs)
            if reason is None:
                runs[family] = metrics
                # reuse the tuned CNN configs so later CNN-bearing families (ensemble) don't re-tune
                for site, sd in metrics['per_site'].items():
                    for task, cfg in sd.get('cnn_cfg', {}).items():
                        if cfg is not None:
                            cache.setdefault((site, task), {'cfg': cfg, 'context': sd['cnn_tuning'][task]})
                continue
        if args.reuse:
            print(f"{family}: retraining ({reason})", flush=True)
        if args.jobs > 1:
            todo.append(family)
        else:
            runs[family] = run_family(family, sites=args.sites, cnn_cache=cache)
    if todo:
        runs.update(run_parallel(todo, args.sites, cache, args.jobs))
    write_comparison({family: runs[family] for family in FAMILY_DIRS}, ROOT / "models" / "comparison_v2")


if __name__ == "__main__":
    main()
