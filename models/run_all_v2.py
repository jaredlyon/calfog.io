#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import multiprocessing
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from models._shared.data import SITES, AQ_SETS, PRIMARY_FAMILIES
from models._shared.families import FAMILIES
from models._shared.study import FAMILY_DIRS, run_family, write_family
from models._shared.report import write_comparison

TASKS = ('nowcast', 'lead_time')


def stale_reason(family, out_dir, metrics):
    """Why saved metrics cannot be reused (None if reusable)."""
    if any('ordinal' not in result for sd in metrics['per_site'].values()
           for results in sd['tasks'].values() for result in results.values()):
        return 'binary-only metrics'
    expected_sets = set(AQ_SETS) | ({'nox_monitor'} if family in PRIMARY_FAMILIES else set())
    if not metrics['per_site'] or any(
        set(sd.get('aqi_ablation', {})) != expected_sets or
        any(task not in sd['aqi_ablation'][aq_set] or family not in sd['aqi_ablation'][aq_set][task]
            for aq_set in expected_sets for task in ('nowcast', 'lead_time'))
        for sd in metrics['per_site'].values()
    ):
        return 'old AQI nesting'
    if not (out_dir / f"{family}_predictions_v2").is_dir():
        return 'metrics without predictions'
    return None


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
                if cnn and family == cnn[0]:
                    for task, cfg in metrics['per_site'][site]['cnn_cfg'].items():
                        cache.setdefault((site, task), cfg)
                    for dependent in cnn[1:]:
                        if (dependent, site) in waiting:
                            waiting.discard((dependent, site))
                            pending[pool.submit(_shard, dependent, site, dict(cache))] = (dependent, site)
                if len(done_sites[family]) == len(sites):
                    runs[family] = write_family({**headers[family], 'per_site': {s: done_sites[family][s] for s in sites}})
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
            metrics = json.loads(saved.read_text(encoding="utf-8"))
            reason = stale_reason(family, out_dir, metrics)
            if reason is None:
                runs[family] = metrics
                # reuse the tuned CNN configs so later CNN-bearing families (ensemble) don't re-tune
                for site, sd in metrics['per_site'].items():
                    for task, cfg in sd.get('cnn_cfg', {}).items():
                        if cfg is not None:
                            cache.setdefault((site, task), cfg)
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
