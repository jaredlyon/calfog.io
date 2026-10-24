"""Task-local eight-trial CNN search; no held-out test labels enter tuning."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import numpy as np
from .families.temporal_cnn import train_cnn


def _trainable(config):
    from ray import tune
    data = np.load(config['data_path'])
    cfg = {k: v for k, v in config.items() if k not in ('data_path', 'seed', 'device')}
    train_cnn(data['Xtr'], data['ctr'], data['Xv'], data['cv'], cfg,
              device=config['device'], max_epochs=3, patience=2, seed=config['seed'],
              report=lambda metrics: tune.report(metrics))


def tune_cnn(prep, site, task, *, art_dir: Path, seed: int) -> dict:
    import ray
    import torch
    from ray import tune
    from ray.tune.search.basic_variant import BasicVariantGenerator

    ddir = Path(art_dir).resolve() / site / 'tune'
    ddir.mkdir(parents=True, exist_ok=True)
    data = ddir / f'{task}.npz'
    # Keep the complete category prior here. train_cnn performs the capped
    # category-17 sampling and inverse-rate loss correction within each trial.
    np.savez_compressed(data, Xtr=prep.Xtr, ctr=prep.ctr, Xv=prep.Xv, cv=prep.cv)
    gpu = int(torch.cuda.is_available())
    space = {'data_path': str(data), 'seed': seed, 'device': 'cuda' if gpu else 'cpu',
             'channels': tune.choice([16, 24, 32, 48]), 'kernel': tune.choice([3, 5]),
             'dropout': tune.choice([.15, .3, .45]), 'lr': tune.loguniform(2e-4, 2e-3),
             'weight_decay': tune.choice([1e-5, 1e-4, 1e-3]),
             'batch_size': tune.choice([256, 512]), 'max_neg': 25000}
    # REPRO-2: Ray's Unix socket paths must stay well below the platform limit.
    with tempfile.TemporaryDirectory(prefix='cf_', dir='/tmp' if Path('/tmp').is_dir() else None) as runtime:
        owned_runtime = not ray.is_initialized()
        if owned_runtime:
            ray.init(num_cpus=min(8, os.cpu_count() or 4), num_gpus=gpu, include_dashboard=False,
                     log_to_driver=False, _temp_dir=runtime)
        try:
            tuner = tune.Tuner(tune.with_resources(_trainable, {'cpu': 2, 'gpu': gpu}), param_space=space,
                               tune_config=tune.TuneConfig(metric='val_nll', mode='min', num_samples=8,
                                   max_concurrent_trials=1, search_alg=BasicVariantGenerator(random_state=seed)),
                               run_config=tune.RunConfig(name=f'{site}_{task}', storage_path=str(ddir / 'results'), verbose=0))
            best = tuner.fit().get_best_result(metric='val_nll', mode='min', scope='all')
            return {k: v for k, v in best.config.items() if k not in ('data_path', 'seed', 'device')}
        finally:
            if owned_runtime:
                ray.shutdown()
