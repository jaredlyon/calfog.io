"""Environment provenance and atomic artifact integrity, without subprocess fitting."""
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from models._shared import artifacts, environment


def test_capture_environment_cached_exact_and_checked(monkeypatch):
    calls = []
    def freeze(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout='numpy==2.5.2\nscipy==1.18.0\n')
    environment.capture_environment.cache_clear()
    monkeypatch.setattr(environment.subprocess, 'run', freeze)
    monkeypatch.setattr(environment.platform, 'platform', lambda: 'synthetic-platform')
    monkeypatch.setattr(environment.platform, 'machine', lambda: 'synthetic-machine')
    monkeypatch.setenv('OMP_NUM_THREADS', '3')
    try:
        observed = environment.capture_environment()
        assert observed is environment.capture_environment()
        assert len(calls) == 1
        assert calls[0][0] == [sys.executable, '-m', 'pip', '--disable-pip-version-check', 'freeze', '--all']
        assert calls[0][1]['check'] is True
        assert set(observed) == {'python', 'platform', 'machine', 'pip_freeze', 'cuda', 'cudnn', 'thread_env'}
        assert observed['platform'] == 'synthetic-platform'
        assert observed['machine'] == 'synthetic-machine'
        assert observed['pip_freeze'] == ['numpy==2.5.2', 'scipy==1.18.0']
        assert observed['thread_env']['OMP_NUM_THREADS'] == '3'
        json.dumps(observed, allow_nan=False)
    finally:
        environment.capture_environment.cache_clear()


def test_capture_environment_propagates_freeze_failure(monkeypatch):
    environment.capture_environment.cache_clear()
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(7, args[0])
    monkeypatch.setattr(environment.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        environment.capture_environment()
    environment.capture_environment.cache_clear()


def test_canonical_hash_atomic_publication_and_verified_bytes(monkeypatch, tmp_path):
    assert artifacts.canonical_hash({'a': 1, 'b': 2}) == artifacts.canonical_hash({'b': 2, 'a': 1})
    assert artifacts.canonical_hash({'a': 2}) != artifacts.canonical_hash({'a': 1})
    path = tmp_path / 'value.json'
    artifacts.atomic_write_json(path, {'old': True})
    digest = artifacts.file_sha256(path)
    assert artifacts.read_verified(path, digest) == path.read_bytes()
    with pytest.raises(ValueError, match='sha256 mismatch'):
        artifacts.read_verified(path, '0' * 64)
    old = path.read_bytes()
    def fail(*args):
        raise OSError('replace failed')
    monkeypatch.setattr(artifacts.os, 'replace', fail)
    with pytest.raises(OSError, match='replace failed'):
        artifacts.atomic_write_json(path, {'new': True})
    assert path.read_bytes() == old
    assert not list(tmp_path.glob('*.part'))


@pytest.mark.parametrize('value', [None, [], {}, {'schema_version': 2}])
def test_completion_rejects_missing_malformed_and_unsupported(tmp_path, value):
    path = artifacts.completion_path(tmp_path, 'xgboost')
    assert artifacts.load_completion(tmp_path, 'xgboost') is None
    path.write_text('not JSON' if value is None else json.dumps(value), encoding='utf-8')
    assert artifacts.load_completion(tmp_path, 'xgboost') is None


@pytest.mark.parametrize('module_name', ['seed_replicates_v2', 'lag_aware_nox_v2'])
def test_prediction_only_units_record_actual_environment(monkeypatch, tmp_path, module_name):
    import importlib
    import pandas as pd
    module = importlib.import_module('models.' + module_name)
    sentinel = {'python': 'producer'}
    monkeypatch.setattr(module, 'capture_environment', lambda: sentinel)
    monkeypatch.setitem(module.FAMILY_DIRS, 'xgboost', tmp_path)
    monkeypatch.setattr(module, 'load_sources', lambda *a, **kw: (pd.DataFrame(), None))
    if module_name == 'lag_aware_nox_v2':
        monkeypatch.setattr(module, 'hourly_visibility_lag', lambda *a: [])
    monkeypatch.setattr(module, 'label_tables', lambda *a: (None, None))
    monkeypatch.setattr(module, 'build_windows', lambda *a, **kw: (None, None, None, None))
    def evaluate(*a, prediction_dir, aq_set, task=None, **kw):
        prediction_dir.mkdir(parents=True, exist_ok=True)
        for arm in ('with', 'without'):
            (prediction_dir / f'{a[2]}__{a[3]}__{aq_set}_{arm}.npz').write_bytes(b'prediction')
        return {}
    monkeypatch.setattr(module, 'evaluate_ablation', evaluate)
    args = ('xgboost', 'location_10', 1) if module_name == 'seed_replicates_v2' else ('xgboost', 'location_10')
    assert module.run_unit(*args) == args
    directory = module.replicate_dir('xgboost', 1) if len(args) == 3 else module.prediction_dir('xgboost')
    manifest = json.loads(next(directory.glob('*.environment.json')).read_text())
    assert manifest['environment'] == sentinel
    assert len(manifest['prediction_files']) == 4
    assert all((directory / name).is_file() for name in manifest['prediction_files'])
