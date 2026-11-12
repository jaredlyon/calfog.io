"""Artifact/environment regressions; CPU only, no training or network fetches."""
import ast
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest
import torch

from ops.retrain import run_retrain as retrain

ROOT = Path(__file__).resolve().parents[2]
PAIRS = [(f"location_{n}", task) for n in range(6, 11)
         for task in ("nowcast", "lead_time")]


@pytest.fixture
def cpu_loader(monkeypatch):
    load = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **kw: load(*a, **{**kw, "map_location": "cpu"}))
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *a, **kw: self)


@pytest.mark.parametrize("site,task", PAIRS)
def test_committed_retrain_bundles_load_and_predict(site, task, cpu_loader, monkeypatch):
    if not retrain.serving_bundle_exists(site, task):
        pytest.skip("ordinal-18 retrain bundles absent; regenerate with run_retrain.py")
    bundle = retrain.load_serving_bundle(site, task)
    rf_file = retrain.SERV / site / f"{task}_random_forest.joblib"  # git-ignored; present only after a local rerun
    assert (bundle["random_forest"] is None) == (not rf_file.exists())
    bundle["random_forest"] = None  # exercise the committed (clean-clone) bundle contract
    cols = bundle["cols"]
    weather = pd.DataFrame({c: np.zeros(24) for c in cols})
    weather["time"] = pd.date_range("2024-01-01", periods=24, freq="h")
    labels = pd.DataFrame({"issue_time": [weather.time.iloc[-1]], "category": [17], "fog": [0],
                           "label_time": [weather.time.iloc[-1] + pd.Timedelta(hours=1)]})
    predict = retrain.temporal_cnn.predict_cnn
    monkeypatch.setattr(retrain.temporal_cnn, "predict_cnn", lambda m, X, device: predict(m, X, "cpu"))
    y, predictions, _ = retrain.predict_proxy(weather, labels, bundle)
    assert set(predictions) == {"climatology", "xgboost", "temporal_cnn"}
    assert all(p.shape == (1,18) and np.isfinite(p).all() and np.allclose(p.sum(1),1) for p in predictions.values())
    manifest = json.loads((retrain.SERV / site / f"{task}_manifest.json").read_text())
    assert set(manifest["models"]) == set(predictions)
    assert manifest["optional_models"] == ["random_forest"]
    # An available optional model must still be scored, not silently dropped.
    bundle["random_forest"] = bundle["xgboost"]
    assert "random_forest" in retrain.predict_proxy(weather, labels, bundle)[1]


def test_resume_requires_every_core_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(retrain, "SERV", tmp_path)
    directory = tmp_path / "location_6"
    directory.mkdir()
    suffixes = ("preprocessor.joblib", "xgboost.joblib", "calibrators.joblib",
                "temporal_cnn.pt", "manifest.json")
    paths = [directory / f"nowcast_{suffix}" for suffix in suffixes]
    for path in paths:
        path.write_text('{"target":"ordinal-18"}' if path.suffix=='.json' else '')
    assert retrain.serving_bundle_exists("location_6", "nowcast")
    for path in paths:
        path.unlink()
        assert not retrain.serving_bundle_exists("location_6", "nowcast"), path.name
        path.write_text('{"target":"ordinal-18"}' if path.suffix=='.json' else '')


@pytest.mark.parametrize("site,task", PAIRS)
def test_shadow_bundle_is_separate_and_rf_free(site, task):
    from ops.shadow.shadow_predict import load_bundle
    directory=ROOT/"ops"/"shadow"/"serving"/site/task
    required=("preprocessing.joblib","metadata.json","xgboost.joblib","temporal_cnn.pt","calibrator_xgboost.joblib","calibrator_temporal_cnn.joblib")
    if not all((directory/name).is_file() for name in required):
        pytest.skip("ordinal-18 shadow bundles absent; regenerate with build_serving.py")
    if json.loads((directory/"metadata.json").read_text()).get("target")!="ordinal-18":
        pytest.skip("shadow bundle predates ordinal-18; regenerate with build_serving.py")
    pre, metadata, xgb, cnn, calx, calc = load_bundle(site, task)
    assert set(pre) == {"imputer", "scaler"}
    assert metadata["site"] == site and metadata["task"] == task
    assert next(cnn.parameters()).device.type == "cpu"


@pytest.mark.parametrize("launcher,script", [
    ("ops/retrain/run_retrain.sh", "ops/retrain/run_retrain.py"),
])
def test_launcher_uses_selected_python_and_own_checkout(tmp_path, launcher, script):
    assert "/home/pa/work" not in (ROOT / launcher).read_text()
    subprocess.run(["bash", "-n", str(ROOT / launcher)], check=True)
    probe = tmp_path / "python-probe"
    harness = ('cd() { printf "CWD=%s\\n" "$(realpath -- "${@: -1}")"; }; '
               'python() { printf "%s\\n" "$@"; }; '
               'exec() { printf "PYTHON=%s\\nSCRIPT=%s\\n" "$1" "$2"; }; '
               'tee() { cat; }; source "$1"')
    completed = subprocess.run(["bash", "-c", harness, "_", str(ROOT / launcher)], cwd=tmp_path,
                               env={**os.environ, "PYTHON": str(probe)},
                               text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert f"CWD={ROOT}\nPYTHON={probe}\nSCRIPT={script}\n" in completed.stdout
    print(completed.stdout, end="")


def test_one_v2_environment_covers_imports():
    requirements = (ROOT / "requirements-v2.txt").read_text()
    for pin in ("torch==2.11.0+cu128", "ray[tune]==2.57.0", "xgboost==3.4.1",
                "scikit-learn==1.9.0", "numpy==2.5.2", "pandas==3.0.5",
                "joblib==1.5.3", "requests==2.34.2", "pytest==9.1.1",
                "matplotlib==3.11.1"):
        assert pin in requirements
    assert "Python 3.12.3" in requirements
    assert "https://download.pytorch.org/whl/cu128" in requirements
    shadow = (ROOT / "ops/shadow/requirements-shadow.txt").read_text()
    assert shadow.strip() == "-r ../../requirements-v2.txt"


def test_ray_starts_from_deep_checkout_path(tmp_path):
    import ray
    import tempfile
    source = ROOT / "models/_shared/tuning.py"
    tree = ast.parse(source.read_text())
    tune = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "tune_cnn")
    call = next(node for node in ast.walk(tune) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "ray" and node.func.attr == "init")
    value = next(kw.value for kw in call.keywords if kw.arg == "_temp_dir")
    assert isinstance(value, ast.Name)
    context = next(item for node in ast.walk(tune) if isinstance(node, ast.With)
                   for item in node.items if isinstance(item.optional_vars, ast.Name)
                   and item.optional_vars.id == value.id)
    deep = tmp_path / ("nested-" * 12)
    deep.mkdir(parents=True)
    old = Path.cwd()
    try:
        os.chdir(deep)
        runtime = eval(compile(ast.Expression(context.context_expr), str(source), "eval"),
                       {"tempfile": tempfile, "Path": Path})
        with runtime as temp_dir:
            try:
                ray.init(num_cpus=1, num_gpus=0, include_dashboard=False,
                         log_to_driver=False, _temp_dir=temp_dir)
                assert ray.is_initialized()
                assert ray.cluster_resources().get("GPU", 0) == 0
                print(f"ray.init OK cwd={deep} temp_dir={temp_dir} GPU=0")
            finally:
                ray.shutdown()
    finally:
        os.chdir(old)


@pytest.fixture
def resume_site(tmp_path,monkeypatch):
    import copy
    monkeypatch.setattr(retrain,'SERV',tmp_path/'serving')
    site='location_6';directory=retrain.SERV/site;directory.mkdir(parents=True)
    for task in ('nowcast','lead_time'):
        for suffix in ('preprocessor.joblib','xgboost.joblib','calibrators.joblib',
                       'temporal_cnn.pt','random_forest.joblib','manifest.json'):
            (directory/f'{task}_{suffix}').write_text('{"target":"ordinal-18"}' if suffix=='manifest.json' else 'original')
    for name in ('feature_spec.json','metadata.json'):(directory/name).write_text('{}')
    inputs={'data':'hash','code':'hash','features':['temperature_2m'],'environment':{'python':'test'}}
    result={'tasks':{task:{'ap':.5} for task in ('nowcast','lead_time')}}
    log={'tasks':{task:{'tuning_trials':8} for task in ('nowcast','lead_time')}}
    result['resume']={'schema_version':1,'inputs':copy.deepcopy(inputs),
        'metrics_sha256':retrain.canonical_hash(result),'log_sha256':retrain.canonical_hash(log),
        'bundle_sha256':retrain.bundle_hashes(site)}
    return site,result,log,inputs


def test_resume_preserves_producer_and_allows_missing_rf(resume_site):
    import copy
    site,result,log,inputs=resume_site;before=copy.deepcopy(result)
    assert retrain.resume_matches(site,result,log,inputs)
    for task in ('nowcast','lead_time'):
        (retrain.SERV/site/f'{task}_random_forest.joblib').unlink()
    assert retrain.resume_matches(site,result,log,inputs)
    assert result==before


@pytest.mark.parametrize('changed',['inputs','metrics','log','missing_task','missing_record',
    'preprocessor.joblib','xgboost.joblib','calibrators.joblib','temporal_cnn.pt',
    'manifest.json','random_forest.joblib','feature_spec.json','metadata.json'])
def test_resume_rejects_changed_generation(resume_site,changed):
    site,result,log,inputs=resume_site
    if changed=='inputs':inputs['data']='changed'
    elif changed=='metrics':result['tasks']['nowcast']['ap']=.4
    elif changed=='log':log['tasks']['nowcast']['tuning_trials']=7
    elif changed=='missing_task':result['tasks'].pop('lead_time')
    elif changed=='missing_record':result.pop('resume')
    else:
        name=changed if changed in ('feature_spec.json','metadata.json') else f'lead_time_{changed}'
        (retrain.SERV/site/name).write_text('modified')
    assert not retrain.resume_matches(site,result,log,inputs)


def test_resume_rejects_unrecorded_rf(resume_site):
    site,result,log,inputs=resume_site
    result['resume']['bundle_sha256'].pop('nowcast_random_forest.joblib')
    assert not retrain.resume_matches(site,result,log,inputs)


def test_checkpoint_is_one_versioned_atomic_record(tmp_path,resume_site,monkeypatch):
    site,result,site_log,_=resume_site
    path=tmp_path/'metrics_retrain.partial.json'
    metrics={'target':'ordinal-18','per_site':{site:result}}
    log={'per_site':{site:site_log}}
    retrain.save_checkpoint(path,metrics,log)
    assert retrain.load_checkpoint(path)==(metrics,log)
    before=path.read_bytes()
    def fail(*args):raise OSError('injected publication failure')
    monkeypatch.setattr(retrain,'atomic_write_json',fail)
    with pytest.raises(OSError,match='publication failure'):retrain.save_checkpoint(path,{}, {})
    assert path.read_bytes()==before
    assert sorted(p.name for p in tmp_path.glob('*.json'))==['metrics_retrain.partial.json']


@pytest.mark.parametrize('checkpoint',[
    {},{'target':'ordinal-18','per_site':{}},
    {'schema_version':2,'metrics':{'target':'ordinal-18','per_site':{}},'log':{'per_site':{}}},
    {'schema_version':1,'metrics':{'target':'ordinal-18','per_site':{'location_6':{}}},'log':{'per_site':{}}}])
def test_legacy_malformed_or_mixed_checkpoint_is_stale(tmp_path,checkpoint):
    path=tmp_path/'metrics_retrain.partial.json';path.write_text(json.dumps(checkpoint))
    assert retrain.load_checkpoint(path) is None


@pytest.mark.parametrize('changed',['weather','visibility','label_code','family_code','features','cutoff','environment'])
def test_producing_fingerprint_binds_inputs_code_config_and_environment(tmp_path,monkeypatch,changed):
    monkeypatch.setattr(retrain,'ROOT',tmp_path)
    monkeypatch.setattr(retrain,'capture_environment',lambda:{'python':'test'})
    paths={'weather':tmp_path/'datasets/weather/location_6_weather_appended.csv',
           'visibility':tmp_path/'datasets/visibility/location_6_visibility_hourly.csv',
           'label_code':tmp_path/'models/_shared/labels.py',
           'family_code':tmp_path/'models/_shared/families/xgboost.py'}
    for path in paths.values():path.parent.mkdir(parents=True,exist_ok=True);path.write_text('original')
    before=retrain.producing_inputs('location_6')
    if changed in paths:paths[changed].write_text('modified')
    elif changed=='features':monkeypatch.setattr(retrain,'APPROVED',retrain.APPROVED+['different'])
    elif changed=='cutoff':monkeypatch.setattr(retrain,'ARCHIVE_CUTOFF','2025-08-26 23:00:00')
    else:monkeypatch.setattr(retrain,'capture_environment',lambda:{'python':'changed'})
    assert retrain.producing_inputs('location_6')!=before
