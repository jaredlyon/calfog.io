"""Shared registry, parameterized windows and one-family orchestration contracts."""
import hashlib
import json
import numpy as np
import pandas as pd
from models._shared import study
from models._shared.data import WEATHER_COLS, CAL_COLS, AQI_COLS
from models._shared.families import FAMILIES, FamilyOutput, seed_for
from models._shared.windows import build_windows, tree_feature_names, tree_view, fit_preprocess
from models._shared.visibility import N_CATEGORIES


def test_registry_contract():
    assert set(FAMILIES) == {'climatology', 'logistic_regression', 'random_forest', 'xgboost',
                             'xgboost_focal_loss', 'temporal_cnn', 'ensemble', 'persistence', 'analog',
                             'xgboost_ordinal', 'xgboost_aft', 'ebm', 'xgboost_pooled'}
    for module in FAMILIES.values():
        assert isinstance(module.NEEDS_CNN_CFG, bool)
        assert callable(module.fit_predict)
    assert set(study.FAMILY_DIRS) == set(FAMILIES) - {'climatology'}
    assert study.PREDICTOR_SETS['xgboost'] == 'weather'
    assert study.PREDICTOR_SETS['persistence'] == 'observed visibility history'
    assert study.PREDICTOR_SETS['xgboost_pooled'] == 'weather + observed visibility (own + 4 neighbors), pooled sites'
    parts = ('xgboost', 'location_10', 'nowcast', 2)
    assert seed_for(*parts) == int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def test_custom_weather_columns_do_not_rebind_default_features():
    times = pd.date_range('2024-01-01', periods=30, freq='h')
    weather = pd.DataFrame({'time': times, 'custom': np.arange(30), **{c: np.zeros(30) for c in CAL_COLS}})
    part = pd.DataFrame({'issue_time': [times[-1]], 'category': [17]})
    X, y, meta, cols = build_windows(weather, part, weather_cols=['custom'])
    assert cols == ['custom'] + CAL_COLS
    assert X.shape == (1, 24, 5)
    assert WEATHER_COLS[0] == 'temperature_2m'
    assert len(tree_feature_names(cols)) == tree_view(X).shape[1]
    np.testing.assert_array_equal(tree_view(X)[0, :len(cols)], X[0, -1])


def test_preprocessing_is_fit_only_on_training_timesteps():
    X = np.array([[[0.], [2.]], [[2.], [4.]]], dtype=np.float32)
    held = np.full((1, 2, 1), 1000., dtype=np.float32)
    (train, val), imputer, scaler = fit_preprocess(X, held)
    np.testing.assert_allclose(scaler.mean_, [2.])
    np.testing.assert_allclose(train.mean(), 0., atol=1e-6)
    assert val.min() > 100


def test_run_family_routes_one_family_and_baseline_without_tuning(monkeypatch, tmp_path):
    times = pd.date_range('2024-01-01', periods=120 * 24, freq='h')
    weather = pd.DataFrame({'time': times, 'custom': np.sin(np.arange(len(times)) / 24),
                             **{c: np.zeros(len(times)) for c in CAL_COLS + AQI_COLS}})
    labels = pd.DataFrame({'issue_time': times, 'fog': np.arange(len(times)) % 2,
                          'category': np.where(np.arange(len(times)) % 2, 2, 17),
                          'row_key': times.astype(str), 'site': 'location_10'})
    monkeypatch.setattr(study, 'load_sources', lambda site, **kw: (weather, None))
    monkeypatch.setattr(study, 'label_tables', lambda *a: (labels, labels.copy()))
    calls = []

    def predict(prep, *, cnn_cfg, seed):
        calls.append((len(prep.cols), seed))
        def pmf(n):
            p = np.zeros((n, N_CATEGORIES))
            p[:, 2] = np.linspace(.1, .9, n)
            p[:, 17] = 1-p[:, 2]
            return p
        return FamilyOutput(pmf(len(prep.cv)), pmf(len(prep.ct)),
                            {name: 100 / len(prep.cols) for name in prep.cols})

    monkeypatch.setattr(FAMILIES['xgboost'], 'fit_predict', predict)
    monkeypatch.setattr(study, 'tuned_cnn_cfg', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('tabular family tuned CNN')))
    from models._shared import report
    monkeypatch.setattr(report, 'write_plots', lambda *a, **kw: None)
    metrics = study.run_family('xgboost', sites=('location_10',), weather_cols=['custom'], out_dir=tmp_path, n_boot=8)
    assert metrics['n_boot'] == 8
    assert metrics['weather_cols'] == ['custom']
    assert metrics['predictor_set'] == 'weather'
    assert len(calls) == 18  # 2 tasks x (3 full-history folds + 2 matched arms x 3 folds)
    sd = metrics['per_site']['location_10']
    for task in ('nowcast', 'lead_time'):
        assert set(sd['tasks'][task]) == {'xgboost', 'climatology'}
        assert sd['cnn_cfg'][task] is None
        assert len(sd['fold_audit'][task]) == 3
        result = sd['tasks'][task]['xgboost']
        assert result['valid_ap_resamples'] <= 8
        assert np.isclose(sum(result['importance'].values()), 100)
        assert result['ordinal']['rps_ci'][0] <= result['ordinal']['rps'] <= result['ordinal']['rps_ci'][1]
        ablation = sd['aqi_ablation'][task]['xgboost']
        assert ablation['bootstrap_resamples'] == 8
        assert ablation['identical_issue_keys']
        assert {'with', 'without'} <= ablation.keys()
        assert ablation['delta_rps_ci'][0] <= ablation['delta_rps'] <= ablation['delta_rps_ci'][1]
    path = tmp_path / 'xgboost_metrics_v2.json'
    assert json.loads(path.read_text())['family'] == 'xgboost'
    assert (tmp_path / 'xgboost_results_v2.txt').is_file()
    text = (tmp_path / 'xgboost_results_v2.txt').read_text(encoding='utf-8')
    assert 'RPS:' in text and 'NLL:' in text and 'Within one category:' in text
    assert 'Median-meters log10 MAE' in text
    assert 'Predictor set: weather' in text
    comparison = report.write_comparison({'xgboost': metrics}, tmp_path / 'comparison')
    assert comparison['aggregate']['nowcast']['xgboost']['mean_rps'] >= 0
    markdown = (tmp_path / 'comparison' / 'model_comparison_v2.md').read_text(encoding='utf-8')
    assert '| RPS (95% CI) |' in markdown and 'Paired delta RPS' in markdown
    assert '| Predictor set |' in markdown


def test_reuse_rejects_binary_only_metrics(monkeypatch, tmp_path):
    import sys
    import pytest
    from models import run_all_v2
    saved = tmp_path / 'xgboost_metrics_v2.json'
    saved.write_text(json.dumps({'per_site': {'location_10': {'tasks': {'nowcast': {
        'xgboost': {'ap': .5}}}}}}), encoding='utf-8')
    monkeypatch.setattr(run_all_v2, 'FAMILY_DIRS', {'xgboost': tmp_path})
    monkeypatch.setattr(sys, 'argv', ['run_all_v2.py', '--reuse'])
    with pytest.raises(ValueError, match='binary-only metrics cannot be reused'):
        run_all_v2.main()
