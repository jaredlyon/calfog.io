"""V1-style family text reports and result-driven cross-family comparisons."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from .data import SITES
from .evaluation import COST_GRID


def aggregate(metrics):
    sites = metrics['per_site']
    out = {'aqi_not_estimable': [], 'aqi_not_applicable': []}
    notes, resolved = [], []
    for task in ('nowcast', 'lead_time'):
        families = sorted(set().union(*(sd['tasks'][task] for sd in sites.values())))
        out[task] = {}
        for family in families:
            values = {site: sd['tasks'][task][family]['ap'] for site, sd in sites.items()
                      if family in sd['tasks'][task]}
            out[task][family] = {'mean_ap': float(np.mean(list(values.values()))), 'site_aps': values}
            rps_values = {site: sd['tasks'][task][family]['ordinal']['rps'] for site, sd in sites.items()
                          if family in sd['tasks'][task]}
            out[task][family].update(mean_rps=float(np.mean(list(rps_values.values()))), site_rps=rps_values)
    out['site_task_summary'], out['aqi_site_summary'] = {}, {}
    for site, sd in sites.items():
        out['site_task_summary'][site], out['aqi_site_summary'][site] = {}, {}
        for task in ('nowcast', 'lead_time'):
            results = sd['tasks'][task]
            out['site_task_summary'][site][task] = {family: {key: result[key] for key in ('ap', 'ap_ci', 'ordinal')}
                                                   for family, result in results.items()}
            out['aqi_site_summary'][site][task] = {family: {key: result[key] for key in ('delta_ap', 'delta_ap_ci', 'delta_rps', 'delta_rps_ci', 'pos_count')}
                                                   for family, result in sd['aqi_ablation'][task].items()
                                                   if result.get('status') not in ('not_estimable', 'not_applicable')}
            learned = [family for family in results if family != 'climatology']
            if learned:
                best = max(learned, key=lambda family: results[family]['ap'])
                result, baseline = results[best], results['climatology']
                strength = 'interval-separated from climatology' if result['ap_ci'][0] > baseline['ap_ci'][1] else 'not interval-separated from climatology'
                notes.append(f'{site} {task}: strongest {best} AP={result["ap"]:.3f} ({strength})')
            for family, result in sd['aqi_ablation'][task].items():
                if result.get('status') in ('not_estimable', 'not_applicable'):
                    out['aqi_' + result['status']].append({'site': site, 'task': task, 'family': family, 'reason': result['reason']})
                    continue
                lo, hi = result['delta_ap_ci']
                if lo > 0 or hi < 0:
                    resolved.append(f'{site}/{task}/{family} (delta={result["delta_ap"]:+.3f}, n+={result["pos_count"]})')
    out['predictability_verdict'] = '; '.join(notes) + '. Results are heterogeneous and retrospective; no site is approved for operations without prospective latency-controlled shadow validation.'
    out['aqi_verdict'] = ('Among estimable comparisons, no paired AQI delta AP interval excluded zero.' if not resolved else
                          'Directionally resolved paired intervals: ' + ', '.join(resolved) + '. These are individual comparisons, not multiplicity-adjusted evidence of a general AQI contribution.')
    if out['aqi_not_estimable']:
        out['aqi_verdict'] += f" {len(out['aqi_not_estimable'])} AQI comparisons were not estimable and excluded."
    if out['aqi_not_applicable']:
        out['aqi_verdict'] += f" {len(out['aqi_not_applicable'])} AQI comparisons were not applicable and excluded."
    return out


def _interval(value, bounds):
    return f'{value:.6f} (95% CI {bounds[0]:.6f}, {bounds[1]:.6f})'


def _meters_error(ordinal):
    value = ordinal['median_meters_mae_log10']
    return f'{value:.6f} (n={ordinal["median_meters_n"]})' if value is not None else 'not estimable (n=0)'


def _cohort_lines(title, result, climatology_ap):
    cal = result['calibration']
    lines = [title, f'Average Precision: {_interval(result["ap"], result["ap_ci"])}',
             f'RPS: {_interval(result["ordinal"]["rps"], result["ordinal"]["rps_ci"])}',
             f'NLL: {result["ordinal"]["nll"]:.6f}',
             f'Within one category: {result["ordinal"]["within_one"]:.6f}',
             f'Median-meters log10 MAE (observed and median categories 1–16 only): {_meters_error(result["ordinal"])}',
             f'ROC-AUC: {_interval(result["roc_auc"], result["roc_auc_ci"])}',
             f'Brier (calibrated): {cal["brier_after"]:.6f}', f'ECE (calibrated): {cal["ece_after"]:.6f}',
             f'Cost-selected threshold (1:1 FN:FP; fold mean): {result["cost_threshold"]:.6f}',
             f'Precision: {result["precision"]:.6f}', f'Recall: {result["recall"]:.6f}', f'F1: {result["f1"]:.6f}',
             f'Climatology AP: {climatology_ap:.6f}']
    for ratio, point in result['cost_operating_points'].items():
        lines.append(f'FN:FP {ratio}:1 threshold={point["mean_threshold"]:.6f}, precision={point["precision"]:.6f}, recall={point["recall"]:.6f}, F1={point["f1"]:.6f}, cost/issue={point["expected_cost_per_issue"]:.6f}')
    if result.get('importance'):
        lines.append('Feature Importance (%):')
        lines.extend(f'  {name}: {value:.6f}%' for name, value in sorted(result['importance'].items(), key=lambda item: (-item[1], item[0])))
    if result.get('ci_warning'):
        lines.append(result['ci_warning'])
    return lines + ['']


def write_family_report(metrics, path):
    family = metrics['family']
    lines = [f'{family.upper()} FOG PREDICTION V2',
             f'Predictor set: {metrics.get("predictor_set", "weather")}',
             'Clock: fixed PST (UTC-8). Hourly label: minimum known GHCNh visibility category; fog = reported <=1 SM.',
             'AP/AUC/RPS intervals are marginal whole-day block intervals, not paired-model comparisons.',
             'Thresholds and isotonic calibration fit validation only; each held-out fold is evaluated once.', '']
    grouped = {}
    for site, sd in metrics['per_site'].items():
        lines += [f'LOCATION {site.removeprefix("location_")} ({SITES.get(site, site)})', '']
        for task in ('nowcast', 'lead_time'):
            lines += [task.upper(), '']
            ablation = sd['aqi_ablation'][task][family]
            blocks = [('WITHOUT AQI (full history)', sd['tasks'][task][family], sd['tasks'][task]['climatology']['ap'])]
            if ablation.get('status') not in ('not_estimable', 'not_applicable'):
                blocks += [('WITH AQI (matched rows)', ablation['with'], ablation['with']['climatology_ap']),
                           ('WITHOUT AQI (matched rows)', ablation['without'], ablation['without']['climatology_ap'])]
            for title, result, baseline in blocks:
                lines += _cohort_lines(title, result, baseline)
                grouped.setdefault((task, title), []).append((result, baseline))
            if ablation.get('status') in ('not_estimable', 'not_applicable'):
                status = ablation['status'].replace('_', ' ')
                lines += [f'AQI ablation: {status} — {ablation["reason"]}', '']
            else:
                lines += [f'Paired AQI delta AP: {_interval(ablation["delta_ap"], ablation["delta_ap_ci"])}',
                          f'Paired AQI delta RPS (with minus without; negative favors AQI): {_interval(ablation["delta_rps"], ablation["delta_rps_ci"])}', '']
    lines += ['AVERAGE METRICS', 'Arithmetic means across sites, not pooled observations; mean site-CI endpoints are descriptive, not an aggregate confidence interval.', '']
    for (task, title), rows in grouped.items():
        lines += [f'{task.upper()} - {title}']
        for key in ('ap', 'roc_auc', 'precision', 'recall', 'f1', 'cost_threshold'):
            lines.append(f'{key}: {np.mean([result[key] for result, _ in rows]):.6f}')
        for key in ('ap_ci', 'roc_auc_ci'):
            lo, hi = np.mean([result[key] for result, _ in rows], axis=0)
            lines.append(f'Mean site {key} endpoints (not aggregate CI): {lo:.6f}, {hi:.6f}')
        for key in ('rps', 'nll', 'within_one', 'median_category_mae'):
            lines.append(f'{key}: {np.mean([result["ordinal"][key] for result, _ in rows]):.6f}')
        meter_rows = [result['ordinal'] for result, _ in rows if result['ordinal']['median_meters_n']]
        meter_count = sum(row['median_meters_n'] for row in meter_rows)
        meter_mae = sum(row['median_meters_mae_log10'] * row['median_meters_n'] for row in meter_rows) / meter_count if meter_count else None
        lines.append(f'Pooled median-meters log10 MAE: {_meters_error({"median_meters_mae_log10": meter_mae, "median_meters_n": meter_count})}')
        for key in ('brier_after', 'ece_after'):
            lines.append(f'{key}: {np.mean([result["calibration"][key] for result, _ in rows]):.6f}')
        lines += [f'Climatology AP: {np.mean([baseline for _, baseline in rows]):.6f}', '']
    lines += [metrics['aggregate']['predictability_verdict'], metrics['aggregate']['aqi_verdict']]
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_plots(metrics, *, art_dir):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    for site, sd in metrics['per_site'].items():
        folder = Path(art_dir) / site
        folder.mkdir(parents=True, exist_ok=True)
        for task in ('nowcast', 'lead_time'):
            results = {family: result for family, result in sd['tasks'][task].items()
                       if result.get('status') not in ('not_estimable', 'not_applicable')}
            if not results:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
            for ax, which, title in ((axes[0], 'reliability_before', 'raw'), (axes[1], 'reliability_after', 'validation-fit isotonic')):
                ax.plot([0, 1], [0, 1], 'k--', lw=1, label='ideal')
                for family, result in results.items():
                    rows = result['calibration'][which]
                    ax.plot([r['mean_probability'] for r in rows], [r['observed_rate'] for r in rows], marker='o', ms=3, label=family)
                ax.set(title=title, xlabel='mean predicted probability', ylabel='observed fog rate', xlim=(0, 1), ylim=(0, 1))
                ax.grid(alpha=.2)
            axes[1].legend(fontsize=7)
            fig.suptitle(f'{site} {task}: reliability')
            fig.tight_layout()
            fig.savefig(folder / f'reliability_{task}.png', dpi=150)
            plt.close(fig)
            nrows = (len(results) + 1) // 2
            fig, axes = plt.subplots(nrows, 2, figsize=(10, 3.5 * nrows), sharex=True, squeeze=False)
            for ax, (family, result) in zip(axes.ravel(), results.items()):
                for ratio in COST_GRID:
                    curve = result['cost_threshold_curves'][str(ratio)]
                    ax.plot(curve['thresholds'], curve['validation_expected_cost'], label=f'{ratio}:1')
                ax.set(title=family, xlabel='threshold', ylabel='validation cost/issue', xlim=(0, 1))
                ax.grid(alpha=.2)
            for ax in axes.ravel()[len(results):]:
                ax.set_visible(False)
            axes[0, 0].legend(title='FN:FP', fontsize=7)
            fig.suptitle(f'{site} {task}: predeclared cost curves')
            fig.tight_layout()
            fig.savefig(folder / f'cost_curves_{task}.png', dpi=150)
            plt.close(fig)


def write_comparison(runs, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = {'families': list(runs),
                'predictor_sets': {'climatology': 'weather',
                                   **{family: metrics.get('predictor_set', 'weather') for family, metrics in runs.items()}},
                'per_site': {}}
    for family, metrics in runs.items():
        for site, sd in metrics['per_site'].items():
            target = combined['per_site'].setdefault(site, {'cohort': sd['cohort'], 'tasks': {}, 'aqi_ablation': {}, 'fold_audit': {}})
            for task in ('nowcast', 'lead_time'):
                target['tasks'].setdefault(task, {}).update(sd['tasks'][task])
                target['aqi_ablation'].setdefault(task, {}).update(sd['aqi_ablation'][task])
                target['fold_audit'].setdefault(task, {})[family] = sd['fold_audit'][task]
    combined['aggregate'] = aggregate(combined)
    lines = ['# CalFog V2 model comparison', '',
             'Fixed PST; hourly minimum GHCNh category; fog = reported <=1 SM. Retrospective expanding-origin evaluation.',
             'AP/AUC/RPS confidence intervals are marginal per-model whole-day block intervals; only AQI deltas use paired draws.',
             'Median-meters log10 error excludes rows where either the observed or median category is censored (0 or 17).', '',
             '| Site | Task | Family | Predictor set | AP (95% CI) | RPS (95% CI) | NLL | Within one | Median-meters log10 MAE (n) | ROC-AUC (95% CI) | Calibrated Brier / ECE |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for site, sd in combined['per_site'].items():
        for task, results in sd['tasks'].items():
            for family, result in results.items():
                cal = result['calibration']
                ordinal = result['ordinal']
                lines.append(f'| {site} | {task} | {family} | {combined["predictor_sets"][family]} | {_interval(result["ap"], result["ap_ci"])} | {_interval(ordinal["rps"], ordinal["rps_ci"])} | {ordinal["nll"]:.6f} | {ordinal["within_one"]:.6f} | {_meters_error(ordinal)} | {_interval(result["roc_auc"], result["roc_auc_ci"])} | {cal["brier_after"]:.6f} / {cal["ece_after"]:.6f} |')
    lines += ['', '## Validation-selected operating points', '',
              '| Site | Task | Family | FN:FP | Mean threshold | Precision | Recall | F1 | Alert rate | Cost/issue |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for site, sd in combined['per_site'].items():
        for task, results in sd['tasks'].items():
            for family, result in results.items():
                for ratio, point in result['cost_operating_points'].items():
                    lines.append(f'| {site} | {task} | {family} | {ratio}:1 | {point["mean_threshold"]:.6f} | {point["precision"]:.6f} | {point["recall"]:.6f} | {point["f1"]:.6f} | {point["alert_rate"]:.6f} | {point["expected_cost_per_issue"]:.6f} |')
    lines += ['', '## Cross-site mean scores', '', '| Task | Family | Mean AP | Mean RPS |', '|---|---|---|---|']
    for task in ('nowcast', 'lead_time'):
        lines += [f'| {task} | {family} | {result["mean_ap"]:.6f} | {result["mean_rps"]:.6f} |' for family, result in combined['aggregate'][task].items()]
    lines += ['', '## Matched AQI comparisons', '', 'RPS delta is with minus without; negative favors AQI.', '',
              '| Site | Task | Family | Paired delta AP (95% CI) | Paired delta RPS (95% CI) | Positives |', '|---|---|---|---|---|---|']
    for site, sd in combined['per_site'].items():
        for task, results in sd['aqi_ablation'].items():
            for family, result in results.items():
                if result.get('status') in ('not_estimable', 'not_applicable'):
                    continue
                lines.append(f'| {site} | {task} | {family} | {_interval(result["delta_ap"], result["delta_ap_ci"])} | {_interval(result["delta_rps"], result["delta_rps_ci"])} | {result["pos_count"]} |')
    for status in ('not_estimable', 'not_applicable'):
        for row in combined['aggregate']['aqi_' + status]:
            lines.append(f'- AQI {status.replace("_", " ")}: {row["site"]}/{row["task"]}/{row["family"]} — {row["reason"]}')
    lines += ['', '## Operational verdict', '', combined['aggregate']['predictability_verdict'], '', combined['aggregate']['aqi_verdict']]
    (out_dir / 'model_comparison_v2.json').write_text(json.dumps(combined, indent=2, allow_nan=False), encoding='utf-8')
    (out_dir / 'model_comparison_v2.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_plots(combined, art_dir=out_dir)
    return combined
