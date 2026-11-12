"""V1-style family text reports and result-driven cross-family comparisons."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from .data import SITES
from .evaluation import COST_GRID

AQ_SET_LABELS = {'nox': 'NOx (primary)', 'pm25': 'PM2.5 (secondary)',
                 'cams': 'CAMS (appendix)', 'nox_monitor': 'NOx conditional on monitor ID (sensitivity)'}


def aggregate(metrics):
    sites = metrics['per_site']
    out = {'aqi_not_estimable': [], 'aqi_not_applicable': []}
    notes, resolved = [], []
    for task in ('nowcast', 'lead_time'):
        families = sorted({family for sd in sites.values() for family, result in sd['tasks'][task].items()
                           if result.get('status') not in ('not_estimable', 'not_applicable')})
        out[task] = {}
        for family in families:
            values = {site: sd['tasks'][task][family]['ap'] for site, sd in sites.items()
                      if family in sd['tasks'][task] and sd['tasks'][task][family].get('status') not in ('not_estimable', 'not_applicable')}
            out[task][family] = {'mean_ap': float(np.mean(list(values.values()))), 'site_aps': values}
            rps_values = {site: sd['tasks'][task][family]['ordinal']['rps'] for site, sd in sites.items()
                          if family in sd['tasks'][task] and sd['tasks'][task][family].get('status') not in ('not_estimable', 'not_applicable')}
            out[task][family].update(mean_rps=float(np.mean(list(rps_values.values()))), site_rps=rps_values)
    out['site_task_summary'], out['aqi_site_summary'] = {}, {}
    for site, sd in sites.items():
        out['site_task_summary'][site], out['aqi_site_summary'][site] = {}, {}
        for task in ('nowcast', 'lead_time'):
            results = sd['tasks'][task]
            out['site_task_summary'][site][task] = {
                family: (dict(result) if result.get('status') in ('not_estimable', 'not_applicable') else
                         {key: result[key] for key in ('ap', 'ap_ci', 'ordinal')})
                for family, result in results.items()}
            learned = [family for family, result in results.items() if family != 'climatology'
                       and result.get('status') not in ('not_estimable', 'not_applicable')]
            if learned:
                best = max(learned, key=lambda family: results[family]['ap'])
                result, baseline = results[best], results['climatology']
                strength = 'interval-separated from climatology' if result['ap_ci'][0] > baseline['ap_ci'][1] else 'not interval-separated from climatology'
                notes.append(f'{site} {task}: strongest {best} AP={result["ap"]:.3f} ({strength})')
        for aq_set, tasks in sd['aqi_ablation'].items():
            out['aqi_site_summary'][site][aq_set] = {}
            for task, results in tasks.items():
                out['aqi_site_summary'][site][aq_set][task] = {
                    family: {key: result[key] for key in ('delta_ap', 'delta_ap_ci', 'delta_rps', 'delta_rps_ci', 'pos_count')}
                    for family, result in results.items()
                    if result.get('status') not in ('not_estimable', 'not_applicable')}
                for family, result in results.items():
                    if result.get('status') in ('not_estimable', 'not_applicable'):
                        out['aqi_' + result['status']].append({
                            'site': site, 'set': aq_set, 'task': task, 'family': family, 'reason': result['reason']})
                        continue
                    lo, hi = result['delta_ap_ci']
                    if lo > 0 or hi < 0:
                        resolved.append(f'{site}/{AQ_SET_LABELS[aq_set]}/{task}/{family} (delta={result["delta_ap"]:+.3f}, n+={result["pos_count"]})')
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
             f'Capped NLL (eps=1e-9, nats): {result["ordinal"]["nll"]:.6f}',
             f'Within one category: {result["ordinal"]["within_one"]:.6f}',
             f'category-representative meters log10 MAE (observed and median categories 1–16 only): {_meters_error(result["ordinal"])}',
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
             'Capped NLL = mean(-log(clip(p_true_category, 1e-9, 1))), in nats.',
             'For NOx conditional on monitor ID, both arms contain identical monitor indicators; only the with-arm adds NOx.',
             'Thresholds and isotonic calibration fit validation only; each held-out fold is evaluated once.', '']
    grouped = {}
    for site, sd in metrics['per_site'].items():
        lines += [f'LOCATION {site.removeprefix("location_")} ({SITES.get(site, site)})', '']
        for task in ('nowcast', 'lead_time'):
            lines += [task.upper(), '']
            main = sd['tasks'][task][family]
            blocks = []
            if main.get('status') in ('not_estimable', 'not_applicable'):
                lines += [f'WITHOUT AQI (full history): {main["status"].replace("_", " ")} — {main["reason"]}', '']
            else:
                blocks.append(('WITHOUT AQI (full history)', main, sd['tasks'][task]['climatology']['ap']))
            for aq_set, tasks in sd['aqi_ablation'].items():
                ablation = tasks[task][family]
                label = AQ_SET_LABELS[aq_set]
                if ablation.get('status') in ('not_estimable', 'not_applicable'):
                    status = ablation['status'].replace('_', ' ')
                    lines += [f'{label} AQI ablation: {status} — {ablation["reason"]}', '']
                    continue
                blocks += [(f'{label} WITH AQI (matched rows)', ablation['with'], ablation['with']['climatology_ap']),
                           (f'{label} WITHOUT AQI (matched rows)', ablation['without'], ablation['without']['climatology_ap'])]
                lines += [f'{label} paired AQI delta AP: {_interval(ablation["delta_ap"], ablation["delta_ap_ci"])}',
                          f'{label} paired AQI delta RPS (with minus without; negative favors AQI): {_interval(ablation["delta_rps"], ablation["delta_rps_ci"])}', '']
            for title, result, baseline in blocks:
                lines += _cohort_lines(title, result, baseline)
                grouped.setdefault((task, title), []).append((result, baseline))
    lines += ['AVERAGE METRICS', 'Arithmetic means across sites, not pooled observations; mean site-CI endpoints are descriptive, not an aggregate confidence interval.', '']
    for (task, title), rows in grouped.items():
        lines += [f'{task.upper()} - {title}']
        for key in ('ap', 'roc_auc', 'precision', 'recall', 'f1', 'cost_threshold'):
            lines.append(f'{key}: {np.mean([result[key] for result, _ in rows]):.6f}')
        for key in ('ap_ci', 'roc_auc_ci'):
            lo, hi = np.mean([result[key] for result, _ in rows], axis=0)
            lines.append(f'Mean site {key} endpoints (not aggregate CI): {lo:.6f}, {hi:.6f}')
        for key in ('rps', 'nll', 'within_one', 'median_category_mae'):
            label = 'Capped NLL (eps=1e-9, nats)' if key == 'nll' else key
            lines.append(f'{label}: {np.mean([result["ordinal"][key] for result, _ in rows]):.6f}')
        meter_rows = [result['ordinal'] for result, _ in rows if result['ordinal']['median_meters_n']]
        meter_count = sum(row['median_meters_n'] for row in meter_rows)
        meter_mae = sum(row['median_meters_mae_log10'] * row['median_meters_n'] for row in meter_rows) / meter_count if meter_count else None
        lines.append(f'Pooled category-representative meters log10 MAE: {_meters_error({"median_meters_mae_log10": meter_mae, "median_meters_n": meter_count})}')
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


def nox_value_rows(combined) -> list[dict]:
    rows = []
    for site, sd in combined['per_site'].items():
        for task, results in sd['aqi_ablation'].get('nox', {}).items():
            for family in ('logistic_regression', 'xgboost', 'ebm'):
                result = results.get(family)
                if result is None or result.get('status') in ('not_estimable', 'not_applicable'):
                    continue
                w, o = result['with'], result['without']
                pi = result['pos_count'] / result['n_test']
                common = {'scope': 'airport', 'site': site, 'task': task, 'family': family}
                for ratio in COST_GRID:
                    cost_without = o['cost_operating_points'][str(ratio)]['expected_cost_per_issue']
                    cost_with = w['cost_operating_points'][str(ratio)]['expected_cost_per_issue']
                    g = min(ratio * pi, 1 - pi)
                    rows.append({**common, 'kind': 'cost', 'ratio': ratio,
                                 'cost_without': cost_without, 'cost_with': cost_with,
                                 'delta_cost': cost_with - cost_without,
                                 'delta_value': (cost_without - cost_with) / g if g else None})
                variance = pi * (1 - pi)
                brier = {'n_test': result['n_test'], 'pos_count': result['pos_count'], 'prevalence': pi,
                         'ap_without': o['ap'], 'ap_with': w['ap'], 'delta_ap': w['ap'] - o['ap'],
                         'relative_ap_change': (w['ap'] - o['ap']) / o['ap'] if o['ap'] else None}
                for prefix, without, with_ in (
                        ('', o['brier'], w['brier']),
                        ('calibrated_', o['calibration']['brier_after'], w['calibration']['brier_after'])):
                    brier.update({prefix + 'brier_without': without, prefix + 'brier_with': with_,
                                  prefix + 'bss_without': 1 - without / variance if variance else None,
                                  prefix + 'bss_with': 1 - with_ / variance if variance else None,
                                  prefix + 'delta_bss': (without - with_) / variance if variance else None})
                rows.append({**common, 'kind': 'brier', **brier})
    groups = {}
    for row in rows:
        key = (row['family'], row['task'], row['kind'], row.get('ratio'))
        groups.setdefault(key, []).append(row)
    for (family, task, kind, ratio), group in groups.items():
        mean = {'scope': 'equal_airport_mean', 'site': None, 'family': family, 'task': task,
                'kind': kind, 'n_airports': len(group), 'n_airports_by_value': {}}
        if ratio is not None:
            mean['ratio'] = ratio
        for key in sorted(group[0].keys() - {'scope', 'site', 'family', 'task', 'kind', 'ratio'}):
            values = [row[key] for row in group if row[key] is not None]
            mean[key] = float(np.mean(values)) if values else None
            mean['n_airports_by_value'][key] = len(values)
        rows.append(mean)
    return rows


def _nox_value_lines(rows):
    lines = ['', '## NOx decision-loss and Brier descriptives (primary families)', '',
             'Descriptive, retrospective misclassification-loss sensitivities at validation-selected thresholds (calibrated probabilities). Cost ratios are assumptions, not measured user costs. Raw Brier uses uncalibrated probabilities; Brier skill is relative to sample climatology. No inference is attached and no value is a practical-significance threshold.']
    tables = (
        ('cost', ('ratio', 'cost_without', 'cost_with', 'delta_cost', 'delta_value')),
        ('brier', ('n_test', 'pos_count', 'prevalence', 'ap_without', 'ap_with', 'delta_ap', 'relative_ap_change',
                   'brier_without', 'brier_with', 'bss_without', 'bss_with', 'delta_bss',
                   'calibrated_brier_without', 'calibrated_brier_with', 'calibrated_bss_without',
                   'calibrated_bss_with', 'calibrated_delta_bss')))
    for scope, title in (('airport', 'Per-airport descriptives'), ('equal_airport_mean', 'Equal-airport means')):
        lines += ['', f'### {title}', '']
        for kind, columns in tables:
            headings = ['Site', 'Task', 'Family'] + list(columns)
            if scope == 'equal_airport_mean':
                headings += ['Airports used (by value column)']
            lines += ['| ' + ' | '.join(headings) + ' |', '|' + '---|' * len(headings)]
            for row in rows:
                if row['scope'] != scope or row['kind'] != kind:
                    continue
                cells = [row['site'] or 'equal-airport mean', row['task'], row['family']]
                cells += ['not estimable' if row[key] is None else f'{row[key]:.6f}' for key in columns]
                if scope == 'equal_airport_mean':
                    cells.append(', '.join(f'{key}={row["n_airports_by_value"][key]}' for key in columns if key != 'ratio'))
                lines.append('| ' + ' | '.join(cells) + ' |')
    return lines


def write_comparison(runs, out_dir):
    rosters = [set(metrics['per_site']) for metrics in runs.values()]
    if any(roster != rosters[0] for roster in rosters[1:]):
        raise ValueError('comparison families have unequal site rosters')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = {'families': list(runs),
                'predictor_sets': {'climatology': 'weather',
                                   **{family: metrics.get('predictor_set', 'weather') for family, metrics in runs.items()}},
                'per_site': {}}
    for family, metrics in runs.items():
        for site, sd in metrics['per_site'].items():
            target = combined['per_site'].setdefault(site, {'cohort': sd['cohort'], 'tasks': {}, 'aqi_ablation': {}, 'fold_audit': {}})
            target.setdefault('environments', {})[family] = sd.get('environment')
            for task in ('nowcast', 'lead_time'):
                target['tasks'].setdefault(task, {}).update(sd['tasks'][task])
                target['fold_audit'].setdefault(task, {})[family] = sd['fold_audit'][task]
            for aq_set, tasks in sd['aqi_ablation'].items():
                for task, results in tasks.items():
                    target['aqi_ablation'].setdefault(aq_set, {}).setdefault(task, {}).update(results)
    combined['aggregate'] = aggregate(combined)
    combined['nox_value_descriptives'] = nox_value_rows(combined)
    lines = ['# CalFog V2 model comparison', '',
             'Fixed PST; hourly minimum GHCNh category; fog = reported <=1 SM. Retrospective expanding-origin evaluation.',
             'AP/AUC/RPS confidence intervals are marginal per-model whole-day block intervals; only AQI deltas use paired draws.',
             'Category-representative meters log10 error excludes observed or median categories 0 and 17; it is not physical-visibility accuracy.',
             'Capped NLL (eps=1e-9, nats) = mean(-log(clip(p_true_category, 1e-9, 1))).', '',
             '| Site | Task | Family | Predictor set | AP (95% CI) | RPS (95% CI) | Capped NLL | Within one | category-representative meters log10 MAE (n) | ROC-AUC (95% CI) | Calibrated Brier / ECE |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for site, sd in combined['per_site'].items():
        for task, results in sd['tasks'].items():
            for family, result in results.items():
                if result.get('status') in ('not_estimable', 'not_applicable'):
                    lines.append(f'- Main {result["status"].replace("_", " ")}: {site}/{task}/{family} — {result["reason"]}')
                    continue
                cal = result['calibration']
                ordinal = result['ordinal']
                lines.append(f'| {site} | {task} | {family} | {combined["predictor_sets"][family]} | {_interval(result["ap"], result["ap_ci"])} | {_interval(ordinal["rps"], ordinal["rps_ci"])} | {ordinal["nll"]:.6f} | {ordinal["within_one"]:.6f} | {_meters_error(ordinal)} | {_interval(result["roc_auc"], result["roc_auc_ci"])} | {cal["brier_after"]:.6f} / {cal["ece_after"]:.6f} |')
    lines += ['', '## Validation-selected operating points', '',
              '| Site | Task | Family | FN:FP | Mean threshold | Precision | Recall | F1 | Alert rate | Cost/issue |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for site, sd in combined['per_site'].items():
        for task, results in sd['tasks'].items():
            for family, result in results.items():
                if result.get('status') in ('not_estimable', 'not_applicable'):
                    continue
                for ratio, point in result['cost_operating_points'].items():
                    lines.append(f'| {site} | {task} | {family} | {ratio}:1 | {point["mean_threshold"]:.6f} | {point["precision"]:.6f} | {point["recall"]:.6f} | {point["f1"]:.6f} | {point["alert_rate"]:.6f} | {point["expected_cost_per_issue"]:.6f} |')
    lines += ['', '## Cross-site mean scores', '', '| Task | Family | Mean AP | Mean RPS |', '|---|---|---|---|']
    for task in ('nowcast', 'lead_time'):
        lines += [f'| {task} | {family} | {result["mean_ap"]:.6f} | {result["mean_rps"]:.6f} |' for family, result in combined['aggregate'][task].items()]
    lines += ['', '## Matched AQI comparisons', '', 'RPS delta is with minus without; negative favors AQI.', '',
              '| Site | Air-quality set | Task | Family | Paired delta AP (95% CI) | Paired delta RPS (95% CI) | Positives |', '|---|---|---|---|---|---|---|']
    for site, sd in combined['per_site'].items():
        for aq_set, tasks in sd['aqi_ablation'].items():
            for task, results in tasks.items():
                for family, result in results.items():
                    if result.get('status') in ('not_estimable', 'not_applicable'):
                        continue
                    lines.append(f'| {site} | {AQ_SET_LABELS[aq_set]} | {task} | {family} | {_interval(result["delta_ap"], result["delta_ap_ci"])} | {_interval(result["delta_rps"], result["delta_rps_ci"])} | {result["pos_count"]} |')
    for status in ('not_estimable', 'not_applicable'):
        for row in combined['aggregate']['aqi_' + status]:
            lines.append(f'- AQI {status.replace("_", " ")}: {row["site"]}/{AQ_SET_LABELS[row["set"]]}/{row["task"]}/{row["family"]} — {row["reason"]}')
    if any('nox_monitor' in sd['aqi_ablation'] for sd in combined['per_site'].values()):
        lines += ['', 'NOx conditional on monitor ID (sensitivity): both arms contain identical weather/calendar inputs and monitor-ID indicators; only the with-arm adds NOx.']
    lines += _nox_value_lines(combined['nox_value_descriptives'])
    lines += ['', '## Operational verdict', '', combined['aggregate']['predictability_verdict'], '', combined['aggregate']['aqi_verdict']]
    (out_dir / 'model_comparison_v2.json').write_text(json.dumps(combined, indent=2, allow_nan=False), encoding='utf-8')
    (out_dir / 'model_comparison_v2.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_plots(combined, art_dir=out_dir)
    return combined
