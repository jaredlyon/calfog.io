## Research status and report versions

**V1 is the historical class-project report, not the current validated scientific account. V2 is an
experimental rebuild under review: NOT publication-ready, NOT production-ready, and NO-GO for autonomous
public fog alerts.** The V2 code uses fixed PST (`Etc/GMT+8`), GHCNh visibility labels translated to 18
ordinal categories, and labels an hour as foggy when any valid visibility report in that hour reports fog.
The committed results in [models/comparison_v2](models/comparison_v2/model_comparison_v2.md) predate the measured
air-quality study design and will be replaced by its combined rerun. The archived pre-fix V2 numbers are superseded,
not validated replacements for V1.

- Historical V1: [class report PDF](report.pdf) and [report source](report.tex).
- Pre-fix V2 archive: [handoff](report_files/v2_prefix/HANDOFF.md),
  [Bakersfield proof of concept](report_files/v2_prefix/poc/poc_report.md) and
  [five-airport study](report_files/v2_prefix/full/report_full.md).
- Operational outputs: [schema-compatible retrain](ops/retrain/report_retrain.md) and
  [shadow evaluation](ops/shadow/report_shadow.md), regenerated on the ordinal GHCNh target.
- [Data sources and attribution](#data-sources-and-attribution) apply separately from the software license.

The following V1 claims are superseded as publication conclusions and are **not re-established by V2**:

1. AQI universally harms prediction or adds no predictive value. V1 did include a matched-row control;
   evaluation defects and missing uncertainty prevent a robust directional conclusion.
2. Stacking is the validated best model or best path forward. The archived V2 studies did not evaluate
   that ensemble; the reworked ensemble's current results are point estimates in the V2 comparison.
3. The reported CNN recall near 97.2% is a validated operational detection rate; V2 uses different tasks,
   cohorts and thresholds, so its scores are not a controlled V1-to-V2 performance delta.
4. High ROC-AUC establishes good probability calibration. Discrimination and calibration are different.
5. Any implication that a PyTorch conversion preserved the original results. V2 redesigns the CNN,
   features and evaluation protocol; adding reworked counterparts does not establish a faithful
   reproduction of the original results.

The historical materials are retained unchanged for provenance. Neither version establishes readiness for
public warning deployment. Distribution shift and prospective issue-time validation remain unresolved.

## Dependent variable

V2 models the **reported airport visibility category**, not uncensored physical visibility.
NOAA GHCNh observations for KMAE, KFAT, KVIS, KHJO and KBFL use one date-free
source/report-profile lookup to recover reportable statute miles; contradictory raw reports
or undecidable encodings remain ambiguous. The common 18-category scale is `<1/4`, `1/4`,
`1/2`, `3/4`, `1`, `1 1/4`, `1 1/2`, `1 3/4`, `2`, `2 1/2`, `3`, `4`, `5`, `6`, `7`,
`8`, `9`, `10+` SM. The low bin is [0, 402.336 m); the high bin is right-censored
at 10 SM. Identified statute miles convert with **1 SM = 1609.344 m**.

Hourly labels take the minimum category bounds in fixed UTC−8 (no daylight saving).
An unresolved hourly category is excluded from model targets. Fog means reported
visibility **≤1 SM** (categories 0–4). Every model predicts a category probability
distribution; visibility summaries and P(fog) derive from that same distribution.
Build labels with `python datasets/scripts/visibility_ghcnh.py --cache <dir>`;
observations retain the full record, and hourly model labels start at 1980-01-01.

## Repository layout (V2)

- `models/<family>/*_v2.py`: reworked model scripts alongside the unchanged originals, including
  logistic regression, calibrated random forest, XGBoost, focal-loss XGBoost, temporal CNN, ensemble,
  persistence/Markov, analog ensemble, threshold-ordinal XGBoost, interval-censored XGBoost AFT,
  Explainable Boosting Machine and pooled five-airport XGBoost.
- `models/_shared/`: sectioned data, labels, splits, windows, evaluation, tuning, family, study and
  reporting modules, with shared tests in `models/_shared/tests/`.
- `models/run_all_v2.py`: runs all model families with shared CNN tuning and writes comparisons to
  `models/comparison_v2/`. `--jobs N` runs (family, airport) shards in parallel worker processes with the same
  seeds and outputs as the serial run.
- `models/seed_replicates_v2.py`: refits the primary NOx ablation for XGBoost and EBM under nine extra model seeds.
- `models/significance_v2.py`: pooled-airport paired air-quality tests (Holm over six primary NOx tests,
  a +0.02 AP equivalence margin, training-seed sensitivity) and family-versus-family significance.
- `eda/long_term_trend/`: weather-adjusted winter fog-day trend, instrument steps, and year-versus-NOx attribution.
- `eda/k-means clustering/*_v2.py`: reworked k-means analysis alongside its original.
- `ops/`: retraining, serving bundles and passive shadow evaluation.
- `report_files/v2_prefix/`: superseded pre-fix PoC/full-study outputs and handoff, retained for provenance.

Predictor sets are recorded in metrics and displayed in family reports and comparisons:
- **weather:** logistic regression, random forest, XGBoost, focal-loss XGBoost, temporal CNN, ensemble,
  analog ensemble, threshold-ordinal XGBoost, XGBoost AFT and EBM retain the existing weather/calendar channels.
- **observed visibility history:** persistence/Markov uses the latest known category from issue−3h
  through issue−1h; the issue hour is never available as a predictor.
- **weather + observed visibility (own + 4 neighbors), pooled sites:** pooled XGBoost uses weather,
  airport identity and causal own/neighbor category lags; preprocessing is fitted once on pooled training
  windows, all sites obey the target site's 24-hour fold embargo, and test rows are target-site only.
  Its per-site matched AQI ablation is not applicable.

Random forest fits temperature/prior-mixture category calibration on validation rows only; fits without
validation retain the raw PMF. The ensemble stacks raw RF cumulative PMFs but blends calibrated RF predictions.

Run from the repository root using the environment in `requirements-v2.txt` (the root
`requirements.txt` remains the historical V1 environment):

```bash
python -m pip install -r requirements-v2.txt
python models/run_all_v2.py --jobs 8
python models/seed_replicates_v2.py --jobs 8
python models/significance_v2.py --n-boot 1000
python eda/long_term_trend/long_term_trend_v2.py
python models/xgboost/xgboost_fog_prediction_v2.py --sites location_10
python -m pytest models/_shared/tests ops "eda/k-means clustering" eda/long_term_trend datasets/scripts -q
```

Family runs write `<family>_metrics_v2.json`, `<family>_results_v2.txt`,
`<family>_artifacts_v2/` and git-ignored `<family>_predictions_v2/` beside their scripts.
Predictions retain issue time, fold, category PMFs and evaluation-only dew-point depression/wind.
`--reuse` keeps only families whose saved metrics have the current air-quality nesting and predictions;
others are retrained. The temporal CNN and every XGBoost family train on the GPU when CUDA is available;
GPU and CPU XGBoost boosters differ slightly. **Numbers are pending the combined rerun.** The public-alert
NO-GO stands regardless.

The air-quality study uses measured EPA hourly **NOx (primary)** and **PM2.5 (secondary)**;
CAMS channels remain appendix comparisons. `datasets/scripts/airdata_epa.py` builds fixed-PST
airport series in `datasets/aq_measured/` using monthly priority-monitor splices and records
monitor switches and acquisition provenance. Each ablation requires at least 20 of 24
nonmissing hours per air-quality channel; training medians impute the remainder, and both arms
use identical rows. Logistic regression, XGBoost and EBM also test NOx with monitor-ID indicators.
Strata (dew-point depression <4.2 °C and wind <7.2 km/h) are evaluation-only, never features.
Because training randomness can move these small differences, XGBoost and EBM are refit under ten model seeds
for the primary NOx tests; the significance report shows the seed spread and how often each verdict repeats.
Logistic regression is deterministic.

---

### This repository began as a class project for UCI's graduate 210P (Statistical Methods I) course and was turned in on March 16th, 2026.

Development will continue as I attempt to create a production-ready predictive model, which will then be linked to https://www.calfog.io for public use.

The original (V1) class report is `report.pdf`, kept unchanged as a historical record.

README is WIP.

## Data sources and attribution

The repository's MIT license does not relicense upstream data. Gridded weather and CAMS air-quality data by
[Open-Meteo](https://open-meteo.com/), under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Place this linked credit next to every Open-Meteo-derived table, figure or released data
supplement, with a link to this section. The historical PDF is retained as an unmodified class artifact;
its attribution has not been retroactively rewritten. A new manuscript or standalone figure must carry
its own credits rather than relying on a repository link alone.

### Weather and atmospheric composition

- **Open-Meteo:** the archive weather and air-quality APIs supply the historical feature channels;
  Forecast and Historical Forecast supply the shadow/proxy inputs. [Data license](https://open-meteo.com/en/licence)
  requires credit, a license link and identification of modifications. Cite Zippenfenig, P. (2023),
  *Open-Meteo.com Weather API*, https://doi.org/10.5281/zenodo.7970649.
  [API access terms](https://open-meteo.com/en/terms) are separate from the data license: the free service
  is non-commercial; commercial service use requires the appropriate access arrangement.
- **Copernicus/ERA5:** Contains modified Copernicus Climate Change Service information 2026.
  The current [ERA5 single-levels catalogue](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)
  uses CC BY 4.0. Catalogue reference: Copernicus Climate Change Service (2023), *ERA5 hourly data on
  single levels from 1940 to present*, https://doi.org/10.24381/cds.adbb2d47 (catalogue consulted
  23 October 2026). This is a source reference, not proof of the precise upstream product/version in
  these exports: the requests do not pin a weather model, and Open-Meteo's archive can combine products.
  The committed `era5_archive` name alone does not establish ERA5 versus ERA5-Land, ERA5T or another
  archive source. Product/version-level provenance remains unresolved and must be established for a
  reproducible rerun; this notice does not repair that gap.
- **CAMS:** Contains modified Copernicus Atmosphere Monitoring Service information 2026.
  The pollutant channels come through Open-Meteo's Air-Quality API, not EPA ambient monitors and not a
  computed EPA AQI index. [Open-Meteo's air-quality attribution instructions](https://open-meteo.com/en/docs/air-quality-api)
  request credit to CAMS ENSEMBLE and Open-Meteo. Credit is given here to CAMS and its ENSEMBLE data
  providers without claiming that the European ENSEMBLE supplied the California data. For the global
  product, the current [CAMS global atmospheric composition forecasts catalogue](https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts?tab=overview)
  uses CC BY 4.0. Source reference: Copernicus Atmosphere Monitoring Service (2021), *CAMS global
  atmospheric composition forecasts*, https://doi.org/10.24381/04a0b097 (catalogue consulted
  23 October 2026). No global reanalysis or European-product lineage is claimed; exact domain/product
  and historical version must be recorded from acquisition evidence for the rerun.

Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus
information or data it contains. The CDS/ADS moved to CC BY 4.0 on 2 July 2025; the older
[Copernicus license](https://apps.ecmwf.int/datasets/licences/copernicus/) also specifies source,
modification and responsibility notices. Historical observation dates are not data acquisition dates.

### Visibility observations

- **NOAA NCEI ISD / Global Hourly:** the V1 and pre-fix V2 visibility labels are manual NCEI exports for Madera,
  Fresno, Visalia, Hanford and Bakersfield. Source citation: NOAA National Centers for Environmental
  Information (2001), *Global Surface Hourly*, five-airport visibility subset.
  [Product metadata and citation](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC00532).
  NOAA-produced data are public domain in the United States under the
  [NCEI Open Data Policy](https://www.ncei.noaa.gov/sites/default/files/2023-12/NCEI%20PD-10-2-02%20-%20Open%20Data%20Policy%20Signed.pdf);
  do not infer a worldwide CC0 license for this legacy product. The V1 report records access to the
  NCEI website on 16 March 2026; the exact export retrieval dates, query IDs and response hashes were
  not preserved in the committed V1 acquisition record.
- **Iowa Environmental Mesonet (IEM), Iowa State University:** [ASOS/METAR archive service](https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py)
  supplies retrospective shadow/retrain-proxy visibility reports. IEM's
  [official policy](https://mesonet.agron.iastate.edu/disclaimer.php) says its materials are public
  domain and attribution is appreciated; no CC license is asserted here. Retrieval parameters,
  timestamps and hashes, where saved, are in the V2 fetch manifests. IEM supplies no accuracy warranty.
- **NOAA/NWS AviationWeather Data API:** [METAR observations](https://aviationweather.gov/data/api/)
  supply recent forward labels and passive snapshots; [NWS reuse policy](https://www.weather.gov/disclaimer).
  These are independent research transformations, not official NOAA/NWS forecasts or endorsed warnings.
- **NOAA NCEI GHCNh (current V2 label source):** Global Historical Climatology Network-Hourly
  period-of-record PSV files for stations USW00093242 (KMAE), USW00093193 (KFAT), USW00093144 (KVIS),
  USW00053119 (KHJO) and USW00023155 (KBFL), retrieved 23 October 2026 and decoded by
  `datasets/scripts/visibility_ghcnh.py`. Its [official metadata](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC01688)
  dedicates the data worldwide under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
  Cite Menne, M. J., et al. (2023), *Global Historical Climatology Network-Hourly (GHCNh)*, NOAA NCEI,
  https://doi.org/10.25921/jp3d-3v19. Do not substitute the GHCN-Daily citation.

### Measured air quality

- **U.S. Environmental Protection Agency (EPA), AirData / Air Quality System (AQS):**
  [pre-generated hourly data files](https://aqs.epa.gov/aqsweb/airdata/download_files.html)
  supply NOx (parameter 42603, `NONOxNOy`) and hourly PM2.5 (88101 and 88502) observations
  for priority monitors near the five airports. Cite U.S. Environmental Protection Agency,
  *AirData: Air Quality Data Collected at Outdoor Monitors Across the US*,
  https://www.epa.gov/outdoor-air-quality-data, with the actual retrieval date from
  `datasets/aq_measured/aq_manifest.json`. The acquisition code excludes superseded PM2.5
  methods 236/238, converts GMT to fixed PST, and splices monitor-months without adjustment.
  These transformed observations are not an EPA AQI index or an EPA-endorsed forecast.

### Modifications and availability

CalFog joins gridded weather/pollutant channels with airport visibility, filters invalid/missing target
reports, aggregates labels to hourly/daily tasks, constructs 24-hour feature windows, performs train-only
preprocessing, and produces model-derived scores, calibration summaries and figures. These are modified
and derived data, not untouched provider outputs. The saved V1 and V2 phases have different clock and
hourly-aggregation implementations; this notice does not claim they are already standardized or valid.
The approved rerun uses fixed PST (`Etc/GMT+8`) keys rather than DST-varying civil time.

Raw CSVs and some acquisition payloads are excluded from git. Committed manifests document only the
requests actually retained; a missing legacy retrieval record is not supplied by the model metrics.
Provider terms/catalogue pages above were checked on 23 October 2026. This is not a claim that all data
were downloaded on that date. Preserve the actual retrieval time, product/version, site/date subset,
query parameters, raw-response checksum, transformations and applicable terms with each new release.

