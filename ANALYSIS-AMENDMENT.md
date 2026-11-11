# CalFog analysis amendment (frozen before the corrected rerun)

Status: written after a first complete set of results had been produced and inspected, and before any corrected result existed. The git commit time of this file is the registration time. This is a disclosed amendment, not a pristine pre-registration.

## Why
A correctness audit of commit 14d2031c found that the first results were computed with measured air-quality values read one hour too late (EPA stamps each hourly value with the hour sampling began), non-converged logistic-regression fits, temporal-CNN tuning that overlapped two NOx test periods, a monitor-ID sensitivity that added NOx and monitor indicators together, and feature importances that included unused boosting rounds. All models and analyses are rerun once after the fixes.

## Unchanged
- Event: reported visibility at or below 1 statute mile (hourly minimum category <= 4), fixed PST.
- Tasks: nowcast; next-morning lead (issued 18:00 PST).
- Primary models: ordinal logistic regression, XGBoost, EBM. Other families are secondary.
- Primary estimand: for each airport, AP with NOx minus AP weather-only on identical held-out rows; unweighted mean over the five airports. Three chronological folds, same model seeds in both arms. The primary fit uses the single registered seed; ten training seeds are reported next to it.
- NOx eligibility: at least 20 of the 24 hourly samples completed by the issue time.
- Test families: six primary tests (3 models x 2 tasks), Holm; the six near-saturation tests and the six lag-aware tests are separate Holm families; all other comparisons are pointwise and exploratory.
- Interval and p-value convention: centred bootstrap interval and two-sided p as in the first analysis; simultaneous interval at 1 - 0.05/6.

## Changed
1. Draws: 50,000 for the air-quality tests, with Monte-Carlo intervals for p, Holm p and the simultaneous upper bound. A decision whose Monte-Carlo band straddles its threshold is reported as MC-unresolved. Family pairs: 10,000.
2. Dependence: the joint calendar-day block bootstrap remains the registered analysis. A second table resamples whole fog years (July-June), 10,000 draws. A result that differs between the two is reported as block-sensitive.
3. Effect size: no practical-significance verdict. Each test reports its estimate, Holm p and the simultaneous upper bound. The +0.02 AP benchmark registered before the first analysis is shown as a labelled column; it is owner-selected and has no literature or cost basis. No preferred threshold will be chosen after the corrected results are seen.
4. Monitor-ID sensitivity: both arms contain monitor indicators; only the with-arm adds NOx.
5. Added sensitivity arms (exploratory): NOx for models that also see the last visibility report; PM2.5 delayed by 24 hours; PM2.5 from FEM methods only.
6. Labels: source quality codes and provenance-matched remarks are applied if fog-hour counts change by less than 1% at every airport and era; otherwise the owner decides before any training.
7. Long-term analysis: ERA5-only winter weather is primary, the mixed-product archive is a comparison; inference uses a dependent wild bootstrap over calendar years with block-length sensitivities, 10,000 draws; the pooled site-demeaned rank trend is removed; two reporting-cadence robustness runs are added; intervals hold the first-stage weather model fixed and say so.
8. Descriptive additions: per-airport estimates, change in misclassification loss at FN:FP ratios 1, 3, 5, 10, 20, change in Brier skill, selection tables.

## Interpretation limits
The short-term comparisons measure incremental retrospective predictive skill for the specified models, inputs and eligible cases. They do not test or confirm an aerosol mechanism or Gray et al.'s timescale separation. Rerunning on the same archive is a correction, not independent validation.
