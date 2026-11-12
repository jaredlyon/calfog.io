# Long-term weather-adjusted fog study

Fixed PST; Nov–Mar winter labeled by November year. Fog = any known hour in categories 0–4.
Every month needs 90% known hourly labels. Diagnostic LR/EBM use realized 00–23 weather, calendar phase and site only; leave one whole winter out across all sites.
Expected and observed use the same days with ≥18 known hours. y = log((observed+0.5)/(expected+0.5)).
Inference: 2,000 joint winter-block draws by default; centered 95% CIs (estimate ± 95th percentile of |draw − estimate|) and the matching two-sided bootstrap p. Singular resamples are excluded and valid draw counts are reported.
NOx requires 65% hourly coverage in each of five months; attribution models share rows. Gaussian OLS AIC counts the variance parameter.
These are diagnostic associations, not causal attribution or operational forecast skill. Gray comparison uses observed−expected fog days per 10 ppb.

## logistic_regression

### Panel and attribution

```json
{
  "panel": {
    "n_rows": 156,
    "n_winters": 46,
    "status": "ok",
    "coefficients": {
      "intercept_location_10": {
        "estimate": 0.08966863605050339,
        "ci95": [
          -0.11512412619111388,
          0.29446139829212065
        ],
        "p_two_sided": 0.4002998500749625
      },
      "intercept_location_6": {
        "estimate": 0.039825399545544364,
        "ci95": [
          -0.04335405411587733,
          0.12300485320696605
        ],
        "p_two_sided": 0.3428285857071464
      },
      "intercept_location_7": {
        "estimate": 0.009955768579164796,
        "ci95": [
          -0.16170755435194334,
          0.18161909151027295
        ],
        "p_two_sided": 0.9135432283858071
      },
      "intercept_location_8": {
        "estimate": 0.05305872568712441,
        "ci95": [
          -0.04939747984240523,
          0.15551493121665405
        ],
        "p_two_sided": 0.3028485757121439
      },
      "intercept_location_9": {
        "estimate": 0.051671414344998744,
        "ci95": [
          -0.08808814217129285,
          0.19143097086129035
        ],
        "p_two_sided": 0.48425787106446777
      },
      "year": {
        "estimate": -0.009541075997033958,
        "ci95": [
          -0.018090534174172317,
          -0.0009916178198955995
        ],
        "p_two_sided": 0.031484257871064465
      },
      "step_location_7": {
        "estimate": -0.18182310023429926,
        "ci95": [
          -0.3852824399382754,
          0.021636239469676827
        ],
        "p_two_sided": 0.07446276861569215
      },
      "step_location_10": {
        "estimate": -0.37826981774976925,
        "ci95": [
          -0.6639900052103929,
          -0.09254963028914559
        ],
        "p_two_sided": 0.01199400299850075
      }
    },
    "r_squared": 0.3737546625078787,
    "aic": 43.6556602084304,
    "valid_draws": 2000,
    "n_boot": 2000,
    "omitted_absorbed_steps": [
      "step_location_8"
    ]
  },
  "post_asos_1996": {
    "n_rows": 122,
    "n_winters": 30,
    "status": "ok",
    "coefficients": {
      "intercept_location_10": {
        "estimate": -0.34324422711513536,
        "ci95": [
          -0.47701440245045756,
          -0.20947405177981315
        ],
        "p_two_sided": 0.0004997501249375312
      },
      "intercept_location_6": {
        "estimate": -0.00872354716241754,
        "ci95": [
          -0.08865993237657756,
          0.07121283805174249
        ],
        "p_two_sided": 0.8415792103948025
      },
      "intercept_location_7": {
        "estimate": -0.22982407066608845,
        "ci95": [
          -0.3331395518648705,
          -0.12650858946730642
        ],
        "p_two_sided": 0.0004997501249375312
      },
      "intercept_location_8": {
        "estimate": 0.0552052647653244,
        "ci95": [
          -0.031119120972538127,
          0.14152965050318692
        ],
        "p_two_sided": 0.20789605197401298
      },
      "intercept_location_9": {
        "estimate": 0.0021999929035786017,
        "ci95": [
          -0.11667583688220406,
          0.12107582268936126
        ],
        "p_two_sided": 0.9735132433783108
      },
      "year": {
        "estimate": -0.011397556398117193,
        "ci95": [
          -0.02160947373992552,
          -0.0011856390563088643
        ],
        "p_two_sided": 0.028985507246376812
      }
    },
    "r_squared": 0.25223375085482314,
    "aic": 47.11289638819554,
    "valid_draws": 2000,
    "n_boot": 2000,
    "omitted_absorbed_steps": [
      "step_location_7",
      "step_location_10",
      "step_location_8"
    ]
  },
  "kvis_1992_step": {
    "n_rows": 156,
    "n_winters": 46,
    "status": "ok",
    "coefficients": {
      "intercept_location_10": {
        "estimate": 0.08966863605050339,
        "ci95": [
          -0.11512412619111388,
          0.29446139829212065
        ],
        "p_two_sided": 0.4002998500749625
      },
      "intercept_location_6": {
        "estimate": 0.039825399545544364,
        "ci95": [
          -0.04335405411587733,
          0.12300485320696605
        ],
        "p_two_sided": 0.3428285857071464
      },
      "intercept_location_7": {
        "estimate": 0.009955768579164796,
        "ci95": [
          -0.16170755435194334,
          0.18161909151027295
        ],
        "p_two_sided": 0.9135432283858071
      },
      "intercept_location_8": {
        "estimate": 0.05305872568712441,
        "ci95": [
          -0.04939747984240523,
          0.15551493121665405
        ],
        "p_two_sided": 0.3028485757121439
      },
      "intercept_location_9": {
        "estimate": 0.051671414344998744,
        "ci95": [
          -0.08808814217129285,
          0.19143097086129035
        ],
        "p_two_sided": 0.48425787106446777
      },
      "year": {
        "estimate": -0.009541075997033958,
        "ci95": [
          -0.018090534174172317,
          -0.0009916178198955995
        ],
        "p_two_sided": 0.031484257871064465
      },
      "step_location_7": {
        "estimate": -0.18182310023429926,
        "ci95": [
          -0.3852824399382754,
          0.021636239469676827
        ],
        "p_two_sided": 0.07446276861569215
      },
      "step_location_10": {
        "estimate": -0.37826981774976925,
        "ci95": [
          -0.6639900052103929,
          -0.09254963028914559
        ],
        "p_two_sided": 0.01199400299850075
      }
    },
    "r_squared": 0.3737546625078787,
    "aic": 43.6556602084304,
    "valid_draws": 2000,
    "n_boot": 2000,
    "omitted_absorbed_steps": [
      "step_location_8"
    ]
  },
  "per_site_trends": {
    "location_10": {
      "status": "ok",
      "n": 45,
      "slope": -0.02091865149180428,
      "ci95": [
        -0.029526984628555665,
        -0.012654646228748095
      ],
      "kendall_tau": -0.47272727272727266,
      "mann_kendall_p": 4.691608676565578e-06
    },
    "location_6": {
      "status": "ok",
      "n": 23,
      "slope": -0.009082041481730004,
      "ci95": [
        -0.023127340267070008,
        0.004585375514132194
      ],
      "kendall_tau": -0.19367588932806323,
      "mann_kendall_p": 0.20739075829593537
    },
    "location_7": {
      "status": "ok",
      "n": 46,
      "slope": -0.016615310454233926,
      "ci95": [
        -0.02110894033700797,
        -0.01163219168327621
      ],
      "kendall_tau": -0.6057971014492755,
      "mann_kendall_p": 2.9111556369917754e-09
    },
    "location_8": {
      "status": "ok",
      "n": 21,
      "slope": -0.009712389580044244,
      "ci95": [
        -0.026891103726228133,
        -0.00052395578315007
      ],
      "kendall_tau": -0.3238095238095238,
      "mann_kendall_p": 0.04196934455920998
    },
    "location_9": {
      "status": "ok",
      "n": 21,
      "slope": -0.009071646984344251,
      "ci95": [
        -0.02679467657410141,
        0.00972101991997687
      ],
      "kendall_tau": -0.14285714285714285,
      "mann_kendall_p": 0.3857814338605447
    }
  },
  "attribution": {
    "status": "ok",
    "common_rows": [
      {
        "site": "location_10",
        "winter": 1980,
        "nox_ppb": 66.83008526187577
      },
      {
        "site": "location_10",
        "winter": 1981,
        "nox_ppb": 51.96247766527695
      },
      {
        "site": "location_10",
        "winter": 1982,
        "nox_ppb": 58.81853983641321
      },
      {
        "site": "location_10",
        "winter": 1983,
        "nox_ppb": 94.40688383527966
      },
      {
        "site": "location_10",
        "winter": 1984,
        "nox_ppb": 88.78513429106808
      },
      {
        "site": "location_10",
        "winter": 1986,
        "nox_ppb": 67.95307193578265
      },
      {
        "site": "location_10",
        "winter": 1995,
        "nox_ppb": 27.98113763631005
      },
      {
        "site": "location_10",
        "winter": 1996,
        "nox_ppb": 27.220821114369503
      },
      {
        "site": "location_10",
        "winter": 1997,
        "nox_ppb": 27.778639618138424
      },
      {
        "site": "location_10",
        "winter": 1998,
        "nox_ppb": 26.066471449487555
      },
      {
        "site": "location_10",
        "winter": 1999,
        "nox_ppb": 29.59719789842382
      },
      {
        "site": "location_10",
        "winter": 2000,
        "nox_ppb": 27.857225600926732
      },
      {
        "site": "location_10",
        "winter": 2001,
        "nox_ppb": 24.413361805979257
      },
      {
        "site": "location_10",
        "winter": 2002,
        "nox_ppb": 25.98757225433526
      },
      {
        "site": "location_10",
        "winter": 2003,
        "nox_ppb": 19.488045427375972
      },
      {
        "site": "location_10",
        "winter": 2004,
        "nox_ppb": 20.178927680798004
      },
      {
        "site": "location_10",
        "winter": 2005,
        "nox_ppb": 36.09256494475963
      },
      {
        "site": "location_10",
        "winter": 2006,
        "nox_ppb": 60.11046680197158
      },
      {
        "site": "location_10",
        "winter": 2007,
        "nox_ppb": 49.10115606936416
      },
      {
        "site": "location_10",
        "winter": 2008,
        "nox_ppb": 48.327283372365336
      },
      {
        "site": "location_10",
        "winter": 2009,
        "nox_ppb": 40.22479721900348
      },
      {
        "site": "location_10",
        "winter": 2010,
        "nox_ppb": 35.77171717171717
      },
      {
        "site": "location_10",
        "winter": 2011,
        "nox_ppb": 43.43805436337625
      },
      {
        "site": "location_10",
        "winter": 2012,
        "nox_ppb": 39.18463768115942
      },
      {
        "site": "location_10",
        "winter": 2014,
        "nox_ppb": 28.295812144780136
      },
      {
        "site": "location_10",
        "winter": 2016,
        "nox_ppb": 26.14366279069768
      },
      {
        "site": "location_10",
        "winter": 2017,
        "nox_ppb": 33.18026086956522
      },
      {
        "site": "location_10",
        "winter": 2018,
        "nox_ppb": 25.448402091807086
      },
      {
        "site": "location_10",
        "winter": 2019,
        "nox_ppb": 25.766377063423107
      },
      {
        "site": "location_10",
        "winter": 2020,
        "nox_ppb": 25.22509417560127
      },
      {
        "site": "location_10",
        "winter": 2021,
        "nox_ppb": 28.093908629441625
      },
      {
        "site": "location_10",
        "winter": 2022,
        "nox_ppb": 22.242558746736293
      },
      {
        "site": "location_10",
        "winter": 2023,
        "nox_ppb": 22.252156411730883
      },
      {
        "site": "location_10",
        "winter": 2024,
        "nox_ppb": 23.85091119467747
      },
      {
        "site": "location_6",
        "winter": 1999,
        "nox_ppb": 33.35001480604087
      },
      {
        "site": "location_6",
        "winter": 2003,
        "nox_ppb": 23.976058394160585
      },
      {
        "site": "location_6",
        "winter": 2005,
        "nox_ppb": 22.198833819241983
      },
      {
        "site": "location_6",
        "winter": 2006,
        "nox_ppb": 22.977239568135396
      },
      {
        "site": "location_6",
        "winter": 2007,
        "nox_ppb": 17.957366589327147
      },
      {
        "site": "location_6",
        "winter": 2008,
        "nox_ppb": 16.594476744186046
      },
      {
        "site": "location_6",
        "winter": 2009,
        "nox_ppb": 14.618891537544696
      },
      {
        "site": "location_6",
        "winter": 2010,
        "nox_ppb": 13.314756174794173
      },
      {
        "site": "location_6",
        "winter": 2012,
        "nox_ppb": 13.681378865979381
      },
      {
        "site": "location_6",
        "winter": 2016,
        "nox_ppb": 8.127062609713283
      },
      {
        "site": "location_6",
        "winter": 2017,
        "nox_ppb": 11.727115666178623
      },
      {
        "site": "location_6",
        "winter": 2018,
        "nox_ppb": 9.38939349545854
      },
      {
        "site": "location_6",
        "winter": 2019,
        "nox_ppb": 8.808408759124088
      },
      {
        "site": "location_6",
        "winter": 2020,
        "nox_ppb": 10.132607416127133
      },
      {
        "site": "location_6",
        "winter": 2021,
        "nox_ppb": 9.464883239728053
      },
      {
        "site": "location_6",
        "winter": 2022,
        "nox_ppb": 8.349336870026525
      },
      {
        "site": "location_6",
        "winter": 2023,
        "nox_ppb": 7.4415177263404635
      },
      {
        "site": "location_6",
        "winter": 2024,
        "nox_ppb": 7.625554224086279
      },
      {
        "site": "location_6",
        "winter": 2025,
        "nox_ppb": 6.994775248102744
      },
      {
        "site": "location_7",
        "winter": 1980,
        "nox_ppb": 106.32058287795992
      },
      {
        "site": "location_7",
        "winter": 1981,
        "nox_ppb": 78.03897685749087
      },
      {
        "site": "location_7",
        "winter": 1982,
        "nox_ppb": 78.13308104698834
      },
      {
        "site": "location_7",
        "winter": 1983,
        "nox_ppb": 84.59177409453653
      },
      {
        "site": "location_7",
        "winter": 1984,
        "nox_ppb": 83.97800776196637
      },
      {
        "site": "location_7",
        "winter": 1985,
        "nox_ppb": 78.07370703004025
      },
      {
        "site": "location_7",
        "winter": 1986,
        "nox_ppb": 116.43059490084985
      },
      {
        "site": "location_7",
        "winter": 1994,
        "nox_ppb": 35.71683226183519
      },
      {
        "site": "location_7",
        "winter": 1995,
        "nox_ppb": 41.70106132075472
      },
      {
        "site": "location_7",
        "winter": 1996,
        "nox_ppb": 36.22445913461539
      },
      {
        "site": "location_7",
        "winter": 1997,
        "nox_ppb": 38.169645656945754
      },
      {
        "site": "location_7",
        "winter": 1998,
        "nox_ppb": 38.99794178182887
      },
      {
        "site": "location_7",
        "winter": 1999,
        "nox_ppb": 44.28068644560791
      },
      {
        "site": "location_7",
        "winter": 2000,
        "nox_ppb": 40.283958602846056
      },
      {
        "site": "location_7",
        "winter": 2001,
        "nox_ppb": 33.79604486422668
      },
      {
        "site": "location_7",
        "winter": 2002,
        "nox_ppb": 35.61832543443918
      },
      {
        "site": "location_7",
        "winter": 2003,
        "nox_ppb": 34.30892182505086
      },
      {
        "site": "location_7",
        "winter": 2004,
        "nox_ppb": 28.05842630651791
      },
      {
        "site": "location_7",
        "winter": 2005,
        "nox_ppb": 31.345708345708346
      },
      {
        "site": "location_7",
        "winter": 2006,
        "nox_ppb": 36.58463810930576
      },
      {
        "site": "location_7",
        "winter": 2007,
        "nox_ppb": 28.982584784601283
      },
      {
        "site": "location_7",
        "winter": 2008,
        "nox_ppb": 30.154962962962962
      },
      {
        "site": "location_7",
        "winter": 2009,
        "nox_ppb": 22.737110311750598
      },
      {
        "site": "location_7",
        "winter": 2010,
        "nox_ppb": 20.953890489913544
      },
      {
        "site": "location_7",
        "winter": 2012,
        "nox_ppb": 24.25542168674699
      },
      {
        "site": "location_7",
        "winter": 2014,
        "nox_ppb": 20.08777811642792
      },
      {
        "site": "location_7",
        "winter": 2015,
        "nox_ppb": 19.74296824368114
      },
      {
        "site": "location_7",
        "winter": 2016,
        "nox_ppb": 18.276494146642015
      },
      {
        "site": "location_7",
        "winter": 2017,
        "nox_ppb": 22.784728773584906
      },
      {
        "site": "location_7",
        "winter": 2018,
        "nox_ppb": 17.84309559939302
      },
      {
        "site": "location_7",
        "winter": 2019,
        "nox_ppb": 17.591957734076903
      },
      {
        "site": "location_7",
        "winter": 2020,
        "nox_ppb": 17.811683343143027
      },
      {
        "site": "location_7",
        "winter": 2021,
        "nox_ppb": 16.8846015037594
      },
      {
        "site": "location_7",
        "winter": 2022,
        "nox_ppb": 16.44575318141462
      },
      {
        "site": "location_7",
        "winter": 2023,
        "nox_ppb": 14.005227606461087
      },
      {
        "site": "location_7",
        "winter": 2024,
        "nox_ppb": 13.070996179841318
      },
      {
        "site": "location_8",
        "winter": 1994,
        "nox_ppb": 39.52385185185185
      },
      {
        "site": "location_8",
        "winter": 1995,
        "nox_ppb": 43.46763848396501
      },
      {
        "site": "location_8",
        "winter": 2006,
        "nox_ppb": 40.16413107080164
      },
      {
        "site": "location_8",
        "winter": 2007,
        "nox_ppb": 30.65805340223945
      },
      {
        "site": "location_8",
        "winter": 2008,
        "nox_ppb": 28.452078521939953
      },
      {
        "site": "location_8",
        "winter": 2009,
        "nox_ppb": 28.40252495596007
      },
      {
        "site": "location_8",
        "winter": 2012,
        "nox_ppb": 24.06021694517737
      },
      {
        "site": "location_8",
        "winter": 2013,
        "nox_ppb": 29.295645863570392
      },
      {
        "site": "location_8",
        "winter": 2014,
        "nox_ppb": 19.114678059200486
      },
      {
        "site": "location_8",
        "winter": 2015,
        "nox_ppb": 16.688927236123092
      },
      {
        "site": "location_8",
        "winter": 2016,
        "nox_ppb": 20.018935837245696
      },
      {
        "site": "location_8",
        "winter": 2018,
        "nox_ppb": 18.25922933499068
      },
      {
        "site": "location_8",
        "winter": 2019,
        "nox_ppb": 16.717759168351144
      },
      {
        "site": "location_8",
        "winter": 2020,
        "nox_ppb": 18.278865534120357
      },
      {
        "site": "location_8",
        "winter": 2022,
        "nox_ppb": 13.174992738890502
      },
      {
        "site": "location_8",
        "winter": 2024,
        "nox_ppb": 13.8213747099768
      },
      {
        "site": "location_9",
        "winter": 1999,
        "nox_ppb": 40.669696020056406
      },
      {
        "site": "location_9",
        "winter": 2004,
        "nox_ppb": 21.912107101280558
      },
      {
        "site": "location_9",
        "winter": 2005,
        "nox_ppb": 27.817204301075268
      },
      {
        "site": "location_9",
        "winter": 2006,
        "nox_ppb": 34.88733233979136
      },
      {
        "site": "location_9",
        "winter": 2007,
        "nox_ppb": 30.65805340223945
      },
      {
        "site": "location_9",
        "winter": 2008,
        "nox_ppb": 28.452078521939953
      },
      {
        "site": "location_9",
        "winter": 2009,
        "nox_ppb": 28.40252495596007
      },
      {
        "site": "location_9",
        "winter": 2011,
        "nox_ppb": 27.403261210410786
      },
      {
        "site": "location_9",
        "winter": 2012,
        "nox_ppb": 22.103939393939395
      },
      {
        "site": "location_9",
        "winter": 2013,
        "nox_ppb": 28.27719298245614
      },
      {
        "site": "location_9",
        "winter": 2015,
        "nox_ppb": 17.651522020725388
      },
      {
        "site": "location_9",
        "winter": 2016,
        "nox_ppb": 13.92640432562331
      },
      {
        "site": "location_9",
        "winter": 2017,
        "nox_ppb": 19.533046573716994
      },
      {
        "site": "location_9",
        "winter": 2018,
        "nox_ppb": 16.19676290463692
      },
      {
        "site": "location_9",
        "winter": 2019,
        "nox_ppb": 15.470949883449881
      },
      {
        "site": "location_9",
        "winter": 2020,
        "nox_ppb": 18.12390158172232
      },
      {
        "site": "location_9",
        "winter": 2021,
        "nox_ppb": 15.828238039673279
      },
      {
        "site": "location_9",
        "winter": 2022,
        "nox_ppb": 14.152518409425626
      },
      {
        "site": "location_9",
        "winter": 2024,
        "nox_ppb": 14.87244117647059
      }
    ],
    "models": {
      "year_only": {
        "n_rows": 124,
        "n_winters": 39,
        "status": "ok",
        "coefficients": {
          "intercept_location_10": {
            "estimate": 0.02279852440717158,
            "ci95": [
              -0.30311015950510245,
              0.34870720831944557
            ],
            "p_two_sided": 0.8904452226113057
          },
          "intercept_location_6": {
            "estimate": 0.022915722307850724,
            "ci95": [
              -0.059117462125689406,
              0.10494890674139085
            ],
            "p_two_sided": 0.5952976488244122
          },
          "intercept_location_7": {
            "estimate": -0.003990574587451512,
            "ci95": [
              -0.22990169791295897,
              0.22192054873805597
            ],
            "p_two_sided": 0.9724862431215607
          },
          "intercept_location_8": {
            "estimate": 0.0719276304704021,
            "ci95": [
              -0.030642799920893485,
              0.17449806086169767
            ],
            "p_two_sided": 0.15657828914457228
          },
          "intercept_location_9": {
            "estimate": 0.05728994256458199,
            "ci95": [
              -0.08912362622517932,
              0.20370351135434328
            ],
            "p_two_sided": 0.4427213606803402
          },
          "year": {
            "estimate": -0.00977893324779442,
            "ci95": [
              -0.019444028496989056,
              -0.00011383799859978304
            ],
            "p_two_sided": 0.048024012006003
          },
          "step_location_7": {
            "estimate": -0.1621728694489544,
            "ci95": [
              -0.40505072559923344,
              0.08070498670132464
            ],
            "p_two_sided": 0.19109554777388693
          },
          "step_location_10": {
            "estimate": -0.28140803207749043,
            "ci95": [
              -0.6600804003660341,
              0.09726433621105324
            ],
            "p_two_sided": 0.15057528764382191
          }
        },
        "r_squared": 0.33393563328678,
        "aic": 28.1270571698048,
        "valid_draws": 1998,
        "n_boot": 2000,
        "omitted_absorbed_steps": [
          "step_location_8"
        ]
      },
      "nox_only": {
        "n_rows": 124,
        "n_winters": 39,
        "status": "ok",
        "coefficients": {
          "intercept_location_10": {
            "estimate": 0.09138645497259493,
            "ci95": [
              -0.18138733374587743,
              0.36416024369106725
            ],
            "p_two_sided": 0.4847423711855928
          },
          "intercept_location_6": {
            "estimate": 0.044182015038429445,
            "ci95": [
              -0.05450310553232217,
              0.14286713560918107
            ],
            "p_two_sided": 0.36418209104552274
          },
          "intercept_location_7": {
            "estimate": -0.016763982087550536,
            "ci95": [
              -0.3089207473806805,
              0.2753927832055794
            ],
            "p_two_sided": 0.9094547273636818
          },
          "intercept_location_8": {
            "estimate": 0.060143882106318915,
            "ci95": [
              -0.03517339731736079,
              0.15546116152999861
            ],
            "p_two_sided": 0.2051025512756378
          },
          "intercept_location_9": {
            "estimate": 0.045840042750945886,
            "ci95": [
              -0.08860202563205394,
              0.1802821111339457
            ],
            "p_two_sided": 0.48774387193596797
          },
          "nox_ppb": {
            "estimate": 0.004702734462731847,
            "ci95": [
              -0.0006425117042860055,
              0.010047980629749698
            ],
            "p_two_sided": 0.0830415207603802
          },
          "step_location_7": {
            "estimate": -0.14520565640163569,
            "ci95": [
              -0.44262899338012307,
              0.1522176805768517
            ],
            "p_two_sided": 0.3021510755377689
          },
          "step_location_10": {
            "estimate": -0.36826730711979916,
            "ci95": [
              -0.6824245038362835,
              -0.05411011040331487
            ],
            "p_two_sided": 0.028514257128564282
          }
        },
        "r_squared": 0.3009901633855423,
        "aic": 34.11359141255499,
        "valid_draws": 1998,
        "n_boot": 2000,
        "omitted_absorbed_steps": [
          "step_location_8"
        ]
      },
      "year_plus_nox": {
        "n_rows": 124,
        "n_winters": 39,
        "status": "ok",
        "coefficients": {
          "intercept_location_10": {
            "estimate": 0.007698332600926153,
            "ci95": [
              -0.3139659590461904,
              0.32936262424804275
            ],
            "p_two_sided": 0.9629814907453726
          },
          "intercept_location_6": {
            "estimate": 0.03503388924800323,
            "ci95": [
              -0.05827914542689462,
              0.1283469239229011
            ],
            "p_two_sided": 0.4612306153076538
          },
          "intercept_location_7": {
            "estimate": -0.0367978478191822,
            "ci95": [
              -0.3043107030628917,
              0.23071500742452727
            ],
            "p_two_sided": 0.7743871935967984
          },
          "intercept_location_8": {
            "estimate": 0.07447073945310682,
            "ci95": [
              -0.029490852315512364,
              0.17843233122172603
            ],
            "p_two_sided": 0.15057528764382191
          },
          "intercept_location_9": {
            "estimate": 0.061110372918353195,
            "ci95": [
              -0.08572180028668305,
              0.20794254612338944
            ],
            "p_two_sided": 0.4042021010505253
          },
          "year": {
            "estimate": -0.008930768882022867,
            "ci95": [
              -0.019792307133298354,
              0.0019307693692526221
            ],
            "p_two_sided": 0.09804902451225612
          },
          "nox_ppb": {
            "estimate": 0.00101954165461597,
            "ci95": [
              -0.0035829149710779403,
              0.005621998280309881
            ],
            "p_two_sided": 0.6438219109554777
          },
          "step_location_7": {
            "estimate": -0.12676542666547688,
            "ci95": [
              -0.4020173150979609,
              0.14848646176700717
            ],
            "p_two_sided": 0.3646823411705853
          },
          "step_location_10": {
            "estimate": -0.26809185179936246,
            "ci95": [
              -0.656367791861491,
              0.120184088262766
            ],
            "p_two_sided": 0.17058529264632316
          }
        },
        "r_squared": 0.3348251154793812,
        "aic": 29.96135320908734,
        "valid_draws": 1998,
        "n_boot": 2000,
        "omitted_absorbed_steps": [
          "step_location_8"
        ]
      }
    },
    "year_slope_change_with_nox": 0.0008481643657715539,
    "gray_comparison": {
      "all": {
        "n_rows": 124,
        "n_winters": 39,
        "status": "ok",
        "coefficients": {
          "intercept_location_10": {
            "estimate": 4.103827547833351,
            "ci95": [
              -6.390213027857584,
              14.597868123524286
            ],
            "p_two_sided": 0.4117058529264632
          },
          "intercept_location_6": {
            "estimate": 1.8189500641029004,
            "ci95": [
              -2.6866781366369943,
              6.3245782648427955
            ],
            "p_two_sided": 0.41870935467733866
          },
          "intercept_location_7": {
            "estimate": 2.461281353973488,
            "ci95": [
              -9.192972282819778,
              14.115534990766754
            ],
            "p_two_sided": 0.639319659829915
          },
          "intercept_location_8": {
            "estimate": 3.110685653205688,
            "ci95": [
              -2.002255136365517,
              8.223626442776894
            ],
            "p_two_sided": 0.2271135567783892
          },
          "intercept_location_9": {
            "estimate": 2.2846859259646846,
            "ci95": [
              -5.336558760519628,
              9.905930612448996
            ],
            "p_two_sided": 0.5457728864432216
          },
          "nox_ppb": {
            "estimate": 0.18050964907509973,
            "ci95": [
              -0.03750620085517628,
              0.3985254990053757
            ],
            "p_two_sided": 0.0945472736368184
          },
          "step_location_7": {
            "estimate": -6.519023861256784,
            "ci95": [
              -19.136964163360467,
              6.098916440846898
            ],
            "p_two_sided": 0.26863431715857927
          },
          "step_location_10": {
            "estimate": -9.247440187951254,
            "ci95": [
              -20.44946030380403,
              1.9545799279015235
            ],
            "p_two_sided": 0.09904952476238119
          }
        },
        "r_squared": 0.24735319786723586,
        "aic": 930.5651868340573,
        "valid_draws": 1998,
        "n_boot": 2000,
        "omitted_absorbed_steps": [
          "step_location_8"
        ],
        "fog_days_per_10ppb": {
          "estimate": 1.8050964907509972,
          "ci95": [
            -0.3750620085517628,
            3.985254990053757
          ],
          "p_two_sided": 0.0945472736368184
        }
      },
      "near_saturation": {
        "n_rows": 19,
        "n_winters": 12,
        "status": "ok",
        "coefficients": {
          "intercept_location_10": {
            "estimate": 0.04073756563588774,
            "ci95": [
              -10.558325135106072,
              10.639800266377849
            ],
            "p_two_sided": 0.9899497487437185
          },
          "intercept_location_6": {
            "estimate": -1.0777772202676854,
            "ci95": [
              -16.83550994868694,
              14.679955508151568
            ],
            "p_two_sided": 0.742713567839196
          },
          "intercept_location_7": {
            "estimate": -0.08113923180763428,
            "ci95": [
              -23.772521609311802,
              23.610243145696536
            ],
            "p_two_sided": 0.9738693467336683
          },
          "intercept_location_8": {
            "estimate": -7.021417428879707,
            "ci95": [
              -20.498859277142625,
              6.456024419383209
            ],
            "p_two_sided": 0.07839195979899498
          },
          "intercept_location_9": {
            "estimate": -10.173776104897533,
            "ci95": [
              -21.85892700940451,
              1.5113747996094453
            ],
            "p_two_sided": 0.06834170854271357
          },
          "nox_ppb": {
            "estimate": 0.34511062757815103,
            "ci95": [
              -0.21897434239995806,
              0.9091955975562601
            ],
            "p_two_sided": 0.10653266331658291
          },
          "step_location_7": {
            "estimate": 4.212215997692065,
            "ci95": [
              -24.28791601586859,
              32.712348011252715
            ],
            "p_two_sided": 0.5557788944723618
          }
        },
        "r_squared": 0.8383838573141821,
        "aic": 128.30380338253832,
        "valid_draws": 994,
        "n_boot": 2000,
        "omitted_absorbed_steps": [
          "step_location_10",
          "step_location_8"
        ],
        "fog_days_per_10ppb": {
          "estimate": 3.4511062757815103,
          "ci95": [
            -2.1897434239995803,
            9.091955975562602
          ],
          "p_two_sided": 0.10653266331658291
        }
      }
    }
  }
}
```

### Winter observations and expectations

| Site | Winter | Observed (all days) | Observed (≥18 h) | Expected | Days |
|---|---:|---:|---:|---:|---:|
| location_10 | 1980 | 52 | 52 | 23.500 | 151 |
| location_10 | 1981 | 33 | 33 | 34.487 | 151 |
| location_10 | 1982 | 46 | 46 | 39.475 | 151 |
| location_10 | 1983 | 38 | 38 | 34.309 | 152 |
| location_10 | 1984 | 64 | 64 | 32.084 | 151 |
| location_10 | 1985 | 61 | 61 | 35.062 | 151 |
| location_10 | 1986 | 37 | 36 | 29.924 | 150 |
| location_10 | 1987 | 36 | 36 | 34.146 | 152 |
| location_10 | 1988 | 44 | 44 | 29.350 | 151 |
| location_10 | 1989 | 36 | 36 | 19.977 | 151 |
| location_10 | 1990 | 27 | 27 | 21.049 | 151 |
| location_10 | 1991 | 47 | 47 | 29.844 | 152 |
| location_10 | 1992 | 29 | 29 | 30.626 | 151 |
| location_10 | 1993 | 33 | 33 | 26.414 | 151 |
| location_10 | 1995 | 34 | 34 | 37.011 | 152 |
| location_10 | 1996 | 37 | 37 | 38.527 | 151 |
| location_10 | 1997 | 36 | 36 | 36.793 | 151 |
| location_10 | 1998 | 37 | 37 | 39.430 | 151 |
| location_10 | 1999 | 8 | 8 | 29.686 | 152 |
| location_10 | 2000 | 26 | 25 | 29.064 | 150 |
| location_10 | 2001 | 41 | 41 | 35.315 | 151 |
| location_10 | 2002 | 39 | 39 | 38.851 | 151 |
| location_10 | 2003 | 23 | 23 | 32.386 | 152 |
| location_10 | 2004 | 38 | 38 | 39.583 | 151 |
| location_10 | 2005 | 29 | 29 | 29.210 | 150 |
| location_10 | 2006 | 15 | 15 | 24.615 | 151 |
| location_10 | 2007 | 20 | 20 | 25.651 | 152 |
| location_10 | 2008 | 29 | 29 | 30.770 | 151 |
| location_10 | 2009 | 25 | 25 | 29.208 | 151 |
| location_10 | 2010 | 24 | 24 | 33.323 | 151 |
| location_10 | 2011 | 8 | 8 | 17.095 | 152 |
| location_10 | 2012 | 12 | 12 | 23.306 | 151 |
| location_10 | 2013 | 3 | 3 | 11.761 | 151 |
| location_10 | 2014 | 33 | 33 | 22.412 | 151 |
| location_10 | 2015 | 11 | 11 | 28.542 | 152 |
| location_10 | 2016 | 13 | 13 | 23.652 | 151 |
| location_10 | 2017 | 12 | 12 | 13.568 | 151 |
| location_10 | 2018 | 19 | 19 | 19.988 | 151 |
| location_10 | 2019 | 11 | 11 | 21.040 | 152 |
| location_10 | 2020 | 9 | 9 | 16.647 | 151 |
| location_10 | 2021 | 27 | 27 | 24.290 | 151 |
| location_10 | 2022 | 14 | 14 | 26.903 | 151 |
| location_10 | 2023 | 15 | 15 | 20.791 | 152 |
| location_10 | 2024 | 17 | 17 | 31.185 | 150 |
| location_10 | 2025 | 43 | 43 | 44.194 | 150 |
| location_6 | 1999 | 44 | 43 | 43.833 | 145 |
| location_6 | 2003 | 50 | 46 | 42.248 | 144 |
| location_6 | 2005 | 58 | 57 | 43.924 | 149 |
| location_6 | 2006 | 42 | 42 | 40.130 | 151 |
| location_6 | 2007 | 41 | 41 | 37.837 | 152 |
| location_6 | 2008 | 50 | 50 | 48.886 | 150 |
| location_6 | 2009 | 52 | 52 | 47.171 | 151 |
| location_6 | 2010 | 45 | 45 | 54.090 | 151 |
| location_6 | 2011 | 50 | 50 | 37.415 | 149 |
| location_6 | 2012 | 44 | 44 | 46.841 | 151 |
| location_6 | 2013 | 17 | 17 | 24.418 | 151 |
| location_6 | 2014 | 68 | 68 | 49.617 | 151 |
| location_6 | 2015 | 42 | 42 | 48.520 | 152 |
| location_6 | 2016 | 35 | 35 | 47.036 | 151 |
| location_6 | 2017 | 46 | 46 | 46.664 | 151 |
| location_6 | 2018 | 41 | 41 | 40.266 | 151 |
| location_6 | 2019 | 50 | 50 | 43.245 | 152 |
| location_6 | 2020 | 37 | 37 | 42.401 | 151 |
| location_6 | 2021 | 61 | 61 | 57.688 | 151 |
| location_6 | 2022 | 29 | 29 | 46.980 | 148 |
| location_6 | 2023 | 35 | 35 | 48.295 | 152 |
| location_6 | 2024 | 59 | 59 | 52.004 | 151 |
| location_6 | 2025 | 68 | 68 | 74.509 | 148 |
| location_7 | 1980 | 63 | 63 | 42.076 | 151 |
| location_7 | 1981 | 61 | 61 | 48.859 | 151 |
| location_7 | 1982 | 58 | 58 | 47.901 | 151 |
| location_7 | 1983 | 59 | 59 | 46.750 | 152 |
| location_7 | 1984 | 58 | 58 | 46.380 | 151 |
| location_7 | 1985 | 63 | 63 | 49.296 | 151 |
| location_7 | 1986 | 46 | 46 | 36.897 | 151 |
| location_7 | 1987 | 48 | 48 | 41.408 | 152 |
| location_7 | 1988 | 54 | 54 | 38.096 | 151 |
| location_7 | 1989 | 58 | 58 | 44.362 | 151 |
| location_7 | 1990 | 36 | 36 | 31.559 | 151 |
| location_7 | 1991 | 55 | 55 | 46.663 | 152 |
| location_7 | 1992 | 41 | 41 | 40.877 | 151 |
| location_7 | 1993 | 50 | 50 | 42.831 | 151 |
| location_7 | 1994 | 52 | 52 | 47.311 | 151 |
| location_7 | 1995 | 50 | 50 | 47.892 | 152 |
| location_7 | 1996 | 47 | 47 | 45.664 | 151 |
| location_7 | 1997 | 59 | 59 | 43.813 | 151 |
| location_7 | 1998 | 54 | 54 | 47.011 | 151 |
| location_7 | 1999 | 28 | 28 | 37.662 | 152 |
| location_7 | 2000 | 38 | 38 | 44.590 | 151 |
| location_7 | 2001 | 46 | 46 | 45.439 | 151 |
| location_7 | 2002 | 46 | 46 | 50.397 | 151 |
| location_7 | 2003 | 33 | 33 | 40.432 | 152 |
| location_7 | 2004 | 48 | 48 | 49.475 | 151 |
| location_7 | 2005 | 42 | 42 | 38.684 | 151 |
| location_7 | 2006 | 25 | 25 | 33.248 | 151 |
| location_7 | 2007 | 31 | 31 | 32.501 | 152 |
| location_7 | 2008 | 42 | 42 | 41.322 | 151 |
| location_7 | 2009 | 29 | 29 | 40.573 | 151 |
| location_7 | 2010 | 29 | 29 | 44.694 | 151 |
| location_7 | 2011 | 28 | 28 | 31.596 | 152 |
| location_7 | 2012 | 32 | 32 | 37.434 | 151 |
| location_7 | 2013 | 7 | 7 | 17.784 | 151 |
| location_7 | 2014 | 49 | 49 | 38.014 | 151 |
| location_7 | 2015 | 24 | 24 | 37.860 | 152 |
| location_7 | 2016 | 25 | 25 | 33.725 | 151 |
| location_7 | 2017 | 22 | 22 | 31.900 | 151 |
| location_7 | 2018 | 17 | 17 | 30.270 | 151 |
| location_7 | 2019 | 31 | 31 | 31.848 | 152 |
| location_7 | 2020 | 18 | 18 | 30.660 | 151 |
| location_7 | 2021 | 45 | 45 | 45.206 | 151 |
| location_7 | 2022 | 22 | 22 | 35.362 | 151 |
| location_7 | 2023 | 12 | 12 | 35.329 | 152 |
| location_7 | 2024 | 37 | 37 | 42.093 | 151 |
| location_7 | 2025 | 49 | 49 | 58.153 | 150 |
| location_8 | 1992 | 54 | 54 | 60.995 | 150 |
| location_8 | 1994 | 61 | 61 | 67.630 | 149 |
| location_8 | 1995 | 64 | 64 | 68.730 | 148 |
| location_8 | 2006 | 48 | 48 | 44.492 | 150 |
| location_8 | 2007 | 52 | 52 | 46.384 | 152 |
| location_8 | 2008 | 74 | 74 | 58.378 | 151 |
| location_8 | 2009 | 66 | 66 | 56.129 | 149 |
| location_8 | 2012 | 43 | 43 | 48.664 | 151 |
| location_8 | 2013 | 27 | 27 | 22.246 | 151 |
| location_8 | 2014 | 78 | 78 | 49.940 | 151 |
| location_8 | 2015 | 62 | 62 | 53.331 | 152 |
| location_8 | 2016 | 52 | 52 | 49.013 | 150 |
| location_8 | 2017 | 53 | 53 | 42.430 | 150 |
| location_8 | 2018 | 50 | 50 | 47.216 | 150 |
| location_8 | 2019 | 42 | 41 | 47.323 | 151 |
| location_8 | 2020 | 45 | 45 | 44.593 | 151 |
| location_8 | 2021 | 53 | 53 | 62.181 | 151 |
| location_8 | 2022 | 37 | 37 | 54.959 | 151 |
| location_8 | 2023 | 42 | 41 | 50.828 | 149 |
| location_8 | 2024 | 54 | 54 | 62.035 | 150 |
| location_8 | 2025 | 62 | 62 | 83.750 | 150 |
| location_9 | 1999 | 41 | 37 | 72.986 | 147 |
| location_9 | 2004 | 81 | 75 | 87.518 | 144 |
| location_9 | 2005 | 83 | 80 | 69.039 | 147 |
| location_9 | 2006 | 56 | 56 | 58.603 | 149 |
| location_9 | 2007 | 66 | 65 | 61.138 | 151 |
| location_9 | 2008 | 78 | 77 | 75.011 | 146 |
| location_9 | 2009 | 84 | 84 | 74.302 | 151 |
| location_9 | 2010 | 75 | 75 | 76.172 | 151 |
| location_9 | 2011 | 78 | 77 | 58.228 | 151 |
| location_9 | 2012 | 87 | 87 | 65.164 | 150 |
| location_9 | 2013 | 62 | 62 | 35.176 | 151 |
| location_9 | 2015 | 82 | 82 | 71.912 | 152 |
| location_9 | 2016 | 70 | 70 | 67.375 | 151 |
| location_9 | 2017 | 71 | 71 | 58.519 | 150 |
| location_9 | 2018 | 60 | 60 | 60.954 | 151 |
| location_9 | 2019 | 67 | 67 | 64.835 | 152 |
| location_9 | 2020 | 56 | 55 | 60.598 | 148 |
| location_9 | 2021 | 77 | 77 | 77.632 | 151 |
| location_9 | 2022 | 51 | 50 | 73.012 | 149 |
| location_9 | 2023 | 53 | 53 | 72.726 | 152 |
| location_9 | 2024 | 58 | 58 | 81.581 | 151 |

### Diagnostic fold availability

```json
[
  {
    "winter": 1980,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1981,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1982,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1983,
    "n_train": 23208,
    "n_test": 304,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1984,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1985,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1986,
    "n_train": 23211,
    "n_test": 301,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1987,
    "n_train": 23208,
    "n_test": 304,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1988,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1989,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1990,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1991,
    "n_train": 23208,
    "n_test": 304,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1992,
    "n_train": 23060,
    "n_test": 452,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1993,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1994,
    "n_train": 23212,
    "n_test": 300,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1995,
    "n_train": 23060,
    "n_test": 452,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1996,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1997,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1998,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1999,
    "n_train": 22916,
    "n_test": 596,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2000,
    "n_train": 23211,
    "n_test": 301,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2001,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2002,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2003,
    "n_train": 23064,
    "n_test": 448,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2004,
    "n_train": 23066,
    "n_test": 446,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2005,
    "n_train": 22915,
    "n_test": 597,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2006,
    "n_train": 22760,
    "n_test": 752,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2007,
    "n_train": 22753,
    "n_test": 759,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2008,
    "n_train": 22763,
    "n_test": 749,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2009,
    "n_train": 22759,
    "n_test": 753,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2010,
    "n_train": 22908,
    "n_test": 604,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2011,
    "n_train": 22908,
    "n_test": 604,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2012,
    "n_train": 22758,
    "n_test": 754,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2013,
    "n_train": 22757,
    "n_test": 755,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2014,
    "n_train": 22908,
    "n_test": 604,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2015,
    "n_train": 22752,
    "n_test": 760,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2016,
    "n_train": 22758,
    "n_test": 754,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2017,
    "n_train": 22759,
    "n_test": 753,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2018,
    "n_train": 22758,
    "n_test": 754,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2019,
    "n_train": 22753,
    "n_test": 759,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2020,
    "n_train": 22760,
    "n_test": 752,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2021,
    "n_train": 22757,
    "n_test": 755,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2022,
    "n_train": 22762,
    "n_test": 750,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2023,
    "n_train": 22755,
    "n_test": 757,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2024,
    "n_train": 22759,
    "n_test": 753,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2025,
    "n_train": 22914,
    "n_test": 598,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  }
]
```

## ebm

### Panel and attribution

```json
{
  "panel": {
    "n_rows": 156,
    "n_winters": 46,
    "status": "ok",
    "coefficients": {
      "intercept_location_10": {
        "estimate": 0.09134424070326425,
        "ci95": [
          -0.13189852402230506,
          0.31458700542883355
        ],
        "p_two_sided": 0.4302848575712144
      },
      "intercept_location_6": {
        "estimate": 0.06784922935434468,
        "ci95": [
          -0.01780417625935865,
          0.153502634968048
        ],
        "p_two_sided": 0.11294352823588207
      },
      "intercept_location_7": {
        "estimate": -0.029677299981987027,
        "ci95": [
          -0.21353877529899798,
          0.1541841753350239
        ],
        "p_two_sided": 0.7651174412793603
      },
      "intercept_location_8": {
        "estimate": 0.0749455923335684,
        "ci95": [
          -0.021058934594190656,
          0.17095011926132747
        ],
        "p_two_sided": 0.12343828085957022
      },
      "intercept_location_9": {
        "estimate": 0.07658896848583062,
        "ci95": [
          -0.06105275576828019,
          0.21423069273994144
        ],
        "p_two_sided": 0.2938530734632684
      },
      "year": {
        "estimate": -0.011237203647839145,
        "ci95": [
          -0.02034975271729639,
          -0.002124654578381898
        ],
        "p_two_sided": 0.015992003998001
      },
      "step_location_7": {
        "estimate": -0.15080279576257816,
        "ci95": [
          -0.35987747640992024,
          0.058271884884763925
        ],
        "p_two_sided": 0.14992503748125938
      },
      "step_location_10": {
        "estimate": -0.4163661320847224,
        "ci95": [
          -0.7124402586354389,
          -0.12029200553400587
        ],
        "p_two_sided": 0.0069965017491254375
      }
    },
    "r_squared": 0.4253205140261034,
    "aic": 44.944382593518455,
    "valid_draws": 2000,
    "n_boot": 2000,
    "omitted_absorbed_steps": [
      "step_location_8"
    ]
  },
  "post_asos_1996": {
    "n_rows": 122,
    "n_winters": 30,
    "status": "ok",
    "coefficients": {
      "intercept_location_10": {
        "estimate": -0.3894632193309255,
        "ci95": [
          -0.5244602963448042,
          -0.2544661423170468
        ],
        "p_two_sided": 0.0004997501249375312
      },
      "intercept_location_6": {
        "estimate": 0.010729677247049799,
        "ci95": [
          -0.0717957353095084,
          0.09325508980360801
        ],
        "p_two_sided": 0.7986006996501749
      },
      "intercept_location_7": {
        "estimate": -0.24662882055421942,
        "ci95": [
          -0.3540597709776698,
          -0.13919787013076906
        ],
        "p_two_sided": 0.0004997501249375312
      },
      "intercept_location_8": {
        "estimate": 0.06908366526848823,
        "ci95": [
          -0.0071477080669084675,
          0.14531503860388495
        ],
        "p_two_sided": 0.07196401799100449
      },
      "intercept_location_9": {
        "estimate": 0.01836110593885011,
        "ci95": [
          -0.0990565413175813,
          0.13577875319528154
        ],
        "p_two_sided": 0.7466266866566716
      },
      "year": {
        "estimate": -0.013467678407704894,
        "ci95": [
          -0.024340378625654788,
          -0.002594978189755001
        ],
        "p_two_sided": 0.01649175412293853
      }
    },
    "r_squared": 0.32227590116010196,
    "aic": 43.4012670586966,
    "valid_draws": 2000,
    "n_boot": 2000,
    "omitted_absorbed_steps": [
      "step_location_7",
      "step_location_10",
      "step_location_8"
    ]
  },
  "kvis_1992_step": {
    "n_rows": 156,
    "n_winters": 46,
    "status": "ok",
    "coefficients": {
      "intercept_location_10": {
        "estimate": 0.09134424070326425,
        "ci95": [
          -0.13189852402230506,
          0.31458700542883355
        ],
        "p_two_sided": 0.4302848575712144
      },
      "intercept_location_6": {
        "estimate": 0.06784922935434468,
        "ci95": [
          -0.01780417625935865,
          0.153502634968048
        ],
        "p_two_sided": 0.11294352823588207
      },
      "intercept_location_7": {
        "estimate": -0.029677299981987027,
        "ci95": [
          -0.21353877529899798,
          0.1541841753350239
        ],
        "p_two_sided": 0.7651174412793603
      },
      "intercept_location_8": {
        "estimate": 0.0749455923335684,
        "ci95": [
          -0.021058934594190656,
          0.17095011926132747
        ],
        "p_two_sided": 0.12343828085957022
      },
      "intercept_location_9": {
        "estimate": 0.07658896848583062,
        "ci95": [
          -0.06105275576828019,
          0.21423069273994144
        ],
        "p_two_sided": 0.2938530734632684
      },
      "year": {
        "estimate": -0.011237203647839145,
        "ci95": [
          -0.02034975271729639,
          -0.002124654578381898
        ],
        "p_two_sided": 0.015992003998001
      },
      "step_location_7": {
        "estimate": -0.15080279576257816,
        "ci95": [
          -0.35987747640992024,
          0.058271884884763925
        ],
        "p_two_sided": 0.14992503748125938
      },
      "step_location_10": {
        "estimate": -0.4163661320847224,
        "ci95": [
          -0.7124402586354389,
          -0.12029200553400587
        ],
        "p_two_sided": 0.0069965017491254375
      }
    },
    "r_squared": 0.4253205140261034,
    "aic": 44.944382593518455,
    "valid_draws": 2000,
    "n_boot": 2000,
    "omitted_absorbed_steps": [
      "step_location_8"
    ]
  },
  "per_site_trends": {
    "location_10": {
      "status": "ok",
      "n": 45,
      "slope": -0.025667049942318342,
      "ci95": [
        -0.03355818635527906,
        -0.016453353643403716
      ],
      "kendall_tau": -0.5353535353535352,
      "mann_kendall_p": 2.1644473968564465e-07
    },
    "location_6": {
      "status": "ok",
      "n": 23,
      "slope": -0.011642620930930018,
      "ci95": [
        -0.025478013317185206,
        0.0028031005521417292
      ],
      "kendall_tau": -0.23320158102766797,
      "mann_kendall_p": 0.12651766910267948
    },
    "location_7": {
      "status": "ok",
      "n": 46,
      "slope": -0.01630671291186891,
      "ci95": [
        -0.021467719143857984,
        -0.010553764628582209
      ],
      "kendall_tau": -0.5903381642512078,
      "mann_kendall_p": 7.249475151815691e-09
    },
    "location_8": {
      "status": "ok",
      "n": 21,
      "slope": -0.007627929263022297,
      "ci95": [
        -0.019987221538558333,
        0.0005106451592676673
      ],
      "kendall_tau": -0.29523809523809524,
      "mann_kendall_p": 0.06493357769830031
    },
    "location_9": {
      "status": "ok",
      "n": 21,
      "slope": -0.01099198400018404,
      "ci95": [
        -0.02951970469376702,
        0.010702934036547956
      ],
      "kendall_tau": -0.1523809523809524,
      "mann_kendall_p": 0.35364385023332723
    }
  },
  "attribution": {
    "status": "ok",
    "common_rows": [
      {
        "site": "location_10",
        "winter": 1980,
        "nox_ppb": 66.83008526187577
      },
      {
        "site": "location_10",
        "winter": 1981,
        "nox_ppb": 51.96247766527695
      },
      {
        "site": "location_10",
        "winter": 1982,
        "nox_ppb": 58.81853983641321
      },
      {
        "site": "location_10",
        "winter": 1983,
        "nox_ppb": 94.40688383527966
      },
      {
        "site": "location_10",
        "winter": 1984,
        "nox_ppb": 88.78513429106808
      },
      {
        "site": "location_10",
        "winter": 1986,
        "nox_ppb": 67.95307193578265
      },
      {
        "site": "location_10",
        "winter": 1995,
        "nox_ppb": 27.98113763631005
      },
      {
        "site": "location_10",
        "winter": 1996,
        "nox_ppb": 27.220821114369503
      },
      {
        "site": "location_10",
        "winter": 1997,
        "nox_ppb": 27.778639618138424
      },
      {
        "site": "location_10",
        "winter": 1998,
        "nox_ppb": 26.066471449487555
      },
      {
        "site": "location_10",
        "winter": 1999,
        "nox_ppb": 29.59719789842382
      },
      {
        "site": "location_10",
        "winter": 2000,
        "nox_ppb": 27.857225600926732
      },
      {
        "site": "location_10",
        "winter": 2001,
        "nox_ppb": 24.413361805979257
      },
      {
        "site": "location_10",
        "winter": 2002,
        "nox_ppb": 25.98757225433526
      },
      {
        "site": "location_10",
        "winter": 2003,
        "nox_ppb": 19.488045427375972
      },
      {
        "site": "location_10",
        "winter": 2004,
        "nox_ppb": 20.178927680798004
      },
      {
        "site": "location_10",
        "winter": 2005,
        "nox_ppb": 36.09256494475963
      },
      {
        "site": "location_10",
        "winter": 2006,
        "nox_ppb": 60.11046680197158
      },
      {
        "site": "location_10",
        "winter": 2007,
        "nox_ppb": 49.10115606936416
      },
      {
        "site": "location_10",
        "winter": 2008,
        "nox_ppb": 48.327283372365336
      },
      {
        "site": "location_10",
        "winter": 2009,
        "nox_ppb": 40.22479721900348
      },
      {
        "site": "location_10",
        "winter": 2010,
        "nox_ppb": 35.77171717171717
      },
      {
        "site": "location_10",
        "winter": 2011,
        "nox_ppb": 43.43805436337625
      },
      {
        "site": "location_10",
        "winter": 2012,
        "nox_ppb": 39.18463768115942
      },
      {
        "site": "location_10",
        "winter": 2014,
        "nox_ppb": 28.295812144780136
      },
      {
        "site": "location_10",
        "winter": 2016,
        "nox_ppb": 26.14366279069768
      },
      {
        "site": "location_10",
        "winter": 2017,
        "nox_ppb": 33.18026086956522
      },
      {
        "site": "location_10",
        "winter": 2018,
        "nox_ppb": 25.448402091807086
      },
      {
        "site": "location_10",
        "winter": 2019,
        "nox_ppb": 25.766377063423107
      },
      {
        "site": "location_10",
        "winter": 2020,
        "nox_ppb": 25.22509417560127
      },
      {
        "site": "location_10",
        "winter": 2021,
        "nox_ppb": 28.093908629441625
      },
      {
        "site": "location_10",
        "winter": 2022,
        "nox_ppb": 22.242558746736293
      },
      {
        "site": "location_10",
        "winter": 2023,
        "nox_ppb": 22.252156411730883
      },
      {
        "site": "location_10",
        "winter": 2024,
        "nox_ppb": 23.85091119467747
      },
      {
        "site": "location_6",
        "winter": 1999,
        "nox_ppb": 33.35001480604087
      },
      {
        "site": "location_6",
        "winter": 2003,
        "nox_ppb": 23.976058394160585
      },
      {
        "site": "location_6",
        "winter": 2005,
        "nox_ppb": 22.198833819241983
      },
      {
        "site": "location_6",
        "winter": 2006,
        "nox_ppb": 22.977239568135396
      },
      {
        "site": "location_6",
        "winter": 2007,
        "nox_ppb": 17.957366589327147
      },
      {
        "site": "location_6",
        "winter": 2008,
        "nox_ppb": 16.594476744186046
      },
      {
        "site": "location_6",
        "winter": 2009,
        "nox_ppb": 14.618891537544696
      },
      {
        "site": "location_6",
        "winter": 2010,
        "nox_ppb": 13.314756174794173
      },
      {
        "site": "location_6",
        "winter": 2012,
        "nox_ppb": 13.681378865979381
      },
      {
        "site": "location_6",
        "winter": 2016,
        "nox_ppb": 8.127062609713283
      },
      {
        "site": "location_6",
        "winter": 2017,
        "nox_ppb": 11.727115666178623
      },
      {
        "site": "location_6",
        "winter": 2018,
        "nox_ppb": 9.38939349545854
      },
      {
        "site": "location_6",
        "winter": 2019,
        "nox_ppb": 8.808408759124088
      },
      {
        "site": "location_6",
        "winter": 2020,
        "nox_ppb": 10.132607416127133
      },
      {
        "site": "location_6",
        "winter": 2021,
        "nox_ppb": 9.464883239728053
      },
      {
        "site": "location_6",
        "winter": 2022,
        "nox_ppb": 8.349336870026525
      },
      {
        "site": "location_6",
        "winter": 2023,
        "nox_ppb": 7.4415177263404635
      },
      {
        "site": "location_6",
        "winter": 2024,
        "nox_ppb": 7.625554224086279
      },
      {
        "site": "location_6",
        "winter": 2025,
        "nox_ppb": 6.994775248102744
      },
      {
        "site": "location_7",
        "winter": 1980,
        "nox_ppb": 106.32058287795992
      },
      {
        "site": "location_7",
        "winter": 1981,
        "nox_ppb": 78.03897685749087
      },
      {
        "site": "location_7",
        "winter": 1982,
        "nox_ppb": 78.13308104698834
      },
      {
        "site": "location_7",
        "winter": 1983,
        "nox_ppb": 84.59177409453653
      },
      {
        "site": "location_7",
        "winter": 1984,
        "nox_ppb": 83.97800776196637
      },
      {
        "site": "location_7",
        "winter": 1985,
        "nox_ppb": 78.07370703004025
      },
      {
        "site": "location_7",
        "winter": 1986,
        "nox_ppb": 116.43059490084985
      },
      {
        "site": "location_7",
        "winter": 1994,
        "nox_ppb": 35.71683226183519
      },
      {
        "site": "location_7",
        "winter": 1995,
        "nox_ppb": 41.70106132075472
      },
      {
        "site": "location_7",
        "winter": 1996,
        "nox_ppb": 36.22445913461539
      },
      {
        "site": "location_7",
        "winter": 1997,
        "nox_ppb": 38.169645656945754
      },
      {
        "site": "location_7",
        "winter": 1998,
        "nox_ppb": 38.99794178182887
      },
      {
        "site": "location_7",
        "winter": 1999,
        "nox_ppb": 44.28068644560791
      },
      {
        "site": "location_7",
        "winter": 2000,
        "nox_ppb": 40.283958602846056
      },
      {
        "site": "location_7",
        "winter": 2001,
        "nox_ppb": 33.79604486422668
      },
      {
        "site": "location_7",
        "winter": 2002,
        "nox_ppb": 35.61832543443918
      },
      {
        "site": "location_7",
        "winter": 2003,
        "nox_ppb": 34.30892182505086
      },
      {
        "site": "location_7",
        "winter": 2004,
        "nox_ppb": 28.05842630651791
      },
      {
        "site": "location_7",
        "winter": 2005,
        "nox_ppb": 31.345708345708346
      },
      {
        "site": "location_7",
        "winter": 2006,
        "nox_ppb": 36.58463810930576
      },
      {
        "site": "location_7",
        "winter": 2007,
        "nox_ppb": 28.982584784601283
      },
      {
        "site": "location_7",
        "winter": 2008,
        "nox_ppb": 30.154962962962962
      },
      {
        "site": "location_7",
        "winter": 2009,
        "nox_ppb": 22.737110311750598
      },
      {
        "site": "location_7",
        "winter": 2010,
        "nox_ppb": 20.953890489913544
      },
      {
        "site": "location_7",
        "winter": 2012,
        "nox_ppb": 24.25542168674699
      },
      {
        "site": "location_7",
        "winter": 2014,
        "nox_ppb": 20.08777811642792
      },
      {
        "site": "location_7",
        "winter": 2015,
        "nox_ppb": 19.74296824368114
      },
      {
        "site": "location_7",
        "winter": 2016,
        "nox_ppb": 18.276494146642015
      },
      {
        "site": "location_7",
        "winter": 2017,
        "nox_ppb": 22.784728773584906
      },
      {
        "site": "location_7",
        "winter": 2018,
        "nox_ppb": 17.84309559939302
      },
      {
        "site": "location_7",
        "winter": 2019,
        "nox_ppb": 17.591957734076903
      },
      {
        "site": "location_7",
        "winter": 2020,
        "nox_ppb": 17.811683343143027
      },
      {
        "site": "location_7",
        "winter": 2021,
        "nox_ppb": 16.8846015037594
      },
      {
        "site": "location_7",
        "winter": 2022,
        "nox_ppb": 16.44575318141462
      },
      {
        "site": "location_7",
        "winter": 2023,
        "nox_ppb": 14.005227606461087
      },
      {
        "site": "location_7",
        "winter": 2024,
        "nox_ppb": 13.070996179841318
      },
      {
        "site": "location_8",
        "winter": 1994,
        "nox_ppb": 39.52385185185185
      },
      {
        "site": "location_8",
        "winter": 1995,
        "nox_ppb": 43.46763848396501
      },
      {
        "site": "location_8",
        "winter": 2006,
        "nox_ppb": 40.16413107080164
      },
      {
        "site": "location_8",
        "winter": 2007,
        "nox_ppb": 30.65805340223945
      },
      {
        "site": "location_8",
        "winter": 2008,
        "nox_ppb": 28.452078521939953
      },
      {
        "site": "location_8",
        "winter": 2009,
        "nox_ppb": 28.40252495596007
      },
      {
        "site": "location_8",
        "winter": 2012,
        "nox_ppb": 24.06021694517737
      },
      {
        "site": "location_8",
        "winter": 2013,
        "nox_ppb": 29.295645863570392
      },
      {
        "site": "location_8",
        "winter": 2014,
        "nox_ppb": 19.114678059200486
      },
      {
        "site": "location_8",
        "winter": 2015,
        "nox_ppb": 16.688927236123092
      },
      {
        "site": "location_8",
        "winter": 2016,
        "nox_ppb": 20.018935837245696
      },
      {
        "site": "location_8",
        "winter": 2018,
        "nox_ppb": 18.25922933499068
      },
      {
        "site": "location_8",
        "winter": 2019,
        "nox_ppb": 16.717759168351144
      },
      {
        "site": "location_8",
        "winter": 2020,
        "nox_ppb": 18.278865534120357
      },
      {
        "site": "location_8",
        "winter": 2022,
        "nox_ppb": 13.174992738890502
      },
      {
        "site": "location_8",
        "winter": 2024,
        "nox_ppb": 13.8213747099768
      },
      {
        "site": "location_9",
        "winter": 1999,
        "nox_ppb": 40.669696020056406
      },
      {
        "site": "location_9",
        "winter": 2004,
        "nox_ppb": 21.912107101280558
      },
      {
        "site": "location_9",
        "winter": 2005,
        "nox_ppb": 27.817204301075268
      },
      {
        "site": "location_9",
        "winter": 2006,
        "nox_ppb": 34.88733233979136
      },
      {
        "site": "location_9",
        "winter": 2007,
        "nox_ppb": 30.65805340223945
      },
      {
        "site": "location_9",
        "winter": 2008,
        "nox_ppb": 28.452078521939953
      },
      {
        "site": "location_9",
        "winter": 2009,
        "nox_ppb": 28.40252495596007
      },
      {
        "site": "location_9",
        "winter": 2011,
        "nox_ppb": 27.403261210410786
      },
      {
        "site": "location_9",
        "winter": 2012,
        "nox_ppb": 22.103939393939395
      },
      {
        "site": "location_9",
        "winter": 2013,
        "nox_ppb": 28.27719298245614
      },
      {
        "site": "location_9",
        "winter": 2015,
        "nox_ppb": 17.651522020725388
      },
      {
        "site": "location_9",
        "winter": 2016,
        "nox_ppb": 13.92640432562331
      },
      {
        "site": "location_9",
        "winter": 2017,
        "nox_ppb": 19.533046573716994
      },
      {
        "site": "location_9",
        "winter": 2018,
        "nox_ppb": 16.19676290463692
      },
      {
        "site": "location_9",
        "winter": 2019,
        "nox_ppb": 15.470949883449881
      },
      {
        "site": "location_9",
        "winter": 2020,
        "nox_ppb": 18.12390158172232
      },
      {
        "site": "location_9",
        "winter": 2021,
        "nox_ppb": 15.828238039673279
      },
      {
        "site": "location_9",
        "winter": 2022,
        "nox_ppb": 14.152518409425626
      },
      {
        "site": "location_9",
        "winter": 2024,
        "nox_ppb": 14.87244117647059
      }
    ],
    "models": {
      "year_only": {
        "n_rows": 124,
        "n_winters": 39,
        "status": "ok",
        "coefficients": {
          "intercept_location_10": {
            "estimate": -0.025871056761811888,
            "ci95": [
              -0.365270727482194,
              0.31352861395857023
            ],
            "p_two_sided": 0.8804402201100551
          },
          "intercept_location_6": {
            "estimate": 0.052718850653280254,
            "ci95": [
              -0.030343725173104928,
              0.13578142647966543
            ],
            "p_two_sided": 0.19909954977488745
          },
          "intercept_location_7": {
            "estimate": -0.08587430282430683,
            "ci95": [
              -0.32387257685061355,
              0.15212397120199989
            ],
            "p_two_sided": 0.4622311155577789
          },
          "intercept_location_8": {
            "estimate": 0.08633079905756912,
            "ci95": [
              -0.011165702535553046,
              0.18382730065069128
            ],
            "p_two_sided": 0.0785392696348174
          },
          "intercept_location_9": {
            "estimate": 0.08536910502436422,
            "ci95": [
              -0.061280406250252925,
              0.23201861629898135
            ],
            "p_two_sided": 0.26313156578289143
          },
          "year": {
            "estimate": -0.012242680202183135,
            "ci95": [
              -0.0221254998367081,
              -0.002359860567658174
            ],
            "p_two_sided": 0.016508254127063533
          },
          "step_location_7": {
            "estimate": -0.09983551035445523,
            "ci95": [
              -0.34256779686262484,
              0.14289677615371435
            ],
            "p_two_sided": 0.42921460730365185
          },
          "step_location_10": {
            "estimate": -0.27414953680937076,
            "ci95": [
              -0.6599527797310845,
              0.11165370611234288
            ],
            "p_two_sided": 0.17958979489744872
          }
        },
        "r_squared": 0.39524483313203385,
        "aic": 26.707768663521833,
        "valid_draws": 1998,
        "n_boot": 2000,
        "omitted_absorbed_steps": [
          "step_location_8"
        ]
      },
      "nox_only": {
        "n_rows": 124,
        "n_winters": 39,
        "status": "ok",
        "coefficients": {
          "intercept_location_10": {
            "estimate": 0.08269638449944383,
            "ci95": [
              -0.21627337178738487,
              0.38166614078627253
            ],
            "p_two_sided": 0.5612806403201601
          },
          "intercept_location_6": {
            "estimate": 0.06826403136419076,
            "ci95": [
              -0.030664441583824198,
              0.16719250431220573
            ],
            "p_two_sided": 0.1520760380190095
          },
          "intercept_location_7": {
            "estimate": -0.06768152804808926,
            "ci95": [
              -0.3689989849866222,
              0.2336359288904437
            ],
            "p_two_sided": 0.6413206603301651
          },
          "intercept_location_8": {
            "estimate": 0.06773356275992423,
            "ci95": [
              -0.01712758274359398,
              0.15259470826344246
            ],
            "p_two_sided": 0.10755377688844422
          },
          "intercept_location_9": {
            "estimate": 0.0658436010624109,
            "ci95": [
              -0.06634714396935852,
              0.1980343460941803
            ],
            "p_two_sided": 0.3196598299149575
          },
          "nox_ppb": {
            "estimate": 0.005227957365002485,
            "ci95": [
              -0.0002747647942052678,
              0.010730679524210238
            ],
            "p_two_sided": 0.06353176588294147
          },
          "step_location_7": {
            "estimate": -0.11518961687212799,
            "ci95": [
              -0.4174194822191893,
              0.18704024847493333
            ],
            "p_two_sided": 0.42621310655327665
          },
          "step_location_10": {
            "estimate": -0.40537679569083573,
            "ci95": [
              -0.7318657120069991,
              -0.0788878793746724
            ],
            "p_two_sided": 0.02001000500250125
          }
        },
        "r_squared": 0.3390492024189298,
        "aic": 37.72587639130661,
        "valid_draws": 1998,
        "n_boot": 2000,
        "omitted_absorbed_steps": [
          "step_location_8"
        ]
      },
      "year_plus_nox": {
        "n_rows": 124,
        "n_winters": 39,
        "status": "ok",
        "coefficients": {
          "intercept_location_10": {
            "estimate": -0.029904095584886197,
            "ci95": [
              -0.3731310064591213,
              0.31332281528934885
            ],
            "p_two_sided": 0.8514257128564282
          },
          "intercept_location_6": {
            "estimate": 0.05595543455640106,
            "ci95": [
              -0.036185620801171714,
              0.14809648991397384
            ],
            "p_two_sided": 0.22961480740370185
          },
          "intercept_location_7": {
            "estimate": -0.09463664229128362,
            "ci95": [
              -0.3708376983964574,
              0.1815644138138902
            ],
            "p_two_sided": 0.4767383691845923
          },
          "intercept_location_8": {
            "estimate": 0.08701002600979352,
            "ci95": [
              -0.010480347417835836,
              0.18450039943742286
            ],
            "p_two_sided": 0.07653826913456728
          },
          "intercept_location_9": {
            "estimate": 0.08638948570133273,
            "ci95": [
              -0.06253436695198802,
              0.23531333835465348
            ],
            "p_two_sided": 0.26013006503251623
          },
          "year": {
            "estimate": -0.012016147992566327,
            "ci95": [
              -0.022888074114450725,
              -0.0011442218706819302
            ],
            "p_two_sided": 0.03151575787893947
          },
          "nox_ppb": {
            "estimate": 0.0002723045592776818,
            "ci95": [
              -0.004103119828354433,
              0.0046477289469097965
            ],
            "p_two_sided": 0.896448224112056
          },
          "step_location_7": {
            "estimate": -0.09037870389769928,
            "ci95": [
              -0.37177006327171536,
              0.19101265547631677
            ],
            "p_two_sided": 0.5087543771885943
          },
          "step_location_10": {
            "estimate": -0.2705929811890286,
            "ci95": [
              -0.6743331266569506,
              0.1331471642788934
            ],
            "p_two_sided": 0.18959479739869936
          }
        },
        "r_squared": 0.39530310667227875,
        "aic": 28.695819584705454,
        "valid_draws": 1998,
        "n_boot": 2000,
        "omitted_absorbed_steps": [
          "step_location_8"
        ]
      }
    },
    "year_slope_change_with_nox": 0.00022653220961680806,
    "gray_comparison": {
      "all": {
        "n_rows": 124,
        "n_winters": 39,
        "status": "ok",
        "coefficients": {
          "intercept_location_10": {
            "estimate": 4.091868201510754,
            "ci95": [
              -6.924417173050963,
              15.108153576072471
            ],
            "p_two_sided": 0.4302151075537769
          },
          "intercept_location_6": {
            "estimate": 2.7094586735958464,
            "ci95": [
              -1.6858576365681035,
              7.104774983759796
            ],
            "p_two_sided": 0.21710855427713857
          },
          "intercept_location_7": {
            "estimate": 1.1218988137742865,
            "ci95": [
              -10.522304136626449,
              12.766101764175023
            ],
            "p_two_sided": 0.8364182091045522
          },
          "intercept_location_8": {
            "estimate": 3.8002970509158396,
            "ci95": [
              -0.6537286623153649,
              8.254322764147044
            ],
            "p_two_sided": 0.09304652326163082
          },
          "intercept_location_9": {
            "estimate": 3.6821310651065717,
            "ci95": [
              -3.8171176877308284,
              11.181379817943972
            ],
            "p_two_sided": 0.3311655827913957
          },
          "nox_ppb": {
            "estimate": 0.1855697146196605,
            "ci95": [
              -0.02926404300958743,
              0.4004034722489084
            ],
            "p_two_sided": 0.08604302151075538
          },
          "step_location_7": {
            "estimate": -6.075213771311152,
            "ci95": [
              -18.77222138488954,
              6.6217938422672376
            ],
            "p_two_sided": 0.29514757378689344
          },
          "step_location_10": {
            "estimate": -10.326770182284223,
            "ci95": [
              -21.586112114170557,
              0.9325717496021113
            ],
            "p_two_sided": 0.0735367683841921
          }
        },
        "r_squared": 0.2808085603031494,
        "aic": 926.0277332390453,
        "valid_draws": 1998,
        "n_boot": 2000,
        "omitted_absorbed_steps": [
          "step_location_8"
        ],
        "fog_days_per_10ppb": {
          "estimate": 1.8556971461966048,
          "ci95": [
            -0.2926404300958743,
            4.0040347224890835
          ],
          "p_two_sided": 0.08604302151075538
        }
      },
      "near_saturation": {
        "n_rows": 19,
        "n_winters": 12,
        "status": "ok",
        "coefficients": {
          "intercept_location_10": {
            "estimate": 0.8035315762161288,
            "ci95": [
              -9.736445821041057,
              11.343508973473314
            ],
            "p_two_sided": 0.7025125628140704
          },
          "intercept_location_6": {
            "estimate": 2.7879909713166113,
            "ci95": [
              -14.780137433868079,
              20.3561193765013
            ],
            "p_two_sided": 0.5487437185929648
          },
          "intercept_location_7": {
            "estimate": -0.13367852624175397,
            "ci95": [
              -22.97246398777813,
              22.70510693529462
            ],
            "p_two_sided": 0.9688442211055276
          },
          "intercept_location_8": {
            "estimate": -3.992792829288908,
            "ci95": [
              -17.342508509641462,
              9.356922851063647
            ],
            "p_two_sided": 0.09547738693467336
          },
          "intercept_location_9": {
            "estimate": -7.59190261837962,
            "ci95": [
              -19.638217686628888,
              4.454412449869648
            ],
            "p_two_sided": 0.12663316582914572
          },
          "nox_ppb": {
            "estimate": 0.32358606038051185,
            "ci95": [
              -0.23735437384161817,
              0.8845264946026419
            ],
            "p_two_sided": 0.10452261306532663
          },
          "step_location_7": {
            "estimate": 3.39970477726861,
            "ci95": [
              -24.45383188545557,
              31.25324143999279
            ],
            "p_two_sided": 0.6120603015075377
          }
        },
        "r_squared": 0.7628124832564132,
        "aic": 130.6409960384497,
        "valid_draws": 994,
        "n_boot": 2000,
        "omitted_absorbed_steps": [
          "step_location_10",
          "step_location_8"
        ],
        "fog_days_per_10ppb": {
          "estimate": 3.2358606038051185,
          "ci95": [
            -2.3735437384161817,
            8.845264946026418
          ],
          "p_two_sided": 0.10452261306532663
        }
      }
    }
  }
}
```

### Winter observations and expectations

| Site | Winter | Observed (all days) | Observed (≥18 h) | Expected | Days |
|---|---:|---:|---:|---:|---:|
| location_10 | 1980 | 52 | 52 | 20.969 | 151 |
| location_10 | 1981 | 33 | 33 | 35.148 | 151 |
| location_10 | 1982 | 46 | 46 | 39.116 | 151 |
| location_10 | 1983 | 38 | 38 | 35.410 | 152 |
| location_10 | 1984 | 64 | 64 | 31.260 | 151 |
| location_10 | 1985 | 61 | 61 | 33.283 | 151 |
| location_10 | 1986 | 37 | 36 | 31.621 | 150 |
| location_10 | 1987 | 36 | 36 | 32.718 | 152 |
| location_10 | 1988 | 44 | 44 | 28.842 | 151 |
| location_10 | 1989 | 36 | 36 | 15.903 | 151 |
| location_10 | 1990 | 27 | 27 | 18.112 | 151 |
| location_10 | 1991 | 47 | 47 | 29.618 | 152 |
| location_10 | 1992 | 29 | 29 | 30.602 | 151 |
| location_10 | 1993 | 33 | 33 | 26.929 | 151 |
| location_10 | 1995 | 34 | 34 | 36.130 | 152 |
| location_10 | 1996 | 37 | 37 | 38.607 | 151 |
| location_10 | 1997 | 36 | 36 | 36.731 | 151 |
| location_10 | 1998 | 37 | 37 | 38.507 | 151 |
| location_10 | 1999 | 8 | 8 | 27.786 | 152 |
| location_10 | 2000 | 26 | 25 | 27.522 | 150 |
| location_10 | 2001 | 41 | 41 | 34.375 | 151 |
| location_10 | 2002 | 39 | 39 | 39.916 | 151 |
| location_10 | 2003 | 23 | 23 | 33.325 | 152 |
| location_10 | 2004 | 38 | 38 | 39.479 | 151 |
| location_10 | 2005 | 29 | 29 | 29.429 | 150 |
| location_10 | 2006 | 15 | 15 | 23.991 | 151 |
| location_10 | 2007 | 20 | 20 | 23.671 | 152 |
| location_10 | 2008 | 29 | 29 | 30.371 | 151 |
| location_10 | 2009 | 25 | 25 | 30.873 | 151 |
| location_10 | 2010 | 24 | 24 | 34.002 | 151 |
| location_10 | 2011 | 8 | 8 | 15.964 | 152 |
| location_10 | 2012 | 12 | 12 | 21.905 | 151 |
| location_10 | 2013 | 3 | 3 | 12.180 | 151 |
| location_10 | 2014 | 33 | 33 | 21.785 | 151 |
| location_10 | 2015 | 11 | 11 | 27.563 | 152 |
| location_10 | 2016 | 13 | 13 | 26.303 | 151 |
| location_10 | 2017 | 12 | 12 | 15.443 | 151 |
| location_10 | 2018 | 19 | 19 | 24.885 | 151 |
| location_10 | 2019 | 11 | 11 | 25.021 | 152 |
| location_10 | 2020 | 9 | 9 | 21.944 | 151 |
| location_10 | 2021 | 27 | 27 | 30.709 | 151 |
| location_10 | 2022 | 14 | 14 | 31.251 | 151 |
| location_10 | 2023 | 15 | 15 | 26.853 | 152 |
| location_10 | 2024 | 17 | 17 | 32.061 | 150 |
| location_10 | 2025 | 43 | 43 | 45.837 | 150 |
| location_6 | 1999 | 44 | 43 | 42.531 | 145 |
| location_6 | 2003 | 50 | 46 | 42.176 | 144 |
| location_6 | 2005 | 58 | 57 | 44.632 | 149 |
| location_6 | 2006 | 42 | 42 | 37.073 | 151 |
| location_6 | 2007 | 41 | 41 | 37.785 | 152 |
| location_6 | 2008 | 50 | 50 | 48.486 | 150 |
| location_6 | 2009 | 52 | 52 | 44.875 | 151 |
| location_6 | 2010 | 45 | 45 | 52.334 | 151 |
| location_6 | 2011 | 50 | 50 | 33.868 | 149 |
| location_6 | 2012 | 44 | 44 | 45.152 | 151 |
| location_6 | 2013 | 17 | 17 | 24.964 | 151 |
| location_6 | 2014 | 68 | 68 | 50.220 | 151 |
| location_6 | 2015 | 42 | 42 | 47.171 | 152 |
| location_6 | 2016 | 35 | 35 | 45.589 | 151 |
| location_6 | 2017 | 46 | 46 | 50.248 | 151 |
| location_6 | 2018 | 41 | 41 | 41.690 | 151 |
| location_6 | 2019 | 50 | 50 | 45.520 | 152 |
| location_6 | 2020 | 37 | 37 | 42.814 | 151 |
| location_6 | 2021 | 61 | 61 | 57.021 | 151 |
| location_6 | 2022 | 29 | 29 | 43.079 | 148 |
| location_6 | 2023 | 35 | 35 | 49.964 | 152 |
| location_6 | 2024 | 59 | 59 | 48.487 | 151 |
| location_6 | 2025 | 68 | 68 | 69.286 | 148 |
| location_7 | 1980 | 63 | 63 | 43.775 | 151 |
| location_7 | 1981 | 61 | 61 | 51.257 | 151 |
| location_7 | 1982 | 58 | 58 | 46.863 | 151 |
| location_7 | 1983 | 59 | 59 | 48.063 | 152 |
| location_7 | 1984 | 58 | 58 | 48.011 | 151 |
| location_7 | 1985 | 63 | 63 | 49.783 | 151 |
| location_7 | 1986 | 46 | 46 | 38.597 | 151 |
| location_7 | 1987 | 48 | 48 | 41.085 | 152 |
| location_7 | 1988 | 54 | 54 | 40.390 | 151 |
| location_7 | 1989 | 58 | 58 | 41.027 | 151 |
| location_7 | 1990 | 36 | 36 | 27.585 | 151 |
| location_7 | 1991 | 55 | 55 | 47.478 | 152 |
| location_7 | 1992 | 41 | 41 | 41.468 | 151 |
| location_7 | 1993 | 50 | 50 | 43.827 | 151 |
| location_7 | 1994 | 52 | 52 | 47.737 | 151 |
| location_7 | 1995 | 50 | 50 | 49.682 | 152 |
| location_7 | 1996 | 47 | 47 | 46.274 | 151 |
| location_7 | 1997 | 59 | 59 | 44.495 | 151 |
| location_7 | 1998 | 54 | 54 | 46.650 | 151 |
| location_7 | 1999 | 28 | 28 | 38.948 | 152 |
| location_7 | 2000 | 38 | 38 | 43.741 | 151 |
| location_7 | 2001 | 46 | 46 | 45.774 | 151 |
| location_7 | 2002 | 46 | 46 | 53.469 | 151 |
| location_7 | 2003 | 33 | 33 | 42.319 | 152 |
| location_7 | 2004 | 48 | 48 | 51.788 | 151 |
| location_7 | 2005 | 42 | 42 | 40.591 | 151 |
| location_7 | 2006 | 25 | 25 | 30.293 | 151 |
| location_7 | 2007 | 31 | 31 | 32.709 | 152 |
| location_7 | 2008 | 42 | 42 | 43.707 | 151 |
| location_7 | 2009 | 29 | 29 | 41.123 | 151 |
| location_7 | 2010 | 29 | 29 | 45.237 | 151 |
| location_7 | 2011 | 28 | 28 | 28.397 | 152 |
| location_7 | 2012 | 32 | 32 | 35.518 | 151 |
| location_7 | 2013 | 7 | 7 | 17.117 | 151 |
| location_7 | 2014 | 49 | 49 | 38.739 | 151 |
| location_7 | 2015 | 24 | 24 | 37.223 | 152 |
| location_7 | 2016 | 25 | 25 | 35.133 | 151 |
| location_7 | 2017 | 22 | 22 | 34.320 | 151 |
| location_7 | 2018 | 17 | 17 | 32.915 | 151 |
| location_7 | 2019 | 31 | 31 | 35.469 | 152 |
| location_7 | 2020 | 18 | 18 | 33.460 | 151 |
| location_7 | 2021 | 45 | 45 | 45.608 | 151 |
| location_7 | 2022 | 22 | 22 | 34.645 | 151 |
| location_7 | 2023 | 12 | 12 | 38.341 | 152 |
| location_7 | 2024 | 37 | 37 | 40.521 | 151 |
| location_7 | 2025 | 49 | 49 | 53.890 | 150 |
| location_8 | 1992 | 54 | 54 | 57.722 | 150 |
| location_8 | 1994 | 61 | 61 | 62.586 | 149 |
| location_8 | 1995 | 64 | 64 | 68.220 | 148 |
| location_8 | 2006 | 48 | 48 | 44.732 | 150 |
| location_8 | 2007 | 52 | 52 | 46.303 | 152 |
| location_8 | 2008 | 74 | 74 | 58.954 | 151 |
| location_8 | 2009 | 66 | 66 | 54.634 | 149 |
| location_8 | 2012 | 43 | 43 | 48.649 | 151 |
| location_8 | 2013 | 27 | 27 | 27.187 | 151 |
| location_8 | 2014 | 78 | 78 | 54.428 | 151 |
| location_8 | 2015 | 62 | 62 | 52.153 | 152 |
| location_8 | 2016 | 52 | 52 | 46.736 | 150 |
| location_8 | 2017 | 53 | 53 | 42.405 | 150 |
| location_8 | 2018 | 50 | 50 | 44.996 | 150 |
| location_8 | 2019 | 42 | 41 | 46.520 | 151 |
| location_8 | 2020 | 45 | 45 | 45.558 | 151 |
| location_8 | 2021 | 53 | 53 | 60.721 | 151 |
| location_8 | 2022 | 37 | 37 | 50.913 | 151 |
| location_8 | 2023 | 42 | 41 | 50.433 | 149 |
| location_8 | 2024 | 54 | 54 | 57.935 | 150 |
| location_8 | 2025 | 62 | 62 | 76.172 | 150 |
| location_9 | 1999 | 41 | 37 | 72.651 | 147 |
| location_9 | 2004 | 81 | 75 | 82.903 | 144 |
| location_9 | 2005 | 83 | 80 | 68.762 | 147 |
| location_9 | 2006 | 56 | 56 | 59.330 | 149 |
| location_9 | 2007 | 66 | 65 | 59.490 | 151 |
| location_9 | 2008 | 78 | 77 | 72.924 | 146 |
| location_9 | 2009 | 84 | 84 | 69.435 | 151 |
| location_9 | 2010 | 75 | 75 | 73.519 | 151 |
| location_9 | 2011 | 78 | 77 | 57.854 | 151 |
| location_9 | 2012 | 87 | 87 | 61.533 | 150 |
| location_9 | 2013 | 62 | 62 | 39.538 | 151 |
| location_9 | 2015 | 82 | 82 | 68.236 | 152 |
| location_9 | 2016 | 70 | 70 | 65.339 | 151 |
| location_9 | 2017 | 71 | 71 | 58.338 | 150 |
| location_9 | 2018 | 60 | 60 | 59.918 | 151 |
| location_9 | 2019 | 67 | 67 | 64.963 | 152 |
| location_9 | 2020 | 56 | 55 | 60.620 | 148 |
| location_9 | 2021 | 77 | 77 | 78.407 | 151 |
| location_9 | 2022 | 51 | 50 | 71.516 | 149 |
| location_9 | 2023 | 53 | 53 | 75.341 | 152 |
| location_9 | 2024 | 58 | 58 | 76.029 | 151 |

### Diagnostic fold availability

```json
[
  {
    "winter": 1980,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1981,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1982,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1983,
    "n_train": 23208,
    "n_test": 304,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1984,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1985,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1986,
    "n_train": 23211,
    "n_test": 301,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1987,
    "n_train": 23208,
    "n_test": 304,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1988,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1989,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1990,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1991,
    "n_train": 23208,
    "n_test": 304,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1992,
    "n_train": 23060,
    "n_test": 452,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1993,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1994,
    "n_train": 23212,
    "n_test": 300,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1995,
    "n_train": 23060,
    "n_test": 452,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1996,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1997,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1998,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 1999,
    "n_train": 22916,
    "n_test": 596,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2000,
    "n_train": 23211,
    "n_test": 301,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2001,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2002,
    "n_train": 23210,
    "n_test": 302,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2003,
    "n_train": 23064,
    "n_test": 448,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2004,
    "n_train": 23066,
    "n_test": 446,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2005,
    "n_train": 22915,
    "n_test": 597,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2006,
    "n_train": 22760,
    "n_test": 752,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2007,
    "n_train": 22753,
    "n_test": 759,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2008,
    "n_train": 22763,
    "n_test": 749,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2009,
    "n_train": 22759,
    "n_test": 753,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2010,
    "n_train": 22908,
    "n_test": 604,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2011,
    "n_train": 22908,
    "n_test": 604,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2012,
    "n_train": 22758,
    "n_test": 754,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2013,
    "n_train": 22757,
    "n_test": 755,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2014,
    "n_train": 22908,
    "n_test": 604,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2015,
    "n_train": 22752,
    "n_test": 760,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2016,
    "n_train": 22758,
    "n_test": 754,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2017,
    "n_train": 22759,
    "n_test": 753,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2018,
    "n_train": 22758,
    "n_test": 754,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2019,
    "n_train": 22753,
    "n_test": 759,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2020,
    "n_train": 22760,
    "n_test": 752,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2021,
    "n_train": 22757,
    "n_test": 755,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2022,
    "n_train": 22762,
    "n_test": 750,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2023,
    "n_train": 22755,
    "n_test": 757,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2024,
    "n_train": 22759,
    "n_test": 753,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  },
  {
    "winter": 2025,
    "n_train": 22914,
    "n_test": 598,
    "status": "ok",
    "weather_columns": [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "precipitation",
      "surface_pressure",
      "vapour_pressure_deficit",
      "wind_speed_10m",
      "wind_speed_100m",
      "wind_gusts_10m",
      "soil_temperature_0_to_7cm",
      "weather_code",
      "cloud_cover_low"
    ]
  }
]
```

## Coverage and measured NOx availability

```json
{
  "coverage": [
    {
      "site": "location_10",
      "winter": 1980,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 1.0,
        "1": 0.9986559139784946,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 1981,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 1.0,
        "1": 0.9986559139784946,
        "2": 0.9985119047619048,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 1982,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 1.0,
        "1": 1.0,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 1983,
      "valid": true,
      "month_coverage": {
        "11": 0.9944444444444445,
        "12": 0.9959677419354839,
        "1": 0.9986559139784946,
        "2": 0.9942528735632183,
        "3": 0.9932795698924731
      }
    },
    {
      "site": "location_10",
      "winter": 1984,
      "valid": true,
      "month_coverage": {
        "11": 0.9958333333333333,
        "12": 0.9932795698924731,
        "1": 0.9919354838709677,
        "2": 0.9880952380952381,
        "3": 0.9838709677419355
      }
    },
    {
      "site": "location_10",
      "winter": 1985,
      "valid": true,
      "month_coverage": {
        "11": 0.9916666666666667,
        "12": 0.9865591397849462,
        "1": 0.9731182795698925,
        "2": 0.9657738095238095,
        "3": 0.9731182795698925
      }
    },
    {
      "site": "location_10",
      "winter": 1986,
      "valid": true,
      "month_coverage": {
        "11": 0.9597222222222223,
        "12": 0.9623655913978495,
        "1": 0.9650537634408602,
        "2": 0.9568452380952381,
        "3": 0.978494623655914
      }
    },
    {
      "site": "location_10",
      "winter": 1987,
      "valid": true,
      "month_coverage": {
        "11": 0.9736111111111111,
        "12": 0.9771505376344086,
        "1": 0.9798387096774194,
        "2": 0.9698275862068966,
        "3": 0.9771505376344086
      }
    },
    {
      "site": "location_10",
      "winter": 1988,
      "valid": true,
      "month_coverage": {
        "11": 0.9722222222222222,
        "12": 0.9731182795698925,
        "1": 0.9771505376344086,
        "2": 0.9821428571428571,
        "3": 0.9690860215053764
      }
    },
    {
      "site": "location_10",
      "winter": 1989,
      "valid": true,
      "month_coverage": {
        "11": 0.9708333333333333,
        "12": 0.9744623655913979,
        "1": 0.9919354838709677,
        "2": 0.9910714285714286,
        "3": 0.9905913978494624
      }
    },
    {
      "site": "location_10",
      "winter": 1990,
      "valid": true,
      "month_coverage": {
        "11": 0.9777777777777777,
        "12": 0.9744623655913979,
        "1": 0.9865591397849462,
        "2": 0.9836309523809523,
        "3": 0.9932795698924731
      }
    },
    {
      "site": "location_10",
      "winter": 1991,
      "valid": true,
      "month_coverage": {
        "11": 0.9930555555555556,
        "12": 0.978494623655914,
        "1": 0.9865591397849462,
        "2": 0.978448275862069,
        "3": 0.9879032258064516
      }
    },
    {
      "site": "location_10",
      "winter": 1992,
      "valid": true,
      "month_coverage": {
        "11": 0.9861111111111112,
        "12": 0.9879032258064516,
        "1": 0.9852150537634409,
        "2": 0.9880952380952381,
        "3": 0.9758064516129032
      }
    },
    {
      "site": "location_10",
      "winter": 1993,
      "valid": true,
      "month_coverage": {
        "11": 0.9833333333333333,
        "12": 0.9758064516129032,
        "1": 0.9798387096774194,
        "2": 0.9747023809523809,
        "3": 0.9744623655913979
      }
    },
    {
      "site": "location_10",
      "winter": 1994,
      "valid": false,
      "month_coverage": {
        "11": 0.8527777777777777,
        "12": 0.967741935483871,
        "1": 0.9771505376344086,
        "2": 0.9568452380952381,
        "3": 0.875
      }
    },
    {
      "site": "location_10",
      "winter": 1995,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9986559139784946,
        "1": 1.0,
        "2": 0.9985632183908046,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 1996,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 0.9986559139784946,
        "1": 0.9973118279569892,
        "2": 0.9985119047619048,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 1997,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9973118279569892,
        "1": 0.9959677419354839,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 1998,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 0.9932795698924731,
        "1": 0.9986559139784946,
        "2": 1.0,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_10",
      "winter": 1999,
      "valid": true,
      "month_coverage": {
        "11": 0.9180555555555555,
        "12": 0.9220430107526881,
        "1": 0.9435483870967742,
        "2": 0.9583333333333334,
        "3": 0.9838709677419355
      }
    },
    {
      "site": "location_10",
      "winter": 2000,
      "valid": true,
      "month_coverage": {
        "11": 0.9458333333333333,
        "12": 0.9690860215053764,
        "1": 0.9838709677419355,
        "2": 0.9895833333333334,
        "3": 0.9811827956989247
      }
    },
    {
      "site": "location_10",
      "winter": 2001,
      "valid": true,
      "month_coverage": {
        "11": 0.9763888888888889,
        "12": 0.9758064516129032,
        "1": 0.956989247311828,
        "2": 0.9866071428571429,
        "3": 0.9932795698924731
      }
    },
    {
      "site": "location_10",
      "winter": 2002,
      "valid": true,
      "month_coverage": {
        "11": 0.9597222222222223,
        "12": 0.9838709677419355,
        "1": 0.9314516129032258,
        "2": 0.9657738095238095,
        "3": 0.9825268817204301
      }
    },
    {
      "site": "location_10",
      "winter": 2003,
      "valid": true,
      "month_coverage": {
        "11": 0.9805555555555555,
        "12": 0.989247311827957,
        "1": 0.967741935483871,
        "2": 0.9870689655172413,
        "3": 0.9919354838709677
      }
    },
    {
      "site": "location_10",
      "winter": 2004,
      "valid": true,
      "month_coverage": {
        "11": 0.9847222222222223,
        "12": 0.9502688172043011,
        "1": 0.9959677419354839,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 2005,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9986559139784946,
        "1": 0.9973118279569892,
        "2": 0.9851190476190477,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_10",
      "winter": 2006,
      "valid": true,
      "month_coverage": {
        "11": 0.9944444444444445,
        "12": 0.9973118279569892,
        "1": 0.9959677419354839,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 2007,
      "valid": true,
      "month_coverage": {
        "11": 0.9888888888888889,
        "12": 1.0,
        "1": 0.9973118279569892,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 2008,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 0.9771505376344086,
        "1": 0.9973118279569892,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 2009,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9959677419354839,
        "1": 0.9852150537634409,
        "2": 0.9985119047619048,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 2010,
      "valid": true,
      "month_coverage": {
        "11": 0.9972222222222222,
        "12": 0.9986559139784946,
        "1": 0.9932795698924731,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 2011,
      "valid": true,
      "month_coverage": {
        "11": 0.9902777777777778,
        "12": 0.9986559139784946,
        "1": 1.0,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 2012,
      "valid": true,
      "month_coverage": {
        "11": 0.9944444444444445,
        "12": 0.9973118279569892,
        "1": 0.9986559139784946,
        "2": 1.0,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_10",
      "winter": 2013,
      "valid": true,
      "month_coverage": {
        "11": 0.9972222222222222,
        "12": 1.0,
        "1": 1.0,
        "2": 1.0,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_10",
      "winter": 2014,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9946236559139785,
        "1": 0.9905913978494624,
        "2": 0.9970238095238095,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 2015,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 1.0,
        "1": 0.9946236559139785,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 2016,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 1.0,
        "1": 0.9959677419354839,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 2017,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 1.0,
        "1": 0.9973118279569892,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 2018,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9959677419354839,
        "1": 0.9986559139784946,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 2019,
      "valid": true,
      "month_coverage": {
        "11": 0.9930555555555556,
        "12": 0.9973118279569892,
        "1": 0.9973118279569892,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 2020,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 1.0,
        "1": 0.9986559139784946,
        "2": 1.0,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_10",
      "winter": 2021,
      "valid": true,
      "month_coverage": {
        "11": 0.9972222222222222,
        "12": 0.9946236559139785,
        "1": 0.9973118279569892,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 2022,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9946236559139785,
        "1": 1.0,
        "2": 0.9985119047619048,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_10",
      "winter": 2023,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 0.9986559139784946,
        "1": 0.9973118279569892,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 2024,
      "valid": true,
      "month_coverage": {
        "11": 0.9902777777777778,
        "12": 0.9865591397849462,
        "1": 1.0,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_10",
      "winter": 2025,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9798387096774194,
        "1": 0.9811827956989247,
        "2": 0.9925595238095238,
        "3": 0.9852150537634409
      }
    },
    {
      "site": "location_10",
      "winter": 2026,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_6",
      "winter": 1980,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_6",
      "winter": 1981,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_6",
      "winter": 1982,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_6",
      "winter": 1983,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_6",
      "winter": 1984,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_6",
      "winter": 1985,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_6",
      "winter": 1986,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_6",
      "winter": 1987,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_6",
      "winter": 1988,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_6",
      "winter": 1989,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_6",
      "winter": 1990,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_6",
      "winter": 1991,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_6",
      "winter": 1992,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_6",
      "winter": 1993,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_6",
      "winter": 1994,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_6",
      "winter": 1995,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_6",
      "winter": 1996,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_6",
      "winter": 1997,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_6",
      "winter": 1998,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.5470430107526881,
        "2": 0.10863095238095238,
        "3": 0.24596774193548387
      }
    },
    {
      "site": "location_6",
      "winter": 1999,
      "valid": true,
      "month_coverage": {
        "11": 0.9625,
        "12": 0.9529569892473119,
        "1": 0.9610215053763441,
        "2": 0.9310344827586207,
        "3": 0.9408602150537635
      }
    },
    {
      "site": "location_6",
      "winter": 2000,
      "valid": false,
      "month_coverage": {
        "11": 0.6555555555555556,
        "12": 0.7513440860215054,
        "1": 0.9301075268817204,
        "2": 0.9717261904761905,
        "3": 0.9758064516129032
      }
    },
    {
      "site": "location_6",
      "winter": 2001,
      "valid": false,
      "month_coverage": {
        "11": 0.9555555555555556,
        "12": 0.9153225806451613,
        "1": 0.9059139784946236,
        "2": 0.9017857142857143,
        "3": 0.8884408602150538
      }
    },
    {
      "site": "location_6",
      "winter": 2002,
      "valid": false,
      "month_coverage": {
        "11": 0.8694444444444445,
        "12": 0.9153225806451613,
        "1": 0.8696236559139785,
        "2": 0.9627976190476191,
        "3": 0.9529569892473119
      }
    },
    {
      "site": "location_6",
      "winter": 2003,
      "valid": true,
      "month_coverage": {
        "11": 0.9805555555555555,
        "12": 0.9018817204301075,
        "1": 0.9274193548387096,
        "2": 0.9770114942528736,
        "3": 0.9919354838709677
      }
    },
    {
      "site": "location_6",
      "winter": 2004,
      "valid": false,
      "month_coverage": {
        "11": 0.9611111111111111,
        "12": 0.8870967741935484,
        "1": 0.9986559139784946,
        "2": 0.9985119047619048,
        "3": 1.0
      }
    },
    {
      "site": "location_6",
      "winter": 2005,
      "valid": true,
      "month_coverage": {
        "11": 0.9944444444444445,
        "12": 0.9986559139784946,
        "1": 0.9435483870967742,
        "2": 1.0,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_6",
      "winter": 2006,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 1.0,
        "1": 0.9973118279569892,
        "2": 1.0,
        "3": 0.9973118279569892
      }
    },
    {
      "site": "location_6",
      "winter": 2007,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 1.0,
        "1": 0.9919354838709677,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_6",
      "winter": 2008,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9986559139784946,
        "1": 1.0,
        "2": 0.9851190476190477,
        "3": 1.0
      }
    },
    {
      "site": "location_6",
      "winter": 2009,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 1.0,
        "1": 0.9986559139784946,
        "2": 1.0,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_6",
      "winter": 2010,
      "valid": true,
      "month_coverage": {
        "11": 0.9972222222222222,
        "12": 1.0,
        "1": 1.0,
        "2": 1.0,
        "3": 0.9946236559139785
      }
    },
    {
      "site": "location_6",
      "winter": 2011,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 1.0,
        "1": 0.9596774193548387,
        "2": 0.9870689655172413,
        "3": 1.0
      }
    },
    {
      "site": "location_6",
      "winter": 2012,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 1.0,
        "1": 1.0,
        "2": 0.9970238095238095,
        "3": 1.0
      }
    },
    {
      "site": "location_6",
      "winter": 2013,
      "valid": true,
      "month_coverage": {
        "11": 0.9972222222222222,
        "12": 1.0,
        "1": 1.0,
        "2": 0.9985119047619048,
        "3": 1.0
      }
    },
    {
      "site": "location_6",
      "winter": 2014,
      "valid": true,
      "month_coverage": {
        "11": 0.9958333333333333,
        "12": 1.0,
        "1": 0.9959677419354839,
        "2": 0.9970238095238095,
        "3": 0.9973118279569892
      }
    },
    {
      "site": "location_6",
      "winter": 2015,
      "valid": true,
      "month_coverage": {
        "11": 0.9972222222222222,
        "12": 0.9946236559139785,
        "1": 1.0,
        "2": 0.9942528735632183,
        "3": 1.0
      }
    },
    {
      "site": "location_6",
      "winter": 2016,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 1.0,
        "1": 1.0,
        "2": 0.9985119047619048,
        "3": 1.0
      }
    },
    {
      "site": "location_6",
      "winter": 2017,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 0.9986559139784946,
        "1": 1.0,
        "2": 0.9985119047619048,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_6",
      "winter": 2018,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 1.0,
        "1": 1.0,
        "2": 0.9985119047619048,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_6",
      "winter": 2019,
      "valid": true,
      "month_coverage": {
        "11": 0.9972222222222222,
        "12": 1.0,
        "1": 1.0,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_6",
      "winter": 2020,
      "valid": true,
      "month_coverage": {
        "11": 0.9958333333333333,
        "12": 0.9959677419354839,
        "1": 1.0,
        "2": 0.9985119047619048,
        "3": 0.989247311827957
      }
    },
    {
      "site": "location_6",
      "winter": 2021,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 0.9959677419354839,
        "1": 0.9986559139784946,
        "2": 0.9985119047619048,
        "3": 1.0
      }
    },
    {
      "site": "location_6",
      "winter": 2022,
      "valid": true,
      "month_coverage": {
        "11": 0.9847222222222223,
        "12": 0.9744623655913979,
        "1": 1.0,
        "2": 0.9627976190476191,
        "3": 0.989247311827957
      }
    },
    {
      "site": "location_6",
      "winter": 2023,
      "valid": true,
      "month_coverage": {
        "11": 0.9972222222222222,
        "12": 1.0,
        "1": 1.0,
        "2": 0.9985632183908046,
        "3": 0.9946236559139785
      }
    },
    {
      "site": "location_6",
      "winter": 2024,
      "valid": true,
      "month_coverage": {
        "11": 0.9916666666666667,
        "12": 1.0,
        "1": 0.9959677419354839,
        "2": 0.9985119047619048,
        "3": 1.0
      }
    },
    {
      "site": "location_6",
      "winter": 2025,
      "valid": true,
      "month_coverage": {
        "11": 0.9805555555555555,
        "12": 1.0,
        "1": 0.9919354838709677,
        "2": 0.9747023809523809,
        "3": 0.9798387096774194
      }
    },
    {
      "site": "location_6",
      "winter": 2026,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_7",
      "winter": 1980,
      "valid": true,
      "month_coverage": {
        "11": 0.9972222222222222,
        "12": 0.9879032258064516,
        "1": 0.9959677419354839,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 1981,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 0.9973118279569892,
        "1": 1.0,
        "2": 0.9970238095238095,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 1982,
      "valid": true,
      "month_coverage": {
        "11": 0.9972222222222222,
        "12": 0.9973118279569892,
        "1": 0.9919354838709677,
        "2": 0.9985119047619048,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_7",
      "winter": 1983,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 0.9973118279569892,
        "1": 0.9986559139784946,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 1984,
      "valid": true,
      "month_coverage": {
        "11": 0.9972222222222222,
        "12": 0.9986559139784946,
        "1": 0.9959677419354839,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 1985,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9825268817204301,
        "1": 0.9905913978494624,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 1986,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9959677419354839,
        "1": 0.9946236559139785,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 1987,
      "valid": true,
      "month_coverage": {
        "11": 0.9958333333333333,
        "12": 0.9986559139784946,
        "1": 0.9946236559139785,
        "2": 0.9985632183908046,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 1988,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9959677419354839,
        "1": 0.9637096774193549,
        "2": 0.9732142857142857,
        "3": 0.9811827956989247
      }
    },
    {
      "site": "location_7",
      "winter": 1989,
      "valid": true,
      "month_coverage": {
        "11": 0.9763888888888889,
        "12": 0.9811827956989247,
        "1": 0.9959677419354839,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 1990,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 1.0,
        "1": 0.9986559139784946,
        "2": 0.9985119047619048,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 1991,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 0.9973118279569892,
        "1": 0.9932795698924731,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 1992,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 0.9946236559139785,
        "1": 0.9959677419354839,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 1993,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9986559139784946,
        "1": 0.9946236559139785,
        "2": 1.0,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_7",
      "winter": 1994,
      "valid": true,
      "month_coverage": {
        "11": 0.9958333333333333,
        "12": 0.9973118279569892,
        "1": 0.9973118279569892,
        "2": 0.9985119047619048,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 1995,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9986559139784946,
        "1": 0.9946236559139785,
        "2": 0.9942528735632183,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 1996,
      "valid": true,
      "month_coverage": {
        "11": 0.9958333333333333,
        "12": 0.9959677419354839,
        "1": 0.9973118279569892,
        "2": 0.9970238095238095,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 1997,
      "valid": true,
      "month_coverage": {
        "11": 0.9972222222222222,
        "12": 0.9973118279569892,
        "1": 0.9932795698924731,
        "2": 1.0,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_7",
      "winter": 1998,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 0.9973118279569892,
        "1": 0.9865591397849462,
        "2": 0.9985119047619048,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_7",
      "winter": 1999,
      "valid": true,
      "month_coverage": {
        "11": 0.9333333333333333,
        "12": 0.9341397849462365,
        "1": 0.9381720430107527,
        "2": 0.9583333333333334,
        "3": 0.9811827956989247
      }
    },
    {
      "site": "location_7",
      "winter": 2000,
      "valid": true,
      "month_coverage": {
        "11": 0.9611111111111111,
        "12": 0.9435483870967742,
        "1": 0.9731182795698925,
        "2": 0.9791666666666666,
        "3": 0.9838709677419355
      }
    },
    {
      "site": "location_7",
      "winter": 2001,
      "valid": true,
      "month_coverage": {
        "11": 0.9694444444444444,
        "12": 0.9731182795698925,
        "1": 0.9610215053763441,
        "2": 0.9717261904761905,
        "3": 0.9919354838709677
      }
    },
    {
      "site": "location_7",
      "winter": 2002,
      "valid": true,
      "month_coverage": {
        "11": 0.9416666666666667,
        "12": 0.967741935483871,
        "1": 0.9301075268817204,
        "2": 0.9702380952380952,
        "3": 0.9825268817204301
      }
    },
    {
      "site": "location_7",
      "winter": 2003,
      "valid": true,
      "month_coverage": {
        "11": 0.9902777777777778,
        "12": 0.9852150537634409,
        "1": 0.9704301075268817,
        "2": 0.9870689655172413,
        "3": 0.989247311827957
      }
    },
    {
      "site": "location_7",
      "winter": 2004,
      "valid": true,
      "month_coverage": {
        "11": 0.9763888888888889,
        "12": 0.9623655913978495,
        "1": 0.9986559139784946,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 2005,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9986559139784946,
        "1": 0.9986559139784946,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 2006,
      "valid": true,
      "month_coverage": {
        "11": 0.9972222222222222,
        "12": 1.0,
        "1": 0.9986559139784946,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 2007,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 0.9986559139784946,
        "1": 0.9973118279569892,
        "2": 0.9971264367816092,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 2008,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 0.9932795698924731,
        "1": 0.9986559139784946,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 2009,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9973118279569892,
        "1": 0.9905913978494624,
        "2": 0.9910714285714286,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 2010,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 0.9973118279569892,
        "1": 0.9905913978494624,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 2011,
      "valid": true,
      "month_coverage": {
        "11": 0.9902777777777778,
        "12": 0.9973118279569892,
        "1": 1.0,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 2012,
      "valid": true,
      "month_coverage": {
        "11": 0.9972222222222222,
        "12": 0.9932795698924731,
        "1": 0.9986559139784946,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 2013,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 1.0,
        "1": 1.0,
        "2": 0.9985119047619048,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 2014,
      "valid": true,
      "month_coverage": {
        "11": 0.9958333333333333,
        "12": 0.9986559139784946,
        "1": 0.9946236559139785,
        "2": 0.9985119047619048,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 2015,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9973118279569892,
        "1": 0.9973118279569892,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 2016,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9973118279569892,
        "1": 0.9986559139784946,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 2017,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9973118279569892,
        "1": 0.9946236559139785,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 2018,
      "valid": true,
      "month_coverage": {
        "11": 0.9916666666666667,
        "12": 1.0,
        "1": 1.0,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 2019,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9919354838709677,
        "1": 0.9973118279569892,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 2020,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9986559139784946,
        "1": 0.9986559139784946,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 2021,
      "valid": true,
      "month_coverage": {
        "11": 0.9888888888888889,
        "12": 0.9919354838709677,
        "1": 0.9959677419354839,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 2022,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 0.9905913978494624,
        "1": 0.9959677419354839,
        "2": 0.9985119047619048,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_7",
      "winter": 2023,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 1.0,
        "1": 0.9932795698924731,
        "2": 1.0,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_7",
      "winter": 2024,
      "valid": true,
      "month_coverage": {
        "11": 0.9972222222222222,
        "12": 0.989247311827957,
        "1": 0.9986559139784946,
        "2": 0.9985119047619048,
        "3": 1.0
      }
    },
    {
      "site": "location_7",
      "winter": 2025,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 0.9879032258064516,
        "1": 0.9838709677419355,
        "2": 0.9895833333333334,
        "3": 0.9865591397849462
      }
    },
    {
      "site": "location_7",
      "winter": 2026,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_8",
      "winter": 1980,
      "valid": false,
      "month_coverage": {
        "11": 0.10972222222222222,
        "12": 0.0013440860215053765,
        "1": 0.0,
        "2": 0.001488095238095238,
        "3": 0.005376344086021506
      }
    },
    {
      "site": "location_8",
      "winter": 1981,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.008064516129032258,
        "1": 0.006720430107526882,
        "2": 0.001488095238095238,
        "3": 0.0013440860215053765
      }
    },
    {
      "site": "location_8",
      "winter": 1982,
      "valid": false,
      "month_coverage": {
        "11": 0.019444444444444445,
        "12": 0.010752688172043012,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_8",
      "winter": 1983,
      "valid": false,
      "month_coverage": {
        "11": 0.6194444444444445,
        "12": 0.6290322580645161,
        "1": 0.668010752688172,
        "2": 0.6422413793103449,
        "3": 0.6384408602150538
      }
    },
    {
      "site": "location_8",
      "winter": 1984,
      "valid": false,
      "month_coverage": {
        "11": 0.6777777777777778,
        "12": 0.6518817204301075,
        "1": 0.6545698924731183,
        "2": 0.6636904761904762,
        "3": 0.6612903225806451
      }
    },
    {
      "site": "location_8",
      "winter": 1985,
      "valid": false,
      "month_coverage": {
        "11": 0.6597222222222222,
        "12": 0.646505376344086,
        "1": 0.657258064516129,
        "2": 0.6622023809523809,
        "3": 0.6653225806451613
      }
    },
    {
      "site": "location_8",
      "winter": 1986,
      "valid": false,
      "month_coverage": {
        "11": 0.6402777777777777,
        "12": 0.6532258064516129,
        "1": 0.625,
        "2": 0.625,
        "3": 0.6048387096774194
      }
    },
    {
      "site": "location_8",
      "winter": 1987,
      "valid": false,
      "month_coverage": {
        "11": 0.675,
        "12": 0.6559139784946236,
        "1": 0.7056451612903226,
        "2": 0.6882183908045977,
        "3": 0.6491935483870968
      }
    },
    {
      "site": "location_8",
      "winter": 1988,
      "valid": false,
      "month_coverage": {
        "11": 0.6583333333333333,
        "12": 0.6948924731182796,
        "1": 0.6801075268817204,
        "2": 0.6860119047619048,
        "3": 0.6653225806451613
      }
    },
    {
      "site": "location_8",
      "winter": 1989,
      "valid": false,
      "month_coverage": {
        "11": 0.6888888888888889,
        "12": 0.6975806451612904,
        "1": 0.6868279569892473,
        "2": 0.6904761904761905,
        "3": 0.6720430107526881
      }
    },
    {
      "site": "location_8",
      "winter": 1990,
      "valid": false,
      "month_coverage": {
        "11": 0.6166666666666667,
        "12": 0.6612903225806451,
        "1": 0.6935483870967742,
        "2": 0.6904761904761905,
        "3": 0.6895161290322581
      }
    },
    {
      "site": "location_8",
      "winter": 1991,
      "valid": false,
      "month_coverage": {
        "11": 0.325,
        "12": 0.4489247311827957,
        "1": 0.8911290322580645,
        "2": 0.5229885057471264,
        "3": 0.5577956989247311
      }
    },
    {
      "site": "location_8",
      "winter": 1992,
      "valid": true,
      "month_coverage": {
        "11": 0.9902777777777778,
        "12": 0.989247311827957,
        "1": 0.9690860215053764,
        "2": 0.9940476190476191,
        "3": 0.989247311827957
      }
    },
    {
      "site": "location_8",
      "winter": 1993,
      "valid": false,
      "month_coverage": {
        "11": 0.9958333333333333,
        "12": 0.9771505376344086,
        "1": 0.8817204301075269,
        "2": 0.8452380952380952,
        "3": 0.9865591397849462
      }
    },
    {
      "site": "location_8",
      "winter": 1994,
      "valid": true,
      "month_coverage": {
        "11": 0.9902777777777778,
        "12": 0.9798387096774194,
        "1": 0.9986559139784946,
        "2": 0.9925595238095238,
        "3": 0.9220430107526881
      }
    },
    {
      "site": "location_8",
      "winter": 1995,
      "valid": true,
      "month_coverage": {
        "11": 0.9527777777777777,
        "12": 0.9905913978494624,
        "1": 0.9946236559139785,
        "2": 0.9985632183908046,
        "3": 0.9596774193548387
      }
    },
    {
      "site": "location_8",
      "winter": 1996,
      "valid": false,
      "month_coverage": {
        "11": 0.7680555555555556,
        "12": 0.8145161290322581,
        "1": 0.6922043010752689,
        "2": 0.7752976190476191,
        "3": 0.6182795698924731
      }
    },
    {
      "site": "location_8",
      "winter": 1997,
      "valid": false,
      "month_coverage": {
        "11": 0.5833333333333334,
        "12": 0.7795698924731183,
        "1": 0.696236559139785,
        "2": 0.7113095238095238,
        "3": 0.7553763440860215
      }
    },
    {
      "site": "location_8",
      "winter": 1998,
      "valid": false,
      "month_coverage": {
        "11": 0.7611111111111111,
        "12": 0.6706989247311828,
        "1": 0.6841397849462365,
        "2": 0.5625,
        "3": 0.696236559139785
      }
    },
    {
      "site": "location_8",
      "winter": 1999,
      "valid": false,
      "month_coverage": {
        "11": 0.8819444444444444,
        "12": 0.8709677419354839,
        "1": 0.9650537634408602,
        "2": 0.9655172413793104,
        "3": 0.9758064516129032
      }
    },
    {
      "site": "location_8",
      "winter": 2000,
      "valid": false,
      "month_coverage": {
        "11": 0.6041666666666666,
        "12": 0.7540322580645161,
        "1": 0.9193548387096774,
        "2": 0.9389880952380952,
        "3": 0.9731182795698925
      }
    },
    {
      "site": "location_8",
      "winter": 2001,
      "valid": false,
      "month_coverage": {
        "11": 0.9152777777777777,
        "12": 0.5282258064516129,
        "1": 0.4650537634408602,
        "2": 0.20535714285714285,
        "3": 0.24059139784946237
      }
    },
    {
      "site": "location_8",
      "winter": 2002,
      "valid": false,
      "month_coverage": {
        "11": 0.5791666666666667,
        "12": 0.7029569892473119,
        "1": 0.728494623655914,
        "2": 0.8482142857142857,
        "3": 0.956989247311828
      }
    },
    {
      "site": "location_8",
      "winter": 2003,
      "valid": false,
      "month_coverage": {
        "11": 0.8972222222222223,
        "12": 0.9220430107526881,
        "1": 0.8548387096774194,
        "2": 0.8477011494252874,
        "3": 0.9838709677419355
      }
    },
    {
      "site": "location_8",
      "winter": 2004,
      "valid": false,
      "month_coverage": {
        "11": 0.9,
        "12": 0.8373655913978495,
        "1": 0.9825268817204301,
        "2": 0.7559523809523809,
        "3": 0.75
      }
    },
    {
      "site": "location_8",
      "winter": 2005,
      "valid": false,
      "month_coverage": {
        "11": 0.7638888888888888,
        "12": 0.6706989247311828,
        "1": 0.6559139784946236,
        "2": 0.9791666666666666,
        "3": 0.46908602150537637
      }
    },
    {
      "site": "location_8",
      "winter": 2006,
      "valid": true,
      "month_coverage": {
        "11": 0.9875,
        "12": 0.9771505376344086,
        "1": 0.9838709677419355,
        "2": 0.9910714285714286,
        "3": 0.9838709677419355
      }
    },
    {
      "site": "location_8",
      "winter": 2007,
      "valid": true,
      "month_coverage": {
        "11": 0.975,
        "12": 0.9825268817204301,
        "1": 1.0,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_8",
      "winter": 2008,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 1.0,
        "1": 1.0,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_8",
      "winter": 2009,
      "valid": true,
      "month_coverage": {
        "11": 0.9263888888888889,
        "12": 1.0,
        "1": 1.0,
        "2": 1.0,
        "3": 0.9932795698924731
      }
    },
    {
      "site": "location_8",
      "winter": 2010,
      "valid": false,
      "month_coverage": {
        "11": 0.8430555555555556,
        "12": 0.9973118279569892,
        "1": 0.946236559139785,
        "2": 0.9196428571428571,
        "3": 0.9838709677419355
      }
    },
    {
      "site": "location_8",
      "winter": 2011,
      "valid": false,
      "month_coverage": {
        "11": 0.9972222222222222,
        "12": 0.9596774193548387,
        "1": 0.9811827956989247,
        "2": 0.7313218390804598,
        "3": 0.9946236559139785
      }
    },
    {
      "site": "location_8",
      "winter": 2012,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9973118279569892,
        "1": 1.0,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_8",
      "winter": 2013,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 1.0,
        "1": 1.0,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_8",
      "winter": 2014,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 1.0,
        "1": 1.0,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_8",
      "winter": 2015,
      "valid": true,
      "month_coverage": {
        "11": 0.9805555555555555,
        "12": 0.9798387096774194,
        "1": 0.978494623655914,
        "2": 0.9827586206896551,
        "3": 0.9798387096774194
      }
    },
    {
      "site": "location_8",
      "winter": 2016,
      "valid": true,
      "month_coverage": {
        "11": 0.9847222222222223,
        "12": 0.9825268817204301,
        "1": 0.9690860215053764,
        "2": 0.9747023809523809,
        "3": 0.9838709677419355
      }
    },
    {
      "site": "location_8",
      "winter": 2017,
      "valid": true,
      "month_coverage": {
        "11": 0.9694444444444444,
        "12": 0.9744623655913979,
        "1": 0.9717741935483871,
        "2": 0.9717261904761905,
        "3": 0.9798387096774194
      }
    },
    {
      "site": "location_8",
      "winter": 2018,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 1.0,
        "1": 1.0,
        "2": 0.9821428571428571,
        "3": 0.9852150537634409
      }
    },
    {
      "site": "location_8",
      "winter": 2019,
      "valid": true,
      "month_coverage": {
        "11": 0.9902777777777778,
        "12": 0.9905913978494624,
        "1": 0.9959677419354839,
        "2": 0.9985632183908046,
        "3": 1.0
      }
    },
    {
      "site": "location_8",
      "winter": 2020,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9973118279569892,
        "1": 0.9986559139784946,
        "2": 1.0,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_8",
      "winter": 2021,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 1.0,
        "1": 0.9986559139784946,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_8",
      "winter": 2022,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9986559139784946,
        "1": 0.9973118279569892,
        "2": 1.0,
        "3": 1.0
      }
    },
    {
      "site": "location_8",
      "winter": 2023,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9973118279569892,
        "1": 1.0,
        "2": 0.9310344827586207,
        "3": 0.9932795698924731
      }
    },
    {
      "site": "location_8",
      "winter": 2024,
      "valid": true,
      "month_coverage": {
        "11": 0.9944444444444445,
        "12": 0.9690860215053764,
        "1": 1.0,
        "2": 0.9970238095238095,
        "3": 0.9946236559139785
      }
    },
    {
      "site": "location_8",
      "winter": 2025,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 1.0,
        "1": 0.9919354838709677,
        "2": 0.9910714285714286,
        "3": 0.9825268817204301
      }
    },
    {
      "site": "location_8",
      "winter": 2026,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_9",
      "winter": 1980,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_9",
      "winter": 1981,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_9",
      "winter": 1982,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_9",
      "winter": 1983,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_9",
      "winter": 1984,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_9",
      "winter": 1985,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_9",
      "winter": 1986,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_9",
      "winter": 1987,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_9",
      "winter": 1988,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_9",
      "winter": 1989,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_9",
      "winter": 1990,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_9",
      "winter": 1991,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_9",
      "winter": 1992,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_9",
      "winter": 1993,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_9",
      "winter": 1994,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_9",
      "winter": 1995,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_9",
      "winter": 1996,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_9",
      "winter": 1997,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    },
    {
      "site": "location_9",
      "winter": 1998,
      "valid": false,
      "month_coverage": {
        "11": 0.9722222222222222,
        "12": 0.9327956989247311,
        "1": 0.8924731182795699,
        "2": 0.9761904761904762,
        "3": 0.9879032258064516
      }
    },
    {
      "site": "location_9",
      "winter": 1999,
      "valid": true,
      "month_coverage": {
        "11": 0.9361111111111111,
        "12": 0.9435483870967742,
        "1": 0.9045698924731183,
        "2": 0.9597701149425287,
        "3": 0.9610215053763441
      }
    },
    {
      "site": "location_9",
      "winter": 2000,
      "valid": false,
      "month_coverage": {
        "11": 0.9555555555555556,
        "12": 0.8373655913978495,
        "1": 0.8776881720430108,
        "2": 0.9613095238095238,
        "3": 0.9045698924731183
      }
    },
    {
      "site": "location_9",
      "winter": 2001,
      "valid": false,
      "month_coverage": {
        "11": 0.8194444444444444,
        "12": 0.9354838709677419,
        "1": 0.9327956989247311,
        "2": 0.9508928571428571,
        "3": 0.9744623655913979
      }
    },
    {
      "site": "location_9",
      "winter": 2002,
      "valid": false,
      "month_coverage": {
        "11": 0.8236111111111111,
        "12": 0.9314516129032258,
        "1": 0.8467741935483871,
        "2": 0.9508928571428571,
        "3": 0.9556451612903226
      }
    },
    {
      "site": "location_9",
      "winter": 2003,
      "valid": false,
      "month_coverage": {
        "11": 0.9833333333333333,
        "12": 0.7943548387096774,
        "1": 0.9435483870967742,
        "2": 0.9813218390804598,
        "3": 0.9811827956989247
      }
    },
    {
      "site": "location_9",
      "winter": 2004,
      "valid": true,
      "month_coverage": {
        "11": 0.9291666666666667,
        "12": 0.9099462365591398,
        "1": 0.9986559139784946,
        "2": 1.0,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_9",
      "winter": 2005,
      "valid": true,
      "month_coverage": {
        "11": 0.9958333333333333,
        "12": 0.9973118279569892,
        "1": 0.9045698924731183,
        "2": 0.9910714285714286,
        "3": 0.9973118279569892
      }
    },
    {
      "site": "location_9",
      "winter": 2006,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9973118279569892,
        "1": 0.9556451612903226,
        "2": 0.9985119047619048,
        "3": 0.9959677419354839
      }
    },
    {
      "site": "location_9",
      "winter": 2007,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 1.0,
        "1": 0.9771505376344086,
        "2": 0.9942528735632183,
        "3": 0.9959677419354839
      }
    },
    {
      "site": "location_9",
      "winter": 2008,
      "valid": true,
      "month_coverage": {
        "11": 0.9166666666666666,
        "12": 0.9986559139784946,
        "1": 1.0,
        "2": 0.9747023809523809,
        "3": 0.9717741935483871
      }
    },
    {
      "site": "location_9",
      "winter": 2009,
      "valid": true,
      "month_coverage": {
        "11": 0.9805555555555555,
        "12": 1.0,
        "1": 0.9986559139784946,
        "2": 0.9940476190476191,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_9",
      "winter": 2010,
      "valid": true,
      "month_coverage": {
        "11": 0.9972222222222222,
        "12": 1.0,
        "1": 1.0,
        "2": 0.9985119047619048,
        "3": 1.0
      }
    },
    {
      "site": "location_9",
      "winter": 2011,
      "valid": true,
      "month_coverage": {
        "11": 0.9847222222222223,
        "12": 0.9986559139784946,
        "1": 1.0,
        "2": 0.9985632183908046,
        "3": 1.0
      }
    },
    {
      "site": "location_9",
      "winter": 2012,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 0.9986559139784946,
        "1": 1.0,
        "2": 0.9985119047619048,
        "3": 0.9704301075268817
      }
    },
    {
      "site": "location_9",
      "winter": 2013,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 1.0,
        "1": 1.0,
        "2": 0.9970238095238095,
        "3": 1.0
      }
    },
    {
      "site": "location_9",
      "winter": 2014,
      "valid": false,
      "month_coverage": {
        "11": 0.9763888888888889,
        "12": 0.4650537634408602,
        "1": 1.0,
        "2": 0.9970238095238095,
        "3": 1.0
      }
    },
    {
      "site": "location_9",
      "winter": 2015,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 1.0,
        "1": 1.0,
        "2": 0.9985632183908046,
        "3": 1.0
      }
    },
    {
      "site": "location_9",
      "winter": 2016,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 1.0,
        "1": 0.9986559139784946,
        "2": 0.9985119047619048,
        "3": 1.0
      }
    },
    {
      "site": "location_9",
      "winter": 2017,
      "valid": true,
      "month_coverage": {
        "11": 0.9958333333333333,
        "12": 0.9905913978494624,
        "1": 1.0,
        "2": 0.9985119047619048,
        "3": 1.0
      }
    },
    {
      "site": "location_9",
      "winter": 2018,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9986559139784946,
        "1": 1.0,
        "2": 1.0,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_9",
      "winter": 2019,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 1.0,
        "1": 1.0,
        "2": 0.9985632183908046,
        "3": 1.0
      }
    },
    {
      "site": "location_9",
      "winter": 2020,
      "valid": true,
      "month_coverage": {
        "11": 0.9722222222222222,
        "12": 1.0,
        "1": 0.9932795698924731,
        "2": 0.9880952380952381,
        "3": 1.0
      }
    },
    {
      "site": "location_9",
      "winter": 2021,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 1.0,
        "1": 0.9986559139784946,
        "2": 0.9985119047619048,
        "3": 1.0
      }
    },
    {
      "site": "location_9",
      "winter": 2022,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 0.9865591397849462,
        "1": 1.0,
        "2": 0.9761904761904762,
        "3": 0.9959677419354839
      }
    },
    {
      "site": "location_9",
      "winter": 2023,
      "valid": true,
      "month_coverage": {
        "11": 0.9986111111111111,
        "12": 1.0,
        "1": 1.0,
        "2": 0.9971264367816092,
        "3": 0.9932795698924731
      }
    },
    {
      "site": "location_9",
      "winter": 2024,
      "valid": true,
      "month_coverage": {
        "11": 1.0,
        "12": 1.0,
        "1": 0.9973118279569892,
        "2": 1.0,
        "3": 0.9986559139784946
      }
    },
    {
      "site": "location_9",
      "winter": 2025,
      "valid": false,
      "month_coverage": {
        "11": 0.8847222222222222,
        "12": 0.8924731182795699,
        "1": 0.9946236559139785,
        "2": 0.9895833333333334,
        "3": 0.9852150537634409
      }
    },
    {
      "site": "location_9",
      "winter": 2026,
      "valid": false,
      "month_coverage": {
        "11": 0.0,
        "12": 0.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0
      }
    }
  ],
  "nox": {
    "location_6": {
      "status": "ok",
      "valid_winters": 28,
      "reason": null
    },
    "location_7": {
      "status": "ok",
      "valid_winters": 36,
      "reason": null
    },
    "location_8": {
      "status": "ok",
      "valid_winters": 35,
      "reason": null
    },
    "location_9": {
      "status": "ok",
      "valid_winters": 35,
      "reason": null
    },
    "location_10": {
      "status": "ok",
      "valid_winters": 35,
      "reason": null
    }
  }
}
```

## Gray optimal-day replication (1980–2026 by default)

Optimal: daily mean DPD <4.2 °C and daily mean wind <7.2 km/h. Undefined shares are null, not zero. Pooled trends demean both axes within site.

```json
{
  "winters": [
    {
      "site": "location_10",
      "winter": 1980,
      "n_days": 151,
      "n_fog": 52,
      "n_optimal": 26,
      "n_both": 10,
      "fog_share_of_optimal": 0.38461538461538464,
      "optimal_share_of_fog": 0.19230769230769232
    },
    {
      "site": "location_10",
      "winter": 1981,
      "n_days": 151,
      "n_fog": 33,
      "n_optimal": 38,
      "n_both": 11,
      "fog_share_of_optimal": 0.2894736842105263,
      "optimal_share_of_fog": 0.3333333333333333
    },
    {
      "site": "location_10",
      "winter": 1982,
      "n_days": 151,
      "n_fog": 46,
      "n_optimal": 67,
      "n_both": 24,
      "fog_share_of_optimal": 0.3582089552238806,
      "optimal_share_of_fog": 0.5217391304347826
    },
    {
      "site": "location_10",
      "winter": 1983,
      "n_days": 152,
      "n_fog": 38,
      "n_optimal": 37,
      "n_both": 17,
      "fog_share_of_optimal": 0.4594594594594595,
      "optimal_share_of_fog": 0.4473684210526316
    },
    {
      "site": "location_10",
      "winter": 1984,
      "n_days": 151,
      "n_fog": 64,
      "n_optimal": 43,
      "n_both": 32,
      "fog_share_of_optimal": 0.7441860465116279,
      "optimal_share_of_fog": 0.5
    },
    {
      "site": "location_10",
      "winter": 1985,
      "n_days": 151,
      "n_fog": 61,
      "n_optimal": 53,
      "n_both": 23,
      "fog_share_of_optimal": 0.4339622641509434,
      "optimal_share_of_fog": 0.3770491803278688
    },
    {
      "site": "location_10",
      "winter": 1986,
      "n_days": 151,
      "n_fog": 37,
      "n_optimal": 33,
      "n_both": 19,
      "fog_share_of_optimal": 0.5757575757575758,
      "optimal_share_of_fog": 0.5135135135135135
    },
    {
      "site": "location_10",
      "winter": 1987,
      "n_days": 152,
      "n_fog": 36,
      "n_optimal": 41,
      "n_both": 22,
      "fog_share_of_optimal": 0.5365853658536586,
      "optimal_share_of_fog": 0.6111111111111112
    },
    {
      "site": "location_10",
      "winter": 1988,
      "n_days": 151,
      "n_fog": 44,
      "n_optimal": 35,
      "n_both": 20,
      "fog_share_of_optimal": 0.5714285714285714,
      "optimal_share_of_fog": 0.45454545454545453
    },
    {
      "site": "location_10",
      "winter": 1989,
      "n_days": 151,
      "n_fog": 36,
      "n_optimal": 9,
      "n_both": 3,
      "fog_share_of_optimal": 0.3333333333333333,
      "optimal_share_of_fog": 0.08333333333333333
    },
    {
      "site": "location_10",
      "winter": 1990,
      "n_days": 151,
      "n_fog": 27,
      "n_optimal": 18,
      "n_both": 11,
      "fog_share_of_optimal": 0.6111111111111112,
      "optimal_share_of_fog": 0.4074074074074074
    },
    {
      "site": "location_10",
      "winter": 1991,
      "n_days": 152,
      "n_fog": 47,
      "n_optimal": 32,
      "n_both": 17,
      "fog_share_of_optimal": 0.53125,
      "optimal_share_of_fog": 0.3617021276595745
    },
    {
      "site": "location_10",
      "winter": 1992,
      "n_days": 151,
      "n_fog": 29,
      "n_optimal": 49,
      "n_both": 15,
      "fog_share_of_optimal": 0.30612244897959184,
      "optimal_share_of_fog": 0.5172413793103449
    },
    {
      "site": "location_10",
      "winter": 1993,
      "n_days": 151,
      "n_fog": 33,
      "n_optimal": 19,
      "n_both": 7,
      "fog_share_of_optimal": 0.3684210526315789,
      "optimal_share_of_fog": 0.21212121212121213
    },
    {
      "site": "location_10",
      "winter": 1995,
      "n_days": 152,
      "n_fog": 34,
      "n_optimal": 53,
      "n_both": 16,
      "fog_share_of_optimal": 0.3018867924528302,
      "optimal_share_of_fog": 0.47058823529411764
    },
    {
      "site": "location_10",
      "winter": 1996,
      "n_days": 151,
      "n_fog": 37,
      "n_optimal": 66,
      "n_both": 24,
      "fog_share_of_optimal": 0.36363636363636365,
      "optimal_share_of_fog": 0.6486486486486487
    },
    {
      "site": "location_10",
      "winter": 1997,
      "n_days": 151,
      "n_fog": 36,
      "n_optimal": 76,
      "n_both": 23,
      "fog_share_of_optimal": 0.3026315789473684,
      "optimal_share_of_fog": 0.6388888888888888
    },
    {
      "site": "location_10",
      "winter": 1998,
      "n_days": 151,
      "n_fog": 37,
      "n_optimal": 57,
      "n_both": 14,
      "fog_share_of_optimal": 0.24561403508771928,
      "optimal_share_of_fog": 0.3783783783783784
    },
    {
      "site": "location_10",
      "winter": 1999,
      "n_days": 152,
      "n_fog": 8,
      "n_optimal": 38,
      "n_both": 3,
      "fog_share_of_optimal": 0.07894736842105263,
      "optimal_share_of_fog": 0.375
    },
    {
      "site": "location_10",
      "winter": 2000,
      "n_days": 151,
      "n_fog": 26,
      "n_optimal": 28,
      "n_both": 6,
      "fog_share_of_optimal": 0.21428571428571427,
      "optimal_share_of_fog": 0.23076923076923078
    },
    {
      "site": "location_10",
      "winter": 2001,
      "n_days": 151,
      "n_fog": 41,
      "n_optimal": 41,
      "n_both": 19,
      "fog_share_of_optimal": 0.4634146341463415,
      "optimal_share_of_fog": 0.4634146341463415
    },
    {
      "site": "location_10",
      "winter": 2002,
      "n_days": 151,
      "n_fog": 39,
      "n_optimal": 46,
      "n_both": 16,
      "fog_share_of_optimal": 0.34782608695652173,
      "optimal_share_of_fog": 0.41025641025641024
    },
    {
      "site": "location_10",
      "winter": 2003,
      "n_days": 152,
      "n_fog": 23,
      "n_optimal": 51,
      "n_both": 13,
      "fog_share_of_optimal": 0.2549019607843137,
      "optimal_share_of_fog": 0.5652173913043478
    },
    {
      "site": "location_10",
      "winter": 2004,
      "n_days": 151,
      "n_fog": 38,
      "n_optimal": 69,
      "n_both": 19,
      "fog_share_of_optimal": 0.2753623188405797,
      "optimal_share_of_fog": 0.5
    },
    {
      "site": "location_10",
      "winter": 2005,
      "n_days": 151,
      "n_fog": 29,
      "n_optimal": 22,
      "n_both": 6,
      "fog_share_of_optimal": 0.2727272727272727,
      "optimal_share_of_fog": 0.20689655172413793
    },
    {
      "site": "location_10",
      "winter": 2006,
      "n_days": 151,
      "n_fog": 15,
      "n_optimal": 16,
      "n_both": 6,
      "fog_share_of_optimal": 0.375,
      "optimal_share_of_fog": 0.4
    },
    {
      "site": "location_10",
      "winter": 2007,
      "n_days": 152,
      "n_fog": 20,
      "n_optimal": 23,
      "n_both": 7,
      "fog_share_of_optimal": 0.30434782608695654,
      "optimal_share_of_fog": 0.35
    },
    {
      "site": "location_10",
      "winter": 2008,
      "n_days": 151,
      "n_fog": 29,
      "n_optimal": 28,
      "n_both": 9,
      "fog_share_of_optimal": 0.32142857142857145,
      "optimal_share_of_fog": 0.3103448275862069
    },
    {
      "site": "location_10",
      "winter": 2009,
      "n_days": 151,
      "n_fog": 25,
      "n_optimal": 39,
      "n_both": 9,
      "fog_share_of_optimal": 0.23076923076923078,
      "optimal_share_of_fog": 0.36
    },
    {
      "site": "location_10",
      "winter": 2010,
      "n_days": 151,
      "n_fog": 24,
      "n_optimal": 45,
      "n_both": 15,
      "fog_share_of_optimal": 0.3333333333333333,
      "optimal_share_of_fog": 0.625
    },
    {
      "site": "location_10",
      "winter": 2011,
      "n_days": 152,
      "n_fog": 8,
      "n_optimal": 14,
      "n_both": 2,
      "fog_share_of_optimal": 0.14285714285714285,
      "optimal_share_of_fog": 0.25
    },
    {
      "site": "location_10",
      "winter": 2012,
      "n_days": 151,
      "n_fog": 12,
      "n_optimal": 18,
      "n_both": 6,
      "fog_share_of_optimal": 0.3333333333333333,
      "optimal_share_of_fog": 0.5
    },
    {
      "site": "location_10",
      "winter": 2013,
      "n_days": 151,
      "n_fog": 3,
      "n_optimal": 4,
      "n_both": 2,
      "fog_share_of_optimal": 0.5,
      "optimal_share_of_fog": 0.6666666666666666
    },
    {
      "site": "location_10",
      "winter": 2014,
      "n_days": 151,
      "n_fog": 33,
      "n_optimal": 11,
      "n_both": 6,
      "fog_share_of_optimal": 0.5454545454545454,
      "optimal_share_of_fog": 0.18181818181818182
    },
    {
      "site": "location_10",
      "winter": 2015,
      "n_days": 152,
      "n_fog": 11,
      "n_optimal": 26,
      "n_both": 3,
      "fog_share_of_optimal": 0.11538461538461539,
      "optimal_share_of_fog": 0.2727272727272727
    },
    {
      "site": "location_10",
      "winter": 2016,
      "n_days": 151,
      "n_fog": 13,
      "n_optimal": 34,
      "n_both": 7,
      "fog_share_of_optimal": 0.20588235294117646,
      "optimal_share_of_fog": 0.5384615384615384
    },
    {
      "site": "location_10",
      "winter": 2017,
      "n_days": 151,
      "n_fog": 12,
      "n_optimal": 3,
      "n_both": 1,
      "fog_share_of_optimal": 0.3333333333333333,
      "optimal_share_of_fog": 0.08333333333333333
    },
    {
      "site": "location_10",
      "winter": 2018,
      "n_days": 151,
      "n_fog": 19,
      "n_optimal": 22,
      "n_both": 8,
      "fog_share_of_optimal": 0.36363636363636365,
      "optimal_share_of_fog": 0.42105263157894735
    },
    {
      "site": "location_10",
      "winter": 2019,
      "n_days": 152,
      "n_fog": 11,
      "n_optimal": 24,
      "n_both": 9,
      "fog_share_of_optimal": 0.375,
      "optimal_share_of_fog": 0.8181818181818182
    },
    {
      "site": "location_10",
      "winter": 2020,
      "n_days": 151,
      "n_fog": 9,
      "n_optimal": 11,
      "n_both": 4,
      "fog_share_of_optimal": 0.36363636363636365,
      "optimal_share_of_fog": 0.4444444444444444
    },
    {
      "site": "location_10",
      "winter": 2021,
      "n_days": 151,
      "n_fog": 27,
      "n_optimal": 26,
      "n_both": 10,
      "fog_share_of_optimal": 0.38461538461538464,
      "optimal_share_of_fog": 0.37037037037037035
    },
    {
      "site": "location_10",
      "winter": 2022,
      "n_days": 151,
      "n_fog": 14,
      "n_optimal": 40,
      "n_both": 10,
      "fog_share_of_optimal": 0.25,
      "optimal_share_of_fog": 0.7142857142857143
    },
    {
      "site": "location_10",
      "winter": 2023,
      "n_days": 152,
      "n_fog": 15,
      "n_optimal": 32,
      "n_both": 8,
      "fog_share_of_optimal": 0.25,
      "optimal_share_of_fog": 0.5333333333333333
    },
    {
      "site": "location_10",
      "winter": 2024,
      "n_days": 151,
      "n_fog": 17,
      "n_optimal": 50,
      "n_both": 12,
      "fog_share_of_optimal": 0.24,
      "optimal_share_of_fog": 0.7058823529411765
    },
    {
      "site": "location_10",
      "winter": 2025,
      "n_days": 151,
      "n_fog": 43,
      "n_optimal": 58,
      "n_both": 29,
      "fog_share_of_optimal": 0.5,
      "optimal_share_of_fog": 0.6744186046511628
    },
    {
      "site": "location_6",
      "winter": 1999,
      "n_days": 152,
      "n_fog": 44,
      "n_optimal": 28,
      "n_both": 12,
      "fog_share_of_optimal": 0.42857142857142855,
      "optimal_share_of_fog": 0.2727272727272727
    },
    {
      "site": "location_6",
      "winter": 2003,
      "n_days": 152,
      "n_fog": 50,
      "n_optimal": 38,
      "n_both": 18,
      "fog_share_of_optimal": 0.47368421052631576,
      "optimal_share_of_fog": 0.36
    },
    {
      "site": "location_6",
      "winter": 2005,
      "n_days": 151,
      "n_fog": 58,
      "n_optimal": 37,
      "n_both": 21,
      "fog_share_of_optimal": 0.5675675675675675,
      "optimal_share_of_fog": 0.3620689655172414
    },
    {
      "site": "location_6",
      "winter": 2006,
      "n_days": 151,
      "n_fog": 42,
      "n_optimal": 18,
      "n_both": 12,
      "fog_share_of_optimal": 0.6666666666666666,
      "optimal_share_of_fog": 0.2857142857142857
    },
    {
      "site": "location_6",
      "winter": 2007,
      "n_days": 152,
      "n_fog": 41,
      "n_optimal": 24,
      "n_both": 16,
      "fog_share_of_optimal": 0.6666666666666666,
      "optimal_share_of_fog": 0.3902439024390244
    },
    {
      "site": "location_6",
      "winter": 2008,
      "n_days": 151,
      "n_fog": 50,
      "n_optimal": 31,
      "n_both": 18,
      "fog_share_of_optimal": 0.5806451612903226,
      "optimal_share_of_fog": 0.36
    },
    {
      "site": "location_6",
      "winter": 2009,
      "n_days": 151,
      "n_fog": 52,
      "n_optimal": 51,
      "n_both": 32,
      "fog_share_of_optimal": 0.6274509803921569,
      "optimal_share_of_fog": 0.6153846153846154
    },
    {
      "site": "location_6",
      "winter": 2010,
      "n_days": 151,
      "n_fog": 45,
      "n_optimal": 53,
      "n_both": 32,
      "fog_share_of_optimal": 0.6037735849056604,
      "optimal_share_of_fog": 0.7111111111111111
    },
    {
      "site": "location_6",
      "winter": 2011,
      "n_days": 152,
      "n_fog": 50,
      "n_optimal": 12,
      "n_both": 8,
      "fog_share_of_optimal": 0.6666666666666666,
      "optimal_share_of_fog": 0.16
    },
    {
      "site": "location_6",
      "winter": 2012,
      "n_days": 151,
      "n_fog": 44,
      "n_optimal": 41,
      "n_both": 22,
      "fog_share_of_optimal": 0.5365853658536586,
      "optimal_share_of_fog": 0.5
    },
    {
      "site": "location_6",
      "winter": 2013,
      "n_days": 151,
      "n_fog": 17,
      "n_optimal": 3,
      "n_both": 1,
      "fog_share_of_optimal": 0.3333333333333333,
      "optimal_share_of_fog": 0.058823529411764705
    },
    {
      "site": "location_6",
      "winter": 2014,
      "n_days": 151,
      "n_fog": 68,
      "n_optimal": 35,
      "n_both": 26,
      "fog_share_of_optimal": 0.7428571428571429,
      "optimal_share_of_fog": 0.38235294117647056
    },
    {
      "site": "location_6",
      "winter": 2015,
      "n_days": 152,
      "n_fog": 42,
      "n_optimal": 36,
      "n_both": 17,
      "fog_share_of_optimal": 0.4722222222222222,
      "optimal_share_of_fog": 0.40476190476190477
    },
    {
      "site": "location_6",
      "winter": 2016,
      "n_days": 151,
      "n_fog": 35,
      "n_optimal": 45,
      "n_both": 13,
      "fog_share_of_optimal": 0.28888888888888886,
      "optimal_share_of_fog": 0.37142857142857144
    },
    {
      "site": "location_6",
      "winter": 2017,
      "n_days": 151,
      "n_fog": 46,
      "n_optimal": 36,
      "n_both": 22,
      "fog_share_of_optimal": 0.6111111111111112,
      "optimal_share_of_fog": 0.4782608695652174
    },
    {
      "site": "location_6",
      "winter": 2018,
      "n_days": 151,
      "n_fog": 41,
      "n_optimal": 44,
      "n_both": 21,
      "fog_share_of_optimal": 0.4772727272727273,
      "optimal_share_of_fog": 0.5121951219512195
    },
    {
      "site": "location_6",
      "winter": 2019,
      "n_days": 152,
      "n_fog": 50,
      "n_optimal": 35,
      "n_both": 23,
      "fog_share_of_optimal": 0.6571428571428571,
      "optimal_share_of_fog": 0.46
    },
    {
      "site": "location_6",
      "winter": 2020,
      "n_days": 151,
      "n_fog": 37,
      "n_optimal": 32,
      "n_both": 21,
      "fog_share_of_optimal": 0.65625,
      "optimal_share_of_fog": 0.5675675675675675
    },
    {
      "site": "location_6",
      "winter": 2021,
      "n_days": 151,
      "n_fog": 61,
      "n_optimal": 53,
      "n_both": 38,
      "fog_share_of_optimal": 0.7169811320754716,
      "optimal_share_of_fog": 0.6229508196721312
    },
    {
      "site": "location_6",
      "winter": 2022,
      "n_days": 151,
      "n_fog": 29,
      "n_optimal": 42,
      "n_both": 20,
      "fog_share_of_optimal": 0.47619047619047616,
      "optimal_share_of_fog": 0.6896551724137931
    },
    {
      "site": "location_6",
      "winter": 2023,
      "n_days": 152,
      "n_fog": 35,
      "n_optimal": 48,
      "n_both": 19,
      "fog_share_of_optimal": 0.3958333333333333,
      "optimal_share_of_fog": 0.5428571428571428
    },
    {
      "site": "location_6",
      "winter": 2024,
      "n_days": 151,
      "n_fog": 59,
      "n_optimal": 39,
      "n_both": 30,
      "fog_share_of_optimal": 0.7692307692307693,
      "optimal_share_of_fog": 0.5084745762711864
    },
    {
      "site": "location_6",
      "winter": 2025,
      "n_days": 151,
      "n_fog": 68,
      "n_optimal": 78,
      "n_both": 52,
      "fog_share_of_optimal": 0.6666666666666666,
      "optimal_share_of_fog": 0.7647058823529411
    },
    {
      "site": "location_7",
      "winter": 1980,
      "n_days": 151,
      "n_fog": 63,
      "n_optimal": 50,
      "n_both": 31,
      "fog_share_of_optimal": 0.62,
      "optimal_share_of_fog": 0.49206349206349204
    },
    {
      "site": "location_7",
      "winter": 1981,
      "n_days": 151,
      "n_fog": 61,
      "n_optimal": 77,
      "n_both": 42,
      "fog_share_of_optimal": 0.5454545454545454,
      "optimal_share_of_fog": 0.6885245901639344
    },
    {
      "site": "location_7",
      "winter": 1982,
      "n_days": 151,
      "n_fog": 58,
      "n_optimal": 78,
      "n_both": 43,
      "fog_share_of_optimal": 0.5512820512820513,
      "optimal_share_of_fog": 0.7413793103448276
    },
    {
      "site": "location_7",
      "winter": 1983,
      "n_days": 152,
      "n_fog": 59,
      "n_optimal": 56,
      "n_both": 35,
      "fog_share_of_optimal": 0.625,
      "optimal_share_of_fog": 0.5932203389830508
    },
    {
      "site": "location_7",
      "winter": 1984,
      "n_days": 151,
      "n_fog": 58,
      "n_optimal": 76,
      "n_both": 48,
      "fog_share_of_optimal": 0.631578947368421,
      "optimal_share_of_fog": 0.8275862068965517
    },
    {
      "site": "location_7",
      "winter": 1985,
      "n_days": 151,
      "n_fog": 63,
      "n_optimal": 74,
      "n_both": 44,
      "fog_share_of_optimal": 0.5945945945945946,
      "optimal_share_of_fog": 0.6984126984126984
    },
    {
      "site": "location_7",
      "winter": 1986,
      "n_days": 151,
      "n_fog": 46,
      "n_optimal": 35,
      "n_both": 23,
      "fog_share_of_optimal": 0.6571428571428571,
      "optimal_share_of_fog": 0.5
    },
    {
      "site": "location_7",
      "winter": 1987,
      "n_days": 152,
      "n_fog": 48,
      "n_optimal": 53,
      "n_both": 27,
      "fog_share_of_optimal": 0.5094339622641509,
      "optimal_share_of_fog": 0.5625
    },
    {
      "site": "location_7",
      "winter": 1988,
      "n_days": 151,
      "n_fog": 54,
      "n_optimal": 43,
      "n_both": 23,
      "fog_share_of_optimal": 0.5348837209302325,
      "optimal_share_of_fog": 0.42592592592592593
    },
    {
      "site": "location_7",
      "winter": 1989,
      "n_days": 151,
      "n_fog": 58,
      "n_optimal": 38,
      "n_both": 19,
      "fog_share_of_optimal": 0.5,
      "optimal_share_of_fog": 0.3275862068965517
    },
    {
      "site": "location_7",
      "winter": 1990,
      "n_days": 151,
      "n_fog": 36,
      "n_optimal": 23,
      "n_both": 9,
      "fog_share_of_optimal": 0.391304347826087,
      "optimal_share_of_fog": 0.25
    },
    {
      "site": "location_7",
      "winter": 1991,
      "n_days": 152,
      "n_fog": 55,
      "n_optimal": 56,
      "n_both": 34,
      "fog_share_of_optimal": 0.6071428571428571,
      "optimal_share_of_fog": 0.6181818181818182
    },
    {
      "site": "location_7",
      "winter": 1992,
      "n_days": 151,
      "n_fog": 41,
      "n_optimal": 53,
      "n_both": 27,
      "fog_share_of_optimal": 0.5094339622641509,
      "optimal_share_of_fog": 0.6585365853658537
    },
    {
      "site": "location_7",
      "winter": 1993,
      "n_days": 151,
      "n_fog": 50,
      "n_optimal": 47,
      "n_both": 25,
      "fog_share_of_optimal": 0.5319148936170213,
      "optimal_share_of_fog": 0.5
    },
    {
      "site": "location_7",
      "winter": 1994,
      "n_days": 151,
      "n_fog": 52,
      "n_optimal": 71,
      "n_both": 42,
      "fog_share_of_optimal": 0.5915492957746479,
      "optimal_share_of_fog": 0.8076923076923077
    },
    {
      "site": "location_7",
      "winter": 1995,
      "n_days": 152,
      "n_fog": 50,
      "n_optimal": 67,
      "n_both": 35,
      "fog_share_of_optimal": 0.5223880597014925,
      "optimal_share_of_fog": 0.7
    },
    {
      "site": "location_7",
      "winter": 1996,
      "n_days": 151,
      "n_fog": 47,
      "n_optimal": 52,
      "n_both": 31,
      "fog_share_of_optimal": 0.5961538461538461,
      "optimal_share_of_fog": 0.6595744680851063
    },
    {
      "site": "location_7",
      "winter": 1997,
      "n_days": 151,
      "n_fog": 59,
      "n_optimal": 68,
      "n_both": 40,
      "fog_share_of_optimal": 0.5882352941176471,
      "optimal_share_of_fog": 0.6779661016949152
    },
    {
      "site": "location_7",
      "winter": 1998,
      "n_days": 151,
      "n_fog": 54,
      "n_optimal": 53,
      "n_both": 37,
      "fog_share_of_optimal": 0.6981132075471698,
      "optimal_share_of_fog": 0.6851851851851852
    },
    {
      "site": "location_7",
      "winter": 1999,
      "n_days": 152,
      "n_fog": 28,
      "n_optimal": 28,
      "n_both": 6,
      "fog_share_of_optimal": 0.21428571428571427,
      "optimal_share_of_fog": 0.21428571428571427
    },
    {
      "site": "location_7",
      "winter": 2000,
      "n_days": 151,
      "n_fog": 38,
      "n_optimal": 36,
      "n_both": 12,
      "fog_share_of_optimal": 0.3333333333333333,
      "optimal_share_of_fog": 0.3157894736842105
    },
    {
      "site": "location_7",
      "winter": 2001,
      "n_days": 151,
      "n_fog": 46,
      "n_optimal": 64,
      "n_both": 30,
      "fog_share_of_optimal": 0.46875,
      "optimal_share_of_fog": 0.6521739130434783
    },
    {
      "site": "location_7",
      "winter": 2002,
      "n_days": 151,
      "n_fog": 46,
      "n_optimal": 63,
      "n_both": 37,
      "fog_share_of_optimal": 0.5873015873015873,
      "optimal_share_of_fog": 0.8043478260869565
    },
    {
      "site": "location_7",
      "winter": 2003,
      "n_days": 152,
      "n_fog": 33,
      "n_optimal": 46,
      "n_both": 19,
      "fog_share_of_optimal": 0.41304347826086957,
      "optimal_share_of_fog": 0.5757575757575758
    },
    {
      "site": "location_7",
      "winter": 2004,
      "n_days": 151,
      "n_fog": 48,
      "n_optimal": 83,
      "n_both": 37,
      "fog_share_of_optimal": 0.4457831325301205,
      "optimal_share_of_fog": 0.7708333333333334
    },
    {
      "site": "location_7",
      "winter": 2005,
      "n_days": 151,
      "n_fog": 42,
      "n_optimal": 43,
      "n_both": 21,
      "fog_share_of_optimal": 0.4883720930232558,
      "optimal_share_of_fog": 0.5
    },
    {
      "site": "location_7",
      "winter": 2006,
      "n_days": 151,
      "n_fog": 25,
      "n_optimal": 16,
      "n_both": 5,
      "fog_share_of_optimal": 0.3125,
      "optimal_share_of_fog": 0.2
    },
    {
      "site": "location_7",
      "winter": 2007,
      "n_days": 152,
      "n_fog": 31,
      "n_optimal": 27,
      "n_both": 15,
      "fog_share_of_optimal": 0.5555555555555556,
      "optimal_share_of_fog": 0.4838709677419355
    },
    {
      "site": "location_7",
      "winter": 2008,
      "n_days": 151,
      "n_fog": 42,
      "n_optimal": 30,
      "n_both": 12,
      "fog_share_of_optimal": 0.4,
      "optimal_share_of_fog": 0.2857142857142857
    },
    {
      "site": "location_7",
      "winter": 2009,
      "n_days": 151,
      "n_fog": 29,
      "n_optimal": 51,
      "n_both": 22,
      "fog_share_of_optimal": 0.43137254901960786,
      "optimal_share_of_fog": 0.7586206896551724
    },
    {
      "site": "location_7",
      "winter": 2010,
      "n_days": 151,
      "n_fog": 29,
      "n_optimal": 57,
      "n_both": 23,
      "fog_share_of_optimal": 0.40350877192982454,
      "optimal_share_of_fog": 0.7931034482758621
    },
    {
      "site": "location_7",
      "winter": 2011,
      "n_days": 152,
      "n_fog": 28,
      "n_optimal": 15,
      "n_both": 7,
      "fog_share_of_optimal": 0.4666666666666667,
      "optimal_share_of_fog": 0.25
    },
    {
      "site": "location_7",
      "winter": 2012,
      "n_days": 151,
      "n_fog": 32,
      "n_optimal": 33,
      "n_both": 14,
      "fog_share_of_optimal": 0.42424242424242425,
      "optimal_share_of_fog": 0.4375
    },
    {
      "site": "location_7",
      "winter": 2013,
      "n_days": 151,
      "n_fog": 7,
      "n_optimal": 6,
      "n_both": 2,
      "fog_share_of_optimal": 0.3333333333333333,
      "optimal_share_of_fog": 0.2857142857142857
    },
    {
      "site": "location_7",
      "winter": 2014,
      "n_days": 151,
      "n_fog": 49,
      "n_optimal": 31,
      "n_both": 13,
      "fog_share_of_optimal": 0.41935483870967744,
      "optimal_share_of_fog": 0.2653061224489796
    },
    {
      "site": "location_7",
      "winter": 2015,
      "n_days": 152,
      "n_fog": 24,
      "n_optimal": 32,
      "n_both": 12,
      "fog_share_of_optimal": 0.375,
      "optimal_share_of_fog": 0.5
    },
    {
      "site": "location_7",
      "winter": 2016,
      "n_days": 151,
      "n_fog": 25,
      "n_optimal": 38,
      "n_both": 10,
      "fog_share_of_optimal": 0.2631578947368421,
      "optimal_share_of_fog": 0.4
    },
    {
      "site": "location_7",
      "winter": 2017,
      "n_days": 151,
      "n_fog": 22,
      "n_optimal": 29,
      "n_both": 14,
      "fog_share_of_optimal": 0.4827586206896552,
      "optimal_share_of_fog": 0.6363636363636364
    },
    {
      "site": "location_7",
      "winter": 2018,
      "n_days": 151,
      "n_fog": 17,
      "n_optimal": 41,
      "n_both": 13,
      "fog_share_of_optimal": 0.3170731707317073,
      "optimal_share_of_fog": 0.7647058823529411
    },
    {
      "site": "location_7",
      "winter": 2019,
      "n_days": 152,
      "n_fog": 31,
      "n_optimal": 43,
      "n_both": 22,
      "fog_share_of_optimal": 0.5116279069767442,
      "optimal_share_of_fog": 0.7096774193548387
    },
    {
      "site": "location_7",
      "winter": 2020,
      "n_days": 151,
      "n_fog": 18,
      "n_optimal": 32,
      "n_both": 15,
      "fog_share_of_optimal": 0.46875,
      "optimal_share_of_fog": 0.8333333333333334
    },
    {
      "site": "location_7",
      "winter": 2021,
      "n_days": 151,
      "n_fog": 45,
      "n_optimal": 57,
      "n_both": 35,
      "fog_share_of_optimal": 0.6140350877192983,
      "optimal_share_of_fog": 0.7777777777777778
    },
    {
      "site": "location_7",
      "winter": 2022,
      "n_days": 151,
      "n_fog": 22,
      "n_optimal": 43,
      "n_both": 14,
      "fog_share_of_optimal": 0.32558139534883723,
      "optimal_share_of_fog": 0.6363636363636364
    },
    {
      "site": "location_7",
      "winter": 2023,
      "n_days": 152,
      "n_fog": 12,
      "n_optimal": 40,
      "n_both": 9,
      "fog_share_of_optimal": 0.225,
      "optimal_share_of_fog": 0.75
    },
    {
      "site": "location_7",
      "winter": 2024,
      "n_days": 151,
      "n_fog": 37,
      "n_optimal": 54,
      "n_both": 27,
      "fog_share_of_optimal": 0.5,
      "optimal_share_of_fog": 0.7297297297297297
    },
    {
      "site": "location_7",
      "winter": 2025,
      "n_days": 151,
      "n_fog": 49,
      "n_optimal": 78,
      "n_both": 44,
      "fog_share_of_optimal": 0.5641025641025641,
      "optimal_share_of_fog": 0.8979591836734694
    },
    {
      "site": "location_8",
      "winter": 1992,
      "n_days": 151,
      "n_fog": 54,
      "n_optimal": 65,
      "n_both": 33,
      "fog_share_of_optimal": 0.5076923076923077,
      "optimal_share_of_fog": 0.6111111111111112
    },
    {
      "site": "location_8",
      "winter": 1994,
      "n_days": 151,
      "n_fog": 61,
      "n_optimal": 90,
      "n_both": 46,
      "fog_share_of_optimal": 0.5111111111111111,
      "optimal_share_of_fog": 0.7540983606557377
    },
    {
      "site": "location_8",
      "winter": 1995,
      "n_days": 152,
      "n_fog": 64,
      "n_optimal": 81,
      "n_both": 53,
      "fog_share_of_optimal": 0.654320987654321,
      "optimal_share_of_fog": 0.828125
    },
    {
      "site": "location_8",
      "winter": 2006,
      "n_days": 151,
      "n_fog": 48,
      "n_optimal": 22,
      "n_both": 12,
      "fog_share_of_optimal": 0.5454545454545454,
      "optimal_share_of_fog": 0.25
    },
    {
      "site": "location_8",
      "winter": 2007,
      "n_days": 152,
      "n_fog": 52,
      "n_optimal": 33,
      "n_both": 18,
      "fog_share_of_optimal": 0.5454545454545454,
      "optimal_share_of_fog": 0.34615384615384615
    },
    {
      "site": "location_8",
      "winter": 2008,
      "n_days": 151,
      "n_fog": 74,
      "n_optimal": 50,
      "n_both": 35,
      "fog_share_of_optimal": 0.7,
      "optimal_share_of_fog": 0.47297297297297297
    },
    {
      "site": "location_8",
      "winter": 2009,
      "n_days": 151,
      "n_fog": 66,
      "n_optimal": 61,
      "n_both": 40,
      "fog_share_of_optimal": 0.6557377049180327,
      "optimal_share_of_fog": 0.6060606060606061
    },
    {
      "site": "location_8",
      "winter": 2012,
      "n_days": 151,
      "n_fog": 43,
      "n_optimal": 33,
      "n_both": 16,
      "fog_share_of_optimal": 0.48484848484848486,
      "optimal_share_of_fog": 0.37209302325581395
    },
    {
      "site": "location_8",
      "winter": 2013,
      "n_days": 151,
      "n_fog": 27,
      "n_optimal": 9,
      "n_both": 2,
      "fog_share_of_optimal": 0.2222222222222222,
      "optimal_share_of_fog": 0.07407407407407407
    },
    {
      "site": "location_8",
      "winter": 2014,
      "n_days": 151,
      "n_fog": 78,
      "n_optimal": 28,
      "n_both": 19,
      "fog_share_of_optimal": 0.6785714285714286,
      "optimal_share_of_fog": 0.24358974358974358
    },
    {
      "site": "location_8",
      "winter": 2015,
      "n_days": 152,
      "n_fog": 62,
      "n_optimal": 45,
      "n_both": 30,
      "fog_share_of_optimal": 0.6666666666666666,
      "optimal_share_of_fog": 0.4838709677419355
    },
    {
      "site": "location_8",
      "winter": 2016,
      "n_days": 151,
      "n_fog": 52,
      "n_optimal": 43,
      "n_both": 21,
      "fog_share_of_optimal": 0.4883720930232558,
      "optimal_share_of_fog": 0.40384615384615385
    },
    {
      "site": "location_8",
      "winter": 2017,
      "n_days": 151,
      "n_fog": 53,
      "n_optimal": 18,
      "n_both": 14,
      "fog_share_of_optimal": 0.7777777777777778,
      "optimal_share_of_fog": 0.2641509433962264
    },
    {
      "site": "location_8",
      "winter": 2018,
      "n_days": 151,
      "n_fog": 50,
      "n_optimal": 56,
      "n_both": 38,
      "fog_share_of_optimal": 0.6785714285714286,
      "optimal_share_of_fog": 0.76
    },
    {
      "site": "location_8",
      "winter": 2019,
      "n_days": 152,
      "n_fog": 42,
      "n_optimal": 45,
      "n_both": 26,
      "fog_share_of_optimal": 0.5777777777777777,
      "optimal_share_of_fog": 0.6190476190476191
    },
    {
      "site": "location_8",
      "winter": 2020,
      "n_days": 151,
      "n_fog": 45,
      "n_optimal": 30,
      "n_both": 20,
      "fog_share_of_optimal": 0.6666666666666666,
      "optimal_share_of_fog": 0.4444444444444444
    },
    {
      "site": "location_8",
      "winter": 2021,
      "n_days": 151,
      "n_fog": 53,
      "n_optimal": 58,
      "n_both": 44,
      "fog_share_of_optimal": 0.7586206896551724,
      "optimal_share_of_fog": 0.8301886792452831
    },
    {
      "site": "location_8",
      "winter": 2022,
      "n_days": 151,
      "n_fog": 37,
      "n_optimal": 54,
      "n_both": 29,
      "fog_share_of_optimal": 0.5370370370370371,
      "optimal_share_of_fog": 0.7837837837837838
    },
    {
      "site": "location_8",
      "winter": 2023,
      "n_days": 152,
      "n_fog": 42,
      "n_optimal": 50,
      "n_both": 26,
      "fog_share_of_optimal": 0.52,
      "optimal_share_of_fog": 0.6190476190476191
    },
    {
      "site": "location_8",
      "winter": 2024,
      "n_days": 151,
      "n_fog": 54,
      "n_optimal": 62,
      "n_both": 40,
      "fog_share_of_optimal": 0.6451612903225806,
      "optimal_share_of_fog": 0.7407407407407407
    },
    {
      "site": "location_8",
      "winter": 2025,
      "n_days": 151,
      "n_fog": 62,
      "n_optimal": 88,
      "n_both": 51,
      "fog_share_of_optimal": 0.5795454545454546,
      "optimal_share_of_fog": 0.8225806451612904
    },
    {
      "site": "location_9",
      "winter": 1999,
      "n_days": 152,
      "n_fog": 41,
      "n_optimal": 28,
      "n_both": 12,
      "fog_share_of_optimal": 0.42857142857142855,
      "optimal_share_of_fog": 0.2926829268292683
    },
    {
      "site": "location_9",
      "winter": 2004,
      "n_days": 151,
      "n_fog": 81,
      "n_optimal": 83,
      "n_both": 52,
      "fog_share_of_optimal": 0.6265060240963856,
      "optimal_share_of_fog": 0.6419753086419753
    },
    {
      "site": "location_9",
      "winter": 2005,
      "n_days": 151,
      "n_fog": 83,
      "n_optimal": 36,
      "n_both": 24,
      "fog_share_of_optimal": 0.6666666666666666,
      "optimal_share_of_fog": 0.2891566265060241
    },
    {
      "site": "location_9",
      "winter": 2006,
      "n_days": 151,
      "n_fog": 56,
      "n_optimal": 10,
      "n_both": 8,
      "fog_share_of_optimal": 0.8,
      "optimal_share_of_fog": 0.14285714285714285
    },
    {
      "site": "location_9",
      "winter": 2007,
      "n_days": 152,
      "n_fog": 66,
      "n_optimal": 24,
      "n_both": 16,
      "fog_share_of_optimal": 0.6666666666666666,
      "optimal_share_of_fog": 0.24242424242424243
    },
    {
      "site": "location_9",
      "winter": 2008,
      "n_days": 151,
      "n_fog": 78,
      "n_optimal": 29,
      "n_both": 19,
      "fog_share_of_optimal": 0.6551724137931034,
      "optimal_share_of_fog": 0.24358974358974358
    },
    {
      "site": "location_9",
      "winter": 2009,
      "n_days": 151,
      "n_fog": 84,
      "n_optimal": 51,
      "n_both": 37,
      "fog_share_of_optimal": 0.7254901960784313,
      "optimal_share_of_fog": 0.44047619047619047
    },
    {
      "site": "location_9",
      "winter": 2010,
      "n_days": 151,
      "n_fog": 75,
      "n_optimal": 50,
      "n_both": 37,
      "fog_share_of_optimal": 0.74,
      "optimal_share_of_fog": 0.49333333333333335
    },
    {
      "site": "location_9",
      "winter": 2011,
      "n_days": 152,
      "n_fog": 78,
      "n_optimal": 11,
      "n_both": 8,
      "fog_share_of_optimal": 0.7272727272727273,
      "optimal_share_of_fog": 0.10256410256410256
    },
    {
      "site": "location_9",
      "winter": 2012,
      "n_days": 151,
      "n_fog": 87,
      "n_optimal": 21,
      "n_both": 17,
      "fog_share_of_optimal": 0.8095238095238095,
      "optimal_share_of_fog": 0.19540229885057472
    },
    {
      "site": "location_9",
      "winter": 2013,
      "n_days": 151,
      "n_fog": 62,
      "n_optimal": 6,
      "n_both": 3,
      "fog_share_of_optimal": 0.5,
      "optimal_share_of_fog": 0.04838709677419355
    },
    {
      "site": "location_9",
      "winter": 2015,
      "n_days": 152,
      "n_fog": 82,
      "n_optimal": 27,
      "n_both": 22,
      "fog_share_of_optimal": 0.8148148148148148,
      "optimal_share_of_fog": 0.2682926829268293
    },
    {
      "site": "location_9",
      "winter": 2016,
      "n_days": 151,
      "n_fog": 70,
      "n_optimal": 28,
      "n_both": 22,
      "fog_share_of_optimal": 0.7857142857142857,
      "optimal_share_of_fog": 0.3142857142857143
    },
    {
      "site": "location_9",
      "winter": 2017,
      "n_days": 151,
      "n_fog": 71,
      "n_optimal": 18,
      "n_both": 15,
      "fog_share_of_optimal": 0.8333333333333334,
      "optimal_share_of_fog": 0.2112676056338028
    },
    {
      "site": "location_9",
      "winter": 2018,
      "n_days": 151,
      "n_fog": 60,
      "n_optimal": 45,
      "n_both": 37,
      "fog_share_of_optimal": 0.8222222222222222,
      "optimal_share_of_fog": 0.6166666666666667
    },
    {
      "site": "location_9",
      "winter": 2019,
      "n_days": 152,
      "n_fog": 67,
      "n_optimal": 44,
      "n_both": 36,
      "fog_share_of_optimal": 0.8181818181818182,
      "optimal_share_of_fog": 0.5373134328358209
    },
    {
      "site": "location_9",
      "winter": 2020,
      "n_days": 151,
      "n_fog": 56,
      "n_optimal": 31,
      "n_both": 25,
      "fog_share_of_optimal": 0.8064516129032258,
      "optimal_share_of_fog": 0.44642857142857145
    },
    {
      "site": "location_9",
      "winter": 2021,
      "n_days": 151,
      "n_fog": 77,
      "n_optimal": 54,
      "n_both": 46,
      "fog_share_of_optimal": 0.8518518518518519,
      "optimal_share_of_fog": 0.5974025974025974
    },
    {
      "site": "location_9",
      "winter": 2022,
      "n_days": 151,
      "n_fog": 51,
      "n_optimal": 49,
      "n_both": 32,
      "fog_share_of_optimal": 0.6530612244897959,
      "optimal_share_of_fog": 0.6274509803921569
    },
    {
      "site": "location_9",
      "winter": 2023,
      "n_days": 152,
      "n_fog": 53,
      "n_optimal": 54,
      "n_both": 33,
      "fog_share_of_optimal": 0.6111111111111112,
      "optimal_share_of_fog": 0.6226415094339622
    },
    {
      "site": "location_9",
      "winter": 2024,
      "n_days": 151,
      "n_fog": 58,
      "n_optimal": 46,
      "n_both": 31,
      "fog_share_of_optimal": 0.6739130434782609,
      "optimal_share_of_fog": 0.5344827586206896
    }
  ],
  "trends": {
    "fog_share_of_optimal": {
      "location_10": {
        "status": "ok",
        "n": 45,
        "slope": -0.0028358208955223873,
        "ci95": [
          -0.005869310029998696,
          -0.0
        ],
        "kendall_tau": -0.21454459414507557,
        "mann_kendall_p": 0.038858286255885506
      },
      "location_6": {
        "status": "ok",
        "n": 23,
        "slope": 0.0029691876750700284,
        "ci95": [
          -0.006039488966318241,
          0.01084183673469388
        ],
        "kendall_tau": 0.1320095050265552,
        "mann_kendall_p": 0.3820140528224012
      },
      "location_7": {
        "status": "ok",
        "n": 46,
        "slope": -0.004669540229885065,
        "ci95": [
          -0.006969531107462142,
          -0.0023856485394946935
        ],
        "kendall_tau": -0.36882934568661346,
        "mann_kendall_p": 0.00030855128906670065
      },
      "location_8": {
        "status": "ok",
        "n": 21,
        "slope": 0.0018078759989890304,
        "ci95": [
          -0.0031893004115226367,
          0.007407407407407408
        ],
        "kendall_tau": 0.14868490295947984,
        "mann_kendall_p": 0.3485592150935216
      },
      "location_9": {
        "status": "ok",
        "n": 21,
        "slope": 0.009028456609101787,
        "ci95": [
          -0.0,
          0.013386880856760356
        ],
        "kendall_tau": 0.31026341346841063,
        "mann_kendall_p": 0.049565421569543786
      },
      "pooled_site_demeaned": {
        "status": "ok",
        "n": 156,
        "slope": -0.0024036492506784627,
        "ci95": [
          -0.004105032842348903,
          -0.000698174006444685
        ],
        "kendall_tau": -0.15327298112400906,
        "mann_kendall_p": 0.004559331924596792
      }
    },
    "optimal_share_of_fog": {
      "location_10": {
        "status": "ok",
        "n": 45,
        "slope": 0.002342803329645435,
        "ci95": [
          -0.0019532637261169125,
          0.006583072100313478
        ],
        "kendall_tau": 0.1255063300219201,
        "mann_kendall_p": 0.2250233264189263
      },
      "location_6": {
        "status": "ok",
        "n": 23,
        "slope": 0.013661202185792353,
        "ci95": [
          0.006995073891625617,
          0.022408963585434205
        ],
        "kendall_tau": 0.4990108793478453,
        "mann_kendall_p": 0.0008719637995609658
      },
      "location_7": {
        "status": "ok",
        "n": 46,
        "slope": 0.0018472906403940884,
        "ci95": [
          -0.002403846153846154,
          0.006189344424638542
        ],
        "kendall_tau": 0.08539625332378178,
        "mann_kendall_p": 0.40448759333355144
      },
      "location_8": {
        "status": "ok",
        "n": 21,
        "slope": 0.00994394119394119,
        "ci95": [
          -0.0019020085209981719,
          0.030743066457352173
        ],
        "kendall_tau": 0.25298401405885795,
        "mann_kendall_p": 0.10933992419429907
      },
      "location_9": {
        "status": "ok",
        "n": 21,
        "slope": 0.01446890445549294,
        "ci95": [
          -0.0005661348430262558,
          0.02825670498084291
        ],
        "kendall_tau": 0.3047619047619048,
        "mann_kendall_p": 0.05636408981183753
      },
      "pooled_site_demeaned": {
        "status": "ok",
        "n": 156,
        "slope": 0.0034531479435651103,
        "ci95": [
          0.0008510392770237815,
          0.006327345735692335
        ],
        "kendall_tau": 0.13879580575886744,
        "mann_kendall_p": 0.010171977773411413
      }
    }
  }
}
```
