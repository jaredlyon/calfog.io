# Air-quality paired tests

Equal-weight mean of per-airport paired deltas; fixed-PST calendar days resampled jointly across airports. CI = estimate ± quantile(|draw − estimate|) at 0.95 (Bonferroni: 1 − 0.05/6). Two-sided p = (1 + count(|draw − estimate| ≥ |estimate|))/(valid draws + 1). AP gains are positive; RPS gains are negative. A pooled draw is valid only if every included airport has rows (and, for AP, positive labels); invalid draws are discarded jointly, never by dropping an airport from the mean.

Primary/near-saturation intervals use α/6; all others use 95%. Rules-out = upper simultaneous AP bound < 0.02. RPS has no equivalence margin. Holm is applied to six primary tests, separately in each stratum and metric.

| Family / set / task / stratum | Metric | Δ with−without | Interval | p | p Holm | Rows/days/draws | Rules-out |
|---|---|---:|---|---:|---:|---:|---|
| logistic_regression / nox / nowcast / all | AP | 0.009054 | [-0.001833, 0.01994] | 0.03996 | 0.1598 | 609425/6706/1000 | true |
| logistic_regression / nox / nowcast / near-saturation | AP | 0.01519 | [0.002404, 0.02798] | 0.000999 | 0.005994 | 88536/3589/1000 | false |
| logistic_regression / nox / nowcast / all | RPS | -0.003598 | [-0.004092, -0.003105] | 0.000999 | 0.005994 | 609425/6706/1000 | — |
| logistic_regression / nox / nowcast / near-saturation | RPS | -0.007052 | [-0.008896, -0.005208] | 0.000999 | 0.005994 | 88536/3589/1000 | — |
| logistic_regression / nox / lead_time / all | AP | 0.004777 | [-0.002807, 0.01236] | 0.1009 | 0.3027 | 23087/6401/1000 | true |
| logistic_regression / nox / lead_time / near-saturation | AP | 0.008605 | [-0.0002995, 0.01751] | 0.01299 | 0.05195 | 6050/2300/1000 | true |
| logistic_regression / nox / lead_time / all | RPS | -0.0009088 | [-0.001295, -0.0005223] | 0.000999 | 0.005994 | 23087/6401/1000 | — |
| logistic_regression / nox / lead_time / near-saturation | RPS | -0.0002662 | [-0.001366, 0.0008335] | 0.4955 | 0.4955 | 6050/2300/1000 | — |
| logistic_regression / pm25 / nowcast / all | AP | -0.06914 | [-0.09045, -0.04782] | 0.000999 | — | 365530/3909/1000 | — |
| logistic_regression / pm25 / nowcast / all | RPS | -0.01037 | [-0.01133, -0.00941] | 0.000999 | — | 365530/3909/1000 | — |
| logistic_regression / pm25 / lead_time / all | AP | 0.01266 | [-0.005828, 0.03114] | 0.2198 | — | 14708/3824/1000 | — |
| logistic_regression / pm25 / lead_time / all | RPS | -0.007536 | [-0.00855, -0.006521] | 0.000999 | — | 14708/3824/1000 | — |
| logistic_regression / cams / nowcast / all | AP | -0.04176 | [-0.06432, -0.01921] | 0.001998 | — | 82274/702/1000 | — |
| logistic_regression / cams / nowcast / all | RPS | 0.001081 | [-0.0008271, 0.002989] | 0.2498 | — | 82274/702/1000 | — |
| logistic_regression / cams / lead_time / all | AP | -0.01005 | [-0.0368, 0.01669] | 0.4515 | — | 3360/699/1000 | — |
| logistic_regression / cams / lead_time / all | RPS | 0.002219 | [3.059e-05, 0.004408] | 0.04895 | — | 3360/699/1000 | — |
| logistic_regression / nox_monitor / nowcast / all | AP | -0.005804 | [-0.01516, 0.003556] | 0.2238 | — | 609425/6706/1000 | — |
| logistic_regression / nox_monitor / nowcast / all | RPS | -0.002187 | [-0.002727, -0.001647] | 0.000999 | — | 609425/6706/1000 | — |
| logistic_regression / nox_monitor / lead_time / all | AP | -0.0212 | [-0.03106, -0.01134] | 0.000999 | — | 23087/6401/1000 | — |
| logistic_regression / nox_monitor / lead_time / all | RPS | 0.008319 | [0.007238, 0.009399] | 0.000999 | — | 23087/6401/1000 | — |
| xgboost / nox / nowcast / all | AP | -0.02456 | [-0.03653, -0.0126] | 0.000999 | 0.005994 | 609425/6706/1000 | true |
| xgboost / nox / nowcast / near-saturation | AP | -0.02031 | [-0.03352, -0.007096] | 0.000999 | 0.005994 | 88536/3589/1000 | true |
| xgboost / nox / nowcast / all | RPS | -0.002082 | [-0.002542, -0.001621] | 0.000999 | 0.005994 | 609425/6706/1000 | — |
| xgboost / nox / nowcast / near-saturation | RPS | -0.002005 | [-0.003608, -0.0004009] | 0.000999 | 0.005994 | 88536/3589/1000 | — |
| xgboost / nox / lead_time / all | AP | -0.01272 | [-0.02733, 0.001885] | 0.03097 | 0.1548 | 23087/6401/1000 | true |
| xgboost / nox / lead_time / near-saturation | AP | -0.008656 | [-0.02555, 0.008233] | 0.1429 | 0.2857 | 6050/2300/1000 | true |
| xgboost / nox / lead_time / all | RPS | -0.0004002 | [-0.0008886, 8.824e-05] | 0.02298 | 0.02298 | 23087/6401/1000 | — |
| xgboost / nox / lead_time / near-saturation | RPS | 0.001688 | [0.0004375, 0.002939] | 0.001998 | 0.005994 | 6050/2300/1000 | — |
| xgboost / pm25 / nowcast / all | AP | 0.03246 | [0.02211, 0.04281] | 0.000999 | — | 365530/3909/1000 | — |
| xgboost / pm25 / nowcast / all | RPS | -0.01271 | [-0.01378, -0.01163] | 0.000999 | — | 365530/3909/1000 | — |
| xgboost / pm25 / lead_time / all | AP | 0.03283 | [0.02083, 0.04484] | 0.000999 | — | 14708/3824/1000 | — |
| xgboost / pm25 / lead_time / all | RPS | -0.006573 | [-0.007259, -0.005887] | 0.000999 | — | 14708/3824/1000 | — |
| xgboost / cams / nowcast / all | AP | -0.02332 | [-0.04155, -0.005091] | 0.01199 | — | 82274/702/1000 | — |
| xgboost / cams / nowcast / all | RPS | -0.002589 | [-0.003367, -0.001811] | 0.000999 | — | 82274/702/1000 | — |
| xgboost / cams / lead_time / all | AP | -0.02157 | [-0.04307, -6.547e-05] | 0.04795 | — | 3360/699/1000 | — |
| xgboost / cams / lead_time / all | RPS | -0.0009028 | [-0.001697, -0.0001092] | 0.02298 | — | 3360/699/1000 | — |
| xgboost / nox_monitor / nowcast / all | AP | -0.02271 | [-0.03206, -0.01336] | 0.000999 | — | 609425/6706/1000 | — |
| xgboost / nox_monitor / nowcast / all | RPS | -0.001807 | [-0.002166, -0.001449] | 0.000999 | — | 609425/6706/1000 | — |
| xgboost / nox_monitor / lead_time / all | AP | -0.01733 | [-0.02829, -0.006368] | 0.000999 | — | 23087/6401/1000 | — |
| xgboost / nox_monitor / lead_time / all | RPS | 0.0006487 | [0.0002918, 0.001006] | 0.001998 | — | 23087/6401/1000 | — |
| ebm / nox / nowcast / all | AP | -0.004584 | [-0.01313, 0.003967] | 0.1479 | 0.3027 | 609425/6706/1000 | true |
| ebm / nox / nowcast / near-saturation | AP | -0.002953 | [-0.01169, 0.005784] | 0.3996 | 0.3996 | 88536/3589/1000 | true |
| ebm / nox / nowcast / all | RPS | -0.002559 | [-0.002982, -0.002137] | 0.000999 | 0.005994 | 609425/6706/1000 | — |
| ebm / nox / nowcast / near-saturation | RPS | -0.00404 | [-0.005537, -0.002544] | 0.000999 | 0.005994 | 88536/3589/1000 | — |
| ebm / nox / lead_time / all | AP | 0.005263 | [-0.006416, 0.01694] | 0.2328 | 0.3027 | 23087/6401/1000 | true |
| ebm / nox / lead_time / near-saturation | AP | 0.00933 | [-0.003438, 0.0221] | 0.04296 | 0.1289 | 6050/2300/1000 | false |
| ebm / nox / lead_time / all | RPS | -0.001335 | [-0.001808, -0.0008619] | 0.000999 | 0.005994 | 23087/6401/1000 | — |
| ebm / nox / lead_time / near-saturation | RPS | -0.001214 | [-0.002566, 0.0001377] | 0.01998 | 0.03996 | 6050/2300/1000 | — |
| ebm / pm25 / nowcast / all | AP | 0.04428 | [0.03635, 0.05221] | 0.000999 | — | 365530/3909/1000 | — |
| ebm / pm25 / nowcast / all | RPS | -0.01565 | [-0.01681, -0.01449] | 0.000999 | — | 365530/3909/1000 | — |
| ebm / pm25 / lead_time / all | AP | 0.01578 | [0.00627, 0.0253] | 0.001998 | — | 14708/3824/1000 | — |
| ebm / pm25 / lead_time / all | RPS | -0.006599 | [-0.007288, -0.00591] | 0.000999 | — | 14708/3824/1000 | — |
| ebm / cams / nowcast / all | AP | -0.02415 | [-0.04304, -0.005261] | 0.01099 | — | 82274/702/1000 | — |
| ebm / cams / nowcast / all | RPS | -0.002443 | [-0.00371, -0.001176] | 0.000999 | — | 82274/702/1000 | — |
| ebm / cams / lead_time / all | AP | -0.02456 | [-0.05313, 0.004009] | 0.09291 | — | 3360/699/1000 | — |
| ebm / cams / lead_time / all | RPS | 0.0002806 | [-0.0004309, 0.000992] | 0.4555 | — | 3360/699/1000 | — |
| ebm / nox_monitor / nowcast / all | AP | -0.01024 | [-0.01659, -0.003883] | 0.002997 | — | 609425/6706/1000 | — |
| ebm / nox_monitor / nowcast / all | RPS | -0.002082 | [-0.002434, -0.001729] | 0.000999 | — | 609425/6706/1000 | — |
| ebm / nox_monitor / lead_time / all | AP | -0.001704 | [-0.01156, 0.008153] | 0.7403 | — | 23087/6401/1000 | — |
| ebm / nox_monitor / lead_time / all | RPS | 0.000188 | [-0.0002522, 0.0006283] | 0.3696 | — | 23087/6401/1000 | — |
| random_forest / nox / nowcast / all | AP | -0.01829 | [-0.028, -0.008588] | 0.001998 | — | 609425/6706/1000 | — |
| random_forest / nox / nowcast / all | RPS | -0.002973 | [-0.003151, -0.002795] | 0.000999 | — | 609425/6706/1000 | — |
| random_forest / nox / lead_time / all | AP | -0.01093 | [-0.02091, -0.0009522] | 0.02897 | — | 23087/6401/1000 | — |
| random_forest / nox / lead_time / all | RPS | -0.002037 | [-0.002269, -0.001805] | 0.000999 | — | 23087/6401/1000 | — |
| random_forest / pm25 / nowcast / all | AP | 0.01239 | [0.0003156, 0.02445] | 0.04396 | — | 365530/3909/1000 | — |
| random_forest / pm25 / nowcast / all | RPS | -0.00729 | [-0.007929, -0.006651] | 0.000999 | — | 365530/3909/1000 | — |
| random_forest / pm25 / lead_time / all | AP | 0.01859 | [0.006794, 0.03038] | 0.003996 | — | 14708/3824/1000 | — |
| random_forest / pm25 / lead_time / all | RPS | -0.005675 | [-0.006181, -0.00517] | 0.000999 | — | 14708/3824/1000 | — |
| random_forest / cams / nowcast / all | AP | -0.02447 | [-0.04111, -0.007825] | 0.002997 | — | 82274/702/1000 | — |
| random_forest / cams / nowcast / all | RPS | -0.001383 | [-0.002274, -0.0004915] | 0.005994 | — | 82274/702/1000 | — |
| random_forest / cams / lead_time / all | AP | -0.01035 | [-0.03626, 0.01555] | 0.4266 | — | 3360/699/1000 | — |
| random_forest / cams / lead_time / all | RPS | -0.001647 | [-0.002393, -0.0009013] | 0.000999 | — | 3360/699/1000 | — |
| xgboost_focal_loss / nox / nowcast / all | AP | -0.02693 | [-0.0363, -0.01756] | 0.000999 | — | 609425/6706/1000 | — |
| xgboost_focal_loss / nox / nowcast / all | RPS | -0.003621 | [-0.003952, -0.00329] | 0.000999 | — | 609425/6706/1000 | — |
| xgboost_focal_loss / nox / lead_time / all | AP | -0.006366 | [-0.01781, 0.005078] | 0.2637 | — | 23087/6401/1000 | — |
| xgboost_focal_loss / nox / lead_time / all | RPS | -0.001723 | [-0.002084, -0.001362] | 0.000999 | — | 23087/6401/1000 | — |
| xgboost_focal_loss / pm25 / nowcast / all | AP | 0.03431 | [0.02329, 0.04533] | 0.000999 | — | 365530/3909/1000 | — |
| xgboost_focal_loss / pm25 / nowcast / all | RPS | -0.01397 | [-0.01505, -0.01289] | 0.000999 | — | 365530/3909/1000 | — |
| xgboost_focal_loss / pm25 / lead_time / all | AP | 0.03747 | [0.02313, 0.05181] | 0.000999 | — | 14708/3824/1000 | — |
| xgboost_focal_loss / pm25 / lead_time / all | RPS | -0.008601 | [-0.009412, -0.00779] | 0.000999 | — | 14708/3824/1000 | — |
| xgboost_focal_loss / cams / nowcast / all | AP | -0.01069 | [-0.02839, 0.007021] | 0.2188 | — | 82274/702/1000 | — |
| xgboost_focal_loss / cams / nowcast / all | RPS | -0.005717 | [-0.006879, -0.004555] | 0.000999 | — | 82274/702/1000 | — |
| xgboost_focal_loss / cams / lead_time / all | AP | 0.01083 | [-0.01546, 0.03712] | 0.4206 | — | 3360/699/1000 | — |
| xgboost_focal_loss / cams / lead_time / all | RPS | 0.0008161 | [-0.0001109, 0.001743] | 0.08192 | — | 3360/699/1000 | — |
| temporal_cnn / nox / nowcast / all | AP | 0.03108 | [0.02077, 0.04139] | 0.000999 | — | 609425/6706/1000 | — |
| temporal_cnn / nox / nowcast / all | RPS | -0.00379 | [-0.004131, -0.003449] | 0.000999 | — | 609425/6706/1000 | — |
| temporal_cnn / nox / lead_time / all | AP | 0.01389 | [0.002988, 0.02478] | 0.007992 | — | 23087/6401/1000 | — |
| temporal_cnn / nox / lead_time / all | RPS | -0.002062 | [-0.002435, -0.001688] | 0.000999 | — | 23087/6401/1000 | — |
| temporal_cnn / pm25 / nowcast / all | AP | 0.02142 | [0.003251, 0.03959] | 0.03097 | — | 365530/3909/1000 | — |
| temporal_cnn / pm25 / nowcast / all | RPS | -0.01255 | [-0.01364, -0.01146] | 0.000999 | — | 365530/3909/1000 | — |
| temporal_cnn / pm25 / lead_time / all | AP | 0.02282 | [0.002795, 0.04285] | 0.02997 | — | 14708/3824/1000 | — |
| temporal_cnn / pm25 / lead_time / all | RPS | -0.009732 | [-0.01082, -0.008642] | 0.000999 | — | 14708/3824/1000 | — |
| temporal_cnn / cams / nowcast / all | AP | -0.01014 | [-0.02501, 0.004729] | 0.1708 | — | 82274/702/1000 | — |
| temporal_cnn / cams / nowcast / all | RPS | 0.0005745 | [-0.0005925, 0.001742] | 0.3506 | — | 82274/702/1000 | — |
| temporal_cnn / cams / lead_time / all | AP | 0.02845 | [-0.009104, 0.066] | 0.1379 | — | 3360/699/1000 | — |
| temporal_cnn / cams / lead_time / all | RPS | -0.01375 | [-0.01551, -0.01199] | 0.000999 | — | 3360/699/1000 | — |
| ensemble / nox / nowcast / all | AP | -0.01045 | [-0.01888, -0.002015] | 0.01798 | — | 609425/6706/1000 | — |
| ensemble / nox / nowcast / all | RPS | -0.001499 | [-0.001726, -0.001273] | 0.000999 | — | 609425/6706/1000 | — |
| ensemble / nox / lead_time / all | AP | -0.003609 | [-0.01327, 0.006051] | 0.4695 | — | 23087/6401/1000 | — |
| ensemble / nox / lead_time / all | RPS | -0.0004233 | [-0.0006491, -0.0001975] | 0.001998 | — | 23087/6401/1000 | — |
| ensemble / pm25 / nowcast / all | AP | 0.02524 | [0.0132, 0.03729] | 0.000999 | — | 365530/3909/1000 | — |
| ensemble / pm25 / nowcast / all | RPS | -0.01246 | [-0.01351, -0.01141] | 0.000999 | — | 365530/3909/1000 | — |
| ensemble / pm25 / lead_time / all | AP | 0.02438 | [0.01032, 0.03845] | 0.001998 | — | 14708/3824/1000 | — |
| ensemble / pm25 / lead_time / all | RPS | -0.00763 | [-0.008457, -0.006803] | 0.000999 | — | 14708/3824/1000 | — |
| ensemble / cams / nowcast / all | AP | -0.01383 | [-0.03488, 0.007229] | 0.1918 | — | 32843/699/1000 | — |
| ensemble / cams / nowcast / all | RPS | -0.0006354 | [-0.001517, 0.0002464] | 0.1489 | — | 32843/699/1000 | — |
| ensemble / cams / lead_time / all | AP | 0.02756 | [0.001344, 0.05377] | 0.04196 | — | 1341/693/1000 | — |
| ensemble / cams / lead_time / all | RPS | -0.002503 | [-0.004059, -0.0009468] | 0.002997 | — | 1341/693/1000 | — |
| persistence / nox / nowcast / all | AP | 0 | [0, 0] | 1 | — | 609425/6706/1000 | — |
| persistence / nox / nowcast / all | RPS | 0 | [0, 0] | 1 | — | 609425/6706/1000 | — |
| persistence / nox / lead_time / all | AP | 0 | [0, 0] | 1 | — | 23087/6401/1000 | — |
| persistence / nox / lead_time / all | RPS | 0 | [0, 0] | 1 | — | 23087/6401/1000 | — |
| persistence / pm25 / nowcast / all | AP | 0 | [0, 0] | 1 | — | 365530/3909/1000 | — |
| persistence / pm25 / nowcast / all | RPS | 0 | [0, 0] | 1 | — | 365530/3909/1000 | — |
| persistence / pm25 / lead_time / all | AP | 0 | [0, 0] | 1 | — | 14708/3824/1000 | — |
| persistence / pm25 / lead_time / all | RPS | 0 | [0, 0] | 1 | — | 14708/3824/1000 | — |
| persistence / cams / nowcast / all | AP | 0 | [0, 0] | 1 | — | 82274/702/1000 | — |
| persistence / cams / nowcast / all | RPS | 0 | [0, 0] | 1 | — | 82274/702/1000 | — |
| persistence / cams / lead_time / all | AP | 0 | [0, 0] | 1 | — | 3360/699/1000 | — |
| persistence / cams / lead_time / all | RPS | 0 | [0, 0] | 1 | — | 3360/699/1000 | — |
| analog / nox / nowcast / all | AP | -0.01947 | [-0.0265, -0.01245] | 0.000999 | — | 609425/6706/1000 | — |
| analog / nox / nowcast / all | RPS | -0.001754 | [-0.00196, -0.001548] | 0.000999 | — | 609425/6706/1000 | — |
| analog / nox / lead_time / all | AP | -0.02021 | [-0.03015, -0.01026] | 0.000999 | — | 23087/6401/1000 | — |
| analog / nox / lead_time / all | RPS | -0.0008237 | [-0.001122, -0.0005251] | 0.000999 | — | 23087/6401/1000 | — |
| analog / pm25 / nowcast / all | AP | 0.02847 | [0.02017, 0.03677] | 0.000999 | — | 365530/3909/1000 | — |
| analog / pm25 / nowcast / all | RPS | -0.01223 | [-0.01334, -0.01111] | 0.000999 | — | 365530/3909/1000 | — |
| analog / pm25 / lead_time / all | AP | 0.0426 | [0.02567, 0.05952] | 0.000999 | — | 14708/3824/1000 | — |
| analog / pm25 / lead_time / all | RPS | -0.008181 | [-0.009221, -0.00714] | 0.000999 | — | 14708/3824/1000 | — |
| analog / cams / nowcast / all | AP | -0.01458 | [-0.03296, 0.003811] | 0.1159 | — | 82274/702/1000 | — |
| analog / cams / nowcast / all | RPS | -0.0005672 | [-0.001395, 0.0002609] | 0.1698 | — | 82274/702/1000 | — |
| analog / cams / lead_time / all | AP | -0.01542 | [-0.04096, 0.01013] | 0.2278 | — | 3360/699/1000 | — |
| analog / cams / lead_time / all | RPS | 0.0004828 | [-0.0005792, 0.001545] | 0.3816 | — | 3360/699/1000 | — |
| xgboost_ordinal / nox / nowcast / all | AP | -0.02408 | [-0.03372, -0.01444] | 0.000999 | — | 609425/6706/1000 | — |
| xgboost_ordinal / nox / nowcast / all | RPS | -0.002603 | [-0.002996, -0.002209] | 0.000999 | — | 609425/6706/1000 | — |
| xgboost_ordinal / nox / lead_time / all | AP | -0.009147 | [-0.01957, 0.001279] | 0.07692 | — | 23087/6401/1000 | — |
| xgboost_ordinal / nox / lead_time / all | RPS | -0.001112 | [-0.001497, -0.0007262] | 0.000999 | — | 23087/6401/1000 | — |
| xgboost_ordinal / pm25 / nowcast / all | AP | 0.03431 | [0.024, 0.04461] | 0.000999 | — | 365530/3909/1000 | — |
| xgboost_ordinal / pm25 / nowcast / all | RPS | -0.01346 | [-0.01459, -0.01234] | 0.000999 | — | 365530/3909/1000 | — |
| xgboost_ordinal / pm25 / lead_time / all | AP | 0.02609 | [0.01293, 0.03926] | 0.000999 | — | 14708/3824/1000 | — |
| xgboost_ordinal / pm25 / lead_time / all | RPS | -0.008154 | [-0.009094, -0.007213] | 0.000999 | — | 14708/3824/1000 | — |
| xgboost_ordinal / cams / nowcast / all | AP | -0.01893 | [-0.03658, -0.001287] | 0.03397 | — | 82274/702/1000 | — |
| xgboost_ordinal / cams / nowcast / all | RPS | -0.003053 | [-0.004025, -0.002082] | 0.000999 | — | 82274/702/1000 | — |
| xgboost_ordinal / cams / lead_time / all | AP | 0.004304 | [-0.02433, 0.03294] | 0.7792 | — | 3360/699/1000 | — |
| xgboost_ordinal / cams / lead_time / all | RPS | -0.0004998 | [-0.001591, 0.0005913] | 0.3876 | — | 3360/699/1000 | — |
| xgboost_aft / nox / nowcast / all | AP | -0.02981 | [-0.03972, -0.0199] | 0.000999 | — | 609425/6706/1000 | — |
| xgboost_aft / nox / nowcast / all | RPS | -0.001921 | [-0.002347, -0.001496] | 0.000999 | — | 609425/6706/1000 | — |
| xgboost_aft / nox / lead_time / all | AP | -0.002807 | [-0.01254, 0.006927] | 0.5504 | — | 23087/6401/1000 | — |
| xgboost_aft / nox / lead_time / all | RPS | -0.001478 | [-0.001951, -0.001005] | 0.000999 | — | 23087/6401/1000 | — |
| xgboost_aft / pm25 / nowcast / all | AP | 0.01868 | [0.005992, 0.03137] | 0.003996 | — | 365530/3909/1000 | — |
| xgboost_aft / pm25 / nowcast / all | RPS | -0.01205 | [-0.01302, -0.01108] | 0.000999 | — | 365530/3909/1000 | — |
| xgboost_aft / pm25 / lead_time / all | AP | 0.01514 | [-0.003535, 0.03382] | 0.1109 | — | 14708/3824/1000 | — |
| xgboost_aft / pm25 / lead_time / all | RPS | -0.006432 | [-0.007453, -0.005412] | 0.000999 | — | 14708/3824/1000 | — |
| xgboost_aft / cams / nowcast / all | AP | -0.00759 | [-0.03715, 0.02197] | 0.6204 | — | 82274/702/1000 | — |
| xgboost_aft / cams / nowcast / all | RPS | -0.002661 | [-0.003739, -0.001584] | 0.000999 | — | 82274/702/1000 | — |
| xgboost_aft / cams / lead_time / all | AP | 0.01199 | [-0.02024, 0.04423] | 0.4685 | — | 3360/699/1000 | — |
| xgboost_aft / cams / lead_time / all | RPS | -0.0009618 | [-0.003106, 0.001182] | 0.3656 | — | 3360/699/1000 | — |

## Primary NOx threshold-Brier decomposition

Thresholds 0–4 are fog; 5–15 are haze. The mean of all 17 thresholds equals ΔRPS. Threshold and range intervals below are pointwise 95% (exploratory).

| Family / task | Threshold or range | Δ Brier | 95% CI |
|---|---|---:|---|
| logistic_regression / nowcast | 0 | -0.0001785 | [-0.0002454, -0.0001116] |
| logistic_regression / nowcast | 1 | -0.0003652 | [-0.0005572, -0.0001731] |
| logistic_regression / nowcast | 2 | -0.0004348 | [-0.0006638, -0.0002059] |
| logistic_regression / nowcast | 3 | -0.0004891 | [-0.0007417, -0.0002365] |
| logistic_regression / nowcast | 4 | -0.0007492 | [-0.001044, -0.000454] |
| logistic_regression / nowcast | 5 | -0.0009446 | [-0.001263, -0.0006262] |
| logistic_regression / nowcast | 6 | -0.001358 | [-0.001708, -0.001008] |
| logistic_regression / nowcast | 7 | -0.001559 | [-0.001931, -0.001188] |
| logistic_regression / nowcast | 8 | -0.002183 | [-0.002607, -0.001758] |
| logistic_regression / nowcast | 9 | -0.002894 | [-0.003382, -0.002406] |
| logistic_regression / nowcast | 10 | -0.004332 | [-0.004904, -0.003759] |
| logistic_regression / nowcast | 11 | -0.005528 | [-0.006174, -0.004882] |
| logistic_regression / nowcast | 12 | -0.006906 | [-0.007647, -0.006166] |
| logistic_regression / nowcast | 13 | -0.006409 | [-0.00715, -0.005668] |
| logistic_regression / nowcast | 14 | -0.01055 | [-0.0113, -0.0098] |
| logistic_regression / nowcast | 15 | -0.009078 | [-0.009808, -0.008347] |
| logistic_regression / nowcast | 16 | -0.007212 | [-0.007918, -0.006506] |
| logistic_regression / nowcast | fog_0_4 | -0.0004434 | [-0.0006438, -0.0002429] |
| logistic_regression / nowcast | haze_5_15 | -0.004704 | [-0.005163, -0.004245] |
| logistic_regression / lead_time | 0 | 6.877e-05 | [-0.0001269, 0.0002644] |
| logistic_regression / lead_time | 1 | 0.0003799 | [9.49e-05, 0.0006649] |
| logistic_regression / lead_time | 2 | 0.0004089 | [0.0001008, 0.000717] |
| logistic_regression / lead_time | 3 | 0.0002717 | [-5.747e-05, 0.0006009] |
| logistic_regression / lead_time | 4 | 3.319e-05 | [-0.0003186, 0.000385] |
| logistic_regression / lead_time | 5 | -6.449e-05 | [-0.0004338, 0.0003048] |
| logistic_regression / lead_time | 6 | -0.0003872 | [-0.000774, -3.299e-07] |
| logistic_regression / lead_time | 7 | -0.0005945 | [-0.0009934, -0.0001957] |
| logistic_regression / lead_time | 8 | -0.0009282 | [-0.001331, -0.0005257] |
| logistic_regression / lead_time | 9 | -0.001065 | [-0.001497, -0.0006319] |
| logistic_regression / lead_time | 10 | -0.001348 | [-0.001791, -0.0009056] |
| logistic_regression / lead_time | 11 | -0.001188 | [-0.001645, -0.0007308] |
| logistic_regression / lead_time | 12 | -0.001367 | [-0.00184, -0.000894] |
| logistic_regression / lead_time | 13 | -0.001078 | [-0.001535, -0.0006204] |
| logistic_regression / lead_time | 14 | -0.003647 | [-0.004139, -0.003156] |
| logistic_regression / lead_time | 15 | -0.002859 | [-0.003328, -0.002391] |
| logistic_regression / lead_time | 16 | -0.002085 | [-0.002561, -0.001609] |
| logistic_regression / lead_time | fog_0_4 | 0.0002325 | [-3.669e-05, 0.0005017] |
| logistic_regression / lead_time | haze_5_15 | -0.001321 | [-0.001643, -0.000998] |
| xgboost / nowcast | 0 | 2.137e-05 | [-3.475e-05, 7.749e-05] |
| xgboost / nowcast | 1 | 0.0002788 | [0.0001117, 0.000446] |
| xgboost / nowcast | 2 | 0.0003648 | [0.0001648, 0.0005648] |
| xgboost / nowcast | 3 | 0.000411 | [0.0001875, 0.0006345] |
| xgboost / nowcast | 4 | 0.0004208 | [0.0001765, 0.0006652] |
| xgboost / nowcast | 5 | 0.0003875 | [0.0001293, 0.0006457] |
| xgboost / nowcast | 6 | 0.0001322 | [-0.0001579, 0.0004223] |
| xgboost / nowcast | 7 | -3.318e-05 | [-0.0003371, 0.0002707] |
| xgboost / nowcast | 8 | -0.00048 | [-0.0008117, -0.0001483] |
| xgboost / nowcast | 9 | -0.001056 | [-0.001444, -0.0006682] |
| xgboost / nowcast | 10 | -0.002257 | [-0.002746, -0.001768] |
| xgboost / nowcast | 11 | -0.003804 | [-0.00441, -0.003199] |
| xgboost / nowcast | 12 | -0.005149 | [-0.005892, -0.004406] |
| xgboost / nowcast | 13 | -0.005039 | [-0.005799, -0.004279] |
| xgboost / nowcast | 14 | -0.00825 | [-0.009196, -0.007303] |
| xgboost / nowcast | 15 | -0.006523 | [-0.007455, -0.005591] |
| xgboost / nowcast | 16 | -0.004812 | [-0.005732, -0.003891] |
| xgboost / nowcast | fog_0_4 | 0.0002994 | [0.0001273, 0.0004715] |
| xgboost / nowcast | haze_5_15 | -0.002916 | [-0.003353, -0.002478] |
| xgboost / lead_time | 0 | -7.466e-05 | [-0.0003834, 0.0002341] |
| xgboost / lead_time | 1 | 0.0005041 | [9.699e-05, 0.0009113] |
| xgboost / lead_time | 2 | 0.0003715 | [-5.266e-05, 0.0007956] |
| xgboost / lead_time | 3 | 0.0002728 | [-0.0001746, 0.0007202] |
| xgboost / lead_time | 4 | 0.000324 | [-0.0001573, 0.0008053] |
| xgboost / lead_time | 5 | 0.0003201 | [-0.0001604, 0.0008005] |
| xgboost / lead_time | 6 | 2.254e-06 | [-0.0004684, 0.0004729] |
| xgboost / lead_time | 7 | 0.0001216 | [-0.0003655, 0.0006087] |
| xgboost / lead_time | 8 | -0.0001965 | [-0.0006709, 0.0002778] |
| xgboost / lead_time | 9 | -0.0003411 | [-0.0008302, 0.0001481] |
| xgboost / lead_time | 10 | -0.0006411 | [-0.001101, -0.0001808] |
| xgboost / lead_time | 11 | -0.0003915 | [-0.0008651, 8.212e-05] |
| xgboost / lead_time | 12 | -0.0005718 | [-0.001105, -3.849e-05] |
| xgboost / lead_time | 13 | -0.0004369 | [-0.0009798, 0.000106] |
| xgboost / lead_time | 14 | -0.002538 | [-0.003381, -0.001695] |
| xgboost / lead_time | 15 | -0.002071 | [-0.002924, -0.001217] |
| xgboost / lead_time | 16 | -0.001457 | [-0.002215, -0.0006987] |
| xgboost / lead_time | fog_0_4 | 0.0002795 | [-9.366e-05, 0.0006528] |
| xgboost / lead_time | haze_5_15 | -0.0006131 | [-0.001018, -0.000208] |
| ebm / nowcast | 0 | 1.743e-05 | [-7.374e-05, 0.0001086] |
| ebm / nowcast | 1 | 0.0001093 | [-6.757e-05, 0.0002862] |
| ebm / nowcast | 2 | 0.0001256 | [-8.823e-05, 0.0003394] |
| ebm / nowcast | 3 | 5.664e-05 | [-0.0001753, 0.0002886] |
| ebm / nowcast | 4 | -0.000134 | [-0.0003903, 0.0001223] |
| ebm / nowcast | 5 | -0.0002619 | [-0.0005322, 8.434e-06] |
| ebm / nowcast | 6 | -0.0005315 | [-0.0008204, -0.0002427] |
| ebm / nowcast | 7 | -0.0006549 | [-0.0009642, -0.0003456] |
| ebm / nowcast | 8 | -0.001114 | [-0.00145, -0.000778] |
| ebm / nowcast | 9 | -0.001618 | [-0.001997, -0.001239] |
| ebm / nowcast | 10 | -0.00293 | [-0.003382, -0.002479] |
| ebm / nowcast | 11 | -0.004088 | [-0.004656, -0.00352] |
| ebm / nowcast | 12 | -0.005555 | [-0.006245, -0.004866] |
| ebm / nowcast | 13 | -0.005026 | [-0.005718, -0.004333] |
| ebm / nowcast | 14 | -0.00898 | [-0.009917, -0.008043] |
| ebm / nowcast | 15 | -0.007296 | [-0.008193, -0.006399] |
| ebm / nowcast | 16 | -0.005628 | [-0.006462, -0.004794] |
| ebm / nowcast | fog_0_4 | 3.5e-05 | [-0.0001486, 0.0002185] |
| ebm / nowcast | haze_5_15 | -0.00346 | [-0.003893, -0.003026] |
| ebm / lead_time | 0 | -0.0003673 | [-0.0005685, -0.0001661] |
| ebm / lead_time | 1 | -0.0002622 | [-0.0005562, 3.188e-05] |
| ebm / lead_time | 2 | -0.0004526 | [-0.0007885, -0.0001166] |
| ebm / lead_time | 3 | -0.0006515 | [-0.001032, -0.0002715] |
| ebm / lead_time | 4 | -0.0009344 | [-0.00137, -0.0004991] |
| ebm / lead_time | 5 | -0.001086 | [-0.001539, -0.0006331] |
| ebm / lead_time | 6 | -0.001612 | [-0.002103, -0.00112] |
| ebm / lead_time | 7 | -0.001434 | [-0.001925, -0.0009432] |
| ebm / lead_time | 8 | -0.00195 | [-0.002477, -0.001422] |
| ebm / lead_time | 9 | -0.002243 | [-0.00275, -0.001737] |
| ebm / lead_time | 10 | -0.002478 | [-0.003011, -0.001945] |
| ebm / lead_time | 11 | -0.00202 | [-0.002559, -0.001481] |
| ebm / lead_time | 12 | -0.001998 | [-0.002583, -0.001413] |
| ebm / lead_time | 13 | -0.001534 | [-0.002084, -0.0009844] |
| ebm / lead_time | 14 | -0.00215 | [-0.002997, -0.001303] |
| ebm / lead_time | 15 | -0.00112 | [-0.001922, -0.0003179] |
| ebm / lead_time | 16 | -0.0003976 | [-0.001069, 0.0002737] |
| ebm / lead_time | fog_0_4 | -0.0005336 | [-0.0008439, -0.0002233] |
| ebm / lead_time | haze_5_15 | -0.001784 | [-0.002215, -0.001354] |

## Secondary per-airport primary-NOx estimates

| Family / task | Airport | Metric | Δ | 95% CI |
|---|---|---|---:|---|
| logistic_regression / nowcast | location_10 | AP | 0.02066 | [0.001841, 0.03947] |
| logistic_regression / nowcast | location_6 | AP | -0.005154 | [-0.02653, 0.01622] |
| logistic_regression / nowcast | location_7 | AP | 0.003778 | [-0.006517, 0.01407] |
| logistic_regression / nowcast | location_8 | AP | 0.01451 | [-0.0003288, 0.02934] |
| logistic_regression / nowcast | location_9 | AP | 0.01148 | [0.002105, 0.02086] |
| logistic_regression / nowcast | location_10 | RPS | -0.0006476 | [-0.001146, -0.000149] |
| logistic_regression / nowcast | location_6 | RPS | -0.005521 | [-0.006065, -0.004977] |
| logistic_regression / nowcast | location_7 | RPS | -0.002732 | [-0.002997, -0.002467] |
| logistic_regression / nowcast | location_8 | RPS | -0.004202 | [-0.004911, -0.003494] |
| logistic_regression / nowcast | location_9 | RPS | -0.004889 | [-0.005437, -0.004341] |
| logistic_regression / lead_time | location_10 | AP | 0.007386 | [-0.01519, 0.02996] |
| logistic_regression / lead_time | location_6 | AP | 0.0247 | [0.01036, 0.03905] |
| logistic_regression / lead_time | location_7 | AP | -0.00125 | [-0.005447, 0.002948] |
| logistic_regression / lead_time | location_8 | AP | -0.0004503 | [-0.004718, 0.003818] |
| logistic_regression / lead_time | location_9 | AP | -0.006505 | [-0.01091, -0.002105] |
| logistic_regression / lead_time | location_10 | RPS | -0.001618 | [-0.002338, -0.0008983] |
| logistic_regression / lead_time | location_6 | RPS | -0.003618 | [-0.004402, -0.002834] |
| logistic_regression / lead_time | location_7 | RPS | -0.0002988 | [-0.0004093, -0.0001883] |
| logistic_regression / lead_time | location_8 | RPS | 0.0003129 | [7.774e-05, 0.0005481] |
| logistic_regression / lead_time | location_9 | RPS | 0.0006784 | [0.0003206, 0.001036] |
| xgboost / nowcast | location_10 | AP | -0.04056 | [-0.06313, -0.01798] |
| xgboost / nowcast | location_6 | AP | -0.00922 | [-0.01896, 0.0005158] |
| xgboost / nowcast | location_7 | AP | -0.01706 | [-0.03318, -0.0009333] |
| xgboost / nowcast | location_8 | AP | -0.03255 | [-0.05137, -0.01373] |
| xgboost / nowcast | location_9 | AP | -0.02343 | [-0.03821, -0.00864] |
| xgboost / nowcast | location_10 | RPS | 0.0004221 | [-0.0001523, 0.0009965] |
| xgboost / nowcast | location_6 | RPS | -0.004866 | [-0.005327, -0.004405] |
| xgboost / nowcast | location_7 | RPS | -0.003317 | [-0.003758, -0.002875] |
| xgboost / nowcast | location_8 | RPS | -0.0006033 | [-0.001188, -1.896e-05] |
| xgboost / nowcast | location_9 | RPS | -0.002044 | [-0.002558, -0.00153] |
| xgboost / lead_time | location_10 | AP | -0.03003 | [-0.06432, 0.004264] |
| xgboost / lead_time | location_6 | AP | -0.00198 | [-0.01884, 0.01488] |
| xgboost / lead_time | location_7 | AP | -0.01871 | [-0.03692, -0.0005024] |
| xgboost / lead_time | location_8 | AP | 0.001968 | [-0.01766, 0.0216] |
| xgboost / lead_time | location_9 | AP | -0.01487 | [-0.02921, -0.0005414] |
| xgboost / lead_time | location_10 | RPS | 0.000732 | [-4.998e-06, 0.001469] |
| xgboost / lead_time | location_6 | RPS | -0.002861 | [-0.003416, -0.002305] |
| xgboost / lead_time | location_7 | RPS | -0.001267 | [-0.001832, -0.0007015] |
| xgboost / lead_time | location_8 | RPS | 0.001017 | [0.0004417, 0.001593] |
| xgboost / lead_time | location_9 | RPS | 0.000377 | [-0.0002259, 0.0009798] |
| ebm / nowcast | location_10 | AP | -0.01288 | [-0.02694, 0.001184] |
| ebm / nowcast | location_6 | AP | 0.002742 | [-0.005919, 0.0114] |
| ebm / nowcast | location_7 | AP | -0.01371 | [-0.02531, -0.002106] |
| ebm / nowcast | location_8 | AP | -0.002148 | [-0.01511, 0.01081] |
| ebm / nowcast | location_9 | AP | 0.003074 | [-0.007333, 0.01348] |
| ebm / nowcast | location_10 | RPS | -0.0009708 | [-0.001516, -0.0004256] |
| ebm / nowcast | location_6 | RPS | -0.004506 | [-0.004968, -0.004045] |
| ebm / nowcast | location_7 | RPS | -0.001777 | [-0.00217, -0.001384] |
| ebm / nowcast | location_8 | RPS | -0.002389 | [-0.00305, -0.001729] |
| ebm / nowcast | location_9 | RPS | -0.003153 | [-0.003701, -0.002606] |
| ebm / lead_time | location_10 | AP | -0.005346 | [-0.03624, 0.02555] |
| ebm / lead_time | location_6 | AP | -0.003774 | [-0.01167, 0.004125] |
| ebm / lead_time | location_7 | AP | -0.00396 | [-0.0215, 0.01358] |
| ebm / lead_time | location_8 | AP | 0.02756 | [0.01491, 0.0402] |
| ebm / lead_time | location_9 | AP | 0.01184 | [0.002358, 0.02132] |
| ebm / lead_time | location_10 | RPS | 0.0004793 | [-0.0002909, 0.00125] |
| ebm / lead_time | location_6 | RPS | -0.002949 | [-0.003425, -0.002474] |
| ebm / lead_time | location_7 | RPS | -0.003038 | [-0.003571, -0.002506] |
| ebm / lead_time | location_8 | RPS | -0.0005721 | [-0.001154, 9.92e-06] |
| ebm / lead_time | location_9 | RPS | -0.0005933 | [-0.001132, -5.488e-05] |
