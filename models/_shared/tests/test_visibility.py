import math

import numpy as np
import pandas as pd
import pytest

from models._shared.visibility import (
    CATEGORY_LOWER_M, CATEGORY_NOMINAL_M, CATEGORY_UPPER_M, SM_TO_M,
    US_SYNOP_KM, decode_ghcnh, decode_statute_miles, hourly_labels,
    parse_rem_prevailing_sm, sm_to_category,
)


def reports(km, sources=None, types=None, rem=None, dates=None):
    n = len(km)
    return pd.DataFrame({
        "DATE": dates if dates is not None else pd.date_range("2026-01-01T08:00:00", periods=n, freq="min"),
        "visibility": km,
        "visibility_Source_Code": sources if sources is not None else ["343"] * n,
        "visibility_Report_Type": types if types is not None else ["FM15"] * n,
        "REM": rem if rem is not None else [""] * n,
    })


@pytest.mark.parametrize("q,category", [(7/8, 4), (1, 4), (9/8, 5), (1/8, 0), (12, 17)])
def test_category_boundaries(q, category):
    assert sm_to_category(q) == category


def test_category_meter_bounds():
    assert CATEGORY_LOWER_M[0] == 0
    assert CATEGORY_UPPER_M[0] == 0.25 * SM_TO_M
    assert math.isnan(CATEGORY_NOMINAL_M[0])
    assert CATEGORY_NOMINAL_M[17] == 10 * SM_TO_M
    assert CATEGORY_UPPER_M[17] == math.inf


def test_one_mile_profiles_and_date_independence():
    frame = reports(
        ["1.6", "1.609", "1.61", "1.6"],
        sources=["335", "343", "413", "223"], types=["SAO", "FM15", "FM15", "FM12"],
        dates=["1928-01-01", "1999-01-01", "2026-01-01", "2026-07-01"],
    )
    frame.index = [8, 3, 6, 1]
    decoded = decode_ghcnh(frame)
    assert decoded.index.equals(frame.index)
    assert decoded.cat_lo.tolist() == [4] * 4
    assert decoded.cat_hi.tolist() == [4] * 4
    assert decoded.q_sm.iloc[:3].tolist() == [1.0] * 3
    assert math.isnan(decoded.q_sm.iloc[3])  # US-VV and native metric agree only on category.
    np.testing.assert_allclose(decoded.canonical_m, SM_TO_M)
    assert decoded.fog.tolist() == [True] * 4
    assert str(decoded.time_utc.dt.tz) == "UTC"
    assert decoded.columns.tolist() == [
        "time_utc", "report_type", "source_code", "visibility_km", "rem_sm", "q_sm",
        "cat_lo", "cat_hi", "status", "canonical_m", "lower_m", "upper_m", "fog", "raw_agrees", "time_from_rem",
    ]


def test_zero_and_truncated_half_mile():
    decoded = decode_ghcnh(reports(["0.0", "0.804"], sources=["343", "223"]))
    assert decoded.status.tolist() == ["censored_low", "identified"]
    assert decoded.cat_lo.tolist() == [0, 2]
    assert decoded.cat_hi.tolist() == [0, 2]
    assert math.isnan(decoded.q_sm.iloc[0])
    assert math.isnan(decoded.canonical_m.iloc[0])
    assert decoded.upper_m.iloc[0] == 0.25 * SM_TO_M
    assert decoded.q_sm.iloc[1] == 0.5
    assert decoded.canonical_m.iloc[1] == 0.5 * SM_TO_M


def test_unknown_missing_high_and_raw_conflict():
    decoded = decode_ghcnh(reports(
        ["0.805", "", "1.7", "16.093", "19.312", "128.747"],
        sources=["223", "343", "413", "343", "315", "315"],
        rem=["KMAE 010800Z 00000KT 2 1/2SM", "", "", "", "", ""],
    ))
    assert decoded.status.tolist() == [
        "ambiguous", "missing", "ambiguous", "censored_high", "censored_high", "censored_high",
    ]
    assert (decoded.cat_lo.iloc[0], decoded.cat_hi.iloc[0]) == (2, 9)
    assert not decoded.raw_agrees.iloc[0]
    assert pd.isna(decoded.fog.iloc[0])
    assert math.isnan(decoded.canonical_m.iloc[0])
    assert pd.isna(decoded.cat_lo.iloc[1])
    assert pd.isna(decoded.fog.iloc[1])
    assert (decoded.cat_lo.iloc[2], decoded.cat_hi.iloc[2]) == (0, 17)
    assert decoded.cat_hi.iloc[-1] == 17  # 80 SM: manual extension, not capped lookup.
    assert decoded.canonical_m.iloc[3] == 10 * SM_TO_M


def test_synop_table_is_not_approximate_miles():
    assert US_SYNOP_KM[4] == 6
    assert US_SYNOP_KM[20] == 30
    decoded = decode_ghcnh(reports(
        ["6.0", "30.0", "32.0"], sources=["223", "223", "335"], types=["FM12", "FM12", "SAO"],
    ))
    assert decoded.q_sm.iloc[:2].isna().all()
    assert decoded.q_sm.iloc[2] == 20.0
    assert decoded.cat_lo.tolist() == [11, 17, 17]


@pytest.mark.parametrize("rem,expected", [
    ("MET040KAAA 010000Z 00000KT 1 1/2SM RMK TWR VIS 1/4SM", 1.5),
    ("KAAA 010000Z 00000KT M1/4SM", 0.0),
    ("KAAA 010000Z 00000KT 0SM", 0.0),
    ("KAAA 010000Z 00000KT P6SM", math.inf),
    ("AWY030AAA SA 0055 CLR 1/2 F 012/50/50/0000", 0.5),
    ("KAAA 010000Z 00000KT1SM", 1.0),
])
def test_prevailing_parser(rem, expected):
    assert parse_rem_prevailing_sm(rem) == expected


@pytest.mark.parametrize("rem", [
    "MET030NEW KAAA 010000Z 00000KT 1SM",
    "KAAA 010000Z 00000KT 0800 R01/0600 RMK VIS N 1/8SM",
    "KAAA 010000Z 00000KT R01/0800FT RMK TWR VIS 1/4SM",
    "RMK VIS 1/4SM", "", "KAAA 010000Z 00000KT 1/0SM",
])
def test_parser_does_not_mine_remarks(rem):
    assert math.isnan(parse_rem_prevailing_sm(rem))


def test_decimal_statute_miles_and_raw_validator():
    decoded = decode_statute_miles(
        pd.Series([0.12, "10+", "M1/4", 1, "bad", 1.7, "P6SM"]),
        raw=pd.Series(["", "", "", "KAAA 010000Z 00000KT 2SM", "", "", ""]),
    )
    assert decoded.cat_lo.iloc[:4].tolist() == [0, 17, 0, 4]
    assert decoded.q_sm.iloc[0] == 1/8
    assert decoded.status.iloc[3] == "ambiguous"
    assert decoded.cat_hi.iloc[3] == 8
    assert decoded.status.iloc[4] == "missing"
    assert decoded.status.iloc[5] == "ambiguous"
    assert decoded.cat_lo.iloc[6] == 17
    assert decoded.columns.tolist() == [
        "rem_sm", "q_sm", "cat_lo", "cat_hi", "status", "canonical_m", "lower_m", "upper_m", "fog", "raw_agrees",
    ]


def test_hourly_fixed_pst_minimum_and_ambiguous_propagation():
    obs = pd.DataFrame({
        "time_utc": [
            "2026-01-01T08:05Z", "2026-01-01T08:55Z", "2026-07-01T08:05Z",
            "2026-07-01T08:55Z", "2026-07-01T09:05Z", "2026-07-01T10:05Z",
        ],
        "cat_lo": pd.array([8, 4, 0, 8, 0, None], dtype="Int64"),
        "cat_hi": pd.array([8, 4, 17, 8, 17, None], dtype="Int64"),
        "status": ["identified", "identified", "ambiguous", "identified", "ambiguous", "missing"],
    })
    out = hourly_labels(obs)
    assert out.columns.tolist() == ["time", "cat_lo", "cat_hi", "category", "fog", "n_reports"]
    assert out.time.tolist() == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-01T01:00")]
    assert out.cat_lo.tolist() == [4, 0, 0]
    assert out.cat_hi.tolist() == [4, 8, 17]
    assert out.n_reports.tolist() == [2, 2, 1]
    assert out.category.iloc[0] == 4
    assert out.category.iloc[1:].isna().all()
    assert out.fog.iloc[0]
    assert out.fog.iloc[1:].isna().all()
    assert str(out.fog.dtype) == "boolean"


def test_empty_hourly_and_statute_outputs_keep_schema():
    out = hourly_labels(pd.DataFrame(columns=["time_utc", "cat_lo", "cat_hi"]))
    assert out.empty
    assert str(out.category.dtype) == "Int64"
    assert str(out.fog.dtype) == "boolean"
    assert decode_statute_miles(pd.Series([], dtype=float)).empty


def test_statute_missing_values_are_missing_not_errors():
    # IEM vsby read by pandas: float column with NaN for missing reports
    decoded = decode_statute_miles(pd.Series([0.25, float("nan"), 10.0]))
    assert decoded.status.tolist()[1] == "missing"
    assert decoded.cat_lo.tolist()[0] == 1 and decoded.cat_lo.tolist()[2] == 17


def test_script_combines_split_hours_and_keeps_full_observations(tmp_path, monkeypatch):
    from pathlib import Path
    import runpy

    script = Path(__file__).resolve().parents[3] / "datasets/scripts/visibility_ghcnh.py"
    convert_site = runpy.run_path(str(script))["convert_site"]
    source = tmp_path / "synthetic.psv"
    reports(
        ["1.609", "3.219", "0.805", "0.805"],
        dates=["1979-12-31T08:05:00", "2026-07-01T08:05:00", "2026-07-01T08:55:00", "2026-07-01T09:00:00"],
        rem=["", "", "KVIS 010855Z 00000KT 1/2SM", "KVIS 010855Z 00000KT 1/2SM"],
    ).to_csv(source, sep="|", index=False)
    original_read = pd.read_csv

    def split_chunks(path, **kwargs):
        if kwargs.get("sep") == "|":
            assert kwargs["usecols"] == [
                "DATE", "visibility", "visibility_Source_Code", "visibility_Report_Type", "REM",
            ]
            assert kwargs["dtype"] is str
            kwargs["chunksize"] = 1
        return original_read(path, **kwargs)

    monkeypatch.setattr(pd, "read_csv", split_chunks)
    convert_site("location_6", source, tmp_path)
    obs = original_read(tmp_path / "location_6_visibility_ghcnh_obs.csv")
    hourly = original_read(tmp_path / "location_6_visibility_hourly.csv")
    assert len(obs) == 3
    assert obs.site.tolist() == ["location_6"] * 3
    assert hourly.category.tolist() == [2]
    assert hourly.n_reports.tolist() == [2]
    assert hourly.time.tolist() == ["2026-07-01 00:00:00"]


def test_observed_fine_grid_values_recover_without_full_range_ambiguity():
    decoded = decode_ghcnh(reports(
        ["5.633", "6.035", "1.3"], sources=["223", "223", "335"],
    ))
    assert decoded.q_sm.tolist() == [3.5, 3.75, 13/16]
    assert decoded.cat_lo.tolist() == [11, 11, 4]
    assert decoded.cat_hi.tolist() == [11, 11, 4]
    assert decoded.status.tolist() == ["identified"] * 3


@pytest.mark.parametrize("report", ["FM12", "SYSA", "SYMT", "SYNO"])
def test_synop_native_metric_candidate(report):
    decoded = decode_ghcnh(reports(
        ["9.0", "12.0", "11.2"], sources=["335", "223", "223"], types=[report] * 3,
    ))
    np.testing.assert_allclose(decoded.q_sm, np.array([9.0, 12.0, 11.2]) / 1.609344)
    assert decoded.cat_lo.tolist() == [13, 15, 14]
    assert decoded.cat_hi.tolist() == [13, 15, 14]
    assert decoded.status.tolist() == ["identified"] * 3


def test_top_code_applies_independent_of_profile():
    decoded = decode_ghcnh(reports(
        ["15.95", "16.0", "19.2", "32.0", "48.0"],
        sources=["unknown"] * 5,
    ))
    assert decoded.status.tolist() == ["censored_high"] * 5
    assert decoded.cat_lo.tolist() == [17] * 5
    assert decoded.cat_hi.tolist() == [17] * 5
    np.testing.assert_allclose(decoded.canonical_m, 10 * SM_TO_M)
    assert not decoded.fog.any()


def test_rem_time_correction_and_exact_duplicate_removal():
    decoded = decode_ghcnh(reports(
        ["1.609", "1.609", "0.805", "3.219"],
        sources=["343", "223", "223", "223"],
        dates=["2006-01-03T22:55:00", "2006-01-03T23:00:00",
               "2006-01-03T23:00:00", "2006-01-03T23:00:00"],
        rem=["KVIS 032255Z 00000KT 1SM", "KVIS 032255Z 00000KT 1SM",
             "KVIS 032255Z 00000KT 1/2SM", "KVIS 032100Z 00000KT 2SM"],
    ))
    assert decoded.index.tolist() == [0, 2, 3]
    assert decoded.time_utc.tolist() == [
        pd.Timestamp("2006-01-03T22:55:00Z"), pd.Timestamp("2006-01-03T22:55:00Z"),
        pd.Timestamp("2006-01-03T23:00:00Z"),
    ]
    assert decoded.time_from_rem.tolist() == [True, True, False]
    hourly = hourly_labels(decoded)
    assert hourly.time.tolist() == [pd.Timestamp("2006-01-03T14:00:00"), pd.Timestamp("2006-01-03T15:00:00")]
    assert hourly.n_reports.tolist() == [2, 1]
    assert hourly.category.tolist() == [2, 8]


@pytest.mark.parametrize("date,token,expected", [
    ("2006-01-01T00:00:00", "312355", "2005-12-31T23:55:00Z"),
    ("2006-01-31T23:55:00", "010005", "2006-02-01T00:05:00Z"),
    ("2006-02-01T00:00:00", "312300", "2006-01-31T23:00:00Z"),
])
def test_rem_time_resolves_adjacent_months_with_inclusive_hour_limit(date, token, expected):
    decoded = decode_ghcnh(reports(
        ["1.609"], dates=[date], rem=[f"KVIS {token}Z 00000KT 1SM"],
    ))
    assert decoded.time_utc.iloc[0] == pd.Timestamp(expected)
    assert decoded.time_from_rem.iloc[0]


def test_low_bin_rem_conflict_remains_ambiguous():
    decoded = decode_ghcnh(reports(
        ["0.402"], rem=["KVIS 010800Z 00000KT M1/4SM"],
    ))
    assert decoded.status.iloc[0] == "ambiguous"
    assert (decoded.cat_lo.iloc[0], decoded.cat_hi.iloc[0]) == (0, 1)
    assert not decoded.raw_agrees.iloc[0]
