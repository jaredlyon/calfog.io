"""Fast contract tests for the v2 data pipeline (no datasets or training)."""

import unittest

import pandas as pd

from v2.pipeline import (
    audit_causal_features,
    chronological_split,
    make_lead_target,
    make_matched_arms,
    parse_target,
)


class TargetContractTests(unittest.TestCase):
    def test_strict_threshold_and_invalid_sentinel(self):
        source = pd.DataFrame(
            {"row": ["fog", "boundary", "invalid"], "visibility_meters": [1609, 1610, 999999]}
        )
        result = parse_target(source)

        self.assertEqual(result["row"].tolist(), ["fog", "boundary"])
        self.assertEqual(result["fog"].tolist(), [1, 0])
        self.assertEqual(len(source), 3, "transformations must not mutate their input")


class SplitContractTests(unittest.TestCase):
    def test_split_is_sorted_disjoint_and_purged(self):
        # Reverse order catches implementations which split before sorting.
        frame = pd.DataFrame(
            {"issue_time": pd.date_range("2024-01-01", periods=10 * 24, freq="h")[::-1], "x": range(10 * 24)}
        )
        split = chronological_split(
            frame,
            validation_start="2024-01-05 00:00",
            test_start="2024-01-08 00:00",
            purge="24h",
        )

        for part in split:
            self.assertTrue(part["issue_time"].is_monotonic_increasing)
        self.assertLess(split.train["issue_time"].max(), split.validation["issue_time"].min())
        self.assertLess(split.validation["issue_time"].max(), split.test["issue_time"].min())
        self.assertGreaterEqual(
            split.validation["issue_time"].min() - split.train["issue_time"].max(), pd.Timedelta("24h")
        )
        self.assertGreaterEqual(
            split.test["issue_time"].min() - split.validation["issue_time"].max(), pd.Timedelta("24h")
        )
        train_times = set(split.train["issue_time"])
        validation_times = set(split.validation["issue_time"])
        test_times = set(split.test["issue_time"])
        self.assertFalse(train_times & validation_times)
        self.assertFalse(train_times & test_times)
        self.assertFalse(validation_times & test_times)


class MatchedArmContractTests(unittest.TestCase):
    def test_arms_use_same_ordered_key_intersection(self):
        with_aqi = pd.DataFrame(
            {"location_id": [1, 1, 1], "time": ["2024-01-03", "2024-01-01", "2024-01-02"], "aqi": [30, 10, 20]}
        )
        without_aqi = pd.DataFrame(
            {"location_id": [1, 1, 1], "time": ["2024-01-04", "2024-01-02", "2024-01-01"], "temperature": [4, 2, 1]}
        )
        arms = make_matched_arms(with_aqi, without_aqi)
        key_cols = ["location_id", "time"]

        pd.testing.assert_frame_equal(arms.with_aqi[key_cols], arms.without_aqi[key_cols])
        self.assertEqual(arms.with_aqi["time"].tolist(), ["2024-01-01", "2024-01-02"])

    def test_duplicate_observational_units_are_rejected(self):
        duplicated = pd.DataFrame({"location_id": [1, 1], "time": ["t", "t"]})
        unique = pd.DataFrame({"location_id": [1], "time": ["t"]})
        with self.assertRaisesRegex(ValueError, "duplicate row keys"):
            make_matched_arms(duplicated, unique)


class LeadTargetContractTests(unittest.TestCase):
    def test_18h_issue_predicts_any_fog_in_next_day_hours_zero_through_nine(self):
        observations = pd.DataFrame(
            {
                "location_id": [7] * 5,
                "time": pd.to_datetime(
                    [
                        "2024-02-02 00:00",  # boundary included, non-fog
                        "2024-02-02 05:00",  # fog makes label positive
                        "2024-02-02 09:59",  # hour 9 included
                        "2024-02-02 10:00",  # excluded
                        "2024-02-03 03:00",  # next verification day, non-fog
                    ]
                ),
                "fog": [0, 1, 0, 1, 0],
            }
        )
        labels = make_lead_target(observations, require_complete_hours=False)

        self.assertEqual(labels["issue_time"].tolist(), [pd.Timestamp("2024-02-01 18:00"), pd.Timestamp("2024-02-02 18:00")])
        self.assertEqual(labels["fog_next_00_09"].tolist(), [1, 0])
        self.assertEqual(labels["verification_end"].iloc[0], pd.Timestamp("2024-02-02 10:00"))


    def test_incomplete_morning_is_not_a_synthetic_negative(self):
        observations = pd.DataFrame({
            "location_id": [10] * 9,
            "time": pd.date_range("2024-02-02 00:00", periods=9, freq="h"),
            "fog": [0] * 9,
        })
        labels = make_lead_target(observations)
        self.assertTrue(labels.empty)



class CausalFeatureContractTests(unittest.TestCase):
    def test_nowcast_allows_feature_at_label_time_but_not_after(self):
        frame = pd.DataFrame(
            {
                "label_time": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00"]),
                "weather_observed_at": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:01"]),
            }
        )
        audit = audit_causal_features(frame, {"weather": "weather_observed_at"}, task="nowcast")

        self.assertFalse(audit.ok)
        self.assertEqual(audit.violations["row_index"].tolist(), [1])

    def test_lead_uses_issue_time_not_future_label_time(self):
        frame = pd.DataFrame(
            {
                "issue_time": pd.to_datetime(["2024-02-01 18:00"]),
                "verification_end": pd.to_datetime(["2024-02-02 10:00"]),
                "feature_observed_at": pd.to_datetime(["2024-02-01 18:01"]),
            }
        )
        audit = audit_causal_features(
            frame,
            ["feature_observed_at"],
            task="lead",
            label_time_col="verification_end",
        )

        self.assertEqual(audit.violation_count, 1)
        with self.assertRaisesRegex(ValueError, "causal feature audit"):
            audit.raise_for_violations()


if __name__ == "__main__":
    unittest.main()
