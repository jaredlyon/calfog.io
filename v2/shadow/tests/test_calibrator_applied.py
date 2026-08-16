"""Serving probabilities must go through the persisted calibrator."""
import numpy as np

from v2.shadow.shadow_common import apply_calibrator


class SquareCalibrator:
    def __init__(self):
        self.calls = 0

    def predict(self, probabilities):
        self.calls += 1
        values = np.asarray(probabilities, dtype=float)
        return values**2


def test_calibrator_monotone_mapping_is_actually_applied():
    calibrator = SquareCalibrator()
    raw = np.array([0.1, 0.4, 0.9])

    calibrated = apply_calibrator(calibrator, raw)

    assert calibrator.calls == 1
    np.testing.assert_allclose(calibrated, [0.01, 0.16, 0.81])
    assert np.all(np.diff(calibrated) >= 0)
    assert not np.allclose(calibrated, raw)
    assert np.all((0.0 <= calibrated) & (calibrated <= 1.0))
