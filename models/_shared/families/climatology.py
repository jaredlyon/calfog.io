from __future__ import annotations

import numpy as np
from . import FamilyOutput
from ..visibility import N_CATEGORIES

NEEDS_CNN_CFG = False

def fit_predict(prep, *, cnn_cfg: dict | None, seed: int) -> FamilyOutput:
    frequencies = np.bincount(prep.ctr, minlength=N_CATEGORIES) / len(prep.ctr)
    return FamilyOutput(np.tile(frequencies, (len(prep.cv), 1)),
                        np.tile(frequencies, (len(prep.ct), 1)))
