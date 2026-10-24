"""Fast checks on the actual metrics-producing window code."""
import numpy as np
import pandas as pd
import pytest
from models._shared.data import WEATHER_COLS, CAL_COLS
from models._shared.windows import build_windows


def _weather(times):
    d={"time":times}
    for c in WEATHER_COLS+CAL_COLS:
        d[c]=np.arange(len(times),dtype=float)
    return pd.DataFrame(d)


def test_production_window_has_24_real_hours_and_ends_at_issue():
    times=pd.date_range("2024-01-01",periods=30,freq="h")
    issue=times[-2]
    part=pd.DataFrame({"issue_time":[issue],"category":[17],"row_key":[str(issue)]})
    X,c,meta,_=build_windows(_weather(times),part)
    assert X.shape==(1,24,len(WEATHER_COLS+CAL_COLS))
    assert meta.issue_time.iloc[0]==issue and c.tolist()==[17]
    np.testing.assert_array_equal(X[0], np.repeat(np.arange(5,29)[:,None],len(WEATHER_COLS+CAL_COLS),axis=1))


def test_production_window_rejects_a_clock_gap():
    times=pd.date_range("2024-01-01",periods=30,freq="h").delete(10)
    issue=times[-2]
    part=pd.DataFrame({"issue_time":[issue],"category":[17],"row_key":[str(issue)]})
    with pytest.raises(AssertionError,match="real-hour"):
        build_windows(_weather(times),part)
