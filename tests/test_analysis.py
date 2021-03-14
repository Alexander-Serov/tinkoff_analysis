from datetime import datetime

from pytz import utc

from analysis import get_figi_history


def test_get_figi_history():
    start = datetime(2021, 3, 8, tzinfo=utc)
    end = datetime(2021, 3, 13, tzinfo=utc)
    interval = "day"
    figi = "BBG005HLTYH9"  # FXIT

    res = get_figi_history(start=start, end=end, interval=interval, figi=figi)
    assert not res.empty
    assert (
        (res["time"].dt.date >= start.date()) & (res["time"].dt.date <= end.date())
    ).all()
    assert (res["figi"] == figi).all()
    assert (res["interval"] == interval).all()
    diff = {"o", "c", "h", "l", "v"}.difference(res.columns)
    assert (
        not diff
    ), f"Some of the required data columns were unexpectedly missing: {diff}."
    assert not res.isna().any(axis=None), "Unexpectedly received nan values."
