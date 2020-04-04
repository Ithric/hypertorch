import pandas as pd
import pytz, datetime
import pendulum as pm
from functools import partial
import numpy as np

def pendulum_to_pytz(dt, ianaTz=None):
    def inner(dt):
        nonlocal ianaTz
        # Convert timezone first
        if ianaTz is None:
            ianaTz = dt.tz.name

        dt = dt.in_tz(ianaTz)

        # Map dt to pytz
        tz = pytz.timezone(ianaTz)
        dt = datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)
        return tz.localize(dt)

    if isinstance(dt, (list,tuple, np.generic,np.ndarray)):
        return list(map(inner, dt))
    else:
        return inner(dt)


def pendulum_to_pytz_naive(dt):
    def map_item(dt):
        return datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)

    if isinstance(dt, (list,tuple, np.generic,np.ndarray)):
        return list(map(map_item, dt))
    else:
        return map_item(dt)

def pytz_to_pendulum(dt):
    assert dt.tzinfo is not None, "Cannot convert naive pytz datetime to pendulum"
    orig_zone = dt.tzinfo.zone
    dt = dt.astimezone(pytz.utc)
    return pm.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond, tz=pm.UTC).in_tz(orig_zone)

def pendulum_to_local(dt : pm.datetime):
    return datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)

def pandas_date_range(from_dt : pm.datetime, to_dt : pm.datetime, freq=None, tz=None):
    try:
        if tz is not None:
            from_dt, to_dt = map(lambda dt: dt.in_tz(tz), (from_dt, to_dt)) # Map from/to tz to target timezone

        from_dt, to_dt = map(partial(pendulum_to_pytz, ianaTz="Europe/Oslo"), (from_dt, to_dt))  
        #print("Target tz=", tz, "given", from_dt, to_dt, from_dt.tzinfo, to_dt.tzinfo)
        df = pd.date_range(from_dt, to_dt, freq=freq, tz=tz)
        return df
    except:
        import traceback
        traceback.print_exc()
        print("PANDAS_DATE_RANGE: from_dt={}, to_dt={}, freq={}, tz={}".format(from_dt, to_dt, freq, tz))
        raise

def apply_timezone(tz, dt : pm.DateTime):
    return pm.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond, tz=tz)