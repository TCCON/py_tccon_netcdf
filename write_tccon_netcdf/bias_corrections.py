import pandas as pd
import netCDF4 as ncdf
import numpy as np
from scipy.interpolate import interp1d

from typing import Optional, Sequence

XLUFT_BIAS_PARAMS = {
    'xco2': (0.353, 0.647),
    'xwco2': (0.154, 0.846),
    'xlco2': (0.849, 0.152),
}

def correct_xco2_from_xluft(ds: ncdf.Dataset, variables: Sequence[str], use_flags=False, use_daily_median=False, daily_extra_days=2):
    lon = ds['long'][:].filled(np.nan)
    xluft = ds['xluft'][:].filled(np.nan)
    times = pd.Timestamp(1970, 1, 1) + pd.TimedeltaIndex(ds['time'][:], unit='s')
    if use_flags:
        flags = ds['flag'][:].filled(999)
    else:
        flags = None
    
    rolled_df = _roll_xluft_for_bias_corr(times=times, xluft=xluft, lon=lon, flags=flags, use_daily_median=use_daily_median, daily_extra_days=daily_extra_days)
    rolled_xluft = rolled_df['y'].to_numpy()

    out = {
        'xluft_rolled': rolled_xluft, 
        'xluft_raw': xluft,
        'flag': flags
    }
    
    for varname in variables:
        xco2 = ds[varname][:].filled(np.nan)
        m, b = XLUFT_BIAS_PARAMS[varname]
        xco2_corrected = _apply_xluft_bc(xco2, rolled_xluft, m, b)
        out[f'{varname}_corr'] = xco2_corrected
        out[f'{varname}_orig'] = xco2

    attrs = {
        'median_window_type': f'{daily_extra_days*2 + 1} days' if use_daily_median else 'rolling',
        'median_used_flag0_only': 'yes' if use_flags else 'no'
    }

    return pd.DataFrame(out, index=times), attrs


def _apply_xluft_bc(xco2, rolling_xluft, m, b):
    return xco2 / (m * rolling_xluft + b)


def _roll_xluft_for_bias_corr(times: pd.DatetimeIndex, xluft: np.ndarray, lon: np.ndarray, flags: Optional[np.ndarray], npts: int = 500,
                              gap: pd.Timedelta = pd.Timedelta(days=30), use_daily_median: bool = False, daily_extra_days: int = 0):
    if flags is None:
        if use_daily_median:
            return _median_by_day(utc_times=times, lons=lon, yvals=xluft, dedup=False)
        else:
            return _roll_data(times=times, yvals=xluft, npts=npts, gap=gap, dedup=False).set_index('times')
    
    # If we're given flags, then we need to do the rolling on only good quality data, then fill back in.
    # If there are duplicate times, roll_data will remove them, then reindexing back to the original
    # times should both expand to the full set of flagged and unflagged data, plus restore the duplicates
    # so that we can reassign the CO2 and unmedianed Xluft
    qq = flags == 0
    if use_daily_median:
        rolled_df = _median_by_day(utc_times=times[qq], lons=lon[qq], yvals=xluft[qq], extra_ndays=daily_extra_days).set_index('times')
    else:
        rolled_df = _roll_data(times=times[qq], yvals=xluft[qq], npts=npts, gap=gap).set_index('times')
    rolled_df = rolled_df.reindex(times)
    rolled_df['filled'] = rolled_df['y'].isna()
    # Surprisingly, this seems okay with out-of-order times in the index, tried:
    #    pd.DataFrame(
    #        {'x': [0.0, 2.0, np.nan]}, 
    #        index=pd.DatetimeIndex(['2020-01-01', '2020-01-03', '2020-01-02'])
    #    ).interpolate(method='time')
    # and it put 1.0 in the 2020-01-02 slot (for pandas v0.24)
    rolled_df['y'] = rolled_df['y'].interpolate(method='time')
    if np.isnan(rolled_df['y'].iloc[0]):
        # Assume that only entries at the beginning weren't filled for some reason.
        # Do a simple backfill as long as that is the case
        ifirst = np.flatnonzero(~rolled_df['y'].isna())[0]
        assert rolled_df['y'].iloc[:ifirst].isna().all()
        assert np.all(rolled_df.index[:ifirst] < rolled_df.index[ifirst])
        rolled_df['y'].fillna(method='backfill', inplace=True)
    return rolled_df


def _roll_data(times: pd.DatetimeIndex, yvals: np.ndarray, npts: int, gap: pd.Timedelta, dedup=True):
    """Compute rolling statistics on data

    Parameters
    ----------
    xvals
        array of x values
    yvals
        array of y values
    npts
        number of points to use in the rolling window
    gap
        a `Pandas Timedelta specifying the longest time difference between adjacent points that a rolling window 
        can operate over. That is, if this is set to "7 days" and there is a data gap a 14 days, the data before 
        and after that gap will have the rolling operation applied to each separately.
    times
        if ``xvals`` is not a time variable, then this input must be the times corresponding to the ``xvals``
        and ``yvals``. It is used to split on gaps.

    Returns
    -------
    output
        A single dataframe with "x", "y", and (if ``times` was given) "time" columns that has the result of the 
        rolling operation.
    """
    df = pd.DataFrame({'times': times, 'y': yvals})
    time_col = 'times'
    
    if dedup:
        df = _dedup_by_times(df)

    grouped_df = _split_by_gaps(df, gap, time_col)

    results = []
    roll_cols = ['y']
    for n, group in grouped_df:
        result = group[roll_cols].rolling(npts, center=True, min_periods=1).median()

        # Times cannot be computed with rolling operations, therefore we need to copy the
        # time column back over. Also copy the unrolled values for comparison.
        result['y_orig'] = group['y']
        result[time_col] = group[time_col]
        results.append(result)
    return pd.concat(results)


def _median_by_day(utc_times: pd.DatetimeIndex, lons: np.ndarray, yvals: np.ndarray, dedup=True, extra_ndays=0):
    local_times = utc_times + pd.TimedeltaIndex(lons / 15.0, unit='h')
    local_dates = local_times.date
    df = pd.DataFrame({'times': utc_times, 'y_orig': yvals, 'date': local_dates})
    if dedup:
        df = _dedup_by_times(df)
        
    if extra_ndays > 0:
        return _median_by_day_windows(df, extra_ndays)
    else:
        return _median_by_day_simple(df)
        

def _median_by_day_simple(df: pd.DataFrame):
    for d, group in df.groupby('date'):
        df.loc[group.index, 'y'] = group['y_orig'].median()
    return df


def _median_by_day_windows(df: pd.DataFrame, extra_ndays: int):
    grouped_df = _split_by_gaps(df, pd.Timedelta(days=30), 'times')
    for _, group_df in grouped_df:
        med_ser = group_df.groupby('date').median()['y_orig']
        smoothed_med_ser = _moving_date_median(med_ser, extra_ndays)
        for date, value in smoothed_med_ser.items():
            df.loc[df['date'] == date, 'y'] = value
    return df


def _moving_date_median(series_in: pd.Series, ndays: int) -> pd.Series:
    """Calculate a moving median across date-wise data 
    
    ndays specifies the half width of the window in normal circumstances; that means with ndays = 1, the window
    will be 3 wide. For points near the beginning or end of the record, the window will shift off-center to try
    to keep the expected number of points. That is, the first ndays + 1 points will use the same window, as will
    the last ndays + 1 points.
    """
    series_in = series_in.sort_index()
    series_out = pd.Series(np.nan, index=series_in.index)
    ntot = series_in.size
    for i, (date, val) in enumerate(series_in.items()):
        if i < ndays:
            # beginning of the record, need to shift the window later to get a full width
            imin = 0
            imax = min(ndays*2 + 1, ntot)
        elif i >= (ntot - ndays):
            # end of the record, need to shift the window earlier to get a full width
            imin = max(ntot - (ndays*2 + 1), 0)
            imax = ntot
        else:
            # middle of the record, no need to shift the window
            imin = max(i - ndays, 0)
            imax = min(i + ndays + 1, ntot)
            
        subset = series_in.iloc[imin:imax]
        series_out[date] = subset.median()
        
    return series_out

    
def _dedup_by_times(df):
    is_dup_time = df.duplicated(subset='times', keep=False)
    is_dup_all = df.duplicated(keep=False)
    if is_dup_time.any() and not (is_dup_time == is_dup_all).all():
        raise NotImplementedError('Cannot deduplicate if some times are duplicates with corresponding non-duplicate y')
    elif is_dup_time.any():
        to_remove = df.duplicated(subset='times', keep='first')
        df = df.loc[~to_remove, :]
        print(f'\nNote: {to_remove.sum()} duplicate times removed')
    return df
    
def _split_by_gaps(df: pd.DataFrame, gap: pd.Timedelta, time):
    """Split an input dataframe with a datetime variable into a groupby dataframe with each group having data without gaps larger than the "gap" variable

    Parameters
    ----------
    df
        pandas dataframe with a datetime column
    gap
        minimum gap length
    time
        column name of the datetime variable

    Returns
    -------
    df_by_gaps
        groupby object with each group having data without gaps larger than the "gap" variable
    """

    df['isgap'] = df[time].diff() > gap
    df['isgap'] = df['isgap'].apply(lambda x: 1 if x else 0).cumsum()
    df_by_gaps = df.groupby('isgap')

    return df_by_gaps


def correct_xn2o_from_pt700(ds, m=0.000646, b=0.782):
    xn2o = ds['xn2o'][:]
    n2o_aicf = ds['xn2o_aicf'][:]

    # We need to remove the AICF because the correction was calculated for pre-AICF XN2O data
    xn2o = xn2o * n2o_aicf
    pt700 = _compute_pt700(ds)
    # Apply the temperature correction. Counterintuitively, we do *NOT* need to reapply the AICF.
    # That is only because for N2O the AICF was calculated as the value of this fit at 310 K.
    # Hence, essentially this is applying a *temperature dependent* AICF. 
    xn2o_corr = xn2o / (m * pt700 + b)

    return xn2o_corr, pt700    


def _compute_pt700(ds):
    t = ds['prior_temperature'][:]
    # convert atm -> hPa
    p = 1013.25 * ds['prior_pressure'][:]
    pt700 = np.full(t.shape[0], np.nan, dtype=t.dtype)
    for (i, (tprof, pprof)) in enumerate(zip(t, p)):
        f = interp1d(pprof, tprof)
        t700_i = f(700.0)
        pt700[i] = t700_i * (1000.0 / 700.0) ** 0.286

    prior_index = ds['prior_index'][:]
    return pt700[prior_index]
