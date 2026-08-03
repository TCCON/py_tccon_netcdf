from netCDF4 import Dataset, default_fillvals
import numpy as np
import pandas as pd
import re

from typing import Sequence, Union


def compute_daily_esf_date(esf_df: pd.DataFrame, as_timestamp: bool = True) -> Union[np.ndarray, pd.DatetimeIndex]:
    """Compute the dates for the error scale factor data

    The error scale factor file has year and day of year in decimal format for the
    last spectrum in a given day, so this function will do its best to round them down
    to the actual start date for the measurements.

    Parameters
    ----------
    esf_df
        DataFrame containing at least "year" and "day" columns containing the
        decimal year and 1-based day-of-year, respectively.

    as_timestamp
        Return an array of floating point Unix timestamps (``True``) or a
        pandas DatetimeIndex (``False``)

    Returns
    -------
    dates
        The ESF dates in the format determined by ``as_timestamp``.
    """
    year = esf_df.year.to_numpy()
    doy = esf_df.day.to_numpy()
    # This should ensure we never accidentally get the next year; subtracting
    # off the day of year should get us withing rounding distance of the actual
    # integer year
    year = np.round(year - doy/365.25)
    dates = pd.to_datetime({'year': year, 'month': 1, 'day': 1})
    dt = pd.to_timedelta(np.floor(doy)-1, unit='days')
    dates = dates + dt
    if as_timestamp:
        dates = (dates - pd.Timestamp(1970,1,1)).dt.total_seconds().astype(int)
    return dates


def write_daily_esf_data(ds: Dataset, esf_df: pd.DataFrame) -> None:
    """Write the daily error scale data to a private netCDF file

    This will create the needed "daily_error_date" dimension as well as the variables.

    Parameters
    ----------
    ds
        The netCDF dataset to write to.

    esf_df
        A dataframe containing columns "year", "day", and "n" followed by xgas/xgas_error
        column pairs.
    """
    dates = compute_daily_esf_date(esf_df)
    date_dimname = 'daily_error_date'
    ds.createDimension(date_dimname, dates.size)

    # Handle the special variables first to ensure they are always added
    var = ds.createVariable(date_dimname, 'f8', (date_dimname,))
    var.setncatts({
        'units': 'seconds since 1970-01-01 00:00:00',
        'note': 'Due to idiosyncrasies of how GGG post processing handles date/times, in some cases, this may be one day after the actual date of the error scale factor'
    })
    var[:] = dates

    var = ds.createVariable('daily_error_doy', 'f8', (date_dimname,))
    var.setncatts({
        'units': 'decimal day of year',
        'description': 'The GGG decimal day of year for the final spectrum on this date'
    })
    var[:] = esf_df.day.to_numpy()

    var = ds.createVariable('daily_error_num_obs', 'i4', (date_dimname,))
    var.setncatts({
        'description': 'The number of observations used to compute the daily error scale factor'
    })
    var[:] = esf_df.n.fillna(default_fillvals['u4']).to_numpy().astype(int)

    for colname, colvals in esf_df.items():
        if colname in {'year', 'day', 'n'}:
            # Special case, already handled
            continue

        if colname.endswith('_error'):
            description = 'Uncertainty in the daily error scale factor, assuming that a day with only two observations very close in time should have 100% uncertainty in the ESF'
        else:
            description = 'Daily error scale factor, computed as a weighted mean ratio of the difference between successive measurements to the quadrature sum of their errors'

        var = ds.createVariable(f'daily_error_{colname}', 'f4', (date_dimname,))
        var.setncatts({
            'units': '1',
            'description': description
        })
        var[:] = colvals.to_numpy()


def get_xgases_for_esf_df(ds: Dataset) -> Sequence[str]:
    """Find the Xgas variables for the given dataset for which we should write daily error scale factor values.
    """
    nc_xgases = [v for v in ds.variables.keys() if re.match('x[a-z0-9]+(_insb|_si)?$', v)]
    for xgas in ['xco2', 'xlco2', 'xwco2']:
        if xgas not in nc_xgases:
            # Probably missed because of the x2007/x2019 suffixes
            nc_xgases.append(xgas)

    # Now handle the insb -> m and _si -> v remapping. Remember that Debra's post processing
    # script only inserts "m" or "v" if the Xgas name from the InSb or Si detector would conflict
    # with and InGaAs one. Start by getting just the InGaAs ones
    esf_xgases = [x for x in nc_xgases if re.match('x[a-z0-9]+$', x)]
    for xgas in nc_xgases:
        xgas_root = xgas.split('_')[0]
        has_ingaas = xgas_root in esf_xgases
        if xgas.endswith('_insb') and has_ingaas:
            esf_xgases.append(f'xm{xgas_root[1:]}')
        elif xgas.endswith('_si') and has_ingaas:
            esf_xgases.append(f'xv{xgas_root[1:]}')
        elif xgas_root not in esf_xgases:
            esf_xgases.append(xgas_root)

    return esf_xgases


def make_dummy_esf_df(ds: Dataset) -> pd.DataFrame:
    """Make a dataframe suitable for :func:`write_daily_esf_data` with fill ESF values.

    The dates and number of observations will be determined from the dataset; the ESF
    values and their uncertainty will be the default fill values for a netCDF float (f4)
    variable.
    """
    xgases = get_xgases_for_esf_df(ds)
    yr_doy_hr = pd.DataFrame({'year': ds['year'][:], 'doy': ds['day'][:], 'hour': ds['hour'][:]})
    esf_date_df = yr_doy_hr.groupby(['year', 'doy']).max()
    esf_n_df = yr_doy_hr.groupby(['year', 'doy']).count()

    # Approximate the post processing date calculation - no need to be exact, as this
    # will be for fill values anyway
    year = esf_date_df.index.get_level_values('year').to_numpy()
    doy = esf_date_df.index.get_level_values('doy').to_numpy()
    hour = esf_date_df.hour.to_numpy()

    doy = doy + hour/24.0
    year = year + doy / 365.25
    dummy_dict = {
        'year': year,
        'day': doy,
        'n': esf_n_df.hour.to_numpy(),
    }
    for xgas in xgases:
        dummy_dict[xgas] = np.full(year.size, default_fillvals['f4'])
        dummy_dict[f'{xgas}_error'] = np.full(year.size, default_fillvals['f4'])
    return pd.DataFrame(dummy_dict)
