import logging
import netCDF4
import numpy as np
import re
import xarray as xr

from .constants import SPECIAL_DESCRIPTION_DICT
from . import get_paths as gp

def add_expanded_aks_to_ds(nc_data, n=500, full_ak_resolution=False, min_extrap=0):
    with netCDF4.Dataset(gp.ak_tables_nc_file()) as ak_nc:
        _add_ak_dim_vars(nc_data, ak_nc)

        xgases_with_aks = [v.strip('_aks') for v in ak_nc.variables.keys() if v.endswith('_aks')]
        for xgas in xgases_with_aks:
            expanded_aks, extrapolation_flags = _create_expanded_aks_from_table(
                ds=nc_data,
                table_ds=ak_nc,
                xgas=xgas,
                n=n,
                full_ak_resolution=full_ak_resolution,
                min_extrap=min_extrap
            )

            ak_varname = f'ak_{xgas}'
            ex_flag_varname = f'extrapolation_flags_ak_{xgas}'

            base_description = ak_nc[f'{xgas}_aks'].description.lower()
            if xgas == 'xlco2':
                description = f'{base_description}. {SPECIAL_DESCRIPTION_DICT["lco2"]}'
            elif xgas == 'xwco2':
                description = f'{base_description}. {SPECIAL_DESCRIPTION_DICT["wco2"]}'
            else:
                description = f'{base_description}.'

            att_dict = {
                'long_name': f'{xgas} column averaging kernel',
                'description': description,
                'units': '',
                'ancillary_variables': ex_flag_varname
            }

            nc_data.createVariable(
                ak_varname, np.float32, ('time', 'ak_altitude'),
                zlib=True, complevel=9
            )
            nc_data[ak_varname].setncatts(att_dict)
            nc_data[ak_varname][:] = expanded_aks.astype(np.float32)

            att_dict = {
                'flag_values': np.array([-2, -1, 0, 1, 2], dtype=extrapolation_flags.dtype),
                'flag_meanings': "clamped_to_min_slant_xgas\nextrapolated_below_lowest_slant_xgas_bin\ninterpolated_normally\nextrapolated_above_largest_slant_xgas_bin\nclamped_to_max_slant_xgas",
                'usage': 'Please see https://tccon-wiki.caltech.edu/Main/GGG2020DataChanges for more information'
            }

            nc_data.createVariable(
                ex_flag_varname, extrapolation_flags.dtype, ('time',),
                zlib=True, complevel=9
            )
            nc_data[ex_flag_varname].setncatts(att_dict)


def add_table_aks_to_ds(nc_data):
    with netCDF4.Dataset(gp.ak_tables_nc_file(),'r') as ak_nc:
        _add_ak_dim_vars(nc_data, ak_nc)
        nbins_ak = ak_nc['slant_xgas_bin'].size

        # dimensions
        nc_data.createDimension('ak_slant_xgas_bin',nbins_ak)

        # coordinate variables
        nc_data.createVariable('ak_slant_xgas_bin',np.int16,('ak_slant_xgas_bin'))
        att_dict = {
            "standard_name": "averaging_kernel_slant_xgas_bin_index",
            "long_name": "averaging kernel slant xgas bin index",
            "description": "Index of the slant xgas bins for the column averaging kernels",
            "units": '',
        }
        nc_data['ak_slant_xgas_bin'].setncatts(att_dict)
        nc_data['ak_slant_xgas_bin'][0:nbins_ak] = np.arange(nbins_ak).astype(np.int16)

        ## create variables

        for ak_bin_var in [i for i in ak_nc.variables if i.startswith('slant') and i!="slant_xgas_bin"]:
            var = 'ak_{}'.format(ak_bin_var)
            nc_data.createVariable(var,np.float32,('ak_slant_xgas_bin'))
            att_dict = {
                "standard_name": var,
                "long_name": var.replace('_',' '),
                "description": ak_nc[ak_bin_var].description.lower()+" (slant_xgas=xgas*airmass)",
                "units": ak_nc[ak_bin_var].units,
            }
            nc_data[var].setncatts(att_dict)
            if var == 'ak_slant_xch4_bin':
                # Need to convert the ppb in the netCDF file to ppm to be consistent with xch4
                nc_data[var][0:nbins_ak] = ak_nc[ak_bin_var][:].data.astype(np.float32) * 1e-3
                nc_data[var].units = 'ppm'
            else:
                nc_data[var][0:nbins_ak] = ak_nc[ak_bin_var][:].data.astype(np.float32)

        for ak_var in [i for i in ak_nc.variables if i.endswith('aks')]:
            var = 'ak_{}'.format(ak_var.strip('_aks'))
            nc_data.createVariable(var,np.float32,('ak_altitude','ak_slant_xgas_bin'))
            att_dict = {
                "standard_name": "{}_column_averaging_kernel".format(ak_var.strip('_aks')),
                "long_name": "{} column averaging kernel".format(ak_var.strip('_aks')),
                "description": ak_nc[ak_var].description.lower()+'. ',
                "units": '',
            }
            if 'lco2' in var:
                att_dict['description'] = att_dict['description']+SPECIAL_DESCRIPTION_DICT['lco2']
            elif 'wco2' in var:
                att_dict['description'] = att_dict['description']+SPECIAL_DESCRIPTION_DICT['wco2']
            nc_data[var].setncatts(att_dict)
            nc_data[var][:] = ak_nc[ak_var][:].data.astype(np.float32)


def expand_aks(ds, xgas, n=500, full_ak_resolution=False, min_extrap=0):
    logging.debug('Expanding AKs for %s', xgas)
    slant_xgas_bins = ds['ak_slant_{}_bin'.format(xgas)][:]
    if xgas == 'xch4' and ds['ak_slant_{}_bin'.format(xgas)].units == 'ppb':
        # XCH4 bins are given in ppb, but XCH4 itself in ppm. Oops!
        slant_xgas_bins = slant_xgas_bins * 1e-3
    aks = ds['ak_{}'.format(xgas)][:]
    return _expand_aks_inner(ds=ds, xgas=xgas, slant_xgas_bins=slant_xgas_bins, aks=aks, n=n, full_ak_resolution=full_ak_resolution, min_extrap=min_extrap)


def _add_ak_dim_vars(nc_data, ak_nc):
    # Add the common dimensions
    nlev_ak = ak_nc['z'].size
    nc_data.createDimension('ak_altitude', nlev_ak) # make it separate from prior_altitude just in case we ever generate new files with different altitudes

    # coordinate variables
    nc_data.createVariable('ak_altitude',np.float32,('ak_altitude'))
    att_dict = {
        "standard_name": "averaging_kernel_altitude_levels",
        "long_name": "averaging kernel altitude levels",
        "description": "Altitude levels for column averaging kernels",
        "units": 'km',
    }
    nc_data['ak_altitude'].setncatts(att_dict)
    nc_data['ak_altitude'][0:nlev_ak] = ak_nc['z'][:].data

    nc_data.createVariable('ak_pressure',np.float32,('ak_altitude'))
    att_dict = {
        "standard_name": "averaging_kernel_pressure_levels",
        "long_name": "averaging kernel pressure levels",
        "description": "Median pressure for the column averaging kernels vertical grid",
        "units": 'hPa',
    }
    nc_data['ak_pressure'].setncatts(att_dict)
    nc_data['ak_pressure'][0:nlev_ak] = ak_nc['pressure'][:].data



def _create_expanded_aks_from_table(ds, table_ds, xgas, n=500, full_ak_resolution=False, min_extrap=0):
    logging.debug('Creating AKs for %s', xgas)
    slant_xgas_bins = table_ds[f'slant_{xgas}_bin'][:].filled(np.nan)
    if xgas == 'xch4' and table_ds[f'slant_{xgas}_bin'].units == 'ppb':
        # XCH4 bins are given in ppb, but XCH4 itself in ppm. Oops!
        slant_xgas_bins = slant_xgas_bins * 1e-3
    aks = table_ds[f'{xgas}_aks'][:].filled(np.nan)

    return _expand_aks_inner(
        ds=ds,
        xgas=xgas,
        slant_xgas_bins=slant_xgas_bins,
        aks=aks,
        n=n,
        full_ak_resolution=full_ak_resolution,
        min_extrap=min_extrap
    )

def _expand_aks_inner(ds, xgas, slant_xgas_bins, aks, n=500, full_ak_resolution=False, min_extrap=0):
    airmass = ds['o2_7885_am_o2'][:]
    if re.match('x[wl]?co2', xgas) and xgas not in ds.variables.keys():
        # The difference in Xgas value between the X2007 and X2019 scale should be minor on the scale
        # of what the AKs care about, so to avoid adding three more AK variables, just use the X2019
        # Xgas value to expand the AKs.
        varname = f'{xgas}_x2019'
        logging.info(f'Using {varname} to compute slant Xgas for {xgas} AKs')
        slant_xgas_values = ds[varname][:] * airmass
    else:
        slant_xgas_values = ds[xgas][:] * airmass
    extrap_flags = np.zeros(slant_xgas_values.shape, dtype=np.int8)
    extrap_flags[slant_xgas_values < min_extrap] = -2
    extrap_flags[(slant_xgas_values >= min_extrap) & (slant_xgas_values < np.min(slant_xgas_bins))] = -1
    extrap_flags[slant_xgas_values > np.max(slant_xgas_bins)] = 2
    if not full_ak_resolution:
        slant_xgas_values = _compute_quantized_slant_xgas(slant_xgas_values, slant_xgas_bins, n=n, min_extrap=min_extrap)
    else:
        slant_xgas_values = np.clip(slant_xgas_values, min_extrap, np.max(slant_xgas_bins))

    expanded_aks = np.full([slant_xgas_values.size, aks.shape[0]], np.nan, dtype=aks.dtype)
    alt = ds['ak_altitude'][:]  # isn't really necessary, but need a coordinate along that dimension anyway
    lookup_aks = xr.DataArray(aks, coords=[alt, slant_xgas_bins], dims=['alt', 'slant_bin'])
    expanded_aks = lookup_aks.interp(slant_bin=slant_xgas_values, kwargs={'fill_value':'extrapolate'})
    expanded_aks = expanded_aks.data.T

    return expanded_aks, extrap_flags


def _compute_quantized_slant_xgas(slant_xgas_values, slant_xgas_bins, n=500, min_extrap=0):
    # Put the individual spectra's slant Xgas values on a smaller number
    # of quantized values ranging between the minimum and maximum values, not allowing
    # the values to go outside of the bins. I decided to base these off of the bins
    # rather than the actual slant xgas values because doing the latter will cause the
    # AKs to change when the public files are updated and there's a wider range of slant
    # xgas variables.
    def quantize(values, minval, maxval, nval):
        si = (values - minval)/(maxval - minval) # normalize to 0 to 1
        si = np.clip(si, 0, 1)
        si = np.round(si * (nval - 1)) # round to values between 0 and (n-1)
        si = si / (nval - 1) * (maxval - minval) + minval # restore original magnitude
        return si

    smin = np.min(slant_xgas_bins)
    smax = np.max(slant_xgas_bins)

    quant_slant = np.full_like(slant_xgas_values, np.nan)

    xx_in = (slant_xgas_values >= smin) & (slant_xgas_values <= smax)
    xx_ex = (slant_xgas_values >= min_extrap) & (slant_xgas_values < smin)
    xx_below = slant_xgas_values < min_extrap
    xx_above = slant_xgas_values > smax

    # First handle values inside the range of the bins
    quant_slant[xx_in] = quantize(slant_xgas_values[xx_in], smin, smax, n)
    # Then the values extrapolated between the bottom bin and 0. Use 10x fewer
    # quantized points that the main region, as this should be a significantly
    # smaller range.
    quant_slant[xx_ex] = quantize(slant_xgas_values[xx_ex], min_extrap, smin, n // 10)
    # Finally set the min and max values
    quant_slant[xx_below] = min_extrap
    quant_slant[xx_above] = smax
    return quant_slant
