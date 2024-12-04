from datetime import datetime
import logging
import netCDF4
import numpy as np
import os
import pandas as pd
import re

from typing import Union

DEFAULT_EFF_PATH_VARNAME = 'effective_path_length'
DEFAULT_O2_DMF_VARNAME = 'o2_mean_mole_fraction'
DEFAULT_NAIR_VARNAME = 'prior_density'
DEFAULT_O2_RET_COL_VARNAME = 'vsw_o2_7885'
DEFAULT_INT_OP_VARNAME = 'integration_operator'
TCCON_PRIOR_XGAS_OVC_VARS = {
    "co2": "co2_6220_ovc_co2",
    "ch4": "ch4_5938_ovc_ch4",
    "n2o": "n2o_4395_ovc_n2o",
    "co": "co_4290_ovc_co",
    "hf": "hf_4038_ovc_hf",
    "h2o": "h2o_4565_ovc_h2o",
    "hdo": "hdo_4054_ovc_hdo",
}
MOLE_FRACTION_CONVERSIONS = {
    '1': 1.0,
    'parts': 1.0,
    'ppm': 1e6,
    'ppb': 1e9,
    'ppt': 1e12,
}

GEOS_VERSION_DESCRIPTIONS = {
    'Met2d': 'two-dimensional meteorological',
    'Met3d': 'three-dimensional meteorological',
    'Chm3d': 'three-dimensional chemical'
}

GEOS_VERSION_EXPECTED_KEYS = ('Met3d', 'Met2d', 'Chm3d')

FPIT_ASSUMED_VERSION = 'fpit (GEOS v5.12.4)*'
IT_ASSUMED_VERSION = 'it (GEOS v5.29.4)*'
# This is needed for constructing character arrays, both for classic files and
# internally for efficient assignation of the inferred versions
GEOS_VER_MAX_LENGTH = max(len(FPIT_ASSUMED_VERSION), len(IT_ASSUMED_VERSION))
IT_CUTOVER_DATE = datetime(2024,4,1)

def raise_and_log(err):
    """
    A kludge for new code to log an error message and raise the normal traceback
    """
    logging.critical(str(err))
    raise err


def create_observation_operator_variable(
        ds: netCDF4.Dataset, eff_path: Union[str, np.ndarray] = DEFAULT_EFF_PATH_VARNAME, o2_dmf: Union[str, float, np.ndarray] = DEFAULT_O2_DMF_VARNAME,
        nair: Union[str, np.ndarray] = DEFAULT_NAIR_VARNAME, ret_o2_col: Union[str, np.ndarray] = DEFAULT_O2_RET_COL_VARNAME,
        varname: str = DEFAULT_INT_OP_VARNAME):
    """Compute the observation operator and add it as a new variable

    Parameters
    ----------
    ds
        The netCDF dataset for a TCCON private file to add the variable to

    eff_path
        The effective path length profiles (2D array, time by altitude) in units of centimeters, or the name of the variable
        in ``ds`` to read for this.

    o2_dmf
        The mean O2 dry mole fractions used in the Xgas calculation, which may be a scale if it is constant in time or a 1D array
        (with dimension "time") if it varies in time. Alternatively, this may be the variable name to read from ``ds`` for this.

    nair
        The number density of air profiles (2D array, time by altitude) in units of molecules/cm3, or the name of the variable
        in ``ds`` to read for this. If the array is provided directly, its first dimension must be time, not prior_time, meaning
        if you read it from a private file, it must be reindexed by ``prior_index``.

    ret_o2_col
        The retrieved O2 column densities, in molecules/cm2 or the name of the variable in ``ds`` to read for this.

    varname
        The name to give the intergration operator variable.
    """
    if isinstance(eff_path, str):
        assert ds[eff_path].units == 'cm'
        eff_path = ds[eff_path][:]
    if isinstance(o2_dmf, str):
        assert ds[o2_dmf].units == '1'
        o2_dmf = ds[o2_dmf][:]
    if isinstance(nair, str):
        assert ds[nair].units == 'molecules.cm-3'
        nair_arr = ds[nair][:]
        if ds[nair].dimensions[0] == 'prior_time':
            pi = ds['prior_index'][:]
            nair = nair_arr[pi]
        else:
            nair = nair_arr
    if isinstance(ret_o2_col, str):
        # This has the wrong units in the private files...
        ret_o2_col = ds[ret_o2_col][:]

    obs_op_atts = {
        'description': ('A vector that, when the dot product is taken with a wet mole fraction profile, applies the TCCON column-average integration. '
                        'This does NOT include the averaging kernel, those must be applied in addition to this vector.'), 
        'units': '1', 
        'usage': 'https://tccon-wiki.caltech.edu/Main/AuxiliaryDataGGG2020'
    }
    # We can't use prior_time as the first dimension, because the at least the effective path lengths and O2 columns change with
    # each spectrum, so we rely on netCDF compression to keep the file size down as much as we can.
    obs_var = ds.createVariable(varname, 'f4', dimensions=('time', 'prior_altitude'), zlib=True, complevel=9)
    obs_var[:] = (eff_path * nair * o2_dmf / ret_o2_col[:, np.newaxis]).astype(np.float32)
    obs_var.setncatts(obs_op_atts)


def create_tccon_prior_xgas_variables(ds, o2_dmf: Union[str, np.ndarray] = DEFAULT_O2_DMF_VARNAME, ret_o2_col: Union[str, np.ndarray] = DEFAULT_O2_RET_COL_VARNAME):
    if isinstance(o2_dmf, str):
        assert ds[o2_dmf].units == '1'
        o2_dmf = ds[o2_dmf][:]
    if isinstance(ret_o2_col, str):
        # This has the wrong units in the private files...
        ret_o2_col = ds[ret_o2_col][:]

    prior_varnames = dict()

    for gas, priv_var in TCCON_PRIOR_XGAS_OVC_VARS.items():
        if priv_var not in ds.variables.keys():
            logging.warning(f'{priv_var} missing from the private file, unexpected for TCCON products')
            continue
        col = ds[priv_var][:]
        xgas = col / ret_o2_col * o2_dmf

        varname = f'prior_x{gas}'
        var = ds.createVariable(varname, 'f4', dimensions=('time',))
        var[:] = xgas
        set_private_name_attrs(var)
        var.setncatts({
            'units': '1',
            'description': f'Column-average mole fraction calculated from the PRIOR profile of {gas}'
        })

        prior_varnames[gas] = varname

    return prior_varnames


def create_one_prior_xgas_variable(ds, gas, col, ret_o2_col, o2_dmf, varname=None, units='1'):
    xgas = col / ret_o2_col * o2_dmf

    if varname is None:
        varname = f'prior_x{gas}'
    var = ds.createVariable(varname, 'f4', dimensions=('time',))
    if units == '1':
        var[:] = xgas
    else:
        var[:] = xgas * MOLE_FRACTION_CONVERSIONS[units]

    set_private_name_attrs(var)
    var.setncatts({
        'units': units,
        'description': f'Column-average mole fraction calculated from the PRIOR profile of {gas}'
    })
    return varname

def set_private_name_attrs(var):
    varname = var.name
    var.standard_name = varname
    var.long_name = varname.replace('_', ' ')


def get_file_format_version(ds):
    try:
        return ds.file_format_version
    except AttributeError:
        logging.warning('file_format_version attribute not found, assuming 2020.A')
        return '2020.A'


def file_fmt_less_than(file_fmt_vers, target_file_fmt_vers):
    major, minor, file_rev = _parse_file_fmt_verse(file_fmt_vers)
    tgt_major, tgt_minor, tgt_rev = _parse_file_fmt_verse(target_file_fmt_vers)
    if major != tgt_major:
        return major < tgt_major
    elif minor != tgt_minor:
        return minor < tgt_minor
    else:
        return file_rev < tgt_rev


def _parse_file_fmt_verse(file_fmt_vers):
    parts = file_fmt_vers.split('.')
    if len(parts) == 3:
        major, minor, file_rev = parts
    elif len(parts) == 2:
        major, file_rev = parts
        minor = '0'
    else:
        raise ValueError(f'cannot parse file format version "{file_fmt_vers}"')
    
    major = int(major)
    minor = int(minor)
    if not re.match(r'[A-Z]', file_rev):
        raise ValueError('file revision in file format is not a single upper case letter')
    

def add_geos_version_variables(nc_data, gv_len, varname, is_classic):
    if is_classic:
        gv_dim = f'a{gv_len}'
        if gv_dim not in nc_data.dimensions.keys():
            nc_data.createDimension(gv_dim, gv_len)
        geos_version_var = nc_data.createVariable(varname, 'S1', ('prior_time', gv_dim))
        geos_version_var._Encoding = 'ascii'
    else:
        geos_version_var = nc_data.createVariable(varname, str, ('prior_time',))
    return geos_version_var


def add_geos_version_var_attrs(nc_data, geos_version_keys):
    for k in geos_version_keys:
        desc = GEOS_VERSION_DESCRIPTIONS.get(k, k)
        att_dict = {
            "description": f"Version information for the Goddard Earth Observing System model that provided the {desc} variables for the priors.",
            "note": "A trailing * indicates that the version information was assumed from the prior time."
        }
        nc_data[geos_version_varname(k)].setncatts(att_dict)

        nc_data[geos_file_varname(k)].description = f"Base name of the {desc} GEOS file used as input for the priors of this observations."
        nc_data[geos_checksum_varname(k)].description = f"MD5 checksum of the {desc} GEOS file used as input for the priors of this observation."

def infer_geos_version_from_modfile_time(mod_file):
    # Fallback if no GEOS version information - must assume that this is
    # an unpatched .mod file and go by the transition date.
    # We expect the mod file name to start with the shorthand for the source met,
    # e.g. "FPIT" or "IT", then an underscore, followed by YYYYMMDDHH.
    m = re.search(r'[a-zA-Z0-9]+_(\d{10})', os.path.basename(mod_file))
    if m is None:
        raise_and_log(ValueError(f'Cannot find date in mod file name "{os.path.basename(mod_file)}"'))
    file_date = datetime.strptime(m.group(1), '%Y%m%d%H')
    if file_date < IT_CUTOVER_DATE:
        versions = {k: FPIT_ASSUMED_VERSION for k in GEOS_VERSION_EXPECTED_KEYS}
    else:
        versions = {k: IT_ASSUMED_VERSION for k in GEOS_VERSION_EXPECTED_KEYS}
    filenames = {k: '' for k in GEOS_VERSION_EXPECTED_KEYS}
    checksums = {k: '' for k in GEOS_VERSION_EXPECTED_KEYS}
    return versions, filenames, checksums


def infer_geos_version_from_prior_time(prior_times: pd.DatetimeIndex):
    cutover = pd.Timestamp(IT_CUTOVER_DATE)
    versions = np.full(prior_times.size, '', dtype=f'<U{GEOS_VER_MAX_LENGTH}')
    filenames = np.full(prior_times.size, '', dtype='<U1')
    checksums = np.full(prior_times.size, '', dtype='<U1')
    xx_fpit = prior_times < cutover
    versions[xx_fpit] = FPIT_ASSUMED_VERSION
    versions[~xx_fpit] = IT_ASSUMED_VERSION
    return versions, filenames, checksums

def geos_version_keys_and_fxns():
    keys = ['geos_versions', 'geos_filenames', 'geos_checksums']
    fxns = [geos_version_varname, geos_file_varname, geos_checksum_varname]
    return zip(keys, fxns)


def geos_version_varname(key):
    return f'geos_{key.lower()}_version'


def geos_file_varname(key):
    return f'geos_{key.lower()}_filename'


def geos_checksum_varname(key):
    return f'geos_{key.lower()}_checksum'