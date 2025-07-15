from argparse import ArgumentParser
from datetime import datetime
import logging
import netCDF4
import numpy as np
import os
import pandas as pd
import re
import subprocess
import sys

from . import __version__
from .constants import (
    DEFAULT_EFF_PATH_VARNAME,
    DEFAULT_INT_OP_VARNAME,
    DEFAULT_NAIR_VARNAME,
    DEFAULT_O2_DMF_VARNAME,
    DEFAULT_O2_RET_COL_VARNAME,
    GEOS_VER_MAX_LENGTH,
    GEOS_VERSION_DESCRIPTIONS,
    GEOS_VERSION_EXPECTED_KEYS,
    FPIT_ASSUMED_VERSION,
    IT_CUTOVER_DATE,
    IT_ASSUMED_VERSION,
    MOLE_FRACTION_CONVERSIONS,
)

from typing import Union, Sequence


def raise_and_log(err):
    """
    A kludge for new code to log an error message and raise the normal traceback
    """
    logging.critical(str(err))
    raise err


def get_version_string():
    pgrm = os.path.basename(sys.argv[0])
    return f'{pgrm} (Version {__version__}; 2024-12-19; SR,JL)'


def setup_logging(log_level, log_file, message='', to_stdout=False):
    """
    Set up the logger to use for this program

    :param log_level: one of the strings "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL" specifying the minimum
     level a message must have to be printed
    :type log_level: str

    :param log_file: file to write all log messages to. This receives all messages, regardless of `log_level`. If this
     is falsey, no log file will be written.
    :type log_file: str or None

    :param message: additional message to write to the log file. An empty string will write nothing.
    :type message: str

    :return: the logger created and a boolean indicating if progress bars should be displayed
    """
    LEVELS = {'PROFILE': 1,
              'DEBUG': logging.DEBUG,
              'INFO': logging.INFO,
              'WARNING': logging.WARNING,
              'ERROR': logging.ERROR,
              'CRITICAL': logging.CRITICAL,
              }

    # add an extra level below DEBUG
    logging.addLevelName(1, 'PROFILE')
    def _log_profile(self, message, *args, **kwargs):
        if self.isEnabledFor(1):
            # it is correct - *args is passed just as args
            self._log(1, message, args, **kwargs)

    def _root_profile(msg, *args, **kwargs):
        logging.log(1, msg, *args, **kwargs)


    logging.Logger.profile = _log_profile
    logging.profile = _root_profile

    # will only display the progress bar for log levels below ERROR
    if LEVELS[log_level] >= 40:
        show_progress = False
    else:
        show_progress = True
    logger = logging.getLogger()
    handlers = [logging.StreamHandler(sys.stdout if to_stdout else sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(handlers=handlers,
                        level="DEBUG",
                        format='\n%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logger.handlers[0].setLevel(LEVELS[log_level])
    if LEVELS[log_level] < logger.level:
        logger.setLevel(LEVELS[log_level])
    logging.info('New write_netcdf log session')
    for handler in logger.handlers:
        if LEVELS[log_level] > LEVELS['PROFILE']:
            handler.setFormatter(logging.Formatter('[%(levelname)s]: %(message)s'))
        else:
            handler.setFormatter(logging.Formatter('[%(levelname)s @ %(asctime)s]: %(message)s'))
    if message:
        logging.info('Note: %s', message)
    logging.info('Running %s', get_version_string())
    proc = subprocess.Popen(['git','rev-parse','--short','HEAD'],cwd=os.path.dirname(__file__),stdout=subprocess.PIPE)
    out, err = proc.communicate()
    HEAD_commit = out.decode("utf-8").strip()
    logging.info('tccon_netcdf repository HEAD: {}'.format(HEAD_commit))
    logging.info('Python executable used: %s', sys.executable)
    logging.info('GGGPATH=%s', get_ggg_path())
    logging.info('cwd=%s', os.getcwd())
    return logger, show_progress, HEAD_commit


def get_ggg_path():
    """
    Get the path to GGG based on the GGGPATH or gggpath environmental variable.

    :return: path to GGG. If both GGGPATH and gggpath are defined in the environment, GGGPATH is preferred.
    :raises: EnvironmentError if neither GGGPATH nor gggpath are defined.
    """
    try:
        GGGPATH = os.environ['GGGPATH']
    except:
        try:
            GGGPATH = os.environ['gggpath']
        except:
            raise EnvironmentError('You need to set a GGGPATH (or gggpath) environment variable')
    return GGGPATH


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
    logging.info('Creating observation operator variable')
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
                        'This does NOT include the averaging kernel, those must be applied in addition to this vector. NOTE: this variable MUST NOT be interpolated. '
                        'It contains a componenent that depends on the spacing of the vertical levels; therefore, interpolating to a different vertical grid will '
                        'made it incorrect.'), 
        'units': '1', 
        'usage': 'https://tccon-wiki.caltech.edu/Main/AuxiliaryDataGGG2020'
    }
    # We can't use prior_time as the first dimension, because the at least the effective path lengths and O2 columns change with
    # each spectrum, so we rely on netCDF compression to keep the file size down as much as we can.
    obs_var = ds.createVariable(varname, 'f4', dimensions=('time', 'prior_altitude'), zlib=True, complevel=9)
    obs_var[:] = (eff_path * nair * o2_dmf[:, np.newaxis] / ret_o2_col[:, np.newaxis]).astype(np.float32)
    obs_var.setncatts(obs_op_atts)


def create_tccon_prior_xgas_variables(ds, o2_dmf: Union[str, np.ndarray] = DEFAULT_O2_DMF_VARNAME, ret_o2_col: Union[str, np.ndarray] = DEFAULT_O2_RET_COL_VARNAME):
    if isinstance(o2_dmf, str):
        assert ds[o2_dmf].units == '1'
        o2_dmf = ds[o2_dmf][:]
    if isinstance(ret_o2_col, str):
        # This has the wrong units in the private files...
        ret_o2_col = ds[ret_o2_col][:]

    prior_varnames = dict()

    for xgas_var, ovc_var in collect_ovc_vars(ds).items():
        if ovc_var not in ds.variables.keys():
            logging.warning(f'{ovc_var} missing from the private file, unexpected for TCCON products')
            continue
        logging.info(f'Creating prior Xgas for {xgas_var} from {ovc_var}')
        desired_units = ds[xgas_var].units
        conv_factor = MOLE_FRACTION_CONVERSIONS[desired_units]
        col = ds[ovc_var][:]
        xgas = col / ret_o2_col * o2_dmf * conv_factor

        varname = f'prior_{xgas_var}'
        var = ds.createVariable(varname, 'f4', dimensions=('time',))
        var[:] = xgas
        set_private_name_attrs(var)
        var.setncatts({
            'units': desired_units,
            'description': f'Column-average mole fraction calculated from the PRIOR profile of {xgas_var}'
        })

        prior_varnames[xgas_var] = varname

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


def collect_xgas_vars(ds):
    # Experimenting with just collecting all variables starting with "x" got also the error, ADCF, ACDF g, ADCF p, and AICF variables.
    # Because "error", "aicf", and "adcf" are distinct enough, we can just filter those out, but "g" and "p" aren't that distinctive
    # as substrings, so we assume that those variables have the "g" or "p" in the same place as the "adcf" in the ADCF variables
    # and use that to subtract out those variables from the list. We also exclude variables with "original" to make sure we don't get
    # "xn2o_original" if we call this on a dataset that already applied a bias reduction to XN2O.
    xgas_vars = set(v for v in ds.variables.keys() if v.startswith('x') and 'error' not in v and 'aicf' not in v and 'original' not in v)
    adcf_vars = set(v for v in xgas_vars if 'adcf' in v)
    g_vars = set(v.replace('adcf', 'g') for v in adcf_vars)
    p_vars = set(v.replace('adcf', 'p') for v in adcf_vars)
    xgas_vars.difference_update(adcf_vars)
    xgas_vars.difference_update(g_vars)
    xgas_vars.difference_update(p_vars)
    return sorted(xgas_vars)


def collect_ovc_vars(ds, xgas_vars=None):
    def is_pri_ovc(v):
        # Want to distinguish 'ch4_6076_ovc_ch4' vs 'ch4_6076_ovc_co2' -·
        # only want ones like the first one that give us the OVC for the primary gas
        # for that window. (First gas name is the primary window gas, second is the
        # gas the OVC is of.)
        parts = v.split('_')
        return parts[0] == parts[3]

    if xgas_vars is None:
        xgas_vars = collect_xgas_vars(ds)
    ovc_vars = [v for v in ds.variables.keys() if 'ovc' in v and is_pri_ovc(v)]
    ovc_by_xgas = dict()
    for v in ovc_vars:
        gas = v.split('_')[0]
        var_list = ovc_by_xgas.setdefault(f'x{gas}', [])
        var_list.append(v)

    # It shouldn't really matter if we use an _si or _insb OVC for an _si or _insb
    # Xgas, the OVC for a given gas should always be the same. But just in case, we'll
    # be consistent. We will assume that the OVC across windows within one detector
    # is the same.
    xgas_to_ovc = dict()
    for xgas_var in xgas_vars:
        if xgas_var.endswith(('_si', '_insb')):
            # For the secondary detector Xgases, their OVC variables are also suffixed
            # with the detector suffix
            xgas, suffix = xgas_var.split('_')
            for ovc_var in ovc_by_xgas[xgas]:
                if ovc_var.endswith(suffix):
                    xgas_to_ovc[xgas_var] = ovc_var
                    break
        elif xgas_var.endswith(('_x2007', '_x2019')):
            # For CO2, the OVCs are the same variables whichever WMO scale we are on,
            # just grab the right co2/wco2/lco2 one (though they should be the same).
            xgas = xgas_var.split('_')[0]
            for ovc_var in ovc_by_xgas[xgas]:
                if not ovc_var.endswith(('_si', '_insb')):
                    xgas_to_ovc[xgas_var] = ovc_var
        else:
            # All other gases must be for the primary detector, so make sure we get the
            # OVC variable that does not have a suffix.
            for ovc_var in ovc_by_xgas[xgas_var]:
                if not ovc_var.endswith(('_si', '_insb')):
                    xgas_to_ovc[xgas_var] = ovc_var

        if xgas_var not in xgas_to_ovc:
            raise KeyError(f'OVC variable not found for {xgas_var}')

    return xgas_to_ovc


class FileFmtVer:
    def __init__(self, file_fmt_vers: str):
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

        self.major = major
        self.minor = minor
        self.file_rev = file_rev

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self}>'

    def __str__(self):
        return f'{self.major}.{self.minor}.{self.file_rev}'

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, FileFmtVer):
            return False
        else:
            return self.major == value.major and self.minor == value.minor and self.file_rev == value.file_rev

    def __lt__(self, value: object) -> bool:
        if not isinstance(value, FileFmtVer):
            return False
        elif self.major != value.major:
            return self.major < value.major
        elif self.minor != value.minor:
            return self.minor < value.minor
        else:
            return self.file_rev < value.file_rev

    def __gt__(self, value: object) -> bool:
        if not isinstance(value, FileFmtVer):
            return False
        elif self == value or self < value:
            return False
        else:
            return True

    def __le__(self, value: object) -> bool:
        if not isinstance(value, FileFmtVer):
            return False
        elif self == value or self < value:
            return True
        else:
            return False

    def __ge__(self, value: object) -> bool:
        if not isinstance(value, FileFmtVer):
            return False
        elif self < value:
            return False
        else:
            return True


def get_file_format_version(ds) -> FileFmtVer:
    try:
        return FileFmtVer(ds.file_format_version)
    except AttributeError:
        logging.warning('file_format_version attribute not found, assuming 2020.A')
        return FileFmtVer('2020.A')


def add_geos_version_variables(nc_data, gv_len, varname, is_classic, allow_string_type=False):
    if is_classic or not allow_string_type:
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


def add_effective_path(ds, is_public):
    if 'effective_path_length' in ds.variables.keys():
        logging.info('Effective path length already present, not recomputing')
        return 
    elif is_public:
        logging.info('Effective path will be merged into an integration_operator for public files')
        return

    prior_nair = ds['prior_density'][:]
    prior_alts = ds['prior_altitude'][:]
    prior_index = ds['prior_index'][:]
    zmin = ds['zmin'][:]
    zmin_quant = np.round(zmin, 5)

    df = pd.DataFrame({'zmin': zmin_quant, 'prior_index': prior_index})
    eff_path = np.full([zmin.size, prior_alts.size], np.nan, dtype='float32')

    logging.info('Computing effective vertical path (takes ~0.1 s per day, be patient)')
    # Because zmin doesn't vary that much, we can *significantly* reduce the amount of time this
    # calculation takes compared to doing the effective path length call for every spectrum by
    # iterating over each unique combination of priors and zmin (since the path length calculation)
    # needs the number density of air and zmin), calculating the path once for that combination, then
    # writing it to every spectrum that has that combination of priors and zmin.
    for (pidx, zm), subdf in df.groupby(['prior_index', 'zmin']):
        # convert km -> cm
        p = 1e5 * _effective_vertical_path(prior_alts, zm, prior_nair[pidx])
        eff_path[subdf.index] = p

    var = ds.createVariable('effective_path_length', 'f4', dimensions=('time', 'prior_altitude'),zlib=True,complevel=9)
    var[:] = eff_path
    # Don't think there's a good standard name for this variable!
    var.setncatts({
        'long_name': 'effective path length',
        'description': 'path length used by GGG when integrating column densities',
        'units': 'cm'
    })
    logging.info('Effective vertical path calculation complete')


def _effective_vertical_path(z, zmin, d):
    """  
    Calculate the effective vertical path used by GFIT for a given z/P/T grid.

    Copied from the GGGUtils repo (https://github.com/joshua-laughner/GGGUtils) on 21 Dec 2022. Should eventually
    make GGGUtils a dependency.

    :param z: altitudes of the vertical levels. May be any unit, but note that the effective paths will be returned in
     the same unit.
    :type z: array-like

    :param zmin: minimum altitude that the light ray reaches. This is given as ``zmin`` in the netCDF files and the .ray
     files. Must be in the same unit as ``z``.
    :type zmin: float

    :param d: number density of air in molec. cm-3
    :type d: array-like

    :return: effective vertical paths in the same units as ``z``
    :rtype: array-like
    """
    def integral(dz_in, lrp_in, sign):
        return dz_in * 0.5 * (1.0 + sign * lrp_in / 3 + lrp_in**2/12 + sign*lrp_in**3/60)

    vpath = np.zeros_like(d)

    # From gfit/compute_vertical_paths.f, we need to find the first level above zmin
    # If there is no such level (which should not happen for TCCON), we treat the top
    # level this way
    try:
        klev = np.flatnonzero(z > zmin)[0]
    except IndexError:
        klev = np.size(z) - 1

    # from gfit/compute_vertical_paths.f, the calculation for level i is
    #   v_i = 0.5 * dz_{i+1} * (1 - l_{i+1}/3 + l_{i+1}**2/12 - l_{i+1}**3/60)
    #       + 0.5 * dz_i * (1 + l_i/3 + l_i**2/12 + l_i**3/60)
    # where
    #   dz_i = z_i - z_{i-1}
    #   l_i  = ln(d_{i-1}/d_i)
    # The top level has no i+1 term. This vector addition duplicates that calculation. The zeros padded to the beginning
    # and end of the difference vectors ensure that when there's no i+1 or i-1 term, it is given a value of 0.
    dz = np.concatenate([[0.0], np.diff(z[klev:]), [0.0]])
    log_rp = np.log(d[klev:-1] / d[klev+1:])
    log_rp = np.concatenate([[0.0], log_rp, [0.0]])

    # The indexing is complicated here, but with how dz and log_rp are constructed, this makes sure that, for vpath[klev],
    # the first integral(...) term uses dz = z[klev+1] - z[klev] and log_rp = ln(d[klev]/d[klev+1]) and the second integral
    # term is 0 (as vpath[klev] needs to account for the surface location below). For all other terms, this combines the
    # contributions from the weight above and below each level, with different integration signs to account for how the
    # weights increase from the level below to the current level and decrease from the current level to the level above.
    vpath[klev:] = integral(dz[1:], log_rp[1:], sign=-1) + integral(dz[:-1], log_rp[:-1], sign=1)

    # Now handle the surface - I don't fully understand how this is constructed mathematically, but the idea is that both
    # the levels in the prior above and below zmin need to contribute to the column, however that contribution needs to be
    # 0 below zmin. 

    dz = z[klev] - z[klev-1]
    xo = (zmin - z[klev-1])/dz
    log_rp = 0.0 if d[klev] <= 0 else np.log(d[klev-1]/d[klev])
    xl = log_rp * (1-xo)
    vpath[klev-1] += dz * (1-xo) * (1-xo-xl*(1+2*xo)/3 + (xl**2)*(1+3*xo)/12 + (xl**3)*(1+4*xo)/60)/2
    vpath[klev] += dz * (1-xo) * (1+xo+xl*(1+2*xo)/3 + (xl**2)*(1+3*xo)/12 - (xl**3)*(1+4*xo)/60)/2

    return vpath


def check_prior_index(ds: netCDF4.Dataset, max_delta_hours: float = 1.55) -> bool:
    """Check that the prior index values are correct

    This will check that the difference between the ZPD time and prior time for each
    spectrum, given the current prior indices, is within `max_delta_hours` hours.
    The default value of 1.55 for that value reflects that the GEOS priors change
    every 3 hours, so no spectrum should be more than half that from its prior time,
    with a small amount of padding to avoid false positives from rounding errors.
    (This is meant to catch egregiously wrong prior indexing.)

    Returns
    -------
    bool
        ``True`` if the prior indices are correct, ``False`` otherwise.
    """
    time = ds['time'][:]
    prior_time = ds['prior_time'][:]
    prior_index = ds['prior_index'][:]

    expanded_prior_time = prior_time[prior_index]
    max_dt_sec = np.ma.max(np.abs(time - expanded_prior_time))
    return max_dt_sec < (max_delta_hours * 3600)


def correct_prior_index(ds: netCDF4.Dataset, assign: bool = True) -> np.ndarray:
    """Correct the prior index values

    This will recalculate the prior index values by finding the index of the
    prior time closest to each ZPD time. Note that this should only be used if
    assigning the prior indices based on the .mav file block headers fails due
    to spectra missing from the runlog or similar reasons, as this may misassign
    prior indices very close to the transition from one prior to the next if your
    Python installation has slightly different numerics than your Fortran one.

    Parameters
    ----------
    ds
        The private netCDF dataset to calculate new prior indices for

    assign
        If ``True``, then the revised prior indices will be assigned to
        the "prior_index" variable in the netCDF file (which must exist).

    Returns
    -------
    np.ndarray
        The new prior indices
    """
    time = ds['time'][:]
    prior_time = ds['prior_time'][:]


    dt = prior_time[np.newaxis,:] - time[:,np.newaxis]
    new_indices = np.ma.argmin(np.abs(dt), axis=1)
    if assign:
        ds['prior_index'][:] = new_indices
    return new_indices


def grammatical_join(elements: Sequence[str], conjunction: str = 'and'):
    n = len(elements)
    if n == 0:
        return ''
    if n == 1:
        return elements[0]
    if n == 2:
        return f'{elements[0]} {conjunction} {elements[1]}'

    comma_list = ', '.join(elements[:-1])
    return f'{comma_list}, {conjunction} {elements[-1]}'
