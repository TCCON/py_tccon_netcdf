from __future__ import print_function

"""
Compile data from the input and output files of GFIT into netCDF files.

For usage info run this code with the --help argument.
"""

import os
import sys
import platform
import subprocess
import netCDF4
import pandas as pd
import numpy as np
import csv
import hashlib
import argparse
from collections import OrderedDict
import time
from datetime import datetime, timedelta
import calendar
import re
import logging
import warnings
import json
from shutil import copyfile
from signal import signal, SIGINT
import gc

wnc_version = 'write_netcdf.py (Version 1.0; 2019-11-15; SR)\n'
file_fmt_version = '2020.B'
std_o2_mole_frac = 0.2095

# Let's try to be CF compliant: http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.pdf
standard_name_dict = {
'year':'year',
'run':'run_number',
'lat':'latitude',
'long':'longitude',
'hour':'decimal_hour',
'azim':'solar_azimuth_angle',
'solzen':'solar_zenith_angle',
'day':'day_of_year',
'wspd':'wind_speed',
'wdir':'wind_direction',
'graw':'spectrum_spectral_point_spacing',
'tins':'instrument_internal_temperature',
'tout':'atmospheric_temperature',
'pins':'instrument_internal_pressure',
'pout':'atmospheric_pressure',
'hout':'atmospheric_humidity',
'h2o_dmf_out':'water_vapour_dry_mole_fraction',
'h2o_dmf_mod':'model_water_vapour_dry_mole_fraction',
'tmod':'model_atmospheric_temperature',
'pmod':'model_atmospheric_pressure',
'sia':'solar_intensity_average',
'fvsi':'fractional_variation_in_solar_intensity',
'zobs':'observation_altitude',
'zmin':'pressure_altitude',
'osds':'observer_sun_doppler_stretch',
'gfit_version':'gfit_version',
'gsetup_version':'gsetup_version',
'fovi':'internal_field_of_view',
'opd':'maximum_optical_path_difference',
'rmsocl':'fit_rms_over_continuum_level',
'cfampocl':'channel_fringe_amplitude_over_continuum_level',
'cfperiod':'channel_fringe_period',
'cfphase':'channel_fringe_phase',
'nit':'number_of_iterations',
'cl':'continuum_level',
'ct':'continuum_tilt',
'cc':'continuum_curvature',
'fs':'frequency_stretch',
'sg':'solar_gas_stretch',
'zo':'zero_level_offset',
'zpres':'pressure_altitude',
'cbf':'continuum_basis_function_coefficient_{}',
'ncbf':'number_of_continuum_basis_functions',
'lsf':'laser_sampling_fraction',
'lse':'laser_sampling_error',
'lsu':'laser_sampling_error_uncertainty',
'lst':'laser_sampling_error_correction_type',
'dip':'dip',
'mvd':'maximum_velocity_displacement',
}

checksum_var_list = ['config','apriori','runlog','levels','mav','ray','isotopologs','windows','telluric_linelists','solar']

standard_name_dict.update({var+'_checksum':var+'_checksum' for var in checksum_var_list})

long_name_dict = {key:val.replace('_',' ') for key,val in standard_name_dict.items()} # standard names without underscores

"""
dimensionless and unspecified units will have empty strings
we could use "1" for dimensionless units instead
both empty string and 1 are recognized as dimensionless units by udunits
but using 1 would differentiate actual dimensionless variables and variables with unspecified units
"""
units_dict = {
'year':'years',
'run':'',
'lat':'degrees_north',
'long':'degrees_east',
'hour':'hours',
'azim':'degrees',
'solzen':'degrees',
'day':'days',
'wspd':'m.s-1',
'wdir':'degrees',
'graw':'cm-1',
'tins':'degrees_Celsius',
'tout':'degrees_Celsius',
'pins':'hPa',
'pout':'hPa',
'hout':'%',
'h2o_dmf_out':'',
'h2o_dmf_mod':'',
'tmod':'degrees_Celsius',
'pmod':'hPa',
'sia':'AU',
'fvsi':'%',
'zobs':'km',
'zmin':'km',
'osds':'ppm',
'gfit_version':'',
'gsetup_version':'',
'fovi':'radians',
'opd':'cm',
'rmsocl':'%',
'cfampocl':'',
'cfperiod':'cm-1',
'cfphase':'radians',
'nit':'',
'cl':'',
'ct':'',
'cc':'',
'fs':'ppm',
'sg':'ppm',
'zo':'%',
'zpres':'km',
'cbf':'',
'ncbf':'',
'prior_effective_latitude':'degrees_north',
'prior_mid_tropospheric_potential_temperature':'degrees_Kelvin',
'prior_equivalent_latitude':'degrees_north',
'prior_temperature':'degrees_Kelvin',
'prior_density':'molecules.cm-3',
'prior_pressure':'atm',
'prior_altitude':'km',
'prior_tropopause_altitude':'km',
'prior_gravity':'m.s-2',
'prior_h2o':'1',
'prior_hdo':'1',
'prior_co2':'ppm',
'prior_n2o':'ppb',
'prior_co':'ppb',
'prior_ch4':'ppb',
'prior_hf':'ppt',
'prior_o2':'1',
}

vmr_scale_dict = {
    '': 1,
    '1': 1,
    'ppm': 1e6,
    'ppb': 1e9,
    'ppt': 1e12
}

special_description_dict = {
    'lco2':' lco2 is the strong CO2 band centered at 4852.87 cm-1 and does not contribute to the xco2 calculation.',
    'wco2':' wco2 is the weak CO2 band centered at 6073.5 and does not contribute to the xco2 calculation.',
    'th2o':' th2o is used for temperature dependent H2O windows and does not contribute to the xh2o calculation.',
    'fco2':' fco2 is used for a spectral window chosen to estimate the channel fringe amplitude and period, it does not contribute to the xco2 calculation',
    'luft':' luft is used for "dry air"',
    'qco2':' qco2 is the strong CO2 band centered at  4974.05 cm-1 and does not contribute to the xco2 calculation.',
    'zco2':' zco2 is used to test zero level offset (zo) fits in the strong CO2 window, zco2_4852 is without zo, and zco2_4852a is with zo. it does not contribute to the xco2 calculation'
}

"""
manual_flag_other = 9
manual_flags_dict = {
    1:"ils",
    2:"tracking",
    3:"surface pressure",
    manual_flag_other:"other"
}
"""
with open(os.path.join(os.path.dirname(__file__), 'release_flag_definitions.json')) as f:
    tmp = json.load(f)
    manual_flags_dict = {v: k for k, v in tmp['definitions'].items()}
    manual_flag_other = tmp['other_flag']


def signal_handler(sig,frame):
    """
    When the program is interruped (e.g. with kill or ctrl-c), write it to the .log file
    """
    logging.critical('The code was interrupted')
    sys.exit()


def raise_and_log(err):
    """
    A kludge for new code to log an error message and raise the normal traceback
    """
    logging.critical(str(err))
    raise err


def get_json_path(env_var, default, default_in_code_dir=True, none_allowed=False):
    if default_in_code_dir:
        code_dir = os.path.dirname(__file__)
        default = os.path.join(code_dir, default)

    # transform e.g. "public_variables.json" to "public variables"
    file_quantity = os.path.splitext(os.path.basename(default))[0].replace('_', ' ')
    json_path = os.getenv(env_var, default)

    if json_path is None and none_allowed:
        return json_path
    elif json_path is None:
        logging.critical('No path defined for %s, aborting.', file_quantity)
        sys.exit(1)

    if not os.path.exists(json_path) and json_path == default:
        logging.critical('The default file for %s (%s) does not exist and no %s environmental variable is defined', file_quantity, default, env_var)
        sys.exit(1)
    elif not os.path.exists(json_path):
        logging.critical('The %s file path given by the %s environmental variable (%s) does not exist. Correct it, or unset the environmental variable to use the default file.', file_quantity, env_var, json_path)
        sys.exit(1)
    else:
        logging.info('Will use %s for %s.', json_path, file_quantity)
        return json_path

def ak_tables_nc_file():
    return get_json_path('TCCON_NETCDF_AK_TABLES', 'ak_tables.nc')


def missing_data_json():
    return get_json_path('TCCON_NETCDF_MISSING_DATA', 'missing_data.json')


def public_variables_json():
    return get_json_path('TCCON_NETCDF_PUB_VARS', 'public_variables.json')


def site_info_json():
    return get_json_path('TCCON_NETCDF_SITE_INFO', 'site_info.json')


def tccon_gases_json():
    return get_json_path('TCCON_NETCDF_GASES', 'tccon_gases.json')


def public_cf_attrs_json():
    return get_json_path('TCCON_NETCDF_PUB_STD_NAMES', 'cf_standard_names.json')


def release_flags_json(cmd_line_value):
    default_in_code_dir = cmd_line_value is None
    cmd_line_value = 'release_flags.json' if cmd_line_value is None else cmd_line_value
    return get_json_path('TCCON_NETCDF_MFLAGS', cmd_line_value, default_in_code_dir=default_in_code_dir)


def read_site_info(siteID):
    with open(site_info_json(),'r') as f:
        try:
            site_data = json.load(f)[siteID]
        except KeyError:
            logging.warning('{} is not in the site_info.json file. Using empty metadata.'.format(siteID))
            site_data = {key:"" for key in ['long_name', 'release_lag', 'location', 'contact', 'site_reference', 'data_doi', 'data_reference', 'data_revision']}
            site_data['release_lag'] = "0"
    site_data['release_lag'] = '{} days'.format(site_data['release_lag'])
    return site_data


def progress(i,tot,bar_length=20,word=''):
    """
    a fancy loadbar to be displayed in the prompt while executing a time consuming loop
    """
    if tot==0:
        tot=1
    percent=float(i+1)/tot
    hashes='#' * int(round(percent*bar_length))
    spaces=' ' * (bar_length - len(hashes))

    if i+1==tot:
        sys.stdout.write("\rProgress:[{0}] {1}%".format(hashes + spaces, int(round(percent * 100)))+" "*50+'\n')
    else:
        sys.stdout.write("\rProgress:[{0}] {1}%".format(hashes + spaces, int(round(percent * 100)))+"    "+str(i+1)+"/"+str(tot)+" "+word+"   ")
    sys.stdout.flush()


def short_error(ex:Exception) -> str:
    """
    Return a short version of a python error message
    """
    return '{0}: {1}'.format(ex.__class__.__name__, ex)


def md5(file_name):
    """
    Reads file_name and get its md5 sum, returns the hexdigest hash string

    file_name: full path to the file
    """
    hash_md5 = hashlib.md5()
    if 'telluric_linelists.md5' not in file_name:
        with open(file_name, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)      
        md5sum = hash_md5.hexdigest()
    else:
        with open(file_name,'r') as f:
            content = f.readlines()
        text = '\n'.join([line.split()[0] for line in content])+'\n'
        md5sum = hashlib.md5(text.encode('utf-8')).hexdigest()

    return md5sum


def checksum(file_name,hexdigest):
    """
    Compare the hash string hexdigest with the md5 sum of the file file_name

    hexdigest: hash string
    file_name: full path to the file
    """

    check = (md5(file_name) == hexdigest)

    if not check:
        print('\n')
        logging.warning('Checksum mismatch for {}\nNew: {}\nOld: {}'.format(file_name,md5(file_name),hexdigest))


def file_info(file_name,delimiter=''):
    """
    Read the first line of a file and get the number of header lines and number of data columns

    file_name: full path to the file
    """
    with open(file_name,'r') as infile:
        if delimiter:
            nhead,ncol = [int(i) for i in infile.readline().strip().split(delimiter)[:2]]
        else:
            nhead,ncol = [int(i) for i in infile.readline().strip().split()[:2]]
    nhead = nhead-1

    return nhead,ncol


def gravity(gdlat,altit):
    """
    This function is just the GGG/src/comn/gravity.f routine translated to python


    Input Parameters:
        gdlat       GeoDetric Latitude (degrees)
        altit       Geometric Altitude (km)
    
    Output Parameter:
        gravity     Effective Gravitational Acceleration (m/s2)
    
    Computes the effective Earth gravity at a given latitude and altitude.
    This is the sum of the gravitational and centripital accelerations.
    These are based on equation I.2.4-(17) in US Standard Atmosphere 1962
    The Earth is assumed to be an oblate ellipsoid, with a ratio of the
    major to minor axes = sqrt(1+con) where con=.006738
    This eccentricity makes the Earth's gravititational field smaller at
    the poles and larger at the equator than if the Earth were a sphere
    of the same mass. [At the equator, more of the mass is directly
    below, whereas at the poles more is off to the sides). This effect
    also makes the local mid-latitude gravity field not point towards
    the center of mass.
    
    The equation used in this subroutine agrees with the International
    Gravitational Formula of 1967 (Helmert's equation) within 0.005%.
    
    Interestingly, since the centripital effect of the Earth's rotation
    (-ve at equator, 0 at poles) has almost the opposite shape to the
    second order gravitational field (+ve at equator, -ve at poles),
    their sum is almost constant so that the surface gravity could be
    approximated (.07%) by the simple expression g=0.99746*GM/radius^2,
    the latitude variation coming entirely from the variation of surface
    r with latitude. This simple equation is not used in this subroutine.
    """

    d2r=3.14159265/180.0    # Conversion from degrees to radians
    gm=3.9862216e+14        # Gravitational constant times Earth's Mass (m3/s2)
    omega=7.292116E-05      # Earth's angular rotational velocity (radians/s)
    con=0.006738            # (a/b)**2-1 where a & b are equatorial & polar radii
    shc=1.6235e-03          # 2nd harmonic coefficient of Earth's gravity field 
    eqrad=6378178.0         # Equatorial Radius (m)

    gclat=np.arctan(np.tan(d2r*gdlat)/(1.0+con))  # radians

    radius=1000.0*altit+eqrad/np.sqrt(1.0+con*np.sin(gclat)**2)
    ff=(radius/eqrad)**2
    hh=radius*omega**2
    ge=gm/eqrad**2                      # = gravity at Re

    gravity=(ge*(1-shc*(3.0*np.sin(gclat)**2-1)/ff)/ff-hh*np.cos(gclat)**2)*(1+0.5*(np.sin(gclat)*np.cos(gclat)*(hh/ge+2.0*shc/ff**2))**2)

    return gravity


def get_geos_versions(mod_file):
    """Extract GEOS versions from the header of a .mod file, if present

    Input:
        - ``mod_file``: full path to a .mod file

    Output:
        - ``versions``: a dictionary with keys of the different GEOS file types
          (e.g. "Met2d", "Chm3d") and values that given the GEOS version description
          (e.g. "fpit (GEOS v5.12.v)").
        - ``found_versions``: a boolean indicating if any versions were found in the header
          (True) or if it had to assume the versions based on the date of the .mod file (False).
          If it had to assume the versions, the version strings themselves will end in an "*".
    """
    versions = dict()
    filenames = dict()
    checksums = dict()
    expected_keys = ('Met3d', 'Met2d', 'Chm3d')
    try:
        with open(mod_file) as f:
            nhead = int(f.readline().split()[0])
            for _ in range(1,nhead):
                line = f.readline()
                if line.startswith('GEOS source'):
                    _, key, vers, name, chksum = [x.strip() for x in line.split(':')]
                    versions[key] = vers
                    filenames[key] = name
                    checksums[key] = chksum
    except FileNotFoundError:
        # If we can't find the files, then we can't actually assume that they
        # have the default versions for their date - it's possible that the user
        # ran with e.g. a file combining GEOS FP-IT met and GEOS IT CO but then
        # deleted it.
        logging.error(f'Could not find .mod file {mod_file}, will use fill values for GEOS file versions. Note that this is NOT ALLOWED for submitted TCCON data!')
        versions = {k: 'UNKNOWN' for k in expected_keys}
        filenames = {k: '' for k in expected_keys}
        checksums = {k: '' for k in expected_keys}

    found_versions = len(versions) > 0
    
    if not found_versions:
        # Fallback if no GEOS version information - must assume that this is
        # an unpatched .mod file and go by the transition date.
        m = re.search(r'FPIT_(\d{10})', os.path.basename(mod_file))
        if m is None:
            raise_and_log(ValueError(f'Cannot find date in mod file name "{os.path.basename(mod_file)}"'))
        file_date = datetime.strptime(m.group(1), '%Y%m%d%H')
        if file_date < datetime(2024,4,1):
            versions = {k: 'fpit (GEOS v5.12.4)*' for k in expected_keys}
        else:
            versions = {k: 'it (GEOS v5.29.4)*' for k in expected_keys}
        filenames = {k: '' for k in expected_keys}
        checksums = {k: '' for k in expected_keys}
    else:
        missing_keys = tuple(k for k in expected_keys if k not in versions)
        if missing_keys:
            s = ', '.join(missing_keys)
            logging.warning(f'.mod file {mod_file} is missing some GEOS version keys: {s}')
            for k in missing_keys:
                versions[k] = 'Unknown'
                filenames[k] = ''
                checksums[k] = ''
    return versions, filenames, checksums, found_versions


def get_geos_version_max_length(mav_data, version_key):
    """Return the length required to store the longest GEOS version string in ``mav_data``.

    Inputs:
        - ``mav_data``: a dictionary of dictionaries, where each child dictionary contains the
          key ``version_key`` which itself points to a dictionary containing the GEOS version strings
          as values, i.e.::

            mav_data = {
                'pa20040721saaaaa.043': {
                    'geos_versions': {'Met2d': 'fpit (GEOS v5.12.4)', 'Met3d': 'fpit (GEOS v5.12.4)', 'Chm3d': 'fpit (GEOS v5.12.4)'},
                    ...
                }
            }

        - ``version_key``: the key that points to the GEOS versions dictionary in each of the
          first-level child dictionaries of ``mav_data``.

    Outputs:
        - ``length``: the length of the longest version string.
    """
    length = 0
    for data_dict in mav_data.values():
        for version in data_dict[version_key].values():
            length = max(length, len(version))
    return max(length, 1)


def get_geos_versions_key_set(mav_data, version_key='geos_versions'):
    """Return the set of keys describing GEOS versions.

    Inputs:
        - ``mav_data``: a dictionary of dictionaries returned by ``read_mav``, see same named
          argument of :func:`get_geos_version_max_length` for details.
        - ``version_key``: the key that points to the GEOS versions dictionary in each of the
          first-level child dictionaries of ``mav_data``.

    Outputs:
        - ``length``: the length of the longest version string.
    """
    keys = set()
    for data_dict in mav_data.values():
        keys.update(data_dict[version_key].keys())
    return sorted(keys)


def geos_version_varname(key):
    return f'geos_{key.lower()}_version'


def geos_file_varname(key):
    return f'geos_{key.lower()}_filename'


def geos_checksum_varname(key):
    return f'geos_{key.lower()}_checksum'


def add_geos_version_variables(nc_data, prior_data, version_key, varname_fxn, geos_version_keys, is_classic):
    geos_version_vars = dict()
    if is_classic:
        gv_len = get_geos_version_max_length(prior_data, version_key)
        gv_dim = f'a{gv_len}'
        if gv_dim not in nc_data.dimensions.keys():
            nc_data.createDimension(gv_dim, gv_len)
        for k in geos_version_keys:
            geos_version_vars[k] = nc_data.createVariable(varname_fxn(k), 'S1', ('prior_time', gv_dim))
            geos_version_vars[k]._Encoding = 'ascii'
    else:
        for k in geos_version_keys:
            geos_version_vars[k] = nc_data.createVariable(geos_version_varname(k), str, ('prior_time',))
    return geos_version_vars


def get_eqlat(mod_file,levels):
    """
    Input:
        - mod_file: full path to a .mod file
        - levels: array of altitude levels from the .mav file
    Output:
        - eqlat: equivalent latitude profile interpolate from the altitude in the .mod file to the 'levels' altitudes 
    """
    try:
        nhead,ncol = file_info(mod_file)
    except FileNotFoundError:
        logging.warning('Could not find model file %s; the equivalent latitude profile will use fill values',mod_file)
        return np.ones(len(levels))*netCDF4.default_fillvals['f4']

    mod_data = pd.read_csv(mod_file,header=nhead,delim_whitespace=True)

    eqlat = np.interp(levels,mod_data['Height'],mod_data['EqL'])

    return eqlat


def get_eflat(vmr_file):
    """
    Input:
        - vmr_file: full path to a .vmr file
    Output:
        - eflat: effective latitude
        - mid_trop_pt: mid troposphere potential temperature, average between 500-700 hPa
    """
    try:
        nhead,ncol = file_info(vmr_file)
    except FileNotFoundError:
        logging.warning('Could not find vmr file %s; the prior effective_latitude and mid-tropospheric potential temperature will use fill values',vmr_file)
        return np.ones(2)*netCDF4.default_fillvals['f4']

    counter = 1
    with open(vmr_file,'r') as f:
        while True:
            line = f.readline().strip()
            if 'EFF_LAT_TROP' in line:
                eflat = float(line.split(':')[1])
            elif 'MIDTROP_THETA' in line:
                mid_trop_pt = float(line.split(':')[1])
                break
            if counter>nhead:
                vmr_error_message = "Did not find EFF_LAT_TROP or MIDTROP_THETA in {}".format(vmr_file)
                logging.critical(vmr_error_message)
                raise RuntimeError(vmr_error_message)
            counter += 1

    return eflat,mid_trop_pt


def get_duplicates(a):
    """
    :param a: an array/list to find duplicates in

    :return: list of duplicated elements in a
    """
    seen = {}
    dupes = []

    for x in a:
        if x not in seen:
            seen[x] = 1
        else:
            if seen[x] == 1:
                dupes.append(x)
            seen[x] += 1
    return dupes


def write_file_fmt_attrs(ds):
    """Insert/update the attributes related to the file format version in a dictionary or on a netCDF dataset
    """
    info_str = 'For a description of the changes between file format versions, see https://tccon-wiki.caltech.edu/Main/GGG2020DataChanges'
    if isinstance(ds, dict):
        ds['file_format_version'] = file_fmt_version
        ds['file_format_information'] = info_str
    else:
        ds.file_format_version = file_fmt_version
        ds.file_format_information = info_str


def read_mav(path,GGGPATH,maxspec,show_progress):
    """
    read .mav files into a dictionary with spectrum filnames as keys (from each "Next spectrum" block in the .mav file)
    values are dataframes with the prior data

    Inputs:
        - path: full path to the .mav file
        - GGGPATH: full path to GGG
        - maxspec: maximum number of spectra expected, will break out of the read loop if it is exceeded
    Outputs:
        - DATA: dataframe with data from the .mav file and the eqlat profile from the .mod files
        - nlev: number of altitudes levels
        - ncell: number of cell levels
    """
    logging.info('Reading MAV file ...')
    DATA = OrderedDict()
    assumed_geos_versions = 0

    with open(path,'r') as infile:
        for i in range(3): # line[0] is gsetup version and line[1] is a "next spectrum" line
            line = infile.readline()
            if i == 1:
                spectrum = line.strip().split(':')[1]
        tropalt = float(infile.readline().split(':')[1])
        oblat = float(infile.readline().split(':')[1])
        vmr_file = infile.readline().strip().split(os.sep)[-1]
        mod_file = infile.readline().strip().split(os.sep)[-1]
        vmr_time = (datetime.strptime(vmr_file.split('_')[1][:-1],'%Y%m%d%H')-datetime(1970,1,1)).total_seconds()

    nhead, ncol, nlev = [int(elem) for elem in line.split()]

    d = pd.read_csv(path,skiprows=nhead+1,delim_whitespace=True)
    d.rename(index=str,columns={'Height':'altitude','Temp':'temperature','Pres':'pressure','Density':'density'},inplace=True)

    mav_block = d[:nlev].apply(pd.to_numeric) # turn all the strings into numbers
    DATA[spectrum] = {
                        'data':mav_block[mav_block['altitude']>=0].copy(deep=True), # don't keep cell levels
                        'time':vmr_time,
                        'tropopause_altitude':tropalt,
                        'cell_data':mav_block[mav_block['altitude']<0].copy(deep=True),
                        'vmr_file':vmr_file,
                        'mod_file':mod_file,
    }
    DATA[spectrum]['data']['gravity'] = DATA[spectrum]['data']['altitude'].apply(lambda z: gravity(oblat,z))
    DATA[spectrum]['data']['equivalent_latitude'] = get_eqlat(os.path.join(GGGPATH,'models','gnd',mod_file),DATA[spectrum]['data']['altitude'])
    
    eflat, mid_trop_pt = get_eflat(os.path.join(GGGPATH,'vmrs','gnd',vmr_file))
    DATA[spectrum]['effective_latitude'] = eflat
    DATA[spectrum]['mid_tropospheric_potential_temperature'] = mid_trop_pt
    DATA[spectrum]['geos_versions'], DATA[spectrum]['geos_filenames'], DATA[spectrum]['geos_checksums'], found_geos_vers = get_geos_versions(os.path.join(GGGPATH, 'models', 'gnd', mod_file))
    if not found_geos_vers:
        assumed_geos_versions += 1
    nlines = d['altitude'].size - nlev # the number of lines in the .mav file starting from "Next spectrum" of the SECOND block
    nblocks = int(nlines/(nlev+7)) # number of mav blacks (starting at the SECOND block)
    ispec = 1
    while True:
        if ispec>maxspec:
            loop_error_message = 'read_mav() tried to iterate more times than the number of spectra in the .tav file ({})'.format(maxspec)
            logging.critical(loop_error_message)
            raise RuntimeError(loop_error_message)

        block_id = ispec*nlev+(ispec-1)*7
        try:
            spectrum = d['temperature'][block_id].split(':')[1]
        except (KeyError, IndexError) as e:
            break
        if show_progress:
            progress(ispec-1,nblocks,word='.mav blocks')
        tropalt = float(d['pressure'][block_id+2])
        oblat = float(d['pressure'][block_id+3])
        vmr_file = os.path.basename(d['altitude'][block_id+4])
        mod_file = os.path.basename(d['altitude'][block_id+5])
        vmr_time = (datetime.strptime(vmr_file.split('_')[1][:-1],'%Y%m%d%H')-datetime(1970,1,1)).total_seconds()
        
        mav_block = d[block_id+7:block_id+7+nlev].apply(pd.to_numeric) # turn all the strings into numbers
        DATA[spectrum] = {
                            'data':mav_block[mav_block['altitude']>=0].copy(deep=True), # don't keep cell levels
                            'time':vmr_time,
                            'tropopause_altitude':tropalt,
                            'cell_data':mav_block[mav_block['altitude']<0].copy(deep=True),
                            'vmr_file':vmr_file,
                            'mod_file':mod_file,
        }
        DATA[spectrum]['data']['gravity'] = DATA[spectrum]['data']['altitude'].apply(lambda z: gravity(oblat,z))
        DATA[spectrum]['data']['equivalent_latitude'] = get_eqlat(os.path.join(GGGPATH,'models','gnd',mod_file),DATA[spectrum]['data']['altitude'])
        
        eflat, mid_trop_pt = get_eflat(os.path.join(GGGPATH,'vmrs','gnd',vmr_file))
        DATA[spectrum]['effective_latitude'] = eflat
        DATA[spectrum]['mid_tropospheric_potential_temperature'] = mid_trop_pt
        DATA[spectrum]['geos_versions'], DATA[spectrum]['geos_filenames'], DATA[spectrum]['geos_checksums'], found_geos_vers = get_geos_versions(os.path.join(GGGPATH, 'models', 'gnd', mod_file))
        if not found_geos_vers:
            assumed_geos_versions += 1

        ispec += 1

    nlev = DATA[spectrum]['data']['altitude'].size # get nlev again without the cell levels
    ncell = DATA[spectrum]['cell_data']['altitude'].size
    mav_vmr_list = [DATA[spec]['vmr_file'] for spec in DATA]
    mav_spec_list = [spec for spec in DATA]
    if len(mav_vmr_list)!=len(set(mav_vmr_list)):
        vmr_dupes = get_duplicates(mav_vmr_list)
        mav_warning = """There are duplicate .mav blocks, typically resulting from spectra not time-ordered in the runlog.
        This can happen if an InSb or Si spectrum listed after an InGaAs spectrum has an earlier time than the InGaAs spectrum.
        Pairs of InSb-InGaAs spectra share output lines in the post_processing outputs.
        In that case the prior_index variable will point to the prior used to process the InGaAs spectrum.
        The duplicated blocks correspond to these vmr files: {}""".format(vmr_dupes)
        logging.warning(mav_warning)
        for vmr_file in vmr_dupes:
            spec_dupes = np.array(mav_spec_list)[np.full(len(mav_vmr_list),vmr_file)==mav_vmr_list]
            for spec in spec_dupes:
                if spec[15]!='a':
                    ingaas_spec = list(spec)
                    ingaas_spec[15] = 'a'
                    ingaas_spec = ''.join(ingaas_spec)
                    logging.warning('{} was processed with {} but the prior_index will refer to the InGaAs spectrum prior {}'.format(spec,vmr_file,DATA[ingaas_spec]['vmr_file']))
                    del DATA[spec]
    if not show_progress: # it's obsolete to print this if we show the progress bar
        logging.info('Finished reading MAV file')
    if assumed_geos_versions > 0:
        logging.warning(f'{assumed_geos_versions} of {ispec} .mod files were missing GEOS versions in the header; they have been assumed from the file dates.')
    return DATA, nlev, ncell


def read_col(col_file,speclength):
    """
    Read a .col file into a dataframe

    :param col_file: full path to the .col file

    :param speclength: maximum length of the spectrum file names

    :return: dataframe with the .col file data
    """

    nhead,ncol = file_info(col_file)
    with open(col_file,'r') as infile:
        content = [infile.readline() for i in range(nhead+1)]
    ggg_line = content[nhead-1]
    gfit_version, gsetup_version = [i.strip().split()[2] for i in content[1:3]]
    ngas = len(ggg_line.split(':')[-1].split())
    widths = [speclength+1,3,6,6,5,6,6,7,7,8]+[7,11,10,8]*ngas # the fixed widths for each variable so we can read with pandas.read_fwf, because sometimes there is no whitespace between numbers
    headers = content[nhead].split()
    col_data = pd.read_fwf(col_file,widths=widths,names=headers,skiprows=nhead+1)
    col_data.rename(str.lower,axis='columns',inplace=True)
    col_data.rename(index=str,columns={'rms/cl':'rmsocl'},inplace=True)

    return col_data, gfit_version, gsetup_version


def get_program_versions(output_file):
    versions = dict()
    with open(output_file) as f:
        for line in f:
            line = line.strip()
            if 'version' in line.lower():
                program, info = line.split(maxsplit=1)
                program = program.strip().lower()
                info = re.sub(r'^[Vv]ersion ', '', info)
                info = re.sub(r'\s+', ' ', info)
                info = re.sub(r'(\d)\s', r'\1; ', info)
                
                versions['{}_version'.format(program)] = info
            elif 'Airmass-Dependent' in line:
                break
            
    return versions


def write_eof(private_nc_file,eof_file,qc_file,nc_var_list,show_progress):
    """
    Convert the private netcdf file into an eof.csv file

    the columns won't be in order of the flag number, but there wil be a "flagged_var_name" column with the name of the flagged variable
    """
    logging.info('Writing eof.csv file ...')

    nhead, ncol = file_info(qc_file)
    with open(qc_file,'r') as f:
        qc_content = f.readlines()[nhead:]
    for i,line in enumerate(qc_content[1:]):
        qc_content[i+1] = '{:2d} {}'.format(i+1,line)

    with netCDF4.Dataset(private_nc_file,'r') as nc, open(eof_file,'w',newline='') as eof:
        writer = csv.writer(eof,delimiter=',',lineterminator=os.linesep)

        nhead = len(qc_content)+3
        ncol = len(nc_var_list)
        nrow = nc['time'].size

        writer.writerow([nhead,ncol,nrow])
        eof.write(wnc_version)
        eof.writelines(qc_content)

        eof_var_list = [] # make a new list of variables for the eof file, including units
        for var in nc_var_list:
            if hasattr(nc[var],'units') and nc[var].units and var not in ['year','day','hour']: # True when the units attribute exists and is different from an empty string
                eof_var_list += ['{}_{}'.format(var,nc[var].units)]
            else:
                eof_var_list += [var]
        writer.writerow(eof_var_list)
        for i in range(nrow):
            if show_progress:
                progress(i,nrow)
            row = [nc[var][i] if not hasattr(nc[var],'precision') else '{:fmt}'.replace('fmt',nc[var].precision[1:]+nc[var].precision[0]).format(nc[var][i]).strip() for var in nc_var_list]
            if row[1]: # if flagged_var_name is not an empty string
                row[1] = eof_var_list[nc_var_list.index(row[1])] # replace the netcdf variable name with the eof variable name
            writer.writerow(row)
           
    logging.info('Finished writing {} {:.2f} MB'.format(eof_file,os.path.getsize(eof_file)/1e6))

    check_eof(private_nc_file,eof_file,nc_var_list,eof_var_list)


def check_eof(private_nc_file, eof_file, nc_var_list, eof_var_list, other_is_nc=False, show_detail=False, ignore=tuple()):
    """
    check that the private netcdf file and eof.csv file contents are equal

    :param private_nc_file: the netCDF file to compare against
    :type private_nc_file: str

    :param eof_file: the .eof.csv file to compare against the netCDF file. Alternatively, a second netCDF file (if
     `other_is_nc` is `True`)
    :type eof_file: str

    :param nc_var_list: sequence of variable names in the private netCDF file to compare against variables in the eof/
     second netCDF file.

    :param eof_var_list: sequence of variable names in the .eof.csv or second netCDF file to compare against the
     original netCDF file. Must match `nc_var_list`, i.e. `nc_var_list[i]` and `eof_var_list[i]` must be the variables
     to compare.

    :param other_is_nc: set to `True` if the second file is a netCDF, rather than .eof.csv file.
    :type other_is_nc: bool

    :param show_detail: set to `True` to show additional detail about the differences between the two files.
     Done separately from the log level because this is conceptually different from logging.
    :type show_detail: bool

    :param ignore: sequence of variable names to ignore (not check for differences). These must be variables
     in ``nc_var_list``
    :type ignore: Sequence[str]

    :return: list of booleans indicating if each variable in the two files matches or not
    """
    logging.info('Checking EOF and NC file contents ...')
    if not other_is_nc:
        nhead, ncol = file_info(eof_file,delimiter=',')
        eof = pd.read_csv(eof_file,header=nhead)
        close_eof = False
    else:
        eof = netCDF4.Dataset(eof_file, 'r')
        ncol = len(eof_var_list)
        close_eof = True

    nc = netCDF4.Dataset(private_nc_file, 'r')
    checks = []
    numeric_diffs = dict()
    try:
        for i,var in enumerate(nc_var_list):
            if var in ignore:
                logging.info('Not checking %s as specified', var)
                checks += [True]
                continue
            elif (var=='spectrum') or ('checksum' in var):
                checks += [np.array_equal(nc[var][:], eof[eof_var_list[i]][:])]
                logging.debug('%s checked with array_equal, result %d', var, checks[-1])
            elif var=='flagged_var_name':
                #checks += [np.array_equal(nc[var][:],eof[eof_var_list[i]][:].replace(np.nan,''))]
                # these are actually different becaus the units are in the variable names for the eof.csv file
                # however, we need an entry, otherwise checks gets out of sync with nc_var_list
                checks += [True]
                logging.debug('%s assumed True: %d', var, checks[-1])
            elif np.issubdtype(nc[var].dtype, np.floating):
                # For floating point values, better to allow for small differences between the two values
                checks += [np.ma.allclose(nc[var][:], eof[eof_var_list[i]][:].astype(nc[var][:].dtype))]
                numeric_diffs[var] = (nc[var][:].filled(np.nan), np.ma.masked_array(eof[eof_var_list[i]][:]).filled(np.nan))
                logging.debug('%s checked with np.ma.allclose, result %d', var, checks[-1])
            else:
                # It seems to be important to convert the eof variable to the datatype of the array,
                # not the netcdf variable, from the first file. With netCDF version 1.4.2, string variables
                # have type "|S1" but the arrays have type "<U32" for some reason.
                checks += [np.array_equal(nc[var][:], eof[eof_var_list[i]][:].astype(nc[var][:].dtype))]
                if np.issubdtype(nc[var].dtype, np.number):
                    numeric_diffs[var] = (nc[var][:].filled(np.nan), np.ma.masked_array(eof[eof_var_list[i]][:]).filled(np.nan))
                logging.debug('%s checked with array_equal after dtype conversion, result %d', var, checks[-1])

    finally:
        nc.close()
        if close_eof:
            eof.close()

    logging.debug('checks length = {}, vars length = {}'.format(len(checks), len(nc_var_list)))

    max_diff = None
    max_perdiff = None
    max_fin_diff = None
    max_fin_perdiff = None

    if False in checks:
        logging.warning('{} / {} different columns:'.format(ncol-np.count_nonzero(checks),ncol))
        for i,var in enumerate(nc_var_list):
            if not checks[i]:
                logging.warning('%s is not identical !',var)
                if var in numeric_diffs and show_detail:
                    this_diff, this_perdiff = print_detailed_diff(var, *numeric_diffs[var])
                    
                    if max_perdiff is None or np.abs(this_perdiff) > np.abs(max_perdiff[0]):
                        max_perdiff = (this_perdiff, var)
                        max_diff = (this_diff, var)
                        logging.debug('Setting max_perdiff to {}'.format(max_perdiff))
                        if np.isfinite(this_perdiff) and (max_fin_perdiff is None or np.abs(this_perdiff) > np.abs(max_fin_perdiff[0])):
                            max_fin_perdiff = (this_perdiff, var)
                            max_fin_diff = (this_diff, var)
                            logging.debug('Setting max_fin_perdiff to {}'.format(max_fin_perdiff))

        if max_perdiff is not None:
            logging.info('Max percent difference ({pdiff}% = {diff}) was in {var}'.format(pdiff=max_perdiff[0], diff=max_diff[0], var=max_perdiff[1]))
        if max_fin_perdiff is not None:
            logging.info('Max FINITE percent difference ({pdiff}% = {diff}) was in {var}'.format(pdiff=max_fin_perdiff[0], diff=max_fin_diff[0], var=max_fin_perdiff[1]))
    else:
        logging.info('Contents are identical')

    return checks


def _nanabsmax(arr, axis=None):
    minimum = np.nanmin(arr, axis=axis)
    maximum = np.nanmax(arr, axis=axis)
    min_larger = np.abs(minimum) > np.abs(maximum)
    return np.where(min_larger, minimum, maximum)


def _nanabsargmax(arr, axis=None):
    minimum = np.nanmin(arr, axis=axis)
    maximum = np.nanmax(arr, axis=axis)
    min_larger = np.abs(minimum) > np.abs(maximum)

    minargs = np.nanargmin(arr, axis=axis)
    maxargs = np.nanargmax(arr, axis=axis)
    return np.where(min_larger, minargs, maxargs)


def _flatten_diffs(diffs, perdiffs, old, new):
    if np.ndim(diffs) == 1:
        return diffs, perdiffs, old, new
    elif np.ndim(diffs) == 2:
        xinds = _nanabsargmax(perdiffs, axis=1)
        # Subtle indexing point: if diffs is m-by-n (and so xinds is an m-element vector)
        # then diffs[:, xinds] becomes m-by-m: the indices in xinds are pulled for each row.
        # However, it seems that if we don't use the and instead give yinds as a concrete
        # vector, it does what we want: pull one element from each row.
        yinds = np.arange(diffs.shape[0])
        return diffs[yinds, xinds], perdiffs[yinds, xinds], old[yinds, xinds], new[yinds, xinds]
    else:
        raise NotImplementedError('Cannot flatten {} dimensional diffs'.format(np.ndim(diffs)))
    


def print_detailed_diff(variable, old_vals, new_vals, threshold=0.01):
    """
    Log detailed differences between old and new values of a variable

    :param variable: the variable name
    :type variable: str

    :param old_vals: the old values
    :type old_vals: Sequence[float or int]

    :param new_vals: the new values, should be same length as ``old_vals``
    :type new_vals: Sequence[float or int]

    :param threshold: the minimum absolute percent difference for which the difference
     for a spectrum should be printed.
    :type threshold: float
    """
    diffs = new_vals - old_vals
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        perdiffs = 100 * diffs / np.abs(old_vals)

    try:
        diffs, perdiffs, old_vals, new_vals = _flatten_diffs(diffs, perdiffs, old_vals, new_vals)
    except NotImplementedError:
        logging.warning('  %s has more than 2 dimensions, not set up to give detailed diffs', variable)
        return

    imax = _nanabsargmax(perdiffs)
    logging.info('  %s has max percent diff (%.4f %%) at spectrum %d', variable, perdiffs[imax], imax+1)
    ispec = 0
    for old, new, delta, pdelta in zip(old_vals, new_vals, diffs, perdiffs):
        ispec += 1
        if np.abs(pdelta) > threshold:
            logging.info('    Spectrum %d (old = %.4g, new = %.4g): Difference = %.4g    Fractional difference = %.4f %%', ispec, old, new, delta, pdelta)

    return _nanabsmax(diffs).item(), _nanabsmax(perdiffs).item()


def write_values(nc_data,var,values,inds=[]):
    """
    :nc_data: the netcdf4 dataset
    :var: the variable name
    :values: the values to write to the variable
    :inds: list of specific indices to write the values in nc_data[var] (must be the same size as values)
    """
    if not inds:
        try:
            nc_data[var][:] = np.array(values).astype(nc_data[var].dtype)
        except ValueError:
            logging.warning('ValueError when writing {} with expected type {} for the following spectra:'.format(var,nc_data[var].dtype))
            for i,val in enumerate(values):
                try:
                    test = np.array([val]).astype(nc_data[var].dtype)
                except ValueError:
                    logging.warning('{} {} = {}'.format(nc_data['spectrum'][i],var,values[i]))
                    nc_data[var][i] = netCDF4.default_fillvals['f4']
                else:
                    nc_data[var][i] = test[0]
            logging.warning('All faulty values have been replaced by the default netcdf fill value for floats: {}'.format(netCDF4.default_fillvals['f4']))
    else:
        full_array = np.full(nc_data[var].size,netCDF4.default_fillvals[nc_data[var].dtype.str[1:]])
        for i,index in enumerate(inds):
            full_array[index] = values[i]
        nc_data[var][:] = full_array.astype(nc_data[var].dtype)


def add_obs_op_variables(private_ds, public_ds, public_slice, mode):
    # This has two parts: converting the OVCs to original Xgas values and generating the
    # "observation operator" vectors, which combine the effective path length, number density,
    # and conversion to column average mole fraction all in one. We will need two of those: one
    # for the regular xgas values and one for the _x2019 ones (since the O2 mole fraction is different).

    # We'll do the observation operator first. 
    ret_o2_col = private_ds['vsw_o2_7885'][:][public_slice]
    x2007_o2 = std_o2_mole_frac
    if mode.lower()=="tccon":
        x2019_o2 = private_ds['o2_mean_mole_fraction_x2019'][:][public_slice]
    eff_path = private_ds['effective_path_length'][:][public_slice]
    # use the public dataset for this one because it's already been expanded to one/spectrum and
    # subset to the kept data
    nair = public_ds['prior_density'][:]


    # This won't work when the experimental variables are in groups, but those won't be standard
    # files anyway.
    x2007_vars = ', '.join(v for v in public_ds.variables.keys() if re.match(r'x[a-z0-9]+$', v) and '_error_' not in v)
    if mode.lower()=="tccon":
        x2019_vars = ', '.join(v for v in public_ds.variables.keys() if re.match(r'x[a-z0-9]+.*_x2019$', v) and '_error_' not in v)

    obs_op_atts = {
        'description': ('A vector that, when the dot product is taken with a wet mole fraction profile, applies the TCCON column-average integration. '
                        'This does NOT include the averaging kernel, those must be applied in addition to this vector. '
                        'The relates_to attribute indicates which Xgas variable to use this operator for when trying to compare against.'), 
        'units': '1', 
        'relates_to': x2007_vars,
        'usage': 'https://tccon-wiki.caltech.edu/Main/AuxiliaryDataGGG2020'
    }
    x2007_obs_var = public_ds.createVariable('integration_operator', 'f4', dimensions=('time', 'prior_altitude'), zlib=True, complevel=9)
    x2007_obs_var[:] = (eff_path * nair * x2007_o2 / ret_o2_col[:, np.newaxis]).astype(np.float32)
    x2007_obs_var.setncatts(obs_op_atts)

    if mode.lower()=="tccon":
        obs_op_atts['relates_to'] = x2019_vars
        x2019_obs_var = public_ds.createVariable('integration_operator_x2019', 'f4', dimensions=('time', 'prior_altitude'))
        x2019_obs_var[:] = (eff_path * nair * x2019_o2[:, np.newaxis] / ret_o2_col[:, np.newaxis]).astype(np.float32)
        x2019_obs_var.setncatts(obs_op_atts)

    # Now let's calculate the a priori column-average mole fractions
    private_vars = {
        "co2": "co2_6220_ovc_co2",
        "ch4": "ch4_5938_ovc_ch4",
        "n2o": "n2o_4395_ovc_n2o",
        "co": "co_4290_ovc_co",
        "hf": "hf_4038_ovc_hf",
        "h2o": "h2o_4565_ovc_h2o",
        "hdo": "hdo_4054_ovc_hdo",
    }
    for gas in list(private_vars.keys()):
        if private_vars[gas] not in private_ds.variables:
            logging.warning(f"{private_vars[gas]} missing from the private file, unexpected for TCCON products.")
            del private_vars[gas]
        

    conversions = {
        'ppm': 1e6,
        'ppb': 1e9,
        'ppt': 1e12,
    }

    for gas, priv_var in private_vars.items():
        col = private_ds[priv_var][:][public_slice]
        unit = public_ds[f'x{gas}'].units
        xgas = col / ret_o2_col * x2007_o2

        var = public_ds.createVariable(f'prior_x{gas}', 'f4', dimensions=('time',))
        var[:] = conversions[unit] * xgas
        var.setncatts({
            'standard_name': public_ds[f'x{gas}'].standard_name,
            'units': unit,
            'description': f'Column-average mole fraction calculated from the PRIOR profile of {gas} using the standard mean O2 mole fraction of {std_o2_mole_frac} appropriate for use when comparing other profiles to non-x2019 variables.'
        })

    if mode.lower()=="tccon":
        # We'll also need to do a special one for co2 with the variable mole fraction
        col = private_ds[private_vars['co2']][:][public_slice]
        unit = public_ds[f'xco2_x2019'].units
        xgas = col / ret_o2_col * x2019_o2

        var = public_ds.createVariable(f'prior_xco2_x2019', 'f4', dimensions=('time',))
        var[:] = conversions[unit] * xgas
        var.setncatts({
            'standard_name': public_ds[f'xco2_x2019'].standard_name,
            'units': unit,
            'description': f'Column-average mole fraction calculated from the PRIOR profile of co2 using the variable mean O2 mole fraction appropriate for use when comparing other profiles to _x2019 variables ONLY.'
        })



def update_attrs_for_public_files(ds, is_public, mode):
    _fix_unspecified_units(ds)
    _fix_inconsistent_units(ds)
    _add_prior_long_units(ds, is_public)
    _fix_public_cf_attributes(ds, is_public)
    _fix_incorrect_attributes(ds)
    _insert_missing_aks(ds, 'xhdo', is_public)
    _add_flag_usage(ds)
    if mode.lower() == "tccon":
        _add_x2019_co2(ds, is_public)
    write_file_fmt_attrs(ds)
    _add_effective_path(ds, is_public)


def _add_x2019_co2(ds, is_public):
    """Adds fields for XCO2, XwCO2, and XlCO2 that are converted to the X2019 CO2 scale and variable O2 mole fraction
    """
    def delta_fo2(xco2_prime, fo2_ref, xco2_ref=400e-6, beta=-1/0.4575):
        # Equation for how the change in O2 mole fraction changes from the reference given. See J. Laughner slides
        # from Dec 2022 telecon or GGG2020 data paper for derivation.
        return (beta - beta*fo2_ref - fo2_ref) * (xco2_prime - xco2_ref) / (1 - xco2_prime - beta * xco2_prime)

    if is_public or "o2_mean_mole_fraction_x2019" in ds.variables:
        # The fields should already be generated in the private files.
        return

    # Updated AICFS were calculated in the 2022-11-16 notebook. I used the ones where I kept what profiles
    # passed filtering with the new (variable O2) Xluft, rather than the ones where I forced the profiles to
    # match because (in theory) the xluft using variable O2 mole fractions should be more accurate.
    old_fo2_ref = std_o2_mole_frac
    new_fo2_ref = 0.209341
    

    new_aicfs = {'xco2': 1.00899, 'xwco2': 0.99957, 'xlco2': 1.00058}
    for k in new_aicfs:
        if ds[k].units != 'ppm':
            raise NotImplementedError(f'Cannot add X2019 CO2 if existing {k} not in ppm! Units were "{ds[k].units}"')


    # The first thing we need is the XCO2 value we use to calculate the O2 mole fractions. For GGG2020.1, what we
    # will end up doing is using the new reference mole fraction of 0.209341 and a simple mean of the airmass corrected
    # XCO2 windows. To approximate that here, since xco2 = co2 / o2 * f_o2 / AICF, we need to multiple by the old AICF
    # and 0.209341 / 0.2095 to switch to the new reference O2 mole fraction. Also convert from ppm.
    xco2_for_o2 = 1e-6 * ds['xco2'][:] * ds['xco2_aicf'][:] * new_fo2_ref / old_fo2_ref
    fo2 = new_fo2_ref + delta_fo2(xco2_for_o2, new_fo2_ref)

    # Now it's straightforward to convert each variable, it's just a matter of multiplying the old values by
    # old_aicf/new_aicf and new_fo2/old_fo2
    for key, new_aicf in new_aicfs.items():
        old_aicf = ds[f'{key}_aicf'][:]

        old_var = ds[key]
        new_var = ds.createVariable(f'{key}_x2019', old_var.dtype, dimensions=old_var.dimensions)
        # just copy the attributes, most of them are the same. We haven't added the scale yet, but do need
        # to update the description
        new_var.setncatts(old_var.__dict__)
        new_var.description = old_var.description.replace('0.2095', 'fo2')
        new_var.note = ('The *_x2019 fields use a variable O2 mole fraction in computing the dry air column mole fraction. '
                        'This is explained in more detail in the GGG2020 paper.')
        new_var.wmo_or_analogous_scale = 'WMO CO2 X2019'
        new_var[:] = old_var[:] * fo2 / old_fo2_ref * old_aicf / new_aicf

        # Repeat for the error, it's also scaled by the O2 mole fraction and AICF
        errkey = key.replace('co2', 'co2_error')
        old_var = ds[errkey]
        new_var = ds.createVariable(f'{errkey}_x2019', old_var.dtype, dimensions=old_var.dimensions)
        new_var.setncatts(old_var.__dict__)
        new_var.note = ('The *_x2019 fields use a variable O2 mole fraction in computing the dry air column mole fraction. '
                        'This is explained in more detail in the GGG2020 paper.')
        new_var[:] = old_var[:] * fo2 / old_fo2_ref * old_aicf / new_aicf

    # Also write the fO2 value, we'll need it in the public file generation for the obs operator
    o2_var = ds.createVariable('o2_mean_mole_fraction_x2019', 'f4', dimensions=('time',))
    o2_var.description = f'O2 mole fraction used when calculating the x*_x2019 variables ONLY. Any Xgas variables without the _x2019 suffix use {std_o2_mole_frac} in their Xgas calculation.'
    o2_var.units = '1'
    o2_var.standard_name = 'dry_atmospheric_mole_fraction_of_oxygen'
    o2_var[:] = fo2.astype('float32')


def _add_prior_long_units(ds, is_public):
    """
    Add a field that describes that the prior gases are wet mole fraction
    """
    logging.info('Adding long_units attributes to prior VMR profile variables')
    unit_long_str = {
        '': 'parts',
        '1': 'parts',
        'ppm': 'parts per million',
        'ppb': 'parts per billion',
        'ppt': 'parts per trillion'
    }
    if is_public:
        # there's not a clear pattern to distinguish gases from other variables
        # in the public files, so must specify the gas names
        regex = re.compile(r'prior_(h2o|co2|n2o|co|ch4|o2|hf|hdo)$')
        h2o_prior = 'prior_h2o'
        units_note = ' (Be sure to convert the H2O and gas priors to compatible units.)'
    else:
        # look for variables of the form "prior_1co2" - must have a number then immediately
        # after "prior_". This excludes things like "prior_gravity"
        regex = re.compile(r'prior_\d+[a-z][a-z0-9]+$')
        h2o_prior = 'prior_1h2o'
        units_note = ''

    for varname in ds.variables.keys():
        if not regex.match(varname):
            continue

        units = ds[varname].units
        long_str = unit_long_str.get(units, units)
        ds[varname].long_units = '{} (wet mole fraction)'.format(long_str)
        if varname == 'prior_h2o':
            ds[varname].note = 'Prior VMRs are given in wet mole fractions. To convert to dry mole fractions, you must calculate H2O_dry = H2O_wet/(1 - H2O_wet).'
        else:
            ds[varname].note = 'Prior VMRs are given in wet mole fractions. To convert to dry mole fractions, you must calculate H2O_dry = H2O_wet/(1 - H2O_wet) and then gas_dry = gas_wet * (1 + H2O_dry), where H2O_wet is the {} variable.{}'.format(h2o_prior, units_note)

    ds['prior_density'].note = "This is the ideal number density for the temperature and pressure at each model level. GGG assumes that this includes water, and so multiplies this by wet mole fractions of trace gases to get those gases' number densities."


def _fix_public_cf_attributes(ds, is_public):
    if not is_public:
        return

    with open(public_cf_attrs_json()) as f:
        pub_cf_attrs = json.load(f)

    for attribute, overrides in pub_cf_attrs['public_overrides'].items():
        for varname, attvalue in overrides.items():
            if varname in ds.variables.keys():
                logging.profile('Updating attribute "{}" on "{}" to "{}"'.format(attribute, varname, attvalue))
                ds[varname].setncattr(attribute, attvalue)

    for attribute, variables in pub_cf_attrs['public_removes'].items():
        for varname in variables:
            if varname in ds.variables.keys() and hasattr(ds[varname], attribute):
                logging.profile('Removing attribute "{}" on "{}"'.format(attribute, varname))
                ds[varname].delncattr(attribute)


def _fix_incorrect_attributes(ds):
    # The default pa_qc.dat file with GGG2020 has the description for XCO2 refer to
    # column_*w*co2. If that slips through, fix it.
    if ds['xco2'].description == '0.2095*column_wco2/column_o2':
        ds['xco2'].description = '0.2095*column_co2/column_o2'
        logging.info('Corrected description of "xco2" variable')

    # The tropopause altitude gets the wrong units in private files created using the version of 
    # write_netcdf distributed with GGG2020. Fix that here
    if ds['prior_tropopause_altitude'].units == 'degrees_north':
        ds['prior_tropopause_altitude'].units = 'km'
        logging.info('Corrected prior_tropopause_altitude units')

    # This isn't incorrect as much as missing, but this is a sensible place to put it
    ak_variables = [v for v in ds.variables.keys() if v.startswith('ak_x')]
    for varname in ak_variables:
        if not hasattr(ds[varname], 'usage'):
            ds[varname].usage = 'Please see https://tccon-wiki.caltech.edu/Main/GGG2020DataChanges for instructions on how to use the AK variables.'


def _fix_inconsistent_units(ds):
    # The slant XCH4 bins were originally given in ppb, while the XCH4 values were in ppm.
    # Make those consistent if that is still the case. 
    if 'ak_slant_xch4_bin' in ds.variables.keys() and ds['ak_slant_xch4_bin'].units == 'ppb':
        ds['ak_slant_xch4_bin'][:] = ds['ak_slant_xch4_bin'][:] * 1e-3
        ds['ak_slant_xch4_bin'].units = 'ppm'
        logging.info('Converted ak_slant_xch4_bin from ppb -> ppm')


def _fix_unspecified_units(ds):
    # UdUnits list of recognized units: https://ncics.org/portfolio/other-resources/udunits2/?
    regexes = [
        re.compile(r'^ak_x'),
        re.compile(r'^prior_\d'),  # only pick trace gas prior and cell variables, which should always be 
        re.compile(r'^cell_\d'),
        re.compile(r'^h2o_dmf_out$'),
        re.compile(r'^h2o_dmf_mod$'),
        re.compile(r'^vsw_'),
        re.compile(r'^xluft$'),
        re.compile(r'^xluft_error'),
        re.compile(r'^ada_x'),
        re.compile(r'_cfampocl$'),
    ]
    for varname, variable in ds.variables.items():
        if any(r.search(varname) for r in regexes):
            logging.profile('Setting {} units to "1"'.format(varname))
            if variable.units == '':
                variable.units = '1'
        
    # Special cases, units that weren't included in the original release but shouldn't just be "1"
    other_units = {
        'sia': 'AU'
    }
    for varname, varunits in other_units.items():
        if ds[varname].units == '':
            logging.profile('Setting {} units to "{}"'.format(varname, varunits))
            ds[varname].units = varunits
            


def _add_flag_usage(ds):
    if 'flag' in ds.variables.keys():
        ds['flag'].comment = "flag == 0 data is good quality, flag > 0 data does not meet TCCON quality standards. If you intend to use flag > 0 data, we STRONGLY encourage you to reach out to the person listed in the contact global attribute. Use of flag > 0 data without consulting the contact person is at your own risk."


def _add_aicf_scale_attr(variable_name, pub_variable, priv_data):
    scale_variable = 'aicf_{}_scale'.format(variable_name)
    if scale_variable not in priv_data.variables:
        logging.warning('Cannot add AICF scale to {}, as {} is not a variable in the private data'.format(variable_name, scale_variable))
        return

    scale = priv_data[scale_variable][0].item()

    if not np.all(scale == priv_data[scale_variable][:]):
        logging.warning('Multiple scale strings found in {}'.format(scale_variable))
        pub_variable.wmo_or_analogous_scale = 'various'
    else:
        if len(scale) == 0:
            scale = 'Not used'
        pub_variable.wmo_or_analogous_scale = scale


def _expand_aks(ds, xgas, n=500, full_ak_resolution=False, min_extrap=0):
    try:
        import xarray as xr
    except ImportError:
        raise ImportError('xarray is required to save per-spectrum AKs. Please install it in this environment.')
    logging.debug('Expanding AKs for %s', xgas)
    airmass = ds['o2_7885_am_o2'][:]
    slant_xgas_values = ds[xgas][:] * airmass
    slant_xgas_bins = ds['ak_slant_{}_bin'.format(xgas)][:]
    if xgas == 'xch4' and ds['ak_slant_{}_bin'.format(xgas)].units == 'ppb':
        # XCH4 bins are given in ppb, but XCH4 itself in ppm. Oops!
        slant_xgas_bins = slant_xgas_bins * 1e-3
    aks = ds['ak_{}'.format(xgas)][:]
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


def _insert_missing_aks(nc_data, xgas, is_public):
    # This duplicates the code in the main function because I didn't want to deal with refactoring reuse this
    # function there and test it. In theory it would be simple to do so though.
    if not xgas.startswith('x'):
        xgas = f'x{xgas}'

    if is_public:
        # Missing AKs must be added to the private files so that the normal AK expansion can happen for the public
        # files.
        return

    slant_xgas_varname = f'ak_slant_{xgas}_bin'
    ak_varname = f'ak_{xgas}'

    with netCDF4.Dataset(ak_tables_nc_file()) as ak_nc:
        if slant_xgas_varname not in nc_data.variables:
            logging.info(f'Adding {xgas} slant bins for AKs')

            ak_bin_var = f'slant_{xgas}_bin'
            nc_data.createVariable(slant_xgas_varname,np.float32,('ak_slant_xgas_bin'))
            att_dict = {
                "standard_name": slant_xgas_varname,
                "long_name": slant_xgas_varname.replace('_',' '),
                "description": ak_nc[ak_bin_var].description.lower()+" (slant_xgas=xgas*airmass)",
                "units": ak_nc[ak_bin_var].units,
            }

            nc_data[slant_xgas_varname].setncatts(att_dict)
            if xgas == 'xch4':
                # Need to convert the ppb in the netCDF file to ppm to be consistent with xch4
                nc_data[slant_xgas_varname][:] = ak_nc[ak_bin_var][:].data.astype(np.float32) * 1e-3
                nc_data[slant_xgas_varname].units = 'ppm'
            else:
                nc_data[slant_xgas_varname][:] = ak_nc[ak_bin_var][:].data.astype(np.float32)

        if ak_varname not in nc_data.variables:
            logging.info(f'Adding {xgas} AK')
            table_ak_var = f'{xgas}_aks'
            nc_data.createVariable(ak_varname,np.float32,('ak_altitude','ak_slant_xgas_bin'))
            att_dict = {
                "standard_name": "{}_column_averaging_kernel".format(table_ak_var.strip('_aks')),
                "long_name": "{} column averaging kernel".format(table_ak_var.strip('_aks')),
                "description": ak_nc[table_ak_var].description.lower()+'. ',
                "units": '',
            }
            if xgas.lower() == 'xlco2':
                att_dict['description'] = att_dict['description']+special_description_dict['lco2']
            elif xgas.lower() == 'xwco2':
                att_dict['description'] = att_dict['description']+special_description_dict['wco2']
            nc_data[ak_varname].setncatts(att_dict)
            nc_data[ak_varname][:] = ak_nc[table_ak_var][:].data.astype(np.float32)

    
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

def _add_effective_path(ds, is_public):
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


def write_public_nc(private_nc_file,code_dir,nc_format,include_experimental=False,public_nc_file=None,remove_if_no_experimental=False,rename_by_dates=True,flag0_only=True,expand_priors=True,expand_aks=True,full_ak_resolution=True,mode="tccon"):
    """
    Take a private netcdf file and write the public file using the public_variables.json file
    """
    logging.profile('Profiling output enabled')
    # factor to convert the prior fields of the public archive into more intuitive units
    factor = {'temperature':1.0,'pressure':1.0,'density':1.0,'gravity':1.0,'1h2o':1.0,'1hdo':1.0,'1co2':1e6,'1n2o':1e9,'1co':1e9,'1ch4':1e9,'1hf':1e12,'1o2':1.0}

    # Using this regex ensures that we only replace "private" in the extension of the netCDF
    # file, not elsewhere in the path. It allows for .private.nc or .private.qc.nc extensions.
    if include_experimental and public_nc_file is None:
        public_nc_file = re.sub(r'\.private((\.qc)?\.nc)$', r'.experimental.public\1', private_nc_file)
    elif not include_experimental and public_nc_file is None:
        public_nc_file = re.sub(r'\.private((\.qc)?\.nc)$', r'.public\1', private_nc_file)

    if not include_experimental:
        remove_if_no_experimental = False

    has_experimental = False
    
    logging.info('Writing {}'.format(public_nc_file))
    with netCDF4.Dataset(private_nc_file,'r') as private_data, netCDF4.Dataset(public_nc_file,'w',format=nc_format) as public_data:
        ## copy all the metadata
        logging.info('Copying attributes')
        private_attributes = private_data.__dict__
        public_attributes = private_attributes.copy()
        manual_flag_attr_list = [i for i in private_attributes if i.startswith('manual_flags')]
        release_flag_attr_list = [i for i in private_attributes if i.startswith('release_flags')]
        pgrm_versions_attr_list = [i for i in private_attributes if i.endswith('_version')]
        for attr in ['flag_info','release_lag','GGGtip','number_of_spectral_windows']+manual_flag_attr_list+pgrm_versions_attr_list: # remove attributes that are only meant for private files
            if attr not in public_attributes:
                continue
            public_attributes.pop(attr)

        # update the history to indicate that the public file is a subset of the private file
        public_attributes['history'] = "Created {} (UTC) from the engineering file {}".format(time.asctime(time.gmtime(time.time())),private_nc_file.split(os.sep)[-1])

        # if any time periods were rejected by the manual or release flags, make a note in the
        # attributes so that users can contact the site PIs to access that data.
        all_flag_attr_list = manual_flag_attr_list + release_flag_attr_list
        if len(all_flag_attr_list) > 0:
            time_periods = []
            for attr in all_flag_attr_list:
                start, end = attr.split('_')[-2:]
                start = '{}-{}-{}'.format(start[:4], start[4:6], start[6:8])
                end = '{}-{}-{}'.format(end[:4], end[4:6], end[6:8])
                time_periods.append('  - {} to {}'.format(start, end))
            time_periods.sort()
            public_attributes['withheld_data'] = ('Some time periods have been withheld due to data quality concerns. '
                                                  'If you require access to this data for your research, please reach out to '
                                                  'the site representative listed in the `contact` attribute. Time periods '
                                                  'withheld (possibly overlapping) are:\n') + '\n'.join(time_periods)

        if 'data_doi' in public_attributes and include_experimental:
            public_attributes['data_doi'] = 'These data are associated with {}'.format(public_attributes['data_doi'])
        # also update the file format attributes
        write_file_fmt_attrs(public_attributes)
        public_data.setncatts(public_attributes)
        logging.info('  -> Done copying attributes')

        # get indices of data to copy based on the release_lag
        logging.info('Finding data to make public based on release lag')
        release_lag = int(private_data.release_lag.split()[0])
        last_public_time = (datetime.utcnow()-datetime(1970,1,1)).total_seconds() - timedelta(days=release_lag).total_seconds()
        release_ids = private_data['time'][:]<last_public_time
        if np.all(~release_ids):
            empty_public_error = f'All the data comes after the public release date ({datetime.utcnow()-timedelta(days=release_lag)}), ' \
                                  'the public file will be empty.\n' \
                                 f'The release lag is set to {release_lag} days, reduce it in site_info.json or in the .private.nc file' \
                                  'to allow more recent data into the public files.'
            logging.critical(empty_public_error)
            raise Exception(empty_public_error)

        # get indices of data with flag = 0 unless we are publishing all of the data
        if flag0_only:
            no_flag_ids = private_data['flag'][:]==0
        else:
            no_flag_ids = np.ones(private_data['flag'].shape, dtype=np.bool_)
        
        nspec = private_data['time'].size
        public_slice = np.array(release_ids & no_flag_ids) # boolean array to slice the private variables on the public ids
        prior_idx = private_data['prior_index'][:][public_slice] # used if expanding priors

        nspec_public = np.sum(public_slice)

        ## copy dimensions
        logging.info('Copying {} dimensions'.format(len(private_data.dimensions)))
        for name, dimension in private_data.dimensions.items():
            if name == 'time':
                public_data.createDimension(name, nspec_public)
            elif expand_priors and name == 'prior_time':
                pass
            elif expand_aks and name == 'ak_slant_xgas_bin':
                pass
            elif name in {'cell_index', 'specname'} or re.match(r'a\d+$', name):
                pass
            else:
                public_data.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
        logging.info('  -> Done copying dimensions')

        ## copy variables based on the info in public_variables.json
        nprivate = len(private_data.variables)
        logging.info('Copying variables. {} variables in private file.'.format(nprivate))
        with open(public_variables_json()) as f:
            public_variables = json.load(f)

        # ensure that the flag variable is included if the public file does not
        # contain only flag == 0 data
        if not flag0_only and "flag" not in public_variables['isequalto']:
            public_variables['isequalto'].append('flag')

        ivariable = -1
        for name,variable in private_data.variables.items():
            ivariable += 1
            if ivariable % 500 == 0:
                logging.info(' - Done copying {} of {} variables'.format(ivariable, nprivate))

            # NB: these don't need to be numpy arrays - could just do any(elem in name for elem in ...)
            contain_check = np.array([elem in name for elem in public_variables['contains']]).any()
            startswith_check = np.array([name.startswith(elem) for elem in public_variables['startswith']]).any()
            endswith_check = np.array([name.endswith(elem) for elem in public_variables['endswith']]).any()
            isequalto_check = np.array([name==elem for elem in public_variables['isequalto']]).any()
            experimental_group_startswith_check = np.array([name.startswith(elem) for elem in public_variables['experimental_group_startswith']]).any()
            insb_group_check = np.array([(name.startswith(elem) and name.endswith('_insb')) for elem in public_variables['insb_group_startswith']]).any()
            si_group_check = np.array([(name.startswith(elem) and name.endswith('_si')) for elem in public_variables['si_group_startswith']]).any()
            remap_check = np.array([name==elem for elem in public_variables['remap']]).any()

            excluded_simple = np.array([elem in name for elem in public_variables['exclude']]).any()
            excluded_regex = np.array([re.search(elem, name) is not None for elem in public_variables.get('exclude_regex', [])]).any()
            excluded = excluded_simple or excluded_regex

            public = np.array([contain_check,isequalto_check,startswith_check,endswith_check,experimental_group_startswith_check,insb_group_check,si_group_check,remap_check]).any() and not excluded

            aicf_scale_check = name in public_variables.get('has_aicf_scale', [])

            if nc_format=='NETCDF4' and experimental_group_startswith_check and 'ingaas_experimental' not in public_data.groups:
                public_data.createGroup('ingaas_experimental')
                public_data['ingaas_experimental'].description = 'These data are EXPERIMENTAL.\nIn the root group of this file, the Xgas variables are obtained by combining columns retrieved from multiple spectral windows.\n In this ingaas_experimental group we include Xgas derived from spectral windows that do not contribute to the Xgas variables of the root group. If you plan to use these data, please work with the site PI.'

            if include_experimental and nc_format=='NETCDF4' and insb_group_check and 'insb_experimental' not in public_data.groups:
                public_data.createGroup('insb_experimental')
                public_data['insb_experimental'].description = 'These data are EXPERIMENTAL.\nIn the root group of this file, all data is obtained from an InGaAs detector while data is this group is obtained from an InSb detector. If you plan to use these data, please work with the site PI.'

            if include_experimental and nc_format=='NETCDF4' and si_group_check and 'si_experimental' not in public_data.groups:
                public_data.createGroup('si_experimental')
                public_data['si_experimental'].description = 'These data are EXPERIMENTAL.\nIn the root group of this file, all data is obtained from an InGaAs detector while data in this group is obtained from an Si detector. If you plan to use these data, please work with the site PI.'

            
            # JLL 2021-09-13: For some reason, it is much faster to read the whole variable
            # in from the private file, then subset it to the public IDs once it is in as
            # a regular numpy (masked) array. There must be either some inefficiency in 
            # the netCDF library to read a subset or an optimization to read the whole array.
            if public and not experimental_group_startswith_check and not (insb_group_check or si_group_check):
                write_atts = True
                outname = name
                extra_atts = dict()

                if expand_priors and name in {'prior_index'}:
                    # We do not need these variables if we are expanding the priors
                    write_atts = False
                elif expand_aks and name.startswith('ak_slant'):
                    # Also don't need these coordinate variables if expanding the AKs
                    write_atts = False
                elif 'time' in variable.dimensions: # only the variables along the 'time' dimension need to be sampled with public_ids
                    # Remapping currently only supported for standard variables
                    outname = public_variables['remap'].get(name, name)
                    public_data.createVariable(outname, variable.datatype, variable.dimensions)
                    this_var_data = private_data[name][:]
                    public_data[outname][:] = this_var_data[public_slice] #private_data[name][public_slice]
                elif expand_priors and 'prior_time' in variable.dimensions:
                    new_dimensions = ('time',) + variable.dimensions[1:]
                    public_data.createVariable(name,variable.datatype,new_dimensions,zlib=True,complevel=9)
                    public_data[name][:] = private_data[name][:][prior_idx]
                elif expand_aks and name.startswith('ak_x'):
                    new_dimensions = ('time',) + tuple([d for d in variable.dimensions if d != 'ak_slant_xgas_bin'])
                    public_data.createVariable(name,variable.datatype,new_dimensions,zlib=True,complevel=9)
                    this_xgas = name.split('_')[1]
                    full_aks, ak_extrap_flags = _expand_aks(private_data, this_xgas, full_ak_resolution=full_ak_resolution)
                    public_data[name][:] = full_aks[public_slice]

                    ex_flag_varname = 'extrapolation_flags_{}'.format(name)
                    ex_flag_var = public_data.createVariable(ex_flag_varname, ak_extrap_flags.dtype, dimensions=('time',),zlib=True,complevel=9)
                    ex_flag_var[:] = ak_extrap_flags[public_slice]
                    ex_flag_var.flag_values = np.array([-2, -1, 0, 1, 2], dtype=ak_extrap_flags.dtype)
                    ex_flag_var.flag_meanings = "clamped_to_min_slant_xgas\nextrapolated_below_lowest_slant_xgas_bin\ninterpolated_normally\nextrapolated_above_largest_slant_xgas_bin\nclamped_to_max_slant_xgas"
                    ex_flag_var.usage = 'Please see https://tccon-wiki.caltech.edu/Main/GGG2020DataChanges for more information'

                    extra_atts.update({'ancillary_variables': (variable.__dict__.get('ancillary_variables', '') + ' ' + ex_flag_varname).strip()})
                else:
                    public_data.createVariable(name, variable.datatype, variable.dimensions)
                    public_data[name][:] = private_data[name][:]

                if write_atts:
                    # copy variable attributes all at once via dictionary
                    public_data[outname].setncatts(private_data[name].__dict__)
                    if extra_atts:
                        public_data[outname].setncatts(extra_atts)

                # add WMO or comparable scale attribute if needed
                if aicf_scale_check:
                    _add_aicf_scale_attr(name, public_data[name], private_data)
            elif nc_format=='NETCDF4' and public and experimental_group_startswith_check: # ingaas experimental variables
                if expand_aks and name.startswith('ak_x'):
                    public_data['ingaas_experimental'].createVariable(name, variable.datatype, variable.dimensions, zlib=True, complevel=9)
                    this_var_data = _expand_aks(private_data, name.split('_')[1], full_ak_resolution=full_ak_resolution)
                else:
                    public_data['ingaas_experimental'].createVariable(name, variable.datatype, variable.dimensions)
                    this_var_data = private_data[name][:]
                public_data['ingaas_experimental'][name][:] = this_var_data[public_slice]
                public_data['ingaas_experimental'][name].setncatts(private_data[name].__dict__)
                
                # add WMO or comparable scale attribute if needed
                if aicf_scale_check:
                    _add_aicf_scale_attr(name, public_data['ingaas_experimental'][name], private_data)
            elif include_experimental and nc_format=='NETCDF4' and public and insb_group_check: # insb experimental variables
                has_experimental = True
                if expand_aks and name.startswith('ak_x'):
                    public_data['insb_experimental'].createVariable(name.replace('_insb',''), variable.datatype, variable.dimensions, zlib=True, complevel=9)
                    this_var_data = _expand_aks(private_data, name.split('_')[1])
                else:
                    public_data['insb_experimental'].createVariable(name.replace('_insb',''), variable.datatype, variable.dimensions)
                    this_var_data = private_data[name][:]
                public_data['insb_experimental'][name.replace('_insb','')][:] = this_var_data[public_slice]
                public_data['insb_experimental'][name.replace('_insb','')].setncatts(private_data[name].__dict__)
                
                # add WMO or comparable scale attribute if needed
                if aicf_scale_check:
                    _add_aicf_scale_attr(name, public_data['insb_experimental'][name], private_data)
            elif include_experimental and nc_format=='NETCDF4' and public and si_group_check: # si experimental variables
                has_experimental = True
                public_name = name.replace('_si', '')
                
                if expand_aks and name.startswith('ak_x'):
                    public_data['si_experimental'].createVariable(public_name, variable.datatype, variable.dimensions, zlib=True, complevel=9)
                    this_var_data = _expand_aks(private_data, name.split('_')[1])
                else:
                    public_data['si_experimental'].createVariable(public_name, variable.datatype, variable.dimensions)
                    this_var_data = private_data[name][:]

                public_data['si_experimental'][public_name][:] = this_var_data[public_slice]
                public_data['si_experimental'][public_name].setncatts(private_data[name].__dict__)
                
                # add WMO or comparable scale attribute if needed
                if aicf_scale_check:
                    _add_aicf_scale_attr(name, public_data['si_experimental'][public_name], private_data)
            elif nc_format=='NETCDF4_CLASSIC' and public and (experimental_group_startswith_check or (include_experimental and (insb_group_check or si_group_check))):
                # For CLASSIC type files, if the variable should go in the public file but it belongs to one of the
                # experimental groups, add extra annotations to its name and attributes. Allow InGaAs "experimental"
                # variables into the standard files, but keep InSb or Si variables to the expt+std files only.
                if insb_group_check or si_group_check:
                    has_experimental = True

                if name.endswith('_x2019'):
                    public_name = name.replace('_x2019', '') + '_experimental_x2019'
                else:
                    public_name = name+'_experimental'

                if expand_aks and name.startswith('ak_x'):
                    public_data.createVariable(public_name, variable.datatype, variable.dimensions, zlib=True, complevel=9)
                    this_var_data = _expand_aks(private_data, name.split('_')[1])
                else:
                    public_data.createVariable(public_name, variable.datatype, variable.dimensions)
                    this_var_data = private_data[name][:]
                public_data[public_name][:] = this_var_data[public_slice]
                public_data[public_name].setncatts(private_data[name].__dict__)
                if hasattr(public_data[public_name],'description'):
                    public_data[public_name].description += ' These data are EXPERIMENTAL. If you plan to use them, please work with the site PI.'
                else:
                    public_data[public_name].description = ' These data are EXPERIMENTAL. If you plan to use them, please work with the site PI.'

                # add WMO or comparable scale attribute if needed
                if aicf_scale_check:
                    _add_aicf_scale_attr(name, public_data[public_name], private_data)
                
            # prior variables
            elif name in ['prior_{}'.format(var) for var in factor.keys()]: # for the a priori profile, only the ones listed in the "factor" dictionary make it to the public file
                public_name = name.replace('_1','_')
                scale_factor = vmr_scale_dict[units_dict[public_name]] # also need to scale them from straight DMF to ppm, ppb, etc.
                if not expand_priors:
                    public_data.createVariable(public_name,variable.datatype,variable.dimensions)
                    public_data[public_name][:] = private_data[name][:] * scale_factor
                else:
                    new_dimensions = ('time',) + variable.dimensions[1:]
                    public_data.createVariable(public_name,variable.datatype,new_dimensions,zlib=True,complevel=9)
                    public_data[public_name][:] = private_data[name][:][prior_idx] * scale_factor
                public_data[public_name].setncatts(private_data[name].__dict__)
                public_data[public_name].description = "a priori profile of {}".format(public_name.replace('prior_',''))
                public_data[public_name].units = units_dict[public_name]

            logging.profile('    > Done copying %s', name)

        private_var_list = [v for v in private_data.variables]

        # special cases
        if 'o2_7885_am_o2' not in private_var_list:
            logging.warning('The O2 window is missing, the "airmass" variable will not be in the public file')
        else:
            public_data.createVariable('airmass',private_data['o2_7885_am_o2'].datatype,private_data['o2_7885_am_o2'].dimensions)
            public_data['airmass'][:] = private_data['o2_7885_am_o2'][:][public_slice]
            public_data['airmass'].setncatts(private_data['o2_7885_am_o2'].__dict__)
            if expand_aks:
                public_data['airmass'].description = "airmass computed as the total vertical column of O2 divided by the total slant column of O2 retrieved from the window centered at 7885 cm-1."
            else:
                public_data['airmass'].description = "airmass computed as the total vertical column of O2 divided by the total slant column of O2 retrieved from the window centered at 7885 cm-1. To compute the slant column of a given gas use Xgas*airmass"
            public_data['airmass'].long_name = 'airmass'
            public_data['airmass'].standard_name = 'airmass'
            public_data['airmass'].units = ''

        # do this before update_attrs so that the standard names can be assigned from cf_standard_names_json
        add_obs_op_variables(private_data, public_data, public_slice, mode=mode)

        logging.info('  --> Done copying variables')
        update_attrs_for_public_files(public_data, is_public=True, mode=mode)

        # Just before we close the file, get the start and end date for the new dates
        public_dates = public_data['time'][[0, -1]]
        public_dates = netCDF4.num2date(public_dates, public_data['time'].units, calendar=public_data['time'].calendar)

    if include_experimental and not has_experimental and remove_if_no_experimental:
        logging.info('{} contains no non-standard TCCON data, removing as requested'.format(public_nc_file))
        os.remove(public_nc_file)
    elif rename_by_dates:
        start_dstr = public_dates[0].strftime('%Y%m%d')
        end_dstr = public_dates[1].strftime('%Y%m%d')
        stem, ext = os.path.basename(public_nc_file).split('.', 1)
        # Construct the new file name out of the site ID and extension of the old name
        new_public_file = os.path.join(os.path.dirname(public_nc_file), f'{stem[:2]}{start_dstr}_{end_dstr}.{ext}')
        if new_public_file != public_nc_file:
            # If the release lag/flagging doesn't change the file name, we don't need to do anything
            if os.path.exists(new_public_file):
                os.remove(new_public_file)
            os.rename(public_nc_file, new_public_file)
        logging.info('Finished writing {} (renamed from {}) {:.2f} MB'.format(new_public_file,public_nc_file,os.path.getsize(new_public_file)/1e6))
    else:
        logging.info('Finished writing {} {:.2f} MB'.format(public_nc_file,os.path.getsize(public_nc_file)/1e6))


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
            # it is correct - *args in is passes just as args
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
    logging.info('Running %s', wnc_version.strip())
    proc = subprocess.Popen(['git','rev-parse','--short','HEAD'],cwd=os.path.dirname(__file__),stdout=subprocess.PIPE)
    out, err = proc.communicate()
    HEAD_commit = out.decode("utf-8").strip()
    logging.info('tccon_netcdf repository HEAD: {}'.format(HEAD_commit))
    logging.info('Python executable used: %s', sys.executable)
    logging.info('GGGPATH=%s', get_ggg_path())
    logging.info('cwd=%s', os.getcwd())
    return logger, show_progress, HEAD_commit


def get_runlog_file(GGGPATH,tav_file,col_file):
    with open(col_file,'r') as infile:
        for i in range(6): # read up to the runlog line in the .col file header
            c = infile.readline().strip()
    runlog_file = c.split()[1]
    if not os.path.exists(runlog_file):
        logging.warning('Could not find {}; now checking under GGGPATH/runlogs/gnd'.format(runlog_file)) 
        runlog_file = os.path.join(GGGPATH,'runlogs','gnd',tav_file.replace('.tav','.grl'))
        if not os.path.exists(runlog_file):
            logging.critical('Could not find {}'.format(runlog_file))
            sys.exit(1)

    if runlog_file.startswith(GGGPATH):
        lse_file = os.path.join(GGGPATH,'lse','gnd',os.path.basename(runlog_file).replace('.grl','.lse'))
        if not os.path.exists(lse_file): # handle egi use case with runlogs and lse files in $GGGPATH/runlog
            lse_file = runlog_file.replace('.grl','.lse')
    else:
        logging.warning('Path to runlog ({}) does not start with GGGPATH ({}). If you did not expect this, make sure your GGGPATH environmental variable is set correctly.'.format(runlog_file, GGGPATH))
        lse_file = runlog_file.replace('.grl','.lse')
        if not os.path.exists(lse_file):
            # If the runlog is at $GGGPATH/runlogs/gnd/<runlog>.grl, then
            # removing the last three parts after splitting on the path
            # separator should give us its GGGPATH
            runlog_parts = runlog_file.split(os.sep)
            runlog_ggg_path = os.sep.join(runlog_parts[:-3])
            lse_file = os.path.join(runlog_ggg_path, 'lse', 'gnd', os.path.basename(runlog_file).replace('.grl','.lse'))
    
    if not os.path.exists(lse_file):
        logging.critical('Could not find the .lse file: {}'.format(lse_file))
        sys.exit(1)

    return runlog_file,lse_file


def set_release_flags(nc_file,flag_file,qc_file=''):
    """
    Use an input .json file to apply release flags to specific time periods.

    Inputs:
        - nc_file: full path to the .private.nc file to add release flags to
        - flag_file: full path to the .json input file for setting manual flags
        - qc_file: full path to the quality controlled file (.private.qc.nc). If
          this is an empty string, `nc_file` is modified in place.

    Notes:
        The `flag_file` needs the following format:

        * The top level is a dictionary where the keys have the form `xx_NN_YYYYMMDD_YYYYMMDD`.
          `xx` must be the site ID. `NN` is a unique number for entry for the same site. The
          two `YYYYMMDD` are start and end dates in year-month-day format. The end date is
          exclusive.
        * The values of the top level are also dictionaries with the keys "value", "name",
          and "comment". "value" must point to a numeric value between 1 and 9. "name" is
          a short string describing the reason for the flag. "comment" is a longer description
          of why this flag was needed.
    """
    site_ID = os.path.basename(nc_file)[:2]
    with open(flag_file,'r') as f:
        flags_data = json.load(f)
    flags_data = {key:val for key,val in flags_data.items() if key.startswith(site_ID)}
    if not flags_data and not qc_file: # empty dictionary and path to output file - ok to abort
        return nc_file

    logging.info("Setting release flags using {}".format(flag_file))
    return _set_extra_flags(nc_file,flags_data,'release',qc_file=qc_file)


def set_manual_flags(nc_file, qc_file='', mflag_file=''):
    siteID = os.path.basename(nc_file)[:2]
    gggpath = get_ggg_path()
    if not mflag_file:
        mflag_file = os.path.join(gggpath, 'tccon', '{}_manual_flagging.dat'.format(siteID))
        if not os.path.exists(mflag_file):
            raise IOError('A manual flagging file ({}) is required even if no periods require manual flagging'.format(mflag_file))

    logging.info('Reading manual flags from {}'.format(mflag_file))
    flags_data = _read_manual_flags_file(mflag_file)
    return _set_extra_flags(nc_file,flags_data,'manual',qc_file=qc_file)    


def _set_extra_flags(nc_file,flags_data,flag_type,qc_file=''):
    """
    Using an input flag dictionary file to apply custom flags to specific time periods

    Inputs:
        - nc_file: full path to the private.nc file
        - flags_data: dictionary with flag periods as keys and flag information as values.
        - flag_type: a string indicating which type of flag this is; usually "manual" or
          "release".
        - qc_file: full path to the quality control file (.private.qc.nc)
    """
    if qc_file:
        logging.info('Creating {}'.format(qc_file))
        copyfile(nc_file,qc_file)
        nc_file = qc_file

    # The manual flags set the 1000s place in the .oof file, so the release flags
    # should set the 10,000s place. We'll reserve the 100,000s place, and if you
    # try to set some other flag type it will set the 1,000,000s place.
    flag_multiplier = {'manual': 1000, 'release': 10000}.get(flag_type, 1000000)
    time_period_list = sorted(flags_data.keys())
    with netCDF4.Dataset(nc_file,'r+') as nc_data:
        # We want to filter which spectra to apply the extra flags to based on the local
        # date in the spectrum names, rather than the UTC date in the "time" variable.
        spectrum_dates = pd.to_datetime([s[2:10] for s in nc_data['spectrum'][tuple()]])
        for i,time_period in enumerate(time_period_list):
            start_dt, end_dt = [datetime.strptime(elem,'%Y%m%d') for elem in time_period.split('_')[2:]]
            start_str, end_str = time_period.split('_')[2:]
            comment = flags_data[time_period].get('comment', '')
            if len(comment) == 0:
                logging.warning('No comment provided for {time}. Consider adding a comment to the {type} flag file.'.format(time=time_period, type=flag_type))

            # must index netCDF datasets for the < comparison to work: comparison between netCDF4.Variable and float not allowed
            # indexing with a tuple() rather than : slightly more robust: a colon won't work for a scalar variable
            # use a set to allow us to compute the intersection between the time indices and the fill indices
            replace_time_ids = list(set(np.where((start_dt <= spectrum_dates) & (spectrum_dates <= end_dt))[0]))
            if not replace_time_ids:
                continue
            start_id = np.min(replace_time_ids)
            end_id = np.max(replace_time_ids)+1 # add 1 so it's included in the slice

            flag_value = flags_data[time_period]['value']
            if flag_value<1 or flag_value>9:
                logging.warning('{type} flag values only allowed between 1 and 9, you tried to set a {type} flag={value}. Setting flag={other} ("other") instead for {time}. Check your {type} flag file'.format(type=flag_type, value=flag_value, other=manual_flag_other, time=time_period))
                flag_value = manual_flag_other
                del flags_data[time_period]['name']               
            if 'name' in flags_data[time_period]:
                flag_name = flags_data[time_period]['name'].lower()
                if flag_name != manual_flags_dict[flag_value]:
                    logging.warning('flag={value} is reserved for "{name}", you tried to set it for "{wrongname}". Setting flag={other} ("other") instead for {time}. Check your {type} flag file'.format(value=flag_value,name=manual_flags_dict[flag_value],wrongname=flag_name,other=manual_flag_other,time=time_period))
                    flag_value = manual_flag_other
                    flag_name = manual_flags_dict[flag_value]
            elif flag_value in manual_flags_dict:
                flag_name = manual_flags_dict[flag_value]
            else:
                logging.warning('You tried setting a new {type} flag ({value}) without a name. Setting flag={other} ("other") instead for {time}'.format(type=flag_type,value=flag_value,other=manual_flag_other,time=time_period))
                flag_value = manual_flag_other
                flag_name = manual_flags_dict[flag_value]

            logging.info("\t- From {start} to {end}: {type} flag={value}; name='{name}'; comment='{comment}'".format(start=start_str,end=end_str,type=flag_type,value=flag_value,name=flag_name,comment=comment))

            nc_data['flag'][start_id:end_id] = nc_data['flag'][start_id:end_id] + flag_multiplier*flag_value
            for i in range(start_id,end_id):
                current_flagged_var = nc_data['flagged_var_name'][i].item()
                if len(current_flagged_var) == 0:
                    nc_data['flagged_var_name'][i] = flag_name
                else:
                    nc_data['flagged_var_name'][i] = '{} + {}'.format(current_flagged_var, flag_name)

            setattr(nc_data,"{type}_flags_{start}_{end}".format(type=flag_type, start=start_str, end=end_str), "flag={value}; name='{name}'; comment='{comment}'".format(value=flag_value,name=flag_name,comment=comment))

    return nc_file


def _read_manual_flags_file(mflag_file):
    siteID = os.path.basename(mflag_file)[:2]

    with open(mflag_file) as f:
        nhead = int(f.readline().split()[0])

        # move past the header
        for _ in range(1, nhead):
            f.readline()

        flags_data = dict()
        
        for iline, line in enumerate(f, start=1):
            if len(line.strip()) == 0:
                # ignore blank lines, which can show up at the end of a file
                continue
            start_str, end_str, flag = line.strip().split()[:3]
            key = '{site}_{idx:02d}_{start}_{end}'.format(site=siteID, idx=iline, start=start_str, end=end_str)
            flag = int(flag)
            if '!' in line:
                comment = line.split('!', maxsplit=1)[1].strip()
            else:
                comment = ''

            flags_data[key] = {'value': flag, 'comment': comment}
    return flags_data



def get_slice(a,b,warn=True):
    """
    Inputs:
        - a: array of unique hashables
        - b: array of unique hashables (all its elements should also be included in a)
        - warn: if True, prints a warning if hash_a[ids]!=hash_b 
    Outputs:
        - ids: array of indices that can be used to slice array a to get its elements that correspond to those in array b (such that a[ids] == b)
    """

    hash_a = hash_array(a)
    hash_b = hash_array(b)

    ids = list(np.where(np.isin(hash_a,hash_b))[0])

    if warn and not np.array_equal(hash_a[ids],hash_b):
        logging.warning('get_slice: it is unexpected that elements in the second array are not included in the first array')

    return ids


def hash_array(x):
    """
    Elementwise hash of x

    Inputs:
        - x: array of unique hashables
    Outputs:
        - hash_x: array of of hashed elements from x
    """

    hash_x = np.array([hash(i) for i in x])
    if not np.unique(hash_x).size==hash_x.size:
        logging.critical("hash_array: could not generate unique hashes for all elements")
        sys.exit(1)

    return hash_x


def main():
    signal(SIGINT,signal_handler)
    code_dir = os.path.dirname(__file__) # path to the tccon_netcdf repository
    GGGPATH = get_ggg_path()

    description = wnc_version + "This writes TCCON outputs in a NETCDF file"
    
    parser = argparse.ArgumentParser(description=description,formatter_class=argparse.RawTextHelpFormatter)
    
    def file_choices(choices,file_name):
        """
        Function handler to check file extensions with argparse

        choices: tuple of accepted file extensions
        file_name: path to the file
        """
        ext = os.path.splitext(file_name)[1][1:]
        if ext not in choices:
            parser.error("file '{}' doesn't end with one of {}".format(file_name, choices))
        if ext == 'nc' and 'private' not in file_name:
            parser.error('The .private.nc file is needed to write the .public.nc file')
        return file_name
    
    parser.add_argument('file',type=lambda file_name:file_choices(('tav','nc'),file_name),help='The .tav file or private.nc file')
    parser.add_argument('--format',default='NETCDF4_CLASSIC',choices=['NETCDF4_CLASSIC','NETCDF4'],help='the format of the NETCDF files')
    parser.add_argument('-r','--read-only',action='store_true',help="Convenience for python interactive shells; sys.exit() right after reading all the input files")
    parser.add_argument('--eof',action='store_true',help='If given, will also write the .eof.csv file')
    parser.add_argument('--public',action='store_true',help='If given, will write a .public.nc or .experimental.public.nc file from the .private.nc')
    parser.add_argument('--publish-all-flags', action='store_true', help='Include all spectra in the public files, not just flag==0')
    parser.add_argument('-s', '--std-only',action='store_true',help='If given with --public, only writes standard TCCON product gases to the public file')
    parser.add_argument('--remove-no-expt',action='store_true',help='If given with --public but without --std-only, if the output file contains only standard TCCON data, it is removed')
    parser.add_argument('--log-level',default='INFO',type=lambda x: x.upper(),help="Log level for the screen (it is always DEBUG for the log file)",choices=['PROFILE', 'DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    parser.add_argument(
        '--log-file',
        default='write_netcdf.log',
        help="""Full path to the log file, by default write_netcdf.log is written to in append mode in the current working directory.
        If you want to write the logs of all your write_netcdf.py runs to a signle file, you can use this argument to specify the path.""",
        )
    parser.add_argument('--log-to-stdout',action='store_true',help='By default, terminal logging pipes through stderr. Use this flag to send it through stdout instead.')
    parser.add_argument('--skip-checksum',action='store_true',help='Option to not make a check on the checksums, for example to run the code on outputs generated by someone else or on a different machine')
    parser.add_argument('-m','--message',default='',help='Add an optional message to be kept in the log file to remember why you ran post-processing e.g. "2020 Eureka R3 processing" ')
    parser.add_argument('--multiggg',default='multiggg.sh',help='Use this argument if you use differently named multiggg.sh files')
    parser.add_argument('--mode',default='TCCON',choices=['TCCON','em27'],help='Will be used to set TCCON specific or em27 specific metadata')
    parser.add_argument('--rflag',action='store_true',help='If given with a private.nc file as input, will create a separate private.qc.nc file with updated flags based on the --rflag-file')
    parser.add_argument('--rflag-file',help='Full path to the .json input file that sets release flags (has no effect without --rflag)')

    parser.add_argument('--no-expand-priors', action='store_false', dest='expand_priors', help='When writing public files, do NOT expand the priors to match the time dimension, leave them on the 3 hourly interval')
    parser.add_argument('--no-expand-aks', action='store_false', dest='expand_aks', help='When writing public files, do NOT expand the AKs to match the time dimension, leave them as lookup tables')
    parser.add_argument('--full-ak-resolution', action='store_true', help='Use the exact slant Xgas for each spectrum when expanding AKs, rather than quantized ones. Output files will be larger.')
    args = parser.parse_args()
    logger, show_progress, HEAD_commit = setup_logging(log_level=args.log_level, log_file=args.log_file, message=args.message, to_stdout=args.log_to_stdout)
    
    if not args.rflag and args.rflag_file is not None:
        logging.warning('Specifying --rflag-file without using --rflag has no effect')
    args.rflag_file = release_flags_json(args.rflag_file)
    

    nc_format = args.format
    classic = nc_format == 'NETCDF4_CLASSIC'
    logging.info('netCDF file format: %s',nc_format)
    if args.eof:
        logging.info('A eof.csv file will be written')
    logging.info('Input file: %s',args.file)

    if args.file.endswith('.nc'):
        private_nc_file = args.file
        if args.rflag:
            # This regex ensures that we only replace the .nc at the end
            # of the filename, never earlier in the path (just in case)
            qc_file = re.sub(r'\.nc$', '.qc.nc', private_nc_file)

            # For public files, we only want to set the release flags now. The regular
            # manual flags require GGG be present with a $GGGPATH/tccon/xx_manual_flags.dat
            # file. Since public file production occurs on tccondata.org, we don't want to
            # rely on a GGG installation.
            private_nc_file = set_release_flags(private_nc_file,args.rflag_file,qc_file=qc_file)
            with netCDF4.Dataset(private_nc_file, 'a') as privds:
                update_attrs_for_public_files(privds, is_public=False, mode=args.mode)
            if not args.public:
                sys.exit()
        logging.info('Writing public file from {}'.format(private_nc_file))
        write_public_nc(private_nc_file,code_dir,nc_format,include_experimental=not args.std_only,remove_if_no_experimental=args.remove_no_expt,flag0_only=not args.publish_all_flags,expand_priors=args.expand_priors,expand_aks=args.expand_aks,full_ak_resolution=args.full_ak_resolution,mode=args.mode)
        sys.exit()

    # input and output file names
    tav_file = args.file
    mav_file = tav_file.replace('.tav','.mav')
    vav_file = tav_file.replace('.tav','.vav')
    vsw_file = tav_file.replace('.tav','.vsw')
    vsw_ada_file = vsw_file+'.ada'
    ada_file = vav_file+'.ada'
    aia_file = ada_file+'.aia'
    esf_file = aia_file+'.daily_error.out'
    eof_file = aia_file+'.eof.csv'
    
    siteID = os.path.basename(tav_file)[:2] # two letter site abbreviation
    qc_file = os.path.join(GGGPATH,'tccon','{}_qc.dat'.format(siteID))
    header_file = os.path.join(GGGPATH,'tccon','{}_oof_header.dat'.format(siteID))
    pth_file = 'extract_pth.out'

    skip_vsw = False
    for input_file in [tav_file,mav_file,vav_file,vsw_file,vsw_ada_file,ada_file,aia_file,esf_file,pth_file,qc_file,header_file]:
        if not os.path.exists(input_file):
            if input_file in [vsw_file,vsw_ada_file]:
                logging.warning('Cannot find input file: {}'.format(input_file))
                skip_vsw = True
            else:
                logging.critical('Cannot find input file: {}'.format(input_file))
                sys.exit(1)

    # need to check that the file ends with .col, not just that .col is in it, because
    # otherwise a .col elsewhere in the file name will cause a problem (e.g. if one is
    # open in vi)
    col_file_list = sorted([i for i in os.listdir(os.getcwd()) if i.endswith('.col')])

    if not col_file_list: # [] evaluates to False
        logging.critical('No .col files in',os.getcwd())
        sys.exit(1)

    runlog_file, lse_file = get_runlog_file(GGGPATH,tav_file,col_file_list[0])

    ## read data, I add the file_name to the data dictionaries for some of them

    # read tccon_gases.json
    with open(tccon_gases_json(),'r') as f:
        tccon_gases = json.load(f)

    # if a gas is shared by InSb and Si, but not InGaAs, then the corresponding .col file should start with 'm' for InSb and 'v' for Si
    insb_only = set([gas for gas in tccon_gases['insb'] if (gas not in tccon_gases['ingaas']) and (gas not in tccon_gases['si'])])
    si_only = set([gas for gas in tccon_gases['si'] if (gas not in tccon_gases['ingaas']) and (gas not in tccon_gases['insb'])])

    # read runlog spectra; only read in the spectrum file names to make checks with the post_processing outputs
    nhead,ncol = file_info(runlog_file)
    runlog_data = pd.read_csv(runlog_file,delim_whitespace=True,skiprows=nhead,usecols=['Spectrum_File_Name','DELTA_NU']).rename(index=str,columns={'Spectrum_File_Name':'spectrum','DELTA_NU':'delta_nu'})
    dnu_set = set(runlog_data['delta_nu'])
    dnu = 0
    if len(dnu_set)>1:
        logging.warning('There are {} different spectral point spacings in the runlog: {}'.format(len(dnu_set),dnu_set))
    else:
        dnu = runlog_data['delta_nu'][0]
    runlog_all_speclist = np.array([spec for spec in runlog_data['spectrum']])
    runlog_insb_speclist = np.array([spec for spec in runlog_data['spectrum'] if spec[15]=='c'])
    runlog_ingaas_speclist = np.array([spec for spec in runlog_data['spectrum'] if spec[15]=='a'])
    runlog_ingaas2_speclist = np.array([spec for spec in runlog_data['spectrum'] if spec[15]=='d']) # second InGaAs detector of em27s
    # use hash() to convert the arrays of strings to arrays of integers for faster array comparisons
    hash_runlog_ingaas_speclist = hash_array(runlog_ingaas_speclist) 
    hash_runlog_ingaas2_speclist = hash_array(runlog_ingaas2_speclist)
    runlog_si_speclist = np.array([spec for spec in runlog_data['spectrum'] if spec[15]=='b'])
    nsi = len(runlog_si_speclist)
    ninsb = len(runlog_insb_speclist)
    ningaas = len(runlog_ingaas_speclist)
    ningaas2 = len(runlog_ingaas2_speclist)
    spec_info = 'The runlog contains:'
    if ningaas:
        spec_info += ' {} InGaAs spectra;'.format(ningaas)
    if ningaas2:
        spec_info += ' {} secondary InGaAs spectra;'.format(ningaas2)
    if ninsb:
        spec_info += ' {} InSb spectra;'.format(ninsb)
    if nsi:
        spec_info += ' {} Si spectra;'.format(nsi)
    logging.info(spec_info)

    if ningaas!=0 and ninsb and ninsb>ningaas:
        logging.critical('Having more InSb than InGaAs spectra is not supported')
        sys.exit(1)
    if ningaas!=0 and nsi and nsi>ningaas:
        logging.critical('Having more Si than InGaAs spectra is not supported')
        sys.exit(1)

    # read site specific data from the tccon_netcdf repository
    # the .apply and .rename bits are just strip the columns from leading and tailing white spaces
    with open(site_info_json(),'r') as f:
        try:
            site_data = json.load(f)[siteID]
        except KeyError:
            logging.warning('{} is not in the site_info.json file. Using empty metadata.'.format(siteID))
            site_data = {key:"" for key in ['long_name', 'release_lag', 'location', 'contact', 'site_reference', 'data_doi', 'data_reference', 'data_revision']}
            site_data['release_lag'] = "0"
    site_data['release_lag'] = '{} days'.format(site_data['release_lag'])

    # multiggg.sh; use it to get the number of windows fitted and check they all have a .col file
    with open(args.multiggg,'r') as infile:
        content = [line for line in infile.readlines() if line[0]!=':' and line.strip()!=''] # the the file without blank lines or commented out lines starting with ':'
    ncol = len(content)
    multiggg_list = [line.split()[1].split('.ggg')[0]+'.col' for line in content]
    if ncol!=len(col_file_list):
        logging.warning('{} has {} command lines but there are {} .col files'.format(args.multiggg,ncol,len(col_file_list)))
        logging.warning('only the data from .col files with a corresponding command in the {} file will be written'.format(args.multiggg))
        for elem in multiggg_list:
            if elem not in col_file_list:
                logging.critical('{} does not exist'.format(elem))
                sys.exit(1)
    col_file_list = multiggg_list
    if 'luft' in col_file_list[0]: # the luft .col file has no checksum for the solar linelist, so if its the first window listed in multiggg.sh, rotate the list for the checksum checks to work
        col_file_list = np.roll(col_file_list,-1)

    # tav file: contains VSFs
    with open(tav_file,'r') as infile:
        nhead,ncol,nspec,naux = np.array(infile.readline().split()).astype(int)
    nhead = nhead-1
    tav_data = pd.read_csv(tav_file,delim_whitespace=True,skiprows=nhead)
    tav_data['file'] = tav_file
    nwin = int((ncol-naux)/2)
    speclength = tav_data['spectrum'].map(len).max() # use the longest spectrum file name length for the specname dimension
    if nspec!=ningaas:
        logging.warning('{} ingaas spectra in runlog; {} spectra in .tav file'.format(ningaas,nspec))

    # read prior data
    prior_data, nlev, ncell = read_mav(mav_file,GGGPATH,tav_data['spectrum'].size,show_progress)
    nprior = len(prior_data.keys())

    logging.info('Reading input files:')
    # header file: it contains general information and comments.
    logging.info('\t- {}'.format(header_file))
    with open(header_file,'r') as infile:
        header_content = infile.read()

    # qc file: it contains information on some variables as well as their flag limits
    logging.info('\t- {}'.format(qc_file))
    nhead, ncol = file_info(qc_file)
    qc_data = pd.read_fwf(qc_file,widths=[15,3,8,7,10,9,10,45],skiprows=nhead+1,names='Variable Output Scale Format Unit Vmin Vmax Description'.split())
    for key in ['Variable','Format','Unit']:
        qc_data[key] = [i.replace('"','') for i in qc_data[key]]
    len_list = len(list(qc_data['Variable']))
    len_set = len(list(set(qc_data['Variable'])))
    if len_list!=len_set:
        dupes = get_duplicates(list(qc_data['Variable']))
        logging.warning('There are {} duplicate variables in the qc.dat file: {}\n flags will be determined based on the first occurence of each duplicate.'.format(len_list-len_set,dupes))
    # the qc.dat file is an input file that gets edited often by users
    # they often mistakenly misalign columns which messes with the fwf read and raises a confusing error when determining flags later
    # check that the Scale, Vmin, and Vmax columns can be converted to floats here    
    for qc_var in ['Scale','Vmin','Vmax']:
        try:
            qc_data[qc_var].astype(float)
        except ValueError as qc_err:
            logging.critical('Could not convert all of {} to floats from the {} file; please check for misaligned columns in the file'.format(qc_var,os.path.basename(qc_file)))
            logging.critical('Python error: "{}"'.format(short_error(qc_err)))
            sys.exit(1)
    
    # error scale factors:
    logging.info('\t- {}'.format(esf_file)) 
    nhead, ncol = file_info(esf_file)
    esf_data = pd.read_csv(esf_file,delim_whitespace=True,skiprows=nhead)

    # lse file: contains laser sampling error data
    logging.info('\t- {}'.format(lse_file))
    nhead, ncol = file_info(lse_file)
    lse_data = pd.read_csv(lse_file,delim_whitespace=True,skiprows=nhead)
    lse_data['file'] = lse_file
    lse_data.rename(index=str,columns={'Specname':'spectrum'},inplace=True) # the other files use 'spectrum'
    # check the .lse file has the same number of spectra as the runlog
    if len(lse_data['spectrum'])!=len(runlog_data['spectrum']):
        logging.critical("Different number of spectra in runlog ({}) and lse ({}) files".format(len(runlog_data['spectrum']),len(lse_data['spectrum'])))
        sys.exit(1)

    # vav file: contains column amounts
    logging.info('\t- {}'.format(vav_file))
    nhead, ncol = file_info(vav_file)
    vav_data = pd.read_csv(vav_file,delim_whitespace=True,skiprows=nhead)
    vav_data['file'] = vav_file
    vav_shape = vav_data.shape[0]
    vav_spec_list = vav_data['spectrum']

    # ada file: contains column-average dry-air mole fractions
    logging.info('\t- {}'.format(ada_file))
    nhead, ncol = file_info(ada_file)
    ada_data = pd.read_csv(ada_file,delim_whitespace=True,skiprows=nhead)
    ada_data['file'] = ada_file
    
    # aia file: ada file with scale factor applied
    logging.info('\t- {}'.format(aia_file))
    nhead, ncol = file_info(aia_file)
    aia_data = pd.read_csv(aia_file,delim_whitespace=True,skiprows=nhead)

    if 'Infinity' in aia_data.values:
        logging.warning('Found "Infinity" values in {}'.format(aia_file))
        object_columns = [i for i in aia_data.select_dtypes(include=np.object).columns if i.startswith('x')]
        logging.warning('Found "Infinity" values in {}'.format(object_columns))
        aia_data[object_columns] = aia_data[object_columns].astype(np.float64)
    if ningaas: # check for consistency with the runlog spectra
        aia_ref_speclist = np.array([i.replace('c.','a.').replace('b.','a.').replace('d.','a.') for i in aia_data['spectrum']]) # this is the .aia spectrum list but with only ingaas names
        if not np.array_equal(hash_array(aia_ref_speclist),hash_runlog_ingaas_speclist):
            logging.warning('The spectra in the .aia file are inconsistent with the runlog spectra:\n {}'.format(set(aia_ref_speclist).symmetric_difference(set(runlog_ingaas_speclist))))       
        ingaas_runlog_slice = get_slice(runlog_data['spectrum'],aia_ref_speclist)
        runlog_slice_dict = {'ingaas':ingaas_runlog_slice}
        aia_slice_dict = {'ingaas':get_slice(aia_ref_speclist,runlog_data['spectrum'],warn=False)}
        if ninsb:
            aia_ref_speclist_insb = np.array([i.replace('a.','c.') for i in aia_data['spectrum']]) # will be used to get .col file spectra indices along the time dimension
            runlog_slice_dict['insb'] = get_slice(runlog_data['spectrum'],aia_ref_speclist_insb)
            aia_slice_dict['insb'] = get_slice(aia_ref_speclist_insb,runlog_data['spectrum'],warn=False)
        if nsi:
            aia_ref_speclist_si = np.array([i.replace('a.','b.') for i in aia_data['spectrum']]) # will be used to get .col file spectra indices along the time dimension
            runlog_slice_dict['si'] = get_slice(runlog_data['spectrum'],aia_ref_speclist_si)
            aia_slice_dict['si'] = get_slice(aia_ref_speclist_si,runlog_data['spectrum'],warn=False)
        if ningaas2:
            aia_ref_speclist_ingaas2 = np.array([i.replace('a.','d.') for i in aia_data['spectrum']])
            runlog_slice_dict['ingaas2'] = get_slice(runlog_data['spectrum'],aia_ref_speclist_ingaas2)
            aia_slice_dict['ingaas2'] = get_slice(aia_ref_speclist_ingaas2,runlog_data['spectrum'],warn=False)

    # read airmass-dependent and -independent correction factors from the header of the .aia file
    aia_data['file'] = aia_file
    with open(aia_file,'r') as f:
        i = 0
        while i<nhead:
            line = f.readline()
            if 'Airmass-Dependent' in line:
                adcf_id = i+1
                nrow_adcf, ncol_adcf = np.array(line.split(':')[1].split()).astype(int)
            elif 'Airmass-Independent' in line:
                aicf_id = i+1
                nrow_aicf, ncol_aicf = np.array(line.split(':')[1].split()).astype(int)
                break 
            i = i+1
        else:
            logging.critical('Could not find the airmass-dependent and airmass-independent correction factors in the header of the .aia file')   
    adcf_data = pd.read_csv(aia_file,delim_whitespace=True,skiprows=adcf_id,nrows=nrow_adcf,names=['xgas','adcf','adcf_error', 'g', 'p'])
    aicf_data = pd.read_csv(aia_file,delim_whitespace=True,skiprows=aicf_id,nrows=nrow_aicf,names=['xgas','aicf','aicf_error','scale'])
    aicf_data.loc[:,'scale'] = aicf_data['scale'].apply(lambda x: '' if type(x)==float else x)

    # read pth data
    logging.info('\t- {}'.format(pth_file))
    nhead,ncol = file_info(pth_file)
    pth_data = pd.read_csv(pth_file,delim_whitespace=True,skiprows=nhead)
    # extract_pth lines correspond to runlog lines, so use the ingaas_runlog_slice to get the values along the time dimension
    pth_data = pth_data.loc[ingaas_runlog_slice]
    pth_data.loc[:,'hout'] = pth_data['hout']/(1-pth_data['hout']) # hout from extract_pth.out is a wet mole fraction; convert wet to dry mole fraction
    pth_data.loc[:,'hmod'] = pth_data['hmod'] # hmod from extract_pth.out is a dry mole fraction
    pth_data.loc[:,'tout'] = pth_data['tout']-273.15 # convert Kelvin to Celcius
    pth_data.loc[:,'tmod'] = pth_data['tmod']-273.15 # convert Kelvin to Celcius

    data_list = [tav_data,vav_data,ada_data,aia_data]

    # vsw files
    if not skip_vsw:
        logging.info('\t- {}'.format(vsw_file))
        nhead,ncol = file_info(vsw_file)
        vsw_sf_check = False
        with open(vsw_file,'r') as infile:
            i = 0
            while i<nhead:
                line = infile.readline()
                if 'sf=' in line:
                    vsw_sf_check = True
                    vsw_sf = (j for j in np.array(line.split()[1:]).astype(np.float))
                    break
                i += 1
        vsw_data = pd.read_csv(vsw_file,delim_whitespace=True,skiprows=nhead)
        vsw_data['file'] = vsw_file
        # vsw.ada file
        logging.info('\t- {}'.format(vsw_ada_file))
        nhead,ncol = file_info(vsw_ada_file)
        vsw_ada_data = pd.read_csv(vsw_ada_file,delim_whitespace=True,skiprows=nhead)
        vsw_ada_data['file'] = vsw_ada_file

        data_list += [vsw_data,vsw_ada_data]    

    ## check all files have the same spectrum lists
    logging.info('Check spectrum array consistency ...')
    hash_vav = hash_array(vav_spec_list)
    check_spec = np.alltrue([np.array_equal(hash_array(data['spectrum']),hash_vav) for data in data_list])
    if not check_spec:
        logging.critical('Files have inconsistent spectrum lists !')
        for data in data_list:
            logging.critical("{} spectra in {}".format(len(data['spectrum']),data['file'][0]))
        sys.exit(1)

    specdates = np.array([datetime(int(round(aia_data['year'][i]-aia_data['day'][i]/366.0)),1,1)+timedelta(days=aia_data['day'][i]-1) for i in range(nspec)])
    start_date = datetime.strftime(specdates[0],'%Y%m%d')
    end_date = datetime.strftime(specdates[-1],'%Y%m%d')

    private_nc_file = '{}{}_{}.private.nc'.format(siteID,start_date,end_date) # the final output file

    # make all the column names consistent between the different files
    for dataframe in [qc_data,esf_data,lse_data,vav_data,ada_data,aia_data]:
        dataframe.rename(str.lower,axis='columns',inplace=True) # all lower case
        if 'doy' in dataframe.columns: # all use 'day' and not 'doy'
            dataframe.rename(index=str,columns={'doy':'day'},inplace=True)
        if 'lon' in dataframe.columns:
            dataframe.rename(index=str,columns={'lon':'long'},inplace=True)

    qc_data['rsc'] = qc_data['scale'].copy()

    gas_list = [i for i in tav_data.columns[naux:] if ('_error' not in i) and ('file' not in i)] # gas names

    mchar = 0
    if 'spectrum' in tav_data.columns:
        mchar = 1

    if args.read_only:
        logging.critical('Code was run in READ ONLY mode, all inputs read, exiting now.')
        sys.exit()

    if os.path.exists(private_nc_file):
        os.remove(private_nc_file)

    logging.info('Writing {} ...'.format(private_nc_file))
    with netCDF4.Dataset(private_nc_file,'w',format=nc_format) as nc_data:
        
        ## global attributes

        # general
        nc_data.source = "Products retrieved from solar absorption spectra using the GGG2020 software"
        nc_data.description = '\n'+header_content
        nc_data.file_creation = "Created with Python {} and the library netCDF4 {}".format(platform.python_version(),netCDF4.__version__)
        nc_data.code_version = "Created using commit {} of the code {}".format(HEAD_commit,wnc_version)
        nc_data.flag_info = 'The Vmin and Vmax attributes of the variables indicate the range of valid values.\nThe values comes from the xx_qc.dat file.\n the variable "flag" stores the index of the variable that contains out of range values.\nThe variable "flagged_var_name" stores the name of that variable'
        
        if args.mode == 'TCCON':
            nc_data.title = "Atmospheric trace gas column-average dry-air mole fractions retrieved from solar absorption spectra measured by ground based Fourier Transform Infrared Spectrometers that are part of the Total Carbon Column Observing Network (TCCON)"
            nc_data.data_use_policy = "https://tccon-wiki.caltech.edu/Network_Policy/Data_Use_Policy"
            nc_data.auxiliary_data_description = "https://tccon-wiki.caltech.edu/Network_Policy/Data_Use_Policy/Auxiliary_Data"
            nc_data.more_information = "https://tccon-wiki.caltech.edu"
            nc_data.tccon_reference = "Wunch, D., G. C. Toon, J.-F. L. Blavier, R. A. Washenfelder, J. Notholt, B. J. Connor, D. W. T. Griffith, V. Sherlock, and P. O. Wennberg (2011), The total carbon column observing network, Philosophical Transactions of the Royal Society - Series A: Mathematical, Physical and Engineering Sciences, 369(1943), 2087-2112, doi:10.1098/rsta.2010.0240. Available from: http://dx.doi.org/10.1098/rsta.2010.0240"
        elif args.mode == 'em27':
            pass # should be updated with general em27 info

        # site specific
        for key,val in site_data.items():
            setattr(nc_data,key,val)

        # other
        nc_data.number_of_spectral_windows = str(len(col_file_list))
       
        if os.path.isdir(os.path.join(GGGPATH,'.hg')):
            try: 
                proc = subprocess.Popen(['hg','summary'],cwd=GGGPATH,stdout=subprocess.PIPE)
            except FileNotFoundError:
                gggtip = ''
                logging.warning('could not use the "hg" command to read the tip revision')
            else:
                out, err = proc.communicate()
                gggtip = out.decode("utf-8")
                logging.info('The output of "hg summary" from the GGG repository:\n %s',gggtip)
        else:
            gggtip = "Could not find .hg in the GGG repository"
            logging.warning('GGGtip %s',gggtip)
        nc_data.GGGtip = "The output of 'hg summary' from the GGG repository:\n"+gggtip
        nc_data.history = "Created {} (UTC)".format(time.asctime(time.gmtime(time.time())))
        # add file format version information
        write_file_fmt_attrs(nc_data)

        logging.info('Creating dimensions and coordinate variables')
        ## create dimensions
        """
        NOTE: when setting the time dimension as unlimited I get a segmentation fault when writing the prior data
        If the time dimension is fixed the writing of the prior data is MUCH faster and does not lead to segmentation fault.
        This is a known issue from the netCDF4 library (or the C library it's built on) and writing to multidimensional variables with unlimited dimensions
        We can fix this later when they fix the problem.
        As far as I am aware this will only be an issue if people want to concatenate multiple netcdf files along the time dimension, they will have to turn the time dimension to an unlimited dimension first
        """
        nc_data.createDimension('time',nspec)
        nc_data.createDimension('prior_time',nprior)
        nc_data.createDimension('prior_altitude',nlev) # used for the prior profiles
        nc_data.createDimension('cell_index',ncell)

        if classic:
            nc_data.createDimension('specname',speclength)
            nc_data.createDimension('a32',32)

        ## create coordinate variables
        nc_data.createVariable('time',np.float64,('time',))
        att_dict = {
            "standard_name": "time",
            "long_name": "time",
            "description": 'UTC time',
            "units": 'seconds since 1970-01-01 00:00:00',
            "calendar": 'gregorian',
        }
        nc_data['time'].setncatts(att_dict)

        nc_data.createVariable('prior_time',np.float64,('prior_time'))
        att_dict = {
            "standard_name": "prior_time",
            "long_name": "prior time",
            "description": 'UTC time for the prior profiles, corresponds to GEOS5 times every 3 hours from 0 to 21',
            "units": 'seconds since 1970-01-01 00:00:00',
            "calendar": 'gregorian',
        }
        nc_data['prior_time'].setncatts(att_dict)

        nc_data.createVariable('cell_index',np.int16,('cell_index'))
        att_dict = {
            "standard_name": "cell_index",
            "long_name": "cell_index",
            "description": "variables with names including 'cell_' will be along dimensions (prior_time,cell_index)",
        }
        nc_data['cell_index'].setncatts(att_dict)
        nc_data['cell_index'][:] = np.arange(ncell)

        nc_data.createVariable('prior_altitude',np.float32,('prior_altitude')) # this one doesn't change between priors
        att_dict = {
            "standard_name": 'prior_altitude_profile',
            "long_name": 'prior altitude profile',
            "units": units_dict['prior_altitude'],
            "description": "altitude levels for the prior profiles, these are the same for all the priors",
        }
        nc_data['prior_altitude'].setncatts(att_dict)
        nc_data['prior_altitude'][0:nlev] = prior_data[list(prior_data.keys())[0]]['data']['altitude'].values

        # averaging kernels
        with netCDF4.Dataset(ak_tables_nc_file(),'r') as ak_nc:
            nlev_ak = ak_nc['z'].size
            nbins_ak = ak_nc['slant_xgas_bin'].size

            # dimensions
            nc_data.createDimension('ak_altitude',nlev_ak) # make it separate from prior_altitude just in case we ever generate new files with different altitudes
            nc_data.createDimension('ak_slant_xgas_bin',nbins_ak)

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
            logging.info('Creating variables')
            logging.info('\t- Averaging kernels')

            nc_data.createVariable('ak_pressure',np.float32,('ak_altitude'))
            att_dict = {
                "standard_name": "averaging_kernel_pressure_levels",
                "long_name": "averaging kernel pressure levels",
                "description": "Median pressure for the column averaging kernels vertical grid",
                "units": 'hPa',
            }
            nc_data['ak_pressure'].setncatts(att_dict)
            nc_data['ak_pressure'][0:nlev_ak] = ak_nc['pressure'][:].data

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
                    att_dict['description'] = att_dict['description']+special_description_dict['lco2']
                elif 'wco2' in var:
                    att_dict['description'] = att_dict['description']+special_description_dict['wco2']
                nc_data[var].setncatts(att_dict)
                nc_data[var][:] = ak_nc[ak_var][:].data.astype(np.float32)
 
        # priors and cell variables
        logging.info('\t- Prior and cell variables')
        nc_data.createVariable('prior_index',np.int16,('time',))
        att_dict = {
            "standard_name": 'prior_index',
            "long_name": 'prior index',
            "units": '',
            "description": 'Index of the prior profile associated with each measurement, it can be used to sample the prior_ and cell_ variables along the prior_time dimension',
        }
        nc_data['prior_index'].setncatts(att_dict)

        prior_var_list = [ i for i in list(prior_data[list(prior_data.keys())[0]]['data'].keys()) if i not in {'altitude', 'geos_versions', 'geos_filenames', 'geos_checksums'}]
        cell_var_list = []
        units_dict.update({'prior_{}'.format(var):'' for var in prior_var_list if 'prior_{}'.format(var) not in units_dict})
        for var in prior_var_list:
            prior_var = 'prior_{}'.format(var)
            nc_data.createVariable(prior_var,np.float32,('prior_time','prior_altitude'))
            att_dict = {}
            att_dict["standard_name"] = '{}_profile'.format(prior_var)
            att_dict["long_name"] = att_dict["standard_name"].replace('_',' ')
            if var in ['temperature','density','pressure','gravity','equivalent_latitude']:
                att_dict["description"] = att_dict["long_name"]
            else:
                att_dict["description"] = 'a priori concentration profile of {}, in parts'.format(var)
            att_dict["units"] = units_dict[prior_var]
            nc_data[prior_var].setncatts(att_dict)

            if var in ['gravity', 'equivalent_latitude']:
                continue
            cell_var = 'cell_{}'.format(var)
            cell_var_list += [cell_var]
            nc_data.createVariable(cell_var,np.float32,('prior_time','cell_index'))
            att_dict = {}
            att_dict["standard_name"] = cell_var
            att_dict["long_name"] = att_dict["standard_name"].replace('_',' ')
            if var in ['temperature','density','pressure','equivalent_latitude']:
                att_dict["description"] = '{} in gas cell'.format(var)
            else:
                att_dict["description"] = 'concentration of {} in gas cell, in parts'.format(var)
            att_dict["units"] = units_dict[prior_var]
            nc_data[cell_var].setncatts(att_dict)

        prior_var_list += ['tropopause_altitude']
        nc_data.createVariable('prior_tropopause_altitude',np.float32,('prior_time'))
        att_dict = {
            "standard_name": 'prior_tropopause_altitude',
            "long_name": 'prior tropopause altitude',
            "description": 'altitude at which the gradient in the prior temperature profile becomes > -2 degrees per km',
            "units": units_dict[prior_var],
        }
        nc_data['prior_tropopause_altitude'].setncatts(att_dict)

        geos_version_keys = get_geos_versions_key_set(prior_data)

        prior_var_list += ['modfile','vmrfile'] + [geos_version_varname(k) for k in geos_version_keys]
        if classic:
            prior_modfile_var = nc_data.createVariable('prior_modfile','S1',('prior_time','a32'))
            prior_modfile_var._Encoding = 'ascii'
            prior_vmrfile_var = nc_data.createVariable('prior_vmrfile','S1',('prior_time','a32'))
            prior_vmrfile_var._Encoding = 'ascii'

            for (vkey, vfxn) in zip(['geos_versions', 'geos_filenames', 'geos_checksums'], [geos_version_varname, geos_file_varname, geos_checksum_varname]):
                add_geos_version_variables(nc_data, prior_data, vkey, vfxn, geos_version_keys, is_classic=True)
        else:
            prior_modfile_var = nc_data.createVariable('prior_modfile',str,('prior_time',))
            prior_vmrfile_var = nc_data.createVariable('prior_vmrfile',str,('prior_time',))
            for (vkey, vfxn) in zip(['geos_versions', 'geos_filenames', 'geos_checksums'], [geos_version_varname, geos_file_varname, geos_checksum_varname]):
                add_geos_version_variables(nc_data, prior_data, vkey, vfxn, geos_version_keys, is_classic=False)
        
        att_dict = {
            "standard_name":'prior_modfile',
            "long_name":'prior modfile',
            "description":'Model file corresponding to a given apriori',
        }
        nc_data['prior_modfile'].setncatts(att_dict)

        att_dict = {
            "standard_name":'prior_vmrfile',
            "long_name":'prior vmrfile',
            "description":'VMR file corresponding to a given apriori',
        }
        nc_data['prior_vmrfile'].setncatts(att_dict)

        prior_var_list += ['effective_latitude','mid_tropospheric_potential_temperature']
        nc_data.createVariable('prior_effective_latitude',np.float32,('prior_time',))
        att_dict = {
            "standard_name": 'prior_effective_latitude',
            "long_name": 'prior effective latitude',
            "description": "latitude at which the mid-tropospheric potential temperature agrees with that from the corresponding 2-week period in a GEOS-FPIT climatology",
            "units": units_dict['prior_effective_latitude'],
        }
        nc_data['prior_effective_latitude'].setncatts(att_dict)

        nc_data.createVariable('prior_mid_tropospheric_potential_temperature',np.float32,('prior_time',))
        att_dict = {
            "standard_name": 'prior_mid_tropospheric_potential_temperature',
            "long_name": 'prior mid-tropospheric potential temperature',
            "description": "average potential temperature between 700-500 hPa",
            "units": units_dict['prior_mid_tropospheric_potential_temperature'],
        }
        nc_data['prior_mid_tropospheric_potential_temperature'].setncatts(att_dict)

        geos_version_descriptions = {
            'Met2d': 'two-dimensional meteorological',
            'Met3d': 'three-dimensional meteorological',
            'Chm3d': 'three-dimensional chemical'
        }
        for k in geos_version_keys:
            desc = geos_version_descriptions.get(k, k)
            att_dict = {
                "description": f"Version information for the Goddard Earth Observing System model that provided the {desc} variables for the priors.",
                "note": "A trailing * indicates that the version information was assumed from the prior time."
            }
            nc_data[geos_version_varname(k)].setncatts(att_dict)

            nc_data[geos_file_varname(k)].description = f"Base name of the {desc} GEOS file used as input for the priors of this observations."
            nc_data[geos_checksum_varname(k)].description = f"MD5 checksum of the {desc} GEOS file used as input for the priors of this observation."

        # checksums
        logging.info('\t- Checksums')
        for var in checksum_var_list:
            if classic:
                checksum_var = nc_data.createVariable(var+'_checksum','S1',('time','a32'))
                checksum_var._Encoding = 'ascii'
            else:
                checksum_var = nc_data.createVariable(var+'_checksum',str,('time',))
            att_dict = {
                "standard_name": standard_name_dict[var+'_checksum'],
                "long_name": long_name_dict[var+'_checksum'],
                "description": 'hexdigest hash string of the md5 sum of the {} file'.format(var),
            }
            checksum_var.setncatts(att_dict)

        # code versions
        logging.info('\t- Code versions')
        nc_data.createVariable('gfit_version',np.float32,('time',))
        att_dict = {
            "description": "version number of the GFIT code that generated the data",
            "standard_name": standard_name_dict['gfit_version'],
            "long_name_dict": long_name_dict['gfit_version'],
        }
        nc_data['gfit_version'].setncatts(att_dict)

        nc_data.createVariable('gsetup_version',np.float32,('time',))
        att_dict = {
            "description": "version number of the GSETUP code that generated the priors",
            "standard_name": standard_name_dict['gsetup_version'],
            "long_name": long_name_dict['gsetup_version'],
        }
        nc_data['gsetup_version'].setncatts(att_dict)

        # flags
        logging.info('\t- Flags')
        nc_data.createVariable('flag',np.int16,('time',))
        att_dict = {
            "description": 'data quality flag, 0 = good',
            "standard_name": 'quality_flag',
            "long_name": 'quality flag',
        }
        nc_data['flag'].setncatts(att_dict)

        if classic:
            v = nc_data.createVariable('flagged_var_name','S1',('time','a32'))
            v._Encoding = 'ascii'
        else:
            nc_data.createVariable('flagged_var_name',str,('time',))
        att_dict = {
            "description": 'name of the variable that caused the data to be flagged; empty string = good',
            "standard_name": 'flagged_variable_name',
            "long_name": 'flagged variable name',
        }
        nc_data['flagged_var_name'].setncatts(att_dict)

        # spectrum file names
        logging.info('\t- Spectrum file names')
        if classic:
            v = nc_data.createVariable('spectrum','S1',('time','specname'))
            v._Encoding = 'ascii'
        else:
            nc_data.createVariable('spectrum',str,('time',))
        att_dict = {
            "standard_name": 'spectrum_file_name',
            "long_name": 'spectrum file name',
            "description": 'spectrum file name',
        }
        nc_data['spectrum'].setncatts(att_dict)

        for i,specname in enumerate(aia_data['spectrum'].values):
            nc_data['spectrum'][i] = specname        

        # auxiliary variables
        logging.info('\t- Auxiliary variables')
        aux_var_list = [tav_data.columns[i] for i in range(1,naux)]
        for var in aux_var_list:
            qc_id = list(qc_data['variable']).index(var)
            #digit = int(qc_data['format'][qc_id].split('.')[-1])
            if var in ['year','day']:
                var_type = np.int16
            else:
                var_type = np.float32 
            nc_data.createVariable(var,var_type,('time',))#,zlib=True)#,least_significant_digit=digit)
            # set attributes using the qc.dat file
            att_dict = {
                "description": qc_data['description'][qc_id],
                "units": qc_data['unit'][qc_id].replace('(','').replace(')','').strip(),
                "vmin": qc_data['vmin'][qc_id],
                "vmax": qc_data['vmax'][qc_id],
                "precision": qc_data['format'][qc_id],
            }
            if var in standard_name_dict.keys():
                att_dict["standard_name"] = standard_name_dict[var]
                att_dict["long_name"] = long_name_dict[var]
                att_dict["units"] = units_dict[var] # reset units here for some of the variables in the qc_file using UDUNITS compatible units
            nc_data[var].setncatts(att_dict)

        nc_data['hour'].description = 'Fractional UT hours (zero path difference crossing time)'

        # get model surface values from the output of extract_pth.f
        logging.info('\t- extract_pth')
        mod_var_dict = {'tmod':'tout','pmod':'pout'}
        for key,val in mod_var_dict.items(): # use a mapping to the equivalent runlog variables to querry their qc.dat info
            qc_id = list(qc_data['variable']).index(val)
            var_type = np.float32 
            nc_data.createVariable(key,var_type,('time'))
            att_dict = {
                "description": 'model {}'.format(qc_data['description'][qc_id]),
                "vmin": qc_data['vmin'][qc_id],
                "vmax": qc_data['vmax'][qc_id],
                "precision": qc_data['format'][qc_id],
            }
            if key in standard_name_dict.keys():
                att_dict["standard_name"] = standard_name_dict[key]
                att_dict["long_name"] = long_name_dict[key]
                att_dict["units"] = units_dict[key]
            nc_data[key].setncatts(att_dict)
            write_values(nc_data,key,np.array(pth_data[key]))
        # Use extract_pth.out hmod and hout to create h2o_dmf_out and h2_dmf_mod
        for key,val in {'h2o_dmf_out':'hout','h2o_dmf_mod':'hmod'}.items():
            nc_data.createVariable(key,np.float32,('time',))
            att_dict = {
                "standard_name":standard_name_dict[key],
                "long_name":long_name_dict[key],
                "units":units_dict[key],
            }
            nc_data[key].setncatts(att_dict)
            write_values(nc_data,key,np.array(pth_data[val]))
        nc_data['h2o_dmf_out'].description = "external water vapour dry mole fraction"
        nc_data['h2o_dmf_mod'].description = "model external water vapour dry mole fraction"
        del pth_data
        gc.collect()

        # write variables from the .vsw and .vsw.ada files
        if not skip_vsw:
            logging.info('\t- .vsw and vsw.ada')
            vsw_var_list = [vsw_data.columns[i] for i in range(naux,len(vsw_data.columns)-1)]  # minus 1 because I added the 'file' column
            full_vsw_var_list = []
            for var in vsw_var_list:
                center_wavenumber = int(''.join([i for i in var.split('_')[1] if i.isdigit()]))
                gas = var.split('_')[0]
                # .vsw file
                if gas in insb_only:
                    varname = 'vsw_{}_insb'.format(var)
                elif center_wavenumber<4000:
                    varname = 'vsw_{}_insb'.format(var[1:])
                elif gas in si_only:
                    varname = 'vsw_{}_si'.format(var)
                elif center_wavenumber>10000:
                    varname = 'vsw_{}_si'.format(var[1:])
                else:
                    varname = 'vsw_{}'.format(var)
                full_vsw_var_list += [varname]
                nc_data.createVariable(varname,np.float32,('time',))
                att_dict = {
                    "standard_name": varname,
                    "long_name": varname.replace('_',' '),
                    "units": '',
                    "precision": 'e12.4',
                }
                if 'error' in varname:
                    att_dict["description"] = "{0} total column density {2} from the window centered at {1} cm-1.".format(*var.split('_'))
                else:
                    att_dict["description"] = "{} total column density from the window centered at {} cm-1".format(*var.split('_'))
                    if vsw_sf_check:
                        # write the data from the vsf= line ine the header of the vsw file
                        if gas in insb_only:
                            sf_var = 'vsw_sf_{}_insb'.format(var)
                        elif center_wavenumber<4000:
                            sf_var = 'vsw_sf_{}_insb'.format(var[1:])
                        elif gas in si_only:
                            sf_var = 'vsw_sf_{}_si'.format(var)
                        elif center_wavenumber>10000:
                            sf_var = 'vsw_sf_{}_si'.format(var[1:])
                        else:
                            sf_var = 'vsw_sf_{}'.format(var)
                        full_vsw_var_list += [sf_var]
                        nc_data.createVariable(sf_var,np.float32,('time',))
                        sf_att_dict = {
                            "standard_name": sf_var,
                            "long_name": sf_var.replace('_',' '),
                            "description": "{} correction factor from the window centered at {} cm-1".format(*var.split('_')),
                            "units": '',
                        }
                        nc_data[sf_var].setncatts(sf_att_dict)
                        nc_data[sf_var][:] = next(vsw_sf)
                nc_data[varname].setncatts(att_dict)
                write_values(nc_data,varname,vsw_data[var])

                # .vsw.ada file
                xvar = 'x'+var
                if gas in insb_only:
                    varname = 'vsw_ada_x{}_insb'.format(var)
                elif center_wavenumber<4000:
                    varname = 'vsw_ada_x{}_insb'.format(var[1:])
                elif gas in si_only:
                    varname = 'vsw_ada_x{}_si'.format(var)
                elif center_wavenumber>10000:
                    varname = 'vsw_ada_x{}_si'.format(var[1:])
                else:
                    varname = 'vsw_ada_'+xvar
                full_vsw_var_list += [varname]
                nc_data.createVariable(varname,np.float32,('time',))
                att_dict = {
                    "standard_name":varname,
                    "long_name":varname.replace('_',' '),
                    "units":'',
                    "precision":'e12.4',
                }
                if 'error' in varname:
                    att_dict["description"] = "{0} column-average mole fraction {2} from the window centered at {1} cm-1, after airmass dependence is removed, but before scaling to WMO.".format(*xvar.split('_'))
                else:
                    att_dict["description"] = "{} column-average mole fraction from the window centered at {} cm-1, after airmass dependence is removed, but before scaling to WMO.".format(*xvar.split('_'))
                nc_data[varname].setncatts(att_dict)
                write_values(nc_data,varname,vsw_ada_data[xvar])
            # end of for var in vsw_var_list
            del vsw_data
            del vsw_ada_data
            gc.collect()
        # end of if not skip_vsw

        # averaged variables (from the different windows of each species)
        logging.info('\t- averaged variables')
        main_var_list = [tav_data.columns[i] for i in range(naux,len(tav_data.columns)-1)]  # minus 1 because I added the 'file' column
        full_main_var_list = []
        for var in main_var_list:
            xvar = 'x'+var
            varname = var
            xvarname = xvar
            qc_id = list(qc_data['variable']).index(xvar)
            gas = var.split('_')[0]

            if var.startswith('m'):
                xvarname = 'x{}_insb'.format(var[1:])
                varname = var[1:]+'_insb'
            elif gas in insb_only:
                xvarname = 'x{}_insb'.format(var)
                varname = var+'_insb'                
            elif var.startswith('v'):
                xvarname = 'x{}_si'.format(var[1:])
                varname = var[1:]+'_si'
            elif gas in si_only:
                xvarname = 'x{}_si'.format(var)
                varname = var+'_si'                                
            
            full_main_var_list += [xvarname]
            nc_data.createVariable(xvarname,np.float32,('time',))
            att_dict = {
                "standard_name": xvar,
                "long_name": xvar.replace('_',' '),
                "description": qc_data['description'][qc_id],
                "units": qc_data['unit'][qc_id].replace('(','').replace(')','').strip(),
                "vmin": qc_data['vmin'][qc_id],
                "vmax": qc_data['vmax'][qc_id],
                "precision": qc_data['format'][qc_id],
            }
            nc_data[xvarname].setncatts(att_dict)
            #nc_data[xvar] will be written from the .aia data further below, not in this loop

            full_main_var_list += ['vsf_'+varname]
            nc_data.createVariable('vsf_'+varname,np.float32,('time',))
            att_dict = {
                "description": varname+" VMR Scale Factor.",
                "precision": 'e12.4',
            }
            nc_data['vsf_'+varname].setncatts(att_dict)
            write_values(nc_data,'vsf_'+varname,tav_data[var].values)
            
            full_main_var_list += ['column_'+varname]
            nc_data.createVariable('column_'+varname,np.float32,('time',))
            att_dict = {
                "description": varname+' column average.',
                "units": 'molecules.cm-2',
                "precision": 'e12.4',
            }
            nc_data['column_'+varname].setncatts(att_dict)
            write_values(nc_data,'column_'+varname,vav_data[var].values)

            full_main_var_list += ['ada_'+xvarname]
            nc_data.createVariable('ada_'+xvarname,np.float32,('time',))
            att_dict = {
                "units": "",
                "precision": 'e12.4',
            }
            if 'error' in varname:
                att_dict["description"] = 'uncertainty associated with ada_{}'.format(xvarname.replace('_error',''))
            else:
                att_dict["description"] = varname+' column-average dry-air mole fraction computed after airmass dependence is removed, but before scaling to WMO.'
            nc_data['ada_'+xvarname].setncatts(att_dict)
            write_values(nc_data,'ada_'+xvarname,ada_data[xvar].values)

            for key in special_description_dict.keys():
                if key in var:
                    for nc_var in [nc_data[xvarname],nc_data['vsf_'+varname],nc_data['column_'+varname],nc_data['ada_'+xvarname]]:
                        nc_var.description += special_description_dict[key]
        del tav_data, vav_data, ada_data
        gc.collect()

        # lse data
        logging.info('\t- .lse')
        lse_dict = {
                    'lst':{'description':'The type of LSE correction applied (0=none; 1=InGaAs (disabled); 2=Si; 3=Dohe et al. (disabled); 4=Other (disabled))',
                            'precision':'e12.4',},
                    'lse':{'description':'Laser sampling error (shift)',
                            'precision':'e12.4',},
                    'lsu':{'description':'Laser sampling error uncertainty',
                            'precision':'e12.4',},
                    'lsf':{'description':'laser sampling fraction',
                            'precision':'e12.4',},
                    'dip':{'description':'A proxy for nonlinearity - the dip at ZPD in the smoothed low-resolution interferogram',
                            'precision':'e12.4',},
                    'mvd':{'description':'Maximum velocity displacement - a measure of how smoothly the scanner is running',
                            'precision':'f9.4',},
                    }

        _, aia_spec_inds, common_spec = np.intersect1d(hash_array(aia_data['spectrum']),hash_array(lse_data['spectrum']),return_indices=True)
        # np.intersect1d sorts the returned indices by the hash value, meaning that by default they will not be in the 
        # correct order to match the spectra. To fix this, get the indices needed to sort the .aia spectra, then use
        # those to sort the LSE data. Will check that this means the .lse spectra match the .aia spectra.
        tmp_inds = np.argsort(aia_spec_inds)
        common_spec = common_spec[tmp_inds]
        if not np.array_equal(np.array(aia_data['spectrum']), np.array(lse_data['spectrum'][common_spec])):
            logging.critical('Unable to match .aia and .lse spectra')
            sys.exit(1)

        for var in lse_dict.keys():
            if var == "dip":
                continue
            nc_data.createVariable(var,np.float32,('time',))
            att_dict = {
                "standard_name":standard_name_dict[var],
                "long_name":long_name_dict[var],
                "description":lse_dict[var]['description'],
                "precision":lse_dict[var]['precision'],
            }
            nc_data[var].setncatts(att_dict)
            write_values(nc_data,var,lse_data[var][common_spec].values)
        # unlike other .lse variables, dip is specific to each detector
        for detector in runlog_slice_dict:
            if detector == 'ingaas':
                # Because GGG2020 processing started before we realized different detector's DIP values were
                # being written to the dip variable, early files already existed with "dip" as the variable,
                # and code exists to plot that variable. Thus we keep "dip" as the variable name for InGaAs
                # data - though this could be changed at the next GGG version.
                #
                # .private.nc files where "dip" does not have the attribute "detector" were written before
                # this fix was implemented.
                varname = 'dip'
            else:
                varname = f"dip_{detector}"
            nc_data.createVariable(varname,np.float32,('time',))
            att_dict = {
                "standard_name":standard_name_dict['dip'],
                "long_name":long_name_dict['dip'],
                "description":lse_dict['dip']['description'],
                "precision":lse_dict['dip']['precision'],
                "detector":detector, 
            }
            nc_data[varname].setncatts(att_dict)
            detector_dip = np.full(nc_data['time'].shape,fill_value = netCDF4.default_fillvals["f4"])
            detector_dip[aia_slice_dict[detector]] = lse_data.set_index(lse_data.index.astype(int)).loc[runlog_slice_dict[detector]]['dip'].values
            nc_data[varname][:] = detector_dip
        del lse_data
        gc.collect()
        
        logging.info('\t- ADCF')
        # airmass-dependent corrections (from the .aia file header)
        correction_var_list = []
        for i,xgas in enumerate(adcf_data['xgas']):
            for var in ['adcf','adcf_error','g','p']:
                varname = '{}_{}'.format(xgas,var)
                correction_var_list += [varname]
                nc_data.createVariable(varname,np.float32,('time',))
                att_dict = {
                    "standard_name": varname,
                    "long_name": varname.replace('_',' '),
                    "precision": 'f9.4',
                }
                if 'error' in var:
                    att_dict["description"] = 'Error of the {} airmass-dependent correction factor'.format(xgas)
                elif var.endswith('g'):
                    att_dict['description'] = 'Polynomial 0 point SZA value for {} airmass-dependent correction factor'.format(xgas)
                elif var.endswith('p'):
                    att_dict['description'] = 'Polynomial order for {} airmass-dependent correction factor'.format(xgas)
                else:
                    att_dict["description"] = '{} airmass-dependent correction factor'.format(xgas)
                nc_data[varname].setncatts(att_dict)
                nc_data[varname][:] = adcf_data[var][i]

        logging.info('\t- AICF')
        # airmass-independent corrections (from the .aia file header)
        for i,xgas in enumerate(aicf_data['xgas']):
            for var in ['aicf','aicf_error']:
                varname = '{}_{}'.format(xgas,var)
                correction_var_list += [varname]
                nc_data.createVariable(varname,np.float32,('time',))
                att_dict = {
                    "standard_name": varname,
                    "long_name": varname.replace('_',' '),
                    "precision": 'f9.4',
                }
                if 'error' in var:
                    att_dict["description"] = 'Error of the {} airmass-independent correction factor'.format(xgas)
                else:
                    att_dict["description"] = '{} airmass-independent correction factor'.format(xgas)
                nc_data[varname].setncatts(att_dict)
                nc_data[varname][:] = aicf_data[var][i]

            varname = 'aicf_{}_scale'.format(xgas)
            if classic:
                v = nc_data.createVariable(varname,'S1',('time','a32'))
                v._Encoding = 'ascii'
            else:
                nc_data.createVariable(varname,str,('time',))
            att_dict = {
                "standard_name":"aicf_scale",
                "long_name":"aicf scale",
                "description":"{} traceability, indicates which WMO scale this gas is tied to".format(xgas),
            }
            nc_data[varname].setncatts(att_dict)
            if aicf_data['scale'][i]=='':
                continue
            for j in range(nc_data['time'].size):
                nc_data[varname][j] = aicf_data['scale'][i]
        del aicf_data,adcf_data
        gc.collect()

        ## write data

        # prior data
        logging.info('Computing prior index ...')
        prior_spec_list = list(prior_data.keys())
        spec_list = nc_data['spectrum'][:]
        if nprior == 1:
            # if there is just one block in the .mav file, set it as the prior index for all spectra
            nc_data['prior_index'][:] = 0
        else:
            prior_runlog_inds = get_slice(runlog_all_speclist, prior_spec_list)
            aia_runlog_inds = get_slice(runlog_all_speclist, spec_list)
            nspec = len(spec_list)
            for spec_id, spectrum in enumerate(spec_list):
                # The .mav blocks should always be in runlog order. Set the prior index to point to
                # the last .mav block with a spectrum that comes before the .aia spectrum in the runlog.
                prior_index = np.flatnonzero(prior_runlog_inds <= aia_runlog_inds[spec_id])[-1]
                nc_data['prior_index'][spec_id] = prior_index

        
        # write prior and cell data
        logging.info('Writing prior data ...')
        special_prior_vars = ['tropopause_altitude','modfile','vmrfile','mid_tropospheric_potential_temperature','effective_latitude']
        special_prior_vars += [geos_version_varname(k) for k in geos_version_keys]
        special_prior_vars += [geos_file_varname(k) for k in geos_version_keys]
        special_prior_vars += [geos_checksum_varname(k) for k in geos_version_keys]
        for prior_spec_id, prior_spectrum in enumerate(prior_spec_list):
            #for var in ['temperature','pressure','density','gravity','1h2o','1hdo','1co2','1n2o','1co','1ch4','1hf','1o2']:
            for var in prior_var_list:
                if var not in special_prior_vars:
                    prior_var = 'prior_{}'.format(var)
                    nc_data[prior_var][prior_spec_id,0:nlev] = prior_data[prior_spectrum]['data'][var].values

                    if var not in ['gravity','equivalent_latitude']:
                        cell_var = 'cell_{}'.format(var)
                        nc_data[cell_var][prior_spec_id,0:ncell] = prior_data[prior_spectrum]['cell_data'][var].values

            nc_data['prior_time'][prior_spec_id] = prior_data[prior_spectrum]['time']
            nc_data['prior_tropopause_altitude'][prior_spec_id] = prior_data[prior_spectrum]['tropopause_altitude']
            nc_data['prior_modfile'][prior_spec_id] = prior_data[prior_spectrum]['mod_file']
            nc_data['prior_vmrfile'][prior_spec_id] = prior_data[prior_spectrum]['vmr_file']
            nc_data['prior_effective_latitude'][prior_spec_id] = prior_data[prior_spectrum]['effective_latitude']
            nc_data['prior_mid_tropospheric_potential_temperature'][prior_spec_id] = prior_data[prior_spectrum]['mid_tropospheric_potential_temperature']
            for k in geos_version_keys:
                nc_data[geos_version_varname(k)][prior_spec_id] = prior_data[prior_spectrum]['geos_versions'][k]
                nc_data[geos_file_varname(k)][prior_spec_id] = prior_data[prior_spectrum]['geos_filenames'][k]
                nc_data[geos_checksum_varname(k)][prior_spec_id] = prior_data[prior_spectrum]['geos_checksums'][k]

        logging.info('Finished writing prior data')
        del prior_data
        gc.collect()

        # update data with new scale factors and determine flags
        logging.info('Writing scaled aia data and determining qc flags')
        for esf_id in range(esf_data['year'].size):
                    
            # indices to slice the data for the concerned spectra
            start = np.sum(esf_data['n'][:esf_id])
            end = start + esf_data['n'][esf_id] 

            """
            If new day, read in the daily error scale factors and compute
            new scale factors (RSC) as weighted averages of the a priori
            ESF factors from the qc.dat file, and the daily values.
            A priori ESF values are the ratio of the xx_error/xxa scale factors
            read in from the qc.dat file, with 100% uncertainties assumed.
            """     
            for gas in gas_list:
                xgas = 'x'+gas
                qc_id = list(qc_data['variable']).index(xgas)
                apesf = qc_data['scale'][qc_id+1]/qc_data['scale'][qc_id] # xx_error/xx
                qc_data.loc[qc_data.index==qc_id+1,'rsc'] = qc_data['scale'][qc_id]*(1.0/apesf+esf_data[xgas][esf_id]/esf_data[xgas+'_error'][esf_id]**2)/(1.0/apesf**2+1.0/esf_data[xgas+'_error'][esf_id]**2)

            """
            Look within each data record to see if any of the data values are
            outside their VMIN to VMAX range. If so, set eflag to the index of
            the variable that was furthest out of range. Then write out the data.
            """
            eflag = np.zeros(end-start)
            kmax = np.zeros(end-start)
            dmax = np.zeros(end-start)
            for var_id,var in enumerate(aia_data.columns[mchar:-1]):
                gas = var.split('_')[0][1:] # [1:] removes the 'x' for gas variables

                if var.startswith('xm'):
                    varname = 'x{}_insb'.format(var[2:])
                    ingaas = False
                elif gas in insb_only:
                    varname = var+'_insb'
                    ingaas = False
                elif var.startswith('xv'):
                    varname = 'x{}_si'.format(var[2:])
                    ingaas = False
                elif gas in si_only:
                    varname = var+'_si'
                    ingaas = False
                else:
                    varname = var
                    ingaas = True

                nnan = np.count_nonzero(np.isnan(aia_data[var]))
                if ingaas and nnan>=1 and esf_id==0:
                    logging.warning('{} NAN values for {}'.format(nnan,var))
                nmiss = len(aia_data[var][aia_data[var]>=9e29])
                if ingaas and nmiss >= 1 and esf_id==0: # only show this for InGaAs spectra
                    logging.warning('{} ({}%) missing values for {}'.format(nmiss,np.round(100*nmiss/nspec,2),var))

                qc_id = list(qc_data['variable']).index(var)
                digit = int(qc_data['format'][qc_id].split('.')[-1])
                fillval = netCDF4.default_fillvals[nc_data[varname].dtype.str[1:]]

                aia_qc_data = np.round(aia_data[var][start:end].values*qc_data['rsc'][qc_id],digit)
                aia_qc_data[aia_qc_data>9e29] = fillval
                nc_data[varname][start:end] = aia_qc_data

                dev = np.abs( (qc_data['rsc'][qc_id]*aia_data[var][start:end].values-qc_data['vmin'][qc_id])/(qc_data['vmax'][qc_id]-qc_data['vmin'][qc_id]) -0.5 )
                dev[np.where(np.isclose(aia_qc_data,fillval))[0]] = 0 # don't flag variables for having missing values
                
                if ingaas: # only set flags based on ingaas data
                    kmax[dev>dmax] = qc_id+1 # add 1 here, otherwise qc_id starts at 0 for 'year'
                    dmax[dev>dmax] = dev[dev>dmax]

            eflag[dmax>0.5] = kmax[dmax>0.5]
            
            # write the flagged variable index
            nc_data['flag'][start:end] = [int(i) if not np.isnan(i) else -1 for i in eflag]

            # write the flagged variable name
            for i in range(start,end):
                if eflag[i-start] == 0:
                    nc_data['flagged_var_name'][i] = ""                    
                else:
                    flagged_var_name = qc_data['variable'][eflag[i-start]-1]
                    if flagged_var_name.startswith('xm'):
                        nc_data['flagged_var_name'][i] = 'x{}_insb'.format(flagged_var_name[2:])
                    elif flagged_var_name.startswith('xv'):
                        nc_data['flagged_var_name'][i] = 'x{}_si'.format(flagged_var_name[2:])
                    else:
                        nc_data['flagged_var_name'][i] = flagged_var_name
        # end of for esf_id in range(esf_data['year'].size)
        del esf_data
        gc.collect()

        flag_list = [i for i in set(nc_data['flag'][:]) if i!=0]
        nflag = np.count_nonzero(nc_data['flag'][:])
        logging.info('Summary of automatic QC flags:')
        logging.info('  #  Parameter              N_flag      %')
        kflag_list = [nc_data['flag'][nc_data['flag']==flag].size for flag in flag_list]
        sorted_kflags_id = np.argsort(kflag_list)[::-1]
        for i in sorted_kflags_id:
            kflag = kflag_list[i]
            flag = flag_list[i]
            logging.info('{:>3}  {:<20} {:>6}   {:>8.3f}'.format(flag,qc_data['variable'][flag-1],kflag,100*kflag/nc_data['time'].size))
        logging.info('     {:<20} {:>6}   {:>8.3f}'.format('TOTAL',nflag,100*nflag/nc_data['time'].size))

        # time
        write_values(nc_data,'year',np.round(aia_data['year'][:].values-aia_data['day'][:].values/366.0))
        write_values(nc_data,'day',np.round(aia_data['day'][:].values-aia_data['hour'][:].values/24.0))
        write_values(nc_data,'time',np.array([elem.total_seconds() for elem in (specdates-datetime(1970,1,1))]))

        # write data from .col and .cbf files
        logging.info('Writing .col and .cbf data:')
        col_var_list = []
        # If there are InSb or Si windows fitted, get the indices along the time dimension of the spectra in the .col file
        insb_col_file_list = [i for i in col_file_list if int(''.join([j for j in i.split('.')[0].split('_')[1] if j.isdigit()]))<4000]
        si_col_file_list = [i for i in col_file_list if int(''.join([j for j in i.split('.')[0].split('_')[1] if j.isdigit()]))>10000]
        if insb_col_file_list:
            col_data, gfit_version, gsetup_version = read_col(insb_col_file_list[0],speclength)
            insb_slice = list(np.where(np.isin(aia_ref_speclist_insb,col_data['spectrum']))[0]) # indices of the InSb spectra along the time dimension
            if np.array_equal(insb_slice,aia_data.index.values):
                # if the insb slice is the full indices, set it to empty list such that write_values writes the whole array at once instead of looping over indices
                insb_slice = []
        if si_col_file_list:
            col_data, gfit_version, gsetup_version = read_col(si_col_file_list[0],speclength)
            si_slice = list(np.where(np.isin(aia_ref_speclist_si,col_data['spectrum']))[0]) # indices of the Si spectra along the time dimension
            if np.array_equal(si_slice,aia_data.index.values):
                # if the si slice is the full indices, set it to empty list such that write_values writes the whole array at once instead of looping over indices
                si_slice = []

        for col_id,col_file in enumerate(col_file_list):
            center_wavenumber = int(''.join([j for j in col_file.split('.')[0].split('_')[1] if j.isdigit()]))
            if show_progress:
                progress(col_id,len(col_file_list),word=col_file)

            cbf_file = col_file.replace('.col','.cbf')
            nhead,ncol = file_info(cbf_file)
            with open(cbf_file,'r') as infile:
                content = [infile.readline() for i in range(nhead+1)]
            headers = content[nhead].split()
            ncbf = len(headers)-4
            if ncbf>0:
                widths = [speclength+2,9,9,7,12]+[9]*(ncbf-1)
            else:
                widths = [speclength+2,9,9,7]
            cbf_data = pd.read_fwf(cbf_file,widths=widths,names=headers,skiprows=nhead+1)
            cbf_data.rename(str.lower,axis='columns',inplace=True)
            cbf_data.rename(index=str,columns={'cf_amp/cl':'cfampocl','cf_period':'cfperiod','cf_phase':'cfphase'},inplace=True)
            cbf_data.rename(index=str,columns={'spectrum_name':'spectrum'},inplace=True)
            cbf_data['spectrum'] = cbf_data['spectrum'].map(lambda x: x.strip('"')) # remove quotes from the spectrum filenames
            
            gas_XXXX = col_file.split('.')[0] # gas_XXXX, suffix for nc_data variable names corresponding to each .col file (i.e. VSF_h2o from the 6220 co2 window becomes co2_6220_VSF_co2)
            gas = gas_XXXX.split('_')[0]
            if gas.startswith('m') or gas.startswith('v'):
                gas_XXXX = gas_XXXX[1:]

            # check if it is insb or ingaas window
            if center_wavenumber<4000: # InSb
                ingaas, insb, si = [False,True,False]
                inds = insb_slice
            elif center_wavenumber>10000: # Si
                ingaas, insb, si = [False,False,True]
                inds = si_slice
            else:
                ingaas, insb, si = [True,False,False]
                inds = []

            # read col_file headers
            col_data, gfit_version, gsetup_version = read_col(col_file,speclength)

            if col_file == col_file_list[0]:
                nhead,ncol = file_info(col_file)
                with open(col_file,'r') as infile:
                    content = [infile.readline() for i in range(nhead+1)]                
                # check that the checksums are right for the files listed in the .col file header
                checksum_dict = OrderedDict((key+'_checksum',None) for key in checksum_var_list)
                # If a line begins with a 32-character MD5 hash, then one or more spaces, then
                # a non-whitespace character, verify the checksum. That corresponds to a line like:
                # 
                #   34136d7d03967a662edc7b0c92b984f1  /home/jlaugh/GGG/ggg-my-devel/config/data_part.lst
                #
                # If not, then there's no checksum or no file following it.
                content_lines = [line for line in content if re.match(r'[a-f0-9]{32}\s+[^\s]', line)]
                for i,line in enumerate(content_lines):
                    csum,fpath = line.split()
                    if not args.skip_checksum:
                        try:
                            checksum(fpath,csum)
                        except FileNotFoundError:
                            logging.warning('Could not find %s. Skip the checksum check ! To silence this, run with the --skip-checksum argument',fpath)
                    checksum_dict[checksum_var_list[i]+'_checksum'] = csum

                nc_data['gfit_version'][:] = gfit_version
                nc_data['gsetup_version'][:] = gsetup_version
                for var in checksum_var_list:
                    checksum_var = var+'_checksum'
                    for i in range(aia_data['spectrum'].size):
                        nc_data[checksum_var][i] = checksum_dict[checksum_var]
                del aia_data
                gc.collect()
            # end of if col_file == col_file_list[0]:

            if col_data.shape[0] != cbf_data.shape[0]: # do this first since it's faster than the check on spectra
                logging.warning('\nDifferent number of spectra in %s and %s, recommend checking this col/cbf pair', col_file, cbf_file)
                continue

            # JLL 2020-05-19: need to check that the shapes are equal first, or get a very confusing error
            col_ref_speclist = np.array([i.replace('c.','a.').replace('b.','a.').replace('d.','a.') for i in col_data['spectrum']]) # this is the .col spectrum list but with only ingaas names
            hash_col = hash_array(col_ref_speclist)
            if ingaas and not (np.array_equal(hash_col,hash_runlog_ingaas_speclist) or np.array_equal(hash_col,hash_runlog_ingaas2_speclist)):
                logging.warning('\nMismatch between .col file spectra and .grl spectra; col_file=%s',col_file)
                continue # contine or exit here ? Might not need to exit if we can add in the results from the faulty col file afterwards

            if ingaas and (col_data.shape[0] != vav_shape):
                inds = get_slice(vav_spec_list,col_data['spectrum'].apply(lambda x: x.replace('d.','a.')))
                dif = col_data['spectrum'].size - len(inds)
                logging.warning('\nThere are {} more spectra in {} than in {}, recommend checking this col/vav file'.format(dif,col_file, vav_file))

            # create window specific variables
            for var in col_data.columns[1:]: # skip the first one ("spectrum")
                if ingaas:
                    varname = '_'.join([gas_XXXX,var])
                elif insb:
                    varname = '_'.join([gas_XXXX,var,'insb'])
                elif si:
                    varname = '_'.join([gas_XXXX,var,'si'])
                col_var_list += [varname]
                nc_data.createVariable(varname,np.float32,('time',))
                
                att_dict = {}               
                if var in standard_name_dict.keys():
                    att_dict['standard_name'] = standard_name_dict[var]
                    att_dict['long_name'] = long_name_dict[var]
                
                if '_' in var:
                    att_dict['description'] = '{} {} retrieved from the {} window centered at {} cm-1.'.format(var.split('_')[1],var.split('_')[0],gas_XXXX.split('_')[0],gas_XXXX.split('_')[1])                    

                    try:
                        iso = int(var.split('_')[1][0])
                    except:
                        pass
                    else:
                        iso_gas = var.split('_')[1]
                        if iso in [1,2,3]:
                            sup = ['st','nd','rd'][iso-1]
                        else:
                            sup = 'th'
                        att_dict['description'] += "{} is the {}{} isotopolog of {} as listed in GGG's isotopologs.dat file.".format(iso_gas,iso,sup,iso_gas[1:])
                else:
                    att_dict['description'] = '{} retrieved from the {} window centered at {} cm-1.'.format(long_name_dict[var],gas_XXXX.split('_')[0],gas_XXXX.split('_')[1])
                for key in special_description_dict.keys():
                    if key in varname:
                        att_dict['description'] += special_description_dict[key]
                if varname.endswith(('fs','sg')):
                    att_dict['units'] = units_dict[varname[-2:]]
                    att_dict['description'] += "The {} (wavenumber shift per spectral point) is in ppm of the spectral point spacing ({:.11f} cm-1)".format(long_name_dict[var],dnu)
                nc_data[varname].setncatts(att_dict)
                write_values(nc_data,varname,col_data[var].values,inds=inds)
            
            # add data from the .cbf file
            if ingaas:
                ncbf_var = '{}_ncbf'.format(gas_XXXX)
            elif insb:
                ncbf_var = '{}_ncbf_insb'.format(gas_XXXX)
            elif si:
                ncbf_var = '{}_ncbf_si'.format(gas_XXXX)
            col_var_list += [ncbf_var]
            nc_data.createVariable(ncbf_var,np.int32,('time',))
            att_dict = {'standard_name':standard_name_dict['ncbf'],'long_name':long_name_dict['ncbf'],'units':units_dict['ncbf']}
            nc_data[ncbf_var].setncatts(att_dict)
            nc_data[ncbf_var][:] = len(cbf_data.columns)-1 # minus 1 because of the spectrum name column

            for var in cbf_data.columns[1:]: # don't use the 'Spectrum' column
                varname = '_'.join([gas_XXXX,var])
                col_var_list += [varname]
                nc_data.createVariable(varname,np.float32,('time',))
                att_dict = {}
                if '_' in var:
                    att_dict['standard_name'] = standard_name_dict[var.split('_')[0]].format(var.split('_')[1])
                    att_dict['long_name'] = long_name_dict[var.split('_')[0]].format(var.split('_')[1])
                else:
                    att_dict['standard_name'] = standard_name_dict[var]
                    att_dict['long_name'] = long_name_dict[var]
                    att_dict['units'] = units_dict[var]
                nc_data[varname].setncatts(att_dict)
                write_values(nc_data,varname,cbf_data[var].values,inds=inds)
        # end of for col_id,col_file in enumerate(col_file_list)
        del col_data, cbf_data
        gc.collect()

        # read the data from missing_data.json and update data with fill values to the netCDF4 default fill value
        """
        It is a dictionary with siteID as keys, values are dictionaries of variable:fill_value

        If a site has different null values defined for different time periods the key has format siteID_ii_YYYYMMDD_YYYYMMDD
        with ii just the period index (e.g. 01 ) so that they come in order when the keys get sorted
        """
        with open(missing_data_json(),'r') as f:
            missing_data = json.load(f)
        missing_data = {key:val for key,val in missing_data.items() if siteID in key}
        if len(missing_data.keys())>1: # if there are different null values for different time periods
            time_period_list = sorted(missing_data.keys())
            for time_period in time_period_list:
                start,end = [(datetime.strptime(elem,'%Y%m%d')-datetime(1970,1,1)).total_seconds() for elem in time_period.split('_')[2:]]

                # must index netCDF datasets for the < comparison to work: comparison between netCDF4.Variable and float not allowed
                # indexing with a tuple() rather than : slightly more robust: a colon won't work for a scalar variable
                # use a set to allow us to compute the intersection between the time indices and the fill indices
                replace_time_ids = set(np.where((start < nc_data['time'][tuple()]) & (nc_data['time'][tuple()] < end))[0])
                for var in missing_data[time_period]:
                    # isclose is more robust for floating point comparisons than ==
                    replace_val_ids = set(np.where(np.isclose(nc_data[var][tuple()], missing_data[time_period][var]))[0])
                    replace_ids = replace_time_ids.intersection(replace_val_ids) # indices for data equal to the fill value in the given time period
                    logging.info(
                        'Convert file value for {} from {} to {} between {} and {}'.format(
                            var,
                            missing_data[time_period][var],
                            netCDF4.default_fillvals['f4'],
                            str(netCDF4.num2date(start,units=nc_data['time'].units,calendar=nc_data['time'].calendar)),
                            str(netCDF4.num2date(end,units=nc_data['time'].units,calendar=nc_data['time'].calendar)),
                            )
                        )
                    for id in replace_ids:
                        nc_data[var][id] = netCDF4.default_fillvals['f4'] 
        elif len(missing_data.keys())==1:
            missing_data = missing_data[siteID]
            for var in missing_data:
                # isclose is more robust for floating point comparisons than ==
                replace_ids = list(np.where(np.isclose(nc_data[var][:], missing_data[var]))[0])
                logging.info('Convert fill value for {} to {}'.format(var,netCDF4.default_fillvals['f4']))
                for id in replace_ids:
                    nc_data[var][id] = netCDF4.default_fillvals['f4']

        pgrm_versions = get_program_versions(aia_file)
        nc_data.setncatts(pgrm_versions)

        # get a list of all the variables written to the private netcdf file, will be used below to check for missing variables before writing an eof.csv file
        private_var_list = [v for v in nc_data.variables]

        update_attrs_for_public_files(nc_data, is_public=False, mode=args.mode)
    # end of the "with open(private_nc_file)" statement
    
    # both function return the path where the flags were written, so 
    # overriding the `private_nc_file` variable ensures any future steps
    # take the correct file.
    private_nc_file = set_manual_flags(private_nc_file)
    private_nc_file = set_release_flags(private_nc_file,args.rflag_file)
    
    logging.info('Finished writing {} {:.2f} MB'.format(private_nc_file,os.path.getsize(private_nc_file)/1e6))

    if args.public:
        write_public_nc(private_nc_file,code_dir,nc_format,include_experimental=not args.std_only,remove_if_no_experimental=args.remove_no_expt,flag0_only=not args.publish_all_flags,expand_priors=args.expand_priors,expand_aks=args.expand_aks,full_ak_resolution=args.full_ak_resolution,mode=args.mode)

    if args.eof:
        ordered_var_list = ['flag','flagged_var_name','spectrum'] # list of variables for writing the eof file
        ordered_var_list += aux_var_list
        ordered_var_list += list(lse_dict.keys())
        ordered_var_list += full_main_var_list
        ordered_var_list += correction_var_list
        ordered_var_list += col_var_list
        if not skip_vsw:
            ordered_var_list += full_vsw_var_list
        ordered_var_list += ['gfit_version','gsetup_version']
        ordered_var_list += [var+'_checksum' for var in checksum_var_list]

        # check that we have all the variables we want
        # the mod_var_dict keys read data from the extract_pth file, these duplicate variables from the aux_var_list
        # the prior_ and ak_ variable are along a different dimensions and can be 2D so we can't include them in the eof.csv file
        # we also don't include "time" as it is split into year/day/hour
        missing_var_list = [var for var in private_var_list if var!='time' and (var not in ordered_var_list) and (('prior_' not in var) and ('ak_' not in var) and ('cell_' not in var)) and (var not in mod_var_dict.keys())]
        if missing_var_list:
            logging.warning('{}/{} variables will not be in the eof.csv file'.format(len(missing_var_list),len(private_var_list)))
            for var in missing_var_list:
                logging.warning('Missing {}'.format(var))

        write_eof(private_nc_file,eof_file,qc_file,ordered_var_list,show_progress)
    for handler in logger.handlers:
        handler.setFormatter(logging.Formatter('[%(levelname)s]: %(message)s at %(asctime)s',datefmt='%m/%d/%Y %I:%M:%S %p'))
    logging.info('Finished write_netcdf log session')


def compare_nc_files(base_file, other_file, log_file=None, log_level='INFO', ignore=('config_checksum','mav_checksum')):
    def get_file_variables(filename):
        with netCDF4.Dataset(filename, 'r') as ds:
            variables = set(ds.variables.keys())
            return variables

    setup_logging(log_level=log_level, log_file=log_file, message='')

    base_variables = get_file_variables(base_file)
    other_variables = get_file_variables(other_file)
    common_variables = base_variables.intersection(other_variables)

    missing_base_vars = base_variables.difference(common_variables)
    if len(missing_base_vars) > 0:
        logging.warning('{} variables ({}) present in the first file ({}) are missing from the second file ({})'
                        .format(len(missing_base_vars), ', '.join(missing_base_vars), base_file, other_file))

    missing_other_vars = other_variables.difference(common_variables)
    if len(missing_other_vars) > 0:
        logging.warning('{} variables present in the second file ({}) were not present in the first file ({})'
                        .format(len(missing_other_vars), ', '.join(missing_other_vars), other_file, base_file))

    common_variables = sorted(list(common_variables))
    checks = check_eof(base_file, other_file, common_variables, common_variables, other_is_nc=True, show_detail=True, ignore=ignore)

    return 1 if False in checks else 0


def compare_nc_files_command_line():
    def csl(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser(description='Check that the variables in two .nc files are equal')
    parser.add_argument('base_file', help='The first .nc file, which will serve as the baseline')
    parser.add_argument('other_file', help='The second .nc file, which will be checked against the base_file')
    parser.add_argument('--log-level', default='INFO', type=lambda x: x.upper(),
                        help="Log level for the screen (it is always DEBUG for the log file)",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--log-file', default=None,
                        help="File to write the logging messages to in addition to the screen. By default, no such "
                             "file is written.")
    parser.add_argument('-i', '--ignore', nargs='?', const='', default='mav_checksum,config_checksum', type=csl,
                        help='A comma separated list of variables not to difference. Default is "%(default)s". '
                             'To ignore no variables, give this flag with no following value')
    parser.epilog = 'An exit code of 1 indicates that there were differences between the files.'
    cl_args = vars(parser.parse_args())
    return compare_nc_files(**cl_args)


if __name__=='__main__': # execute only when the code is run by itself, and not when it is imported
    main()
