from datetime import datetime
import json
import os

from . import __version__

WNC_VERSION = f'write_netcdf.py (Version {__version__}; 2024-12-19; SR,JL)\n'
STD_O2_MOLE_FRAC = 0.2095
FILE_FMT_V2020pC = '2020.C'
FILE_FMT_V2020p1pA = '2020.1.A'

# Allowed choices for the --mode flag which determines some behavior (e.g. choice of AICFs)
NETWORK_MODES = ('TCCON', 'em27')
LOG_LEVEL_CHOICES = ('PROFILE', 'DEBUG','INFO','WARNING','ERROR','CRITICAL')

# Let's try to be CF compliant: http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.pdf
STANDARD_NAME_DICT = {
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

CHECKSUM_VAR_LIST = ['config','apriori','runlog','levels','mav','ray','isotopologs','windows','telluric_linelists','solar']

STANDARD_NAME_DICT.update({var+'_checksum':var+'_checksum' for var in CHECKSUM_VAR_LIST})

LONG_NAME_DICT = {key:val.replace('_',' ') for key,val in STANDARD_NAME_DICT.items()} # standard names without underscores

"""
dimensionless and unspecified units will have empty strings
we could use "1" for dimensionless units instead
both empty string and 1 are recognized as dimensionless units by udunits
but using 1 would differentiate actual dimensionless variables and variables with unspecified units
"""
UNITS_DICT = {
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

MOLE_FRACTION_CONVERSIONS = {
    '': 1.0,
    '1': 1.0,
    'parts': 1.0,
    'ppm': 1e6,
    'ppb': 1e9,
    'ppt': 1e12,
}

SPECIAL_DESCRIPTION_DICT = {
    'lco2':' lco2 is the strong CO2 band centered at 4852.87 cm-1 and does not contribute to the xco2 calculation.',
    'wco2':' wco2 is the weak CO2 band centered at 6073.5 and does not contribute to the xco2 calculation.',
    'th2o':' th2o is used for temperature dependent H2O windows and does not contribute to the xh2o calculation.',
    'fco2':' fco2 is used for a spectral window chosen to estimate the channel fringe amplitude and period, it does not contribute to the xco2 calculation',
    'luft':' luft is used for "dry air"',
    'qco2':' qco2 is the strong CO2 band centered at  4974.05 cm-1 and does not contribute to the xco2 calculation.',
    'zco2':' zco2 is used to test zero level offset (zo) fits in the strong CO2 window, zco2_4852 is without zo, and zco2_4852a is with zo. it does not contribute to the xco2 calculation'
}

with open(os.path.join(os.path.dirname(__file__), 'release_flag_definitions.json')) as f:
    _tmp = json.load(f)
    MANUAL_FLAGS_DICT = {v: k for k, v in _tmp['definitions'].items()}
    MANUAL_FLAG_OTHER = _tmp['other_flag']


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