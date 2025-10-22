from argparse import ArgumentParser
import json
import logging
from netCDF4 import Dataset
import numpy as np
import os
import pandas as pd
import re

from . import common_utils as cu
from . import get_paths as gp
from . import daily_error
from .make_ggg2020_1 import driver as ggg2020c_to_ggg2020p1a
from .constants import AUX_DATA_URL, FILE_FMT_V2020pC, FILE_FMT_V2020p1pA, GGG2020_REFERENCE, LOG_LEVEL_CHOICES, SPECIAL_DESCRIPTION_DICT, NETWORK_MODES, TCCON_DATA_POLICY_URL

# define these once so we don't have to parse the string every time
v2020A = cu.FileFmtVer('2020.A')
v2020B = cu.FileFmtVer('2020.B')
v2020C = cu.FileFmtVer(FILE_FMT_V2020pC)
v2020p1A = cu.FileFmtVer(FILE_FMT_V2020p1pA)
vCurrent = v2020p1A

def main():
    p = ArgumentParser(description='Update a TCCON netCDF file format version')
    p.add_argument('netcdf_file', help='Path to the netCDF file to update')
    p.add_argument('--mode', choices=NETWORK_MODES, default='TCCON',
                   help='Which type of instrument/network this file is for; used to set certain data and metadata value')
    p.add_argument('--tgt-ver', type=cu.FileFmtVer, default=v2020p1A,
                   help='Version to upgrade the file to, consisting of MAJOR.MINOR.REV e.g. '
                        '2020.1.A. The MINOR value may be omitted if it is 0, so 2020.B and 2020.0.B are '
                        'equivalent. The default is "%(default)s".')
    p.add_argument('--log-level', default='INFO', type=lambda x: x.upper(), choices=LOG_LEVEL_CHOICES,
                   help="Log level for the screen (it is always DEBUG for the log file)")
    p.add_argument('--pdb', action='store_true', help='Start in the Python debugger')
    clargs = vars(p.parse_args())
    if clargs.pop('pdb', False):
        import pdb
        pdb.set_trace()
    log_level = clargs.pop('log_level')
    cu.setup_logging(log_level=log_level, log_file=None)
    driver(**clargs)


def driver(netcdf_file: os.PathLike, mode: str = 'TCCON', tgt_ver: cu.FileFmtVer = v2020p1A):
    # Open the file once to get necessary information about it
    with Dataset(netcdf_file) as ds:
        is_public = 'flag' not in ds.variables.keys()
        file_version = cu.get_file_format_version(ds)

    if file_version < v2020C and tgt_ver >= v2020C:
        logging.info(f'Converting from v{file_version} to v2020.0.C')
        with Dataset(netcdf_file, 'a') as ds:
            ggg2020a_to_ggg2020c(ds, is_public, mode)

    if file_version < v2020p1A and tgt_ver >= v2020p1A:
        logging.info('Converting from v2020.0.C to v2020.1.A')
        ggg2020c_to_ggg2020p1a(netcdf_file, mode=mode, in_place=True)


def is_file_current_version(netcdf_file: os.PathLike):
    with Dataset(netcdf_file) as ds:
        try:
            file_fmt_str = ds.file_format_version
        except AttributeError:
            return False
        else:
            file_fmt_ver = cu.FileFmtVer(file_fmt_str)
            return file_fmt_ver == vCurrent


def ggg2020a_to_ggg2020c(ds, is_public, mode):
    # fix the prior index if necessary first so that any later calculations that
    # depend on it are done correctly
    if not is_public:
        check_and_fix_prior_index(ds)

    _update_links(ds)
    # inserting missing AKs should come early so they get the missing attributes added
    # and incorrect units fixed just like the other AK variables.
    _insert_missing_aks(ds, 'xhdo', is_public)
    _fix_unspecified_units(ds)
    _fix_inconsistent_units(ds)
    _add_prior_long_units(ds, is_public)
    _fix_public_cf_attributes(ds, is_public)
    _fix_incorrect_attributes(ds)
    _add_flag_usage(ds)
    if not is_public:
        # The public file should not include the detailed GEOS file information;
        # so don't try to re-add these variables when working with public files.
        add_geos_versions_by_date(ds)

        # The public file also should not include the daily error data
        _insert_daily_error_variables(ds)
    # Starting with GGG2020.1, x2019 variables will be handled by the separate program to
    # convert GGG2020 files to GGG2020.1 - this just makes things easier than trying to
    # undo the GGG2020 X2019 variable O2 mole fraction.
    write_file_fmt_attrs(ds, FILE_FMT_V2020pC)
    cu.add_effective_path(ds, is_public)


def check_and_fix_prior_index(ds):
    if not cu.check_prior_index(ds):
        logging.warning('The existing prior indices are incorrect based on the time/prior_time differences; they will be recalculated')
        cu.correct_prior_index(ds)
    else:
        logging.info('The existing prior indices appear correct, not recalculating')



def add_geos_versions_by_date(ds):
    is_classic = ds.data_model != 'NETCDF4'
    prior_times = pd.Timestamp(1970, 1, 1) + pd.to_timedelta(ds['prior_time'][:], unit='s')
    versions, filenames, checksums = cu.infer_geos_version_from_prior_time(prior_times)
    # Since we don't have the original .mod files, we have to just assume that it would have all of the expected keys
    geos_version_keys = cu.GEOS_VERSION_EXPECTED_KEYS
    # First, find out which variables we need
    for (vkey, vfxn) in cu.geos_version_keys_and_fxns():
        if vkey == 'geos_versions':
            gv_max_len = cu.GEOS_VER_MAX_LENGTH 
            gv_values = versions
        elif vkey == 'geos_filenames':
            gv_max_len = 1
            gv_values = filenames
        elif vkey == 'geos_checksums':
            gv_max_len = 1
            gv_values = checksums
        else:
            raise NotImplementedError(f'GEOS information key {vkey}')

        for gkey in geos_version_keys:
            gv_varname = vfxn(gkey)
            if gv_varname not in ds.variables.keys():
                var = cu.add_geos_version_variables(ds, gv_max_len, gv_varname, is_classic)
                # Must convert to a byte array otherwise the UTF-encoded array tries to use 4
                # bytes per character. Guessing this is an issue between ASCII and UTF encoding.
                var[:] = gv_values.astype('S')

    # This will overwrite the attributes if any of this variables existed. That's fine;
    # we're using the same function to write the attributes here as we do in the main writer
    # so they will be the same either way.
    cu.add_geos_version_var_attrs(ds, geos_version_keys)


def _fix_public_cf_attributes(ds, is_public):
    if not is_public:
        return

    with open(gp.public_cf_attrs_json()) as f:
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

    # Last, there's a few things it's easiest to scan all the variables for
    n_prior_notes_updated = 0
    for varname, var in ds.variables.items():
        # The column densities and their errors should be in molecules per cm2, not per m2
        # This was fixed in file format version GGG2020.C, but we need this for updating the
        # already-uploaded files.
        if varname.startswith('column') and var.units == 'molecules.m-2':
            var.units = 'molecules.cm-2'

        # It's annoyingly easy to have the wrong gas name in the description. Find all the Xgas variables whose
        # description matches that pattern of "o2_dmf * column_X / column_o2 ... " and ensure that the correct 
        # gas is in the description.
        if varname.startswith('x') and hasattr(var, 'description') and re.search(r'column_[a-z0-9]+/column_o2', var.description):
            # The gas name should be everything up to the first underscore (e.g., for _insb gases)
            # without the leading "x"
            old_description = var.description
            gas_name = varname.split('_')[0][1:]
            correct_description = re.sub(r'column_[a-z0-9]+/column_o2', f'column_{gas_name}/column_o2', old_description)
            if old_description != correct_description:
                var.description = correct_description
                logging.info(f'Corrected description of "{varname}" variable: "{old_description}" -> "{correct_description}"')

        # Not so much wrong as there's an easier way: the prior profile's describe how to dry
        # them, but use a more complicated equation than needed. 
        if varname.startswith('prior_') and hasattr(var, 'note'):
            old_note = var.note
            if old_note == 'Prior VMRs are given in wet mole fractions. To convert to dry mole fractions, you must calculate H2O_dry = H2O_wet/(1 - H2O_wet) and then gas_dry = gas_wet * (1 + H2O_dry), where H2O_wet is the prior_1h2o variable.':
                # Replace the two equations with one
                var.note = 'Prior VMRs are given in wet mole fractions. To convert to dry mole fractions, you must calculate gas_dry = gas_wet / (1 - H2O_wet), where H2O_wet is the prior_1h2o variable.'
                logging.debug(f'Updated note on {varname} to use simpler drying equation')
                n_prior_notes_updated += 1

    if n_prior_notes_updated > 0:
        logging.info(f'Updated notes on {n_prior_notes_updated} prior variable(s) to use the simpler drying equation.')


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
            logging.debug('Setting {} units to "1"'.format(varname))
            if variable.units == '':
                variable.units = '1'

    # Special cases, units that weren't included in the original release but shouldn't just be "1"
    other_units = {
        'sia': 'AU'
    }
    for varname, varunits in other_units.items():
        if ds[varname].units == '':
            logging.debug('Setting {} units to "{}"'.format(varname, varunits))
            ds[varname].units = varunits


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


def _add_flag_usage(ds):
    if 'flag' in ds.variables.keys():
        ds['flag'].comment = "flag == 0 data is good quality, flag > 0 data does not meet TCCON quality standards. If you intend to use flag > 0 data, we STRONGLY encourage you to reach out to the person listed in the contact global attribute. Use of flag > 0 data without consulting the contact person is at your own risk."



def _update_links(ds):
    # Ensure that all files have the correct URL for auxiliary data - it was
    # out of date on older files, and the EM27s were missing it entirely.
    ds.auxiliary_data_description = AUX_DATA_URL
    # Add in the retrieval reference if it is missing
    ds.retrieval_reference = GGG2020_REFERENCE

    # Only update the data use policy if the attribute is present - EM27s
    # don't necessarily follow the TCCON data use policy.
    if hasattr(ds, 'data_use_policy'):
        ds.data_use_policy = TCCON_DATA_POLICY_URL


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

    with Dataset(gp.ak_tables_nc_file()) as ak_nc:
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
                att_dict['description'] = att_dict['description']+SPECIAL_DESCRIPTION_DICT['lco2']
            elif xgas.lower() == 'xwco2':
                att_dict['description'] = att_dict['description']+SPECIAL_DESCRIPTION_DICT['wco2']
            nc_data[ak_varname].setncatts(att_dict)
            nc_data[ak_varname][:] = ak_nc[table_ak_var][:].data.astype(np.float32)


def _insert_daily_error_variables(nc_data):
    if 'daily_error_date' in nc_data.dimensions.keys():
        logging.info('Daily error dimension present, assuming the daily error variables have already been written')
        return

    logging.info('Adding daily error variables with fill values')
    dummy_esf_df = daily_error.make_dummy_esf_df(nc_data)
    daily_error.write_daily_esf_data(nc_data, dummy_esf_df)


def write_file_fmt_attrs(ds, file_fmt_version):
    """Insert/update the attributes related to the file format version in a dictionary or on a netCDF dataset
    """
    info_str = 'For a description of the changes between file format versions, see https://tccon-wiki.caltech.edu/Main/GGG2020DataChanges'
    if isinstance(ds, dict):
        ds['file_format_version'] = file_fmt_version
        ds['file_format_information'] = info_str
    else:
        ds.file_format_version = file_fmt_version
        ds.file_format_information = info_str
