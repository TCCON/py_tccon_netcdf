from argparse import ArgumentParser
import logging
import netCDF4 as ncdf
import numpy as np
import pandas as pd
import os

from ginput.priors.tccon_priors import O2MeanMoleFractionRecord

from . import nc_ops, common_utils
from . import bias_corrections as bc
from .common_utils import set_private_name_attrs
from .constants import FILE_FMT_V2020p1pA

# First value is the AICF, second is its error
GGG2020p1_TCCON_AICFS = {
    'xco2_x2007': (1.00950, 0.00044),  # WMO X2007 scale
    'xco2_x2019': (1.00901, 0.00045),
    'xwco2_x2007': (1.00008, 0.00045), # WMO X2007 scale
    'xwco2_x2019': (0.99959, 0.00047),
    'xlco2_x2007': (1.00106, 0.00067), # WMO X2007 scale
    'xlco2_x2019': (1.00059, 0.00068),
    'xch4': (1.00214, 0.00133),
    'xh2o': (0.98676, 0.01701),
    'xn2o': (0.9821,  0.0098), # apply the same AICF to XN2O so that it gets the new O2 mole fractions
    'xco': (1.0000, 0.0526), # ditto for XCO
}

GGG2020p1_EM27_AICFS = {
    'xco2_x2007': (None, None),
    'xco2_x2019': (1.0062, 0.0003),
    'xch4': (0.9984, 0.0003),
    'xco': (1.0000, 0.032)
}

VAR_RENAMES = {
    'xco2': 'xco2_x2007',
    'xco2_error': 'xco2_error_x2007',
    'xco2_aicf': 'xco2_aicf_x2007',
    'xco2_aicf_error': 'xco2_aicf_error_x2007',
    'aicf_xco2_scale': 'aicf_xco2_x2007_scale',

    'xwco2': 'xwco2_x2007',
    'xwco2_error': 'xwco2_error_x2007',
    'xwco2_aicf': 'xwco2_aicf_x2007',
    'xwco2_aicf_error': 'xwco2_aicf_error_x2007',
    'aicf_xwco2_scale': 'aicf_xwco2_x2007_scale',

    'xlco2': 'xlco2_x2007',
    'xlco2_error': 'xlco2_error_x2007',
    'xlco2_aicf': 'xlco2_aicf_x2007',
    'xlco2_aicf_error': 'xlco2_aicf_error_x2007',
    'aicf_xlco2_scale': 'aicf_xlco2_x2007_scale',
}

VARS_TO_REMOVE = {
    'xco2_x2019',
    'xco2_error_x2019',
    'xwco2_x2019',
    'xwco2_error_x2019',
    'xlco2_x2019',
    'xlco2_error_x2019',
    'o2_mean_mole_fraction_x2019',
}

OLD_O2_DMF = 0.2095

def main():
    p = ArgumentParser(description='Create a GGG2020.1 netCDF file from a GGG2020 file with the 2020.C format file.')
    p.add_argument('input_file', help='The GGG2020.C format netCDF file to start from.')
    p.add_argument('mode', choices=('TCCON', 'em27'), help='Which type of data is being processed (determines AICFs)')
    p.add_argument('--pdb', action='store_true', help='Launch Python debugger')

    out_grp = p.add_mutually_exclusive_group(required=True)
    out_grp.add_argument('--in-place', action='store_true', help='Modify INPUT_FILE in place. Note that one of this or --output-file are required')
    out_grp.add_argument('-o', '--output-file', help='Output the GGG2020.1 file to this path instead of modifying it in place.')

    clargs = vars(p.parse_args())
    if clargs.pop('pdb'):
        import pdb
        pdb.set_trace()
    driver(**clargs)


def driver(input_file, mode, output_file=None, in_place=False):
    if not os.path.exists(input_file):
        raise IOError(f'Input file {input_file} does not exist')

    if in_place:
        output_file = f'{input_file}.tmp'

    nc_ops.copy_netcdf(
        input_nc_file=input_file,
        output_file=output_file,
        clobber=True,
        var_renames=VAR_RENAMES,
        exclude_vars=VARS_TO_REMOVE
    )

    with ncdf.Dataset(output_file, 'a') as ds:
        update_o2_and_aicfs(ds, mode)
        bias_correct_xn2o(ds)

    if in_place:
        os.rename(output_file, input_file)


def update_o2_and_aicfs(ds, mode):
    new_o2_dmfs = _get_new_o2_dmfs(ds)
    # For every gas and its error, it was calculated as X = V * fO2 / AICF, so we
    # need to multiple by new_fO2 / old_fO2 and old_AICF / new_AICF
    if mode.lower() == 'tccon':
        aicfs = GGG2020p1_TCCON_AICFS
    elif mode.lower() == 'em27':
        aicfs = GGG2020p1_EM27_AICFS
    else:
        raise NotImplementedError(f'mode = {mode}')
    
    # We have to check here whether the x2019 variables were removed correctly
    # because inside the next for loop they'll be added back in as each CO2 variable
    # comes up.
    for xgas_varname in aicfs.keys():
        if xgas_varname.endswith('x2019') and xgas_varname in ds.variables.keys():
            raise RuntimeError('x2019 variables must not be present in the file')
    
    for xgas_varname, (new_aicf, new_aicf_error) in aicfs.items():
        var_is_missing = xgas_varname not in ds.variables.keys()
        if xgas_varname.endswith('x2019'):
            # These will be handled by their respective x2007 variables.
            continue
        elif var_is_missing:
            logging.warning(f'{xgas_varname} missing - if this is not an EM27 file, something may have gone wrong during post processing')
            continue

        if xgas_varname.startswith(('xco2_', 'xwco2_', 'xlco2_')):
            gas, scale = xgas_varname.split('_', 1)
            aicf_varname = f'{gas}_aicf_{scale}'
            aicf_error_varname = f'{gas}_aicf_error_{scale}'
            error_varname = f'{gas}_error_{scale}'
            create_x2019 = True
        else:
            aicf_varname = f'{xgas_varname}_aicf'
            aicf_error_varname = f'{xgas_varname}_aicf_error'
            error_varname = f'{xgas_varname}_error'
            create_x2019 = False

        # First the Xgas values, ensure that the standard and long name attributes match the variable name
        old_aicfs = ds[aicf_varname][:]
        unscaled_values = ds[xgas_varname][:] * old_aicfs / OLD_O2_DMF * new_o2_dmfs
        ds[xgas_varname][:] = unscaled_values / new_aicf
        set_private_name_attrs(ds[xgas_varname])

        # Then the error values, ditto on the attributes
        unscaled_error_values = ds[error_varname][:] * old_aicfs / OLD_O2_DMF * new_o2_dmfs
        ds[error_varname][:] = unscaled_error_values / new_aicf
        set_private_name_attrs(ds[error_varname])

        # We also have to update the AICF values themselves and the error
        ds[aicf_varname][:] = new_aicf
        set_private_name_attrs(ds[aicf_varname])

        ds[aicf_error_varname][:] = new_aicf_error
        set_private_name_attrs(ds[aicf_varname])

        if create_x2019:
            xgas = xgas_varname.split('_')[0]
            add_x2019_xco2(ds, xgas, unscaled_values, unscaled_error_values, aicfs)

    # Confirm that all of the expected x2019 gases were added
    for xgas_varname in aicfs.keys():
        missed_xgases = []
        if xgas_varname.endswith('x2019') and xgas_varname not in ds.variables.keys():
            missed_xgases.append(xgas_varname)
        if len(missed_xgases) > 0:
            raise RuntimeError(f'Some x2019 gases were not added {", ".join(missed_xgases)}')

    # Create a new variable that stores the O2 DMF
    o2_var = ds.createVariable('o2_mean_mole_fraction', 'f4', dimensions=('time',))
    o2_var[:] = new_o2_dmfs
    o2_var.description = 'Global mean O2 dry mole fraction used to calculate the Xgas column averages; Xgas = column_gas / column_2 * o2_dmf'
    o2_var.units = '1'
    o2_var.standard_name = 'dry_atmospheric_mole_fraction_of_oxygen'

    # We might as well create the observation operator and prior Xgas variables here,
    # that way they're available in both the private and public files.
    common_utils.create_observation_operator_variable(ds)
    common_utils.create_tccon_prior_xgas_variables(ds)

    # Update the file format version and add an algorithm version 
    ds.file_format_version = FILE_FMT_V2020p1pA
    ds.algorithm_version = 'GGG2020.1'


def _get_new_o2_dmfs(ds):
    time_index = pd.Timestamp(1970, 1, 1) + pd.to_timedelta(ds['time'][:], unit='s')
    o2_dmf_record = O2MeanMoleFractionRecord(auto_update_fo2_file=True)
    o2_dmfs = np.full(time_index.size, np.nan)
    for i, time in enumerate(time_index):
        o2_dmfs[i] = o2_dmf_record.get_o2_mole_fraction(time)
    return o2_dmfs
    
def add_x2019_xco2(ds, xgas, unscaled_xgas_values, unscaled_xgas_error_values, aicfs):
    x2007_xgas_varname = f'{xgas}_x2007'
    x2007_error_varname = f'{xgas}_error_x2007'
    x2007_aicf_varname = f'{xgas}_aicf_x2007'
    x2007_aicf_error_varname = f'{xgas}_aicf_error_x2007'
    x2007_aicf_scale_varname = f'aicf_{xgas}_x2007_scale'
    # Calculate the X2019 values from the already-read-in X2007 values,
    # copy the existing variable's attributes and update the standard and long name
    # then replace the values
    new_aicf, new_aicf_error = aicfs[f'{xgas}_x2019']
    _add_x2019_variable(ds, x2007_xgas_varname, unscaled_xgas_values / new_aicf)
    _add_x2019_variable(ds, x2007_error_varname, unscaled_xgas_error_values / new_aicf)

    # Now add in the ancillary variables: the AICF and its error, plus the new scale variables
    _add_x2019_variable(ds, x2007_aicf_varname, new_aicf)
    _add_x2019_variable(ds, x2007_aicf_error_varname, new_aicf_error)
    # For the scale, since it can be tricky to get the right array type for character, just read in the existing
    # array and replace its values
    new_scale_values = ds[x2007_aicf_scale_varname][:]
    new_scale_values[:] = 'WMO CO2 X2019'
    _add_x2019_variable(ds, x2007_aicf_scale_varname, new_scale_values, is_text=True)

def _add_x2019_variable(ds, x2007_varname, new_values, is_text=False):
    dtype = 'S1' if is_text else ds[x2007_varname].dtype
    x2019_varname = x2007_varname.replace('x2007', 'x2019')
    x2019_var = ds.createVariable(x2019_varname, dtype, ds[x2007_varname].dimensions)
    if is_text:
        x2019_var._Encoding = 'ascii'
        # not entirely sure why the conversion to byte string type ('S') is necessary, maybe
        # because it's ASCII encoded? Without this, it makes each character try to use 4 bytes
        x2019_var[:] = new_values.astype('S')
    else:
        x2019_var[:] = new_values
    x2019_var.setncatts(ds[x2007_varname].__dict__)
    set_private_name_attrs(x2019_var)


def bias_correct_xn2o(ds):
    logging.info('Applying PT700 bias correction to XN2O')
    # This m and b value were computed from XN2O that uses the new O2 mole fraction.
    # Therefore, the XN2O in ``ds`` MUST have bee converted to the new O2 mole fractions
    # before calling this function.
    xn2o_corr, pt700 = bc.correct_xn2o_from_pt700(ds, m=0.000626, b=0.787)

    var_xn2o_orig = ds.createVariable('xn2o_original', ds['xn2o'].dtype, dimensions=ds['xn2o'].dimensions)
    var_xn2o_orig.setncatts(ds['xn2o'].__dict__)
    var_xn2o_orig.note = 'This variable contains the XN2O values BEFORE the temperature bias correction is applied but AFTER the new O2 mole fractions were applied'
    var_xn2o_orig[:] = ds['xn2o'][:]

    var_xn2o_new = ds['xn2o']
    var_xn2o_new.note = 'This variable contains the XN2O values with a bias correction applied based on the prior potential temperature at 700 hPa'
    var_xn2o_new.ancillary_variables = 'potential_temperature_700hPa'
    var_xn2o_new[:] = xn2o_corr

    var_pt700 = ds.createVariable('potential_temperature_700hPa', ds['xn2o'].dtype, dimensions=ds['xn2o'].dimensions)
    var_pt700.standard_name = 'potential_temperature'
    var_pt700.long_name = 'potential temperature at 700 hPa'
    var_pt700.units = 'degrees_Kelvin'
    var_pt700.note = 'This is the a priori potential temperature at 700 hPa used to bias correct XN2O'
    var_pt700[:] = pt700


if __name__ == '__main__':
    main()
