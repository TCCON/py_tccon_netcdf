from argparse import ArgumentParser
import logging
import netCDF4 as ncdf
import os
import shutil

from . import bias_corrections as bc


def main():
    p = ArgumentParser(description='Create a GGG2020.1 netCDF file from a GGG2020 file with the 2020.B format file.')
    p.add_argument('input_file', help='The GGG2020.B format netCDF file to start from.')
    p.add_argument('--pdb', action='store_true', help='Launch Python debugger')

    out_grp = p.add_mutually_exclusive_group(required=True)
    out_grp.add_argument('--in-place', action='store_true', help='Modify INPUT_FILE in place. Note that one of this or --output-file are required')
    out_grp.add_argument('-o', '--output-file', help='Output the GGG2020.1 file to this path instead of modifying it in place.')

    clargs = vars(p.parse_args())
    if clargs.pop('pdb'):
        import pdb
        pdb.set_trace()
    driver(**clargs)


def driver(input_file, output_file=None, in_place=False):
    if not os.path.exists(input_file):
        raise IOError(f'Input file {input_file} does not exist')

    if in_place:
        nc_path = input_file
    elif output_file is not None:
        if os.path.exists(output_file) and os.path.samefile(input_file, output_file):
            raise IOError('input_file cannot be the same as output_file. Use in_place=True to modify the input file directly.')
        shutil.copy2(input_file, output_file)
        nc_path = output_file
    else:
        raise TypeError('Either in_place must be True or output_file must not be None')


    with ncdf.Dataset(nc_path, 'a') as ds:
        _bias_correct_xco2(ds)
        _bias_correct_xn2o(ds)

    
def _bias_correct_xco2(ds, variables=('xco2', 'xwco2', 'xlco2')):
    pretty_names = {'xco2': 'XCO2', 'xwco2': 'XwCO2', 'xlco2': 'XlCO2'}
    corrected_df, xluft_attrs = bc.correct_xco2_from_xluft(ds, variables)

    # We want to store the original XCO2 for comparison, along with the rolling Xluft used for the correction
    # which requires making a couple new variables
    for varname in variables:
        logging.info(f'Applying Xluft bias correction to {varname}')
        xco2_orig = ds.createVariable(f'{varname}_original', ds[varname].dtype, dimensions=ds[varname].dimensions)
        xco2_orig.setncatts(ds[varname].__dict__)
        xco2_orig.note = f'This variable contains the {pretty_names[varname]} values from the .aia file BEFORE the Xluft bias correction is applied.'
        xco2_orig[:] = ds[varname][:]

        xco2_new = ds[varname]
        xco2_new.note = f'This variable contains the {pretty_names[varname]} values with a bias correction applied based on a moving median of Xluft.'
        xco2_new.ancillary_variables = 'xluft_for_bias_correction'
        xco2_new[:] = corrected_df[f'{varname}_corr'].to_numpy()

    xluft_rolling = ds.createVariable('xluft_for_bias_correction', ds['xluft'].dtype, dimensions=ds['xluft'].dimensions)
    xluft_rolling.setncatts(ds['xluft'].__dict__)
    xluft_rolling.setncatts(xluft_attrs)
    xluft_rolling.note = 'This is the moving median Xluft used for the XCO2 bias corrections'
    xluft_rolling[:] = corrected_df['xluft_rolled'].to_numpy()


def _bias_correct_xn2o(ds):
    logging.info('Applying PT700 bias correction to XN2O')
    xn2o_corr, pt700 = bc.correct_xn2o_from_pt700(ds)

    var_xn2o_orig = ds.createVariable('xn2o_original', ds['xn2o'].dtype, dimensions=ds['xn2o'].dimensions)
    var_xn2o_orig.setncatts(ds['xn2o'].__dict__)
    var_xn2o_orig.note = 'This variable contains the XN2O values from the .aia file BEFORE the temperature bias correction is applied'
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
