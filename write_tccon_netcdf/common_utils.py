import netCDF4
import numpy as np
from typing import Union

def create_observation_operator_variable(
        ds: netCDF4.Dataset, eff_path: Union[str, np.ndarray] = 'effective_path_length', o2_dmf: Union[str, float, np.ndarray] = 'o2_mean_mole_fraction',
        nair: Union[str, np.ndarray] = 'prior_density', ret_o2_col: Union[str, np.ndarray] = 'vsw_o2_7885', varname: str = 'integration_operator'):
    """Compute the observation operator and add it as a new variable

    Parameters
    ----------
    ds
        The netCDF dataset to add the variable to

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