import logging
import numpy as np

from .constants import UNITS_DICT
from . import common_utils as cu

def add_unexpanded_priors(nc_data, prior_data, speclist, runlog_all_speclist, nlev: int, ncell: int, classic: bool):
    _add_coords(nc_data=nc_data, prior_data=prior_data, nlev=nlev, ncell=ncell, priors_expanded=False)
    prior_var_list, geos_version_keys, _ = _create_priors_vars(nc_data, prior_data, nlev, ncell, classic, expand_priors=False)
    logging.info('Writing prior data ...')
    _fill_unexpanded_prior_vars(
        nc_data=nc_data, speclist=speclist, runlog_all_speclist=runlog_all_speclist, prior_data=prior_data, prior_var_list=prior_var_list,
        geos_version_keys=geos_version_keys, nlev=nlev, ncell=ncell
    )


def add_expanded_priors(nc_data, prior_data, speclist, runlog_all_speclist, nlev: int, ncell: int, classic: bool):
    _add_coords(nc_data=nc_data, prior_data=prior_data, nlev=nlev, ncell=ncell, priors_expanded=True)
    prior_var_list, geos_version_keys, unexpanded_size_arrays = _create_priors_vars(
        nc_data=nc_data, prior_data=prior_data, nlev=nlev, ncell=ncell, classic=classic, expand_priors=True
    )
    logging.info('Writing prior data ...')
    # To avoid _too_ much code duplication, we pass in our dictionary of arrays to pretend to be the netcdf
    # data. This function then writes to it as if it were writing to the netCDF file, and we can then use the
    # prior index in that same dictionary to reindex the data into expanded priors.
    unexpanded_size_arrays['prior_index'] = np.full(len(speclist), -999)
    _fill_unexpanded_prior_vars(
        nc_data=unexpanded_size_arrays, speclist=speclist, runlog_all_speclist=runlog_all_speclist, prior_data=prior_data, prior_var_list=prior_var_list,
        geos_version_keys=geos_version_keys, nlev=nlev, ncell=ncell
    )
    # We'll also cheat about the effective path length; since it uses the prior index to calculate itself more
    # efficiently, do that now. This will mean that the file format update call doesn't have to deal with it.
    cu.add_effective_path(unexpanded_size_arrays, is_public=False, zmin=nc_data['zmin'][:], prior_alts=nc_data['prior_altitude'][:])
    cu.add_effective_path_variable(nc_data)
    prior_index = unexpanded_size_arrays.pop('prior_index')
    for varname, unexpanded_arr in unexpanded_size_arrays.items():
        try:
            if isinstance(unexpanded_arr[0], str):
                for i_nc, i_p in enumerate(prior_index):
                    nc_data[varname][i_nc] = unexpanded_arr[i_p]
            else:
                nc_data[varname][:] = unexpanded_arr[prior_index]
        except ValueError as e:
            print(nc_data[varname].dtype, unexpanded_arr.dtype)
            raise ValueError(f'{varname}, {e}')


def _create_priors_vars(nc_data, prior_data, nlev: int, ncell: int, classic: bool, expand_priors: bool):
    time_dim = 'time' if expand_priors else 'prior_time'
    # We need this if expanding arrays to have a place to collect the unexpanded priors
    # prior_time is created in _add_coords because it is a coordinate, so it is one array
    # we need to make here as a special case.
    nprior = len(prior_data)
    prior_var_arrays = dict() if expand_priors else None
    if expand_priors:
        prior_var_arrays['prior_time'] = np.full(nprior, np.nan, dtype=np.float64)

    prior_var_list = [ i for i in list(prior_data[list(prior_data.keys())[0]]['data'].keys()) if i not in {'altitude', 'geos_versions', 'geos_filenames', 'geos_checksums'}]
    cell_var_list = []
    UNITS_DICT.update({'prior_{}'.format(var):'' for var in prior_var_list if 'prior_{}'.format(var) not in UNITS_DICT})
    for var in prior_var_list:
        prior_var = 'prior_{}'.format(var)
        nc_data.createVariable(prior_var, np.float32, (time_dim,'prior_altitude'), zlib=True, complevel=9)
        att_dict = {}
        att_dict["standard_name"] = '{}_profile'.format(prior_var)
        att_dict["long_name"] = att_dict["standard_name"].replace('_',' ')
        if var in ['temperature','density','pressure','gravity','equivalent_latitude']:
            att_dict["description"] = att_dict["long_name"]
        else:
            att_dict["description"] = 'a priori concentration profile of {}, in parts'.format(var)
        att_dict["units"] = UNITS_DICT[prior_var]
        nc_data[prior_var].setncatts(att_dict)
        if expand_priors:
            prior_var_arrays[prior_var] = np.full([nprior, nlev], np.nan, dtype=np.float32)

        if var in ['gravity', 'equivalent_latitude']:
            continue
        cell_var = 'cell_{}'.format(var)
        cell_var_list += [cell_var]
        nc_data.createVariable(cell_var, np.float32, (time_dim,'cell_index'), zlib=True, complevel=9)
        att_dict = {}
        att_dict["standard_name"] = cell_var
        att_dict["long_name"] = att_dict["standard_name"].replace('_',' ')
        if var in ['temperature','density','pressure','equivalent_latitude']:
            att_dict["description"] = '{} in gas cell'.format(var)
        else:
            att_dict["description"] = 'concentration of {} in gas cell, in parts'.format(var)
        att_dict["units"] = UNITS_DICT[prior_var]
        nc_data[cell_var].setncatts(att_dict)
        if expand_priors:
            prior_var_arrays[cell_var] = np.full([nprior, ncell], np.nan, dtype=np.float32)

    prior_var_list += ['tropopause_altitude']
    nc_data.createVariable('prior_tropopause_altitude', np.float32, (time_dim,), zlib=True, complevel=9)
    att_dict = {
        "standard_name": 'prior_tropopause_altitude',
        "long_name": 'prior tropopause altitude',
        "description": 'altitude at which the gradient in the prior temperature profile becomes > -2 degrees per km',
        "units": UNITS_DICT[prior_var],
    }
    nc_data['prior_tropopause_altitude'].setncatts(att_dict)
    if expand_priors:
        prior_var_arrays['prior_tropopause_altitude'] = np.full([nprior], np.nan, dtype=np.float32)

    geos_version_keys = _get_geos_versions_key_set(prior_data)

    prior_var_list += ['modfile','vmrfile'] + [cu.geos_version_varname(k) for k in geos_version_keys]
    if classic:
        prior_modfile_var = nc_data.createVariable('prior_modfile', 'S1', (time_dim,'a32'), zlib=True, complevel=9)
        prior_modfile_var._Encoding = 'ascii'
        if expand_priors:
            prior_var_arrays['prior_modfile'] = np.full([nprior], None)
        prior_vmrfile_var = nc_data.createVariable('prior_vmrfile','S1',(time_dim,'a32'), zlib=True, complevel=9)
        prior_vmrfile_var._Encoding = 'ascii'
        if expand_priors:
            prior_var_arrays['prior_vmrfile'] = np.full([nprior], None)

        for (vkey, vfxn) in cu.geos_version_keys_and_fxns():
            gv_max_len = _get_geos_version_max_length(prior_data, vkey)
            for gkey in geos_version_keys:
                gv_varname = vfxn(gkey)
                cu.add_geos_version_variables(nc_data, gv_max_len, gv_varname, is_classic=True, time_dim=time_dim)
                if expand_priors:
                    prior_var_arrays[gv_varname] = np.full([nprior], None)
    else:
        prior_modfile_var = nc_data.createVariable('prior_modfile', str, (time_dim,), zlib=True, complevel=9)
        if expand_priors:
            prior_var_arrays['prior_modfile'] = np.full([nprior], None)
        prior_vmrfile_var = nc_data.createVariable('prior_vmrfile', str, (time_dim,), zlib=True, complevel=9)
        if expand_priors:
            prior_var_arrays['prior_vmrfile'] = np.full([nprior], None)
        for (vkey, vfxn) in cu.geos_version_keys_and_fxns():
            gv_max_len = _get_geos_version_max_length(prior_data, vkey)
            for gkey in geos_version_keys:
                gv_varname = vfxn(gkey)
                cu.add_geos_version_variables(nc_data, gv_max_len, gv_varname, is_classic=False, time_dim=time_dim)
                if expand_priors:
                    prior_var_arrays[gv_varname] = np.full([nprior], None)

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
    nc_data.createVariable('prior_effective_latitude', np.float32, (time_dim,), zlib=True, complevel=9)
    att_dict = {
        "standard_name": 'prior_effective_latitude',
        "long_name": 'prior effective latitude',
        "description": "latitude at which the mid-tropospheric potential temperature agrees with that from the corresponding 2-week period in a GEOS-FPIT climatology",
        "units": UNITS_DICT['prior_effective_latitude'],
    }
    nc_data['prior_effective_latitude'].setncatts(att_dict)
    if expand_priors:
        prior_var_arrays['prior_effective_latitude'] = np.full([nprior], np.nan, dtype=np.float32)

    nc_data.createVariable('prior_mid_tropospheric_potential_temperature', np.float32, (time_dim,), zlib=True, complevel=9)
    att_dict = {
        "standard_name": 'prior_mid_tropospheric_potential_temperature',
        "long_name": 'prior mid-tropospheric potential temperature',
        "description": "average potential temperature between 700-500 hPa",
        "units": UNITS_DICT['prior_mid_tropospheric_potential_temperature'],
    }
    nc_data['prior_mid_tropospheric_potential_temperature'].setncatts(att_dict)
    if expand_priors:
        prior_var_arrays['prior_mid_tropospheric_potential_temperature'] = np.full([nprior], np.nan, dtype=np.float32)

    cu.add_geos_version_var_attrs(nc_data, geos_version_keys)

    return prior_var_list, geos_version_keys, prior_var_arrays



def _add_coords(nc_data, prior_data, nlev: int, ncell: int, priors_expanded: bool):
    nprior = len(prior_data)

    if not priors_expanded:
        nc_data.createDimension('prior_time',nprior)
        time_dim = 'prior_time'
    else:
        time_dim = 'time'

    nc_data.createDimension('prior_altitude',nlev) # used for the prior profiles
    nc_data.createDimension('cell_index',ncell)

    nc_data.createVariable('prior_time',np.float64,(time_dim,))
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
        "description": f"variables with names including 'cell_' will be along dimensions ({time_dim}, cell_index)",
    }
    nc_data['cell_index'].setncatts(att_dict)
    nc_data['cell_index'][:] = np.arange(ncell)

    nc_data.createVariable('prior_altitude',np.float32,('prior_altitude')) # this one doesn't change between priors
    att_dict = {
        "standard_name": 'prior_altitude_profile',
        "long_name": 'prior altitude profile',
        "units": UNITS_DICT['prior_altitude'],
        "description": "altitude levels for the prior profiles, these are the same for all the priors",
    }
    nc_data['prior_altitude'].setncatts(att_dict)
    nc_data['prior_altitude'][0:nlev] = prior_data[list(prior_data.keys())[0]]['data']['altitude'].values

    if not priors_expanded:
        nc_data.createVariable('prior_index',np.int16,('time',))
        att_dict = {
            "standard_name": 'prior_index',
            "long_name": 'prior index',
            "units": '',
            "description": 'Index of the prior profile associated with each measurement, it can be used to sample the prior_ and cell_ variables along the prior_time dimension',
        }
        nc_data['prior_index'].setncatts(att_dict)


def _fill_unexpanded_prior_vars(nc_data, speclist, runlog_all_speclist, prior_data, prior_var_list, geos_version_keys, nlev, ncell):
    prior_spec_list = list(prior_data.keys())
    prior_index = _compute_prior_index_by_order(speclist=speclist, runlog_all_speclist=runlog_all_speclist, prior_spec_list=prior_spec_list)
    # TODO: check if prior_index is valid and recalculate if not. Or just calculate by time difference to start.
    nc_data['prior_index'][:] = prior_index

    special_prior_vars, special_vars_map = _special_prior_vars_map(geos_version_keys)
    for prior_spec_id, prior_spectrum in enumerate(prior_spec_list):
        #for var in ['temperature','pressure','density','gravity','1h2o','1hdo','1co2','1n2o','1co','1ch4','1hf','1o2']:
        for var in prior_var_list:
            if var not in special_prior_vars:
                prior_var = 'prior_{}'.format(var)
                nc_data[prior_var][prior_spec_id,0:nlev] = prior_data[prior_spectrum]['data'][var].values

                if var not in ['gravity','equivalent_latitude']:
                    cell_var = 'cell_{}'.format(var)
                    nc_data[cell_var][prior_spec_id,0:ncell] = prior_data[prior_spectrum]['cell_data'][var].values


        for nc_var, dict_index in special_vars_map.items():
            nc_data[nc_var][prior_spec_id] = dict_index.get(prior_data[prior_spectrum])


def _special_prior_vars_map(geos_version_keys):
    # Grr - modfile and vmrfile must not have underscores in the special variable list,
    # because they don't have underscores in the output variable name. But they _do_ in
    # the prior_data dict, so they have to get special treatment.
    spv = ['time', 'tropopause_altitude', 'modfile', 'vmrfile', 'effective_latitude', 'mid_tropospheric_potential_temperature']
    spv_map = {f'prior_{v}': cu.DictMultiIndex(v) for v in spv if v not in {'modfile', 'vmrfile'}}
    spv_map['prior_modfile'] = cu.DictMultiIndex('mod_file')
    spv_map['prior_vmrfile'] = cu.DictMultiIndex('vmr_file')
    for k in geos_version_keys:
        spv.append(cu.geos_version_varname(k))
        spv_map[cu.geos_version_varname(k)] = cu.DictMultiIndex('geos_versions', k)
        spv.append(cu.geos_file_varname(k))
        spv_map[cu.geos_file_varname(k)] = cu.DictMultiIndex('geos_filenames', k)
        spv.append(cu.geos_checksum_varname(k))
        spv_map[cu.geos_checksum_varname(k)] = cu.DictMultiIndex('geos_checksums', k)
    return spv, spv_map


def _compute_prior_index_by_order(speclist, prior_spec_list, runlog_all_speclist):
    logging.info('Computing prior index ...')
    if len(prior_spec_list) == 1:
        # if there is just one block in the .mav file, set it as the prior index for all spectra
        return np.zeros(np.size(speclist), dtype=int)
    else:
        prior_index_arr = np.full(np.size(speclist), -999)
        prior_runlog_inds = cu.get_slice(runlog_all_speclist, prior_spec_list)
        aia_runlog_inds = cu.get_slice(runlog_all_speclist, speclist)
        for spec_id, spectrum in enumerate(speclist):
            # The .mav blocks should always be in runlog order. Set the prior index to point to
            # the last .mav block with a spectrum that comes before the .aia spectrum in the runlog.
            prior_index = np.flatnonzero(prior_runlog_inds <= aia_runlog_inds[spec_id])[-1]
            prior_index_arr[spec_id] = prior_index
        return prior_index_arr


def _get_geos_version_max_length(mav_data, version_key):
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


def _get_geos_versions_key_set(mav_data, version_key='geos_versions'):
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
