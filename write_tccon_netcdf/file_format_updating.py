import pandas as pd

from . import common_utils as cu

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
                var = cu.add_geos_version_variable(ds, gv_max_len, gv_varname, is_classic)
                var[:] = gv_values

    # This will overwrite the attributes if any of this variables existed. That's fine;
    # we're using the same function to write the attributes here as we do in the main writer
    # so they will be the same either way.
    cu.add_geos_version_var_attrs(ds, geos_version_keys)
