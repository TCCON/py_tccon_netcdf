import logging
import os
import sys

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