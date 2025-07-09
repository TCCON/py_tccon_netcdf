import logging
import netCDF4
import numpy as np
import os

from typing import Optional

def copy_netcdf(input_nc_file, output_file, clobber=False, var_renames=None, exclude_vars=None):
    if var_renames is None:
        var_renames = dict()
    if exclude_vars is None:
        exclude_vars = set()

    if os.path.exists(output_file) and os.path.samefile(input_nc_file, output_file):
        raise IOError('Cannot overwrite the input_nc_file directly, must specify a different output file')
    if os.path.exists(output_file) and not clobber:
        raise IOError('Output file {} exists, will not overwrite by default. Use the clobber flag to permit overwriting.'.format(output_file))

    with netCDF4.Dataset(input_nc_file, 'r') as ncin, netCDF4.Dataset(output_file, 'w') as ncout:
        # First check that there are no groups other than root - not dealing with those for now, so need to warn
        if len(ncin.groups) > 0:
            logging.warning('Input netCDF file has groups other than the root group, copying these is not yet implemented!')

        # That done we can start copying data to the new file
        logging.info('Copying global attributes')
        _copy_group_attrs(ncin=ncin, ncout=ncout)
        logging.info('Copying global dimensions')
        _copy_group_dims(ncin=ncin, ncout=ncout, time_inds=None, prior_time_inds=None)
        logging.info('Copying variables')
        for ivar, varname in enumerate(ncin.variables.keys()):
            if varname not in exclude_vars:
                new_name = var_renames.get(varname, varname)
                _copy_variable(ncin=ncin, ncout=ncout, varname=varname, new_varname=new_name, time_inds=None, prior_time_inds=None)
            else:
                logging.info(f'Not copying excluded variable {varname}')


def _copy_group_attrs(ncin, ncout):
    group_attrs = ncin.__dict__.copy()
    # now = datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
    # group_attrs['file_creation'] = "Created with Python {}, the library netCDF4 {}, and subset_netcdf {}".format(platform.python_version(), netCDF4.__version__, __version__)
    # group_attrs['history'] = "Created from {} with subset_netcdf {} on {}".format(ncin.filepath(), __version__, now)
    ncout.setncatts(group_attrs)


def _copy_group_dims(ncin, ncout, time_inds: Optional[np.ndarray], prior_time_inds: Optional[np.ndarray]):
    for name, dimension in ncin.dimensions.items():
        if name == 'time':
            length = dimension.size if time_inds is None else np.sum(time_inds)
        elif name == 'prior_time':
            length = dimension.size if prior_time_inds is None else np.sum(prior_time_inds)
        elif dimension.isunlimited():
            length = None
        else:
            length = dimension.size

        ncout.createDimension(name, length)


def _copy_variable(ncin: netCDF4.Dataset, ncout: netCDF4.Dataset, varname: str, new_varname: str,
                   time_inds: Optional[np.ndarray], prior_time_inds: Optional[np.ndarray]):
    def check_dim_position(v, dimname):
        if v.dimensions.index(dimname) != 0:
            raise NotImplementedError('Error copying variable "{}": {} dimension is not first'.format(v.name, dimname))

    def copy_var_data(v_in, v_out, inds):
        if v_in.dtype == str:
            # String variables must be copied in a loop as far as I know
            if v_in.ndim != 1:
                raise NotImplementedError('Copying string variables with >1 dimension not implemented')
            data = v_in[:]
            for i_out, i_in in enumerate(inds):
                v_out[i_out] = data[i_in]
            return

        if v_in.dtype.kind == 'U':
            # This should be very similar to copying string arrays, but I'm not entirely sure how unicode types
            # get encoded in netCDF, so we'll leave that until we have an example
            raise NotImplementedError('Copying unicode variables not implemented')

        if v_in.dtype.kind == 'S':
            # Character variables present a bit of a challenge - they get read from the netCDF file as 1D arrays,
            # but need written as 2D (regular index + string length). So we need to convert them from 1D string
            # arrays to 2D char arrays.
            if v_in.ndim != 2:
                raise NotImplementedError('Copying string variables with other than 2 dimensions not implemented')
            dtype = 'S{}'.format(v_in.shape[1])
            char_array = netCDF4.stringtochar(np.array(v_in[inds], dtype))
            v_out[:] = char_array
            return

        # Default for numeric data
        v_out[:] = v_in[inds]

    var_in = ncin.variables[varname]
    if 'time' in var_in.dimensions and 'prior_time' in var_in.dimensions:
        raise NotImplementedError('Cannot copy/subset a variable ({}) with both time and prior_time as dimensions!'.format(varname))

    var_out = ncout.createVariable(new_varname, var_in.datatype, var_in.dimensions)
    var_out.setncatts(var_in.__dict__)
    if 'a32' in var_in.dimensions or 'specname' in var_in.dimensions:
        var_out._Encoding = 'ascii'

    if varname == 'prior_index' and time_inds is not None:
        # This is a special case: we know it has the "time" dimension and is not a string
        # variable (so don't need to use `copy_var_data`) but also needs the values adjusted
        # to point to the correct index in the prior variables
        tmp_inds = var_in[time_inds]
        tmp_inds = tmp_inds - np.min(tmp_inds)
        var_out[:] = tmp_inds
    elif 'time' in var_in.dimensions and time_inds is not None:
        check_dim_position(var_in, 'time')
        copy_var_data(var_in, var_out, time_inds)
    elif 'prior_time' in var_in.dimensions and prior_time_inds is not None:
        check_dim_position(var_out, 'prior_time')
        copy_var_data(var_in, var_out, prior_time_inds)
    else:
        # use a tuple to indicate copy all variable data - works
        # even for scalars
        copy_var_data(var_in, var_out, tuple())
