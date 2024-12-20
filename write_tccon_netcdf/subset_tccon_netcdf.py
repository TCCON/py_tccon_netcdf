from __future__ import print_function

from argparse import ArgumentParser
from datetime import datetime, timezone
import logging
import netCDF4
import numpy as np
import os
from numpy.core.fromnumeric import var
import pandas as pd
import platform
import re
import subprocess
import sys

__version__ = '1.0.0'


def setup_logging(log_level, log_file=None, message=''):
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
    LEVELS = {'DEBUG': logging.DEBUG,
              'INFO': logging.INFO,
              'WARNING': logging.WARNING,
              'ERROR': logging.ERROR,
              'CRITICAL': logging.CRITICAL,
              }
    # will only display the progress bar for log levels below ERROR
    if LEVELS[log_level] >= 40:
        show_progress = False
    else:
        show_progress = True
    logger = logging.getLogger()
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(handlers=handlers,
                        level="DEBUG",
                        format='\n%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logger.handlers[0].setLevel(LEVELS[log_level])
    logging.info('New subset_tccon_netcdf log session')
    for handler in logger.handlers:
        handler.setFormatter(logging.Formatter('[%(levelname)s]: %(message)s'))
    if message:
        logging.info('Note: %s', message)
    logging.info('Running subset_tccon version %s', __version__.strip())
    proc = subprocess.Popen(['git','rev-parse','--short','HEAD'],cwd=os.path.dirname(__file__),stdout=subprocess.PIPE)
    out, err = proc.communicate()
    HEAD_commit = out.decode("utf-8").strip()
    logging.info('tccon_netcdf repository HEAD: {}'.format(HEAD_commit))
    logging.info('Python executable used: %s', sys.executable)
    logging.info('cwd=%s', os.getcwd())
    return logger, show_progress, HEAD_commit


def progress(i,tot,bar_length=20,word='',simple=False):
    if simple:
        _simple_progress(i,tot)
    else:
        _fancy_progress(i,tot,bar_length=bar_length,word=word)


def _simple_progress(i,tot,freq=250):
    if (i % freq == 0) or i == (tot-1):
        percent = i / tot * 100
        sys.stdout.write(' {:.0f}% complete\n'.format(percent))
        sys.stdout.flush()


def _fancy_progress(i,tot,bar_length=20,word=''):
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
        sys.stdout.write("\rProgress:[{0}] {1}%".format(hashes + spaces, int(round(percent * 100)))+"    "+str(i+1)+"/"+str(tot)+" "+word+" "*30)
    sys.stdout.flush()


def find_subset_time_inds(ncin, start_date, end_date):
    spec_dates = netCDF4.chartostring(ncin['spectrum'][:, 2:10])
    subset_dates = pd.date_range(start_date, end_date)
    subset_dates = np.array([d.strftime('%Y%m%d') for d in subset_dates])
    return np.isin(spec_dates, subset_dates)


def find_subset_prior_time_inds(ncin, time_inds):
    prior_inds = np.arange(ncin['prior_time'].size)
    subset_prior_inds = ncin['prior_index'][time_inds]
    return np.isin(prior_inds, subset_prior_inds)


def copy_group_attrs(ncin, ncout):
    group_attrs = ncin.__dict__.copy()
    now = datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
    group_attrs['file_creation'] = "Created with Python {}, the library netCDF4 {}, and subset_netcdf {}".format(platform.python_version(), netCDF4.__version__, __version__)
    group_attrs['history'] = "Created from {} with subset_netcdf {} on {}".format(ncin.filepath(), __version__, now)
    ncout.setncatts(group_attrs)


def copy_group_dims(ncin, ncout, time_inds, prior_time_inds):
    for name, dimension in ncin.dimensions.items():
        if name == 'time':
            length = np.sum(time_inds)
        elif name == 'prior_time':
            length = np.sum(prior_time_inds)
        elif dimension.isunlimited():
            length = None
        else:
            length = dimension.size

        ncout.createDimension(name, length)


def copy_variable(ncin, ncout, varname, time_inds, prior_time_inds):
    def check_dim_position(v, dimname):
        if v.dimensions.index(dimname) != 0:
            raise NotImplementedError('Error copying variable "{}": {} dimension is not first'.format(v.name, dimname))

    def copy_var_data(v_in, v_out, inds):
        if v_in.dtype.kind == 'U':
            # This should be very similar to copying string arrays, but I'm not entirely sure how unicode types
            # get encoded in netCDF, so we'll leave that until we have an example
            raise NotImplementedError('Copying unicode variables not implemented')

        if v_in.dtype.kind == 'S':
            # String variables present a bit of a challenge - they get read from the netCDF file as 1D arrays,
            # but need written as 2D (regular index + string length). So we need to convert them from 1D string
            # arrays to 2D char arrays.
            if v_in.ndim != 2:
                raise NotImplementedError('Copying string variables with other than 2 dimensions not implemented')
            dtype = 'S{}'.format(v_in.shape[1])
            char_array = netCDF4.stringtochar(np.array(v_in[inds], dtype))
            v_out[:] = char_array
        else:
            v_out[:] = v_in[inds]

    var_in = ncin.variables[varname]
    if 'time' in var_in.dimensions and 'prior_time' in var_in.dimensions:
        raise NotImplementedError('Cannot subset a variable ({}) with both time and prior_time as dimensions!'.format(varname))

    var_out = ncout.createVariable(varname, var_in.datatype, var_in.dimensions)
    var_out.setncatts(var_in.__dict__)
    if 'a32' in var_in.dimensions or 'specname' in var_in.dimensions:
        var_out._Encoding = 'ascii'

    if varname == 'prior_index':
        # This is a special case: we know it has the "time" dimension and is not a string
        # variable (so don't need to use `copy_var_data`) but also needs the values adjusted
        # to point to the correct index in the prior variables
        tmp_inds = var_in[time_inds]
        tmp_inds = tmp_inds - np.min(tmp_inds)
        var_out[:] = tmp_inds
    elif 'time' in var_in.dimensions:
        check_dim_position(var_in, 'time')
        copy_var_data(var_in, var_out, time_inds)
    elif 'prior_time' in var_in.dimensions:
        check_dim_position(var_out, 'prior_time')
        copy_var_data(var_in, var_out, prior_time_inds)
    else:
        # use a tuple to indicate copy all variable data - works
        # even for scalars
        copy_var_data(var_in, var_out, tuple())


def parse_args():
    p = ArgumentParser(description='Subset a TCCON .private.nc or .private.qc.nc file to a date or range of dates')
    p.add_argument('input_nc_file', help='Original .private.nc or .private.qc.nc file to subset')
    p.add_argument('start_date', help='First date in to include in the subset, in YYYYMMDD format')
    p.add_argument('end_date', nargs='?', help='Last date to include in the subset (inclusive). If omitted, then only START_DATE is included in the subset')
    p.add_argument('-o', '--output-file', help='Where to write the output file. By default, it is written to the current directory with a name determined from the start and end dates.')
    p.add_argument('-c', '--clobber', action='store_true', help='Allow overwriting an existing output file. Note: will never be allowed to overwrite the input file directly.')
    p.add_argument('--simple-progress',action='store_true',help='Print percentage complete every 250 variables copied instead of the fancy progress bar. (Useful for logging to a file.)')
    p.add_argument('--tb', action='store_true', help='Print a full traceback if an error occurs')
    p.add_argument('--pdb', action='store_true', help='Launch Python debugger')

    return vars(p.parse_args())


def driver(input_nc_file, start_date, end_date=None, output_file=None, clobber=False, simple_progress=False):
    setup_logging('INFO')

    if end_date is None:
        end_date = start_date

    if not re.match(r'\d{8}', start_date):
        raise TypeError('start_date must be in YYYYMMDD format')
    if not re.match(r'\d{8}', end_date):
        raise TypeError('end_date must be in YYYYMMDD format')

    if output_file is None:
        site = os.path.basename(input_nc_file)[:2]
        ext = os.path.basename(input_nc_file).split('.', 1)[1]
        output_file = '{site}{start}_{end}.{ext}'.format(site=site, start=start_date, end=end_date, ext=ext)

    if os.path.exists(output_file) and os.path.samefile(input_nc_file, output_file):
        raise IOError('Cannot overwrite the input_nc_file directly, must specify a different output file')
    if os.path.exists(output_file) and not clobber:
        raise IOError('Output file {} exists, will not overwrite by default. Use the clobber flag to permit overwriting.'.format(output_file))

    logging.info('Will save subset file to %s', output_file)

    with netCDF4.Dataset(input_nc_file, 'r') as ncin, netCDF4.Dataset(output_file, 'w') as ncout:
        # First check that there are no groups other than root - not dealing with those for now, so need to warn
        if len(ncin.groups) > 0:
            logging.warn('Input netCDF file has groups other than the root group, copying these is not yet implemented!')

        # Next figure out the time and prior_time indices we need to keep 
        time_inds = find_subset_time_inds(ncin, start_date, end_date)
        prior_time_inds = find_subset_prior_time_inds(ncin, time_inds)

        nspectra = np.sum(time_inds)
        nprior = np.sum(prior_time_inds)
        if nspectra > 0:
            logging.info('{} spectra and {} prior values will be retained'.format(nspectra, nprior))
        else:
            logging.warn('No spectra found in the date range {} to {}!'.format(start_date, end_date))

        # That done we can start copying data to the new file
        logging.info('Copying global attributes')
        copy_group_attrs(ncin=ncin, ncout=ncout)
        logging.info('Copying global dimensions')
        copy_group_dims(ncin=ncin, ncout=ncout, time_inds=time_inds, prior_time_inds=prior_time_inds)
        logging.info('Copying variables')
        nvar = len(ncin.variables)
        for ivar, varname in enumerate(ncin.variables.keys()):
            progress(ivar, nvar, word=varname, simple=simple_progress)
            copy_variable(ncin=ncin, ncout=ncout, varname=varname, time_inds=time_inds, prior_time_inds=prior_time_inds)


def main():
    clargs = parse_args()
    full_tb = clargs.pop('tb')
    if clargs.pop('pdb'):
        import pdb
        pdb.set_trace()

    try:
        driver(**clargs)
    except Exception as err:
        if full_tb:
            raise
        else:
            logging.critical('An error occurred: %s', err)
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
