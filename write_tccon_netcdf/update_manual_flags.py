from argparse import ArgumentParser
import netCDF4 as ncdf
import numpy as np
import os
import sys
from pathlib import Path
from shutil import copyfile

from .write_netcdf import set_manual_flags


def update_manual_flags(nc_file_in, nc_file_out=None, mflags_file=None):
    if nc_file_out is not None:
        if Path(nc_file_in).resolve() == Path(nc_file_out).resolve():
            raise IOError('Input and output files point to the same file.')

        copyfile(nc_file_in, nc_file_out)
        nc_file = nc_file_out
    else:
        nc_file = nc_file_in

    # First clear any existing flags
    with ncdf.Dataset(nc_file, 'r+') as ds:
        if np.any(ds['flag'][:] > 9999):
            # This is mainly because it is difficult to parse apart the flagged var name entries when multiple
            # rounds of additional flags have been added. Really, TCCON partners should not be modifying files
            # after they've been released flagged - they should update their local copy, then reupload that to
            # tccondata.org, where it will have new release flags added.
            raise RuntimeError('Cannot update manual flags in a file that already has release (or other post-manual) flags set.')

        _clear_manual_flags(ds)

    # Then go ahead and add in manual flags with the new file. Do intercept the error message for a missing manual
    # flagging file under GGGPATH/tccon to make the error message more relevant.
    try:
        set_manual_flags(nc_file, mflag_file=mflags_file)
    except IOError as err:
        if err.args[0].startswith('A manual flagging file'):
            site_id = os.path.basename(nc_file)
            gggpath = os.getenv('GGGPATH', 'no GGGPATH defined')
            raise IOError('Could not find a manual flagging file for site "{}" under GGGPATH = {}'.format(site_id, gggpath))
        else:
            raise

    if nc_file_out is None:
        print('Finished. Flags in {} have been updated.'.format(nc_file))
    else:
        print('Finished. A new copy of {} with updated flags has been written to {}'.format(nc_file_in, nc_file))
        
    

def _clear_manual_flags(nc_data, nc_slice=slice(None)):
    full_flags = nc_data['flag'][nc_slice]
    if np.any(full_flags > 9999):
        raise RuntimeError('Cannot clear manual flags from a file that has release flags applied.')
    mflags = (full_flags // 1000) % 10
    full_flags -= (mflags * 1000)
    nc_data['flag'][nc_slice] = full_flags  # not sure if this is needed?
    nc_indices = np.arange(nc_data['flag'].size)[nc_slice]

    for islice, (m, f) in enumerate(zip(mflags, full_flags)):
        ifull = nc_indices[islice]
        if m > 0:
            # This means we need to replace at least part of the flagged variable name
            s = nc_data['flagged_var_name'][ifull].item()
            if (f % 1000) > 0:
                # These was an automatic variable name at the beginning of this flagged_var_name. We
                # need to keep that, but remove any extra names following on after a "+"
                nc_data['flagged_var_name'][ifull] = s.split(' + ', 1)[0]
            else:
                # No automatic variable name, so we can just clear the whole flagged_var_name
                nc_data['flagged_var_name'][ifull] = ''

    for att_name in nc_data.ncattrs():
        if att_name.startswith('manual_flags_'):
            nc_data.delncattr(att_name)


def parse_args():
    p = ArgumentParser(description='Replace the manual flags in a TCCON .private.nc file with new ones')
    p.add_argument('nc_file_in', help='The original .private.nc file to update flags in')
    out_grp = p.add_mutually_exclusive_group(required=True)
    out_grp.add_argument('-o', '--output-file', dest='nc_file_out', help='Path to write a new netCDF file to with the updated flags')
    out_grp.add_argument('--inplace', action='store_true', help='Update the flags in NC_FILE_IN without making a copy')
    p.add_argument('--mflags-file', help='Manual flagging .dat file to read the new manual flags from. If not given, '
                                         'then this program will attempt to find one under $GGGPATH/tccon with the '
                                         'site ID given by the first two letters of the input file name.')
    p.add_argument('--tb', '--traceback', dest='traceback', action='store_true', help='For debugging, show the full stack trace of errors')

    return vars(p.parse_args())


def main():
    clargs = parse_args()
    clargs.pop('inplace')  # strictly a command line argument to help the user recognize that they are modifying the existing file
    show_tb = clargs.pop('traceback')
    try:
        update_manual_flags(**clargs)
    except Exception as err:
        if show_tb:
            raise
        else:
            print('ERROR: {}'.format(err), file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()
