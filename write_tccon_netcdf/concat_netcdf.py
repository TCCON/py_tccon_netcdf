import sys
import platform
import os
import numpy as np
import netCDF4
import re
import shutil
import argparse
from contextlib import ExitStack
import time
from datetime import datetime

"""
Concatenate a set of TCCON netcdf files along the time dimension.

This code was updated on 2021-09-30 to allow different windows, e.g. to support concatenating
files with InGaAs+Si and InGaAs+InSb data. Several of the Caltech-run sites changed from Si to
InSb detectors partway through their lifetime, so this is necessary.

Don't mix .private.nc and .public.nc files.

There are consistency checks for dimensions that will cause a crash if not fulfilled. Mismatched
global attributes cause a warning only. Variable attributes are currently not checked.
"""

# -------------- #
# CODE FOR TESTS #
# -------------- #

_DEFAULT_TEST_VARS = ('time', 'xluft', 'xco2', 'xch4',
                      'xocs_insb', 'vsf_ocs_insb', 'ocs_2051_ovc_ocs_insb',
                      'xao2_si', 'vsf_ao2_si', 'ao2_13082_ovc_ao2_si',
                      'prior_time', 'prior_1co2', 'prior_effective_latitude',
                      'ak_xco2', 'ak_xch4', 'ak_slant_xco2_bin', 'ak_slant_xch4_bin')

def test_main(orig_files, concat_file, test_variables=_DEFAULT_TEST_VARS, verbose=True):
    ecode = 0
    with ExitStack() as stack:
        concat_ds = stack.enter_context(netCDF4.Dataset(concat_file))
        orig_ds = [stack.enter_context(netCDF4.Dataset(f)) for f in orig_files]
        concat_spec_inds = [np.isin(concat_ds['spectrum'][:], orig['spectrum'][:]) for orig in orig_ds]
        concat_prior_inds = [np.isin(concat_ds['prior_modfile'][:], orig['prior_modfile'][:]) for orig in orig_ds]

        for ivar, var in enumerate(test_variables):
            failures = 0
            for iorig, orig in enumerate(orig_ds):
                if 'time' in concat_ds[var].dimensions:
                    inds = concat_spec_inds[iorig]
                elif 'prior_time' in concat_ds[var].dimensions:
                    inds = concat_prior_inds[iorig]
                else:
                    inds = tuple()

                if not compare_variables(concat_ds, orig, inds, var, verbose=verbose):
                    failures += 1

            if failures == 0:
                print('{}: PASS'.format(var))
            else:
                print('{}: FAIL ({} of {} files do not match)'.format(var, failures, len(orig_files)))
                ecode = 1

            if verbose:
                print('')  # put a gap before the next section

    if ecode == 0:
        print('{} PASSES'.format(concat_file))
    else:
        print('{} FAILS'.format(concat_file))

    return ecode
                

                
def compare_variables(concat_ds, orig_ds, inds, varname, verbose=True):
    if varname not in orig_ds.variables.keys():
        # This variable wasn't in this original file, so the concatenated dataset should be all
        # fill values
        ok = np.all(concat_ds[varname][:][inds].mask)
        if verbose:
            if ok:
                print('{} absent from {} and is all fill values in concatenated file (correct)'.format(varname, orig_ds.filepath()))
            else:
                print('FAIL: {} absent from {} and but is NOT all fill values in concatenated file'.format(varname, orig_ds.filepath()))
        return ok

    else:
        c_vals = concat_ds[varname][:][inds]
        o_vals = orig_ds[varname][:]
        val_ok = np.ma.allclose(c_vals, o_vals)
        mask_ok = np.array_equal(normalize_mask(c_vals), normalize_mask(o_vals))
        ok = val_ok and mask_ok
        if verbose:
            if ok:
                print('Variable {} matches between {} and concatenated file (correct)'.format(varname, orig_ds.filepath()))
            elif not val_ok:
                print('FAIL: Variable {} DOES NOT match between {} and concatenated file'.format(varname, orig_ds.filepath()))
            elif not mask_ok:
                print('FAIL: Fill values locations in variable {} DO NOT match between {} and concatenated file'.format(varname, orig_ds.filepath()))
            print('   Original: {}'.format(o_vals))
            print('   Concatenated: {}'.format(c_vals))
        return ok


def normalize_mask(arr):
    # Masked arrays with no masked values will have a mask that is a scalar `False`. 
    # This will cause array_equal comparison failures if one file has a scalar and
    # the other a full vector of `False` values, so force the scalar into an array.
    mask = arr.mask
    shape = arr.shape
    if np.ndim(mask) == 0:
        return np.full(shape, mask)
    else:
        return mask


# --------- #
# MAIN CODE #
# --------- #

def num2date(x,units,calendar):
    """
    Convert netcdf time variable to python datetime (rather than cftime object returned by netCDF4.num2date)

    Inputs:
        - x: array
        - units: time units
        - calendar: calendar used

    Outputs:
        - x as python datetime
    """
    return np.array([datetime(*cftime.timetuple()[:6]) for cftime in netCDF4.num2date(x,units=units,calendar=calendar)])


def progress(i,tot,bar_length=20,word='',simple=False):
    if simple:
        _simple_progress(i,tot)
    else:
        _fancy_progress(i,tot,bar_length=bar_length,word=word)


def _simple_progress(i,tot,freq=250):
    if (i % freq == 0) or i == (tot-1):
        percent = i / tot * 100
        sys.stdout.write('Concatenation {:.0f}% complete\n'.format(percent))
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


def print_collection_diffs(name1, col1, name2, col2, stream=sys.stdout):
    col1 = set(col1)
    col2 = set(col2)
    print('Present in {} but not {}:'.format(name1, name2), file=stream)
    for v in col1.difference(col2):
        print('  - {}'.format(v), file=stream)
    print('Missing from {} but not {}:'.format(name1, name2), file=stream)
    for v in col2.difference(col1):
        print('  - {}'.format(v), file=stream)


def print_differing_values(name1, vals1, name2, vals2, common_vals=None, stream=sys.stdout):
    if common_vals is None:
        common_vals = sorted(set(vals1.keys()).intersection(vals2.keys()))

    print('Differing values for {} vs. {}:'.format(name1, name2), file=stream)
    for cv in common_vals:
        print('  - {}: {} vs {}'.format(cv, vals1[cv], vals2[cv]), file=stream)


def get_first_variable(ncin_list, varname):
    for ncin in ncin_list:
        if varname in ncin.variables.keys():
            return ncin[varname]

    raise RuntimeError('Could not find a variable named "{}" in any of the input files'.format(varname))



def main():
    parser = argparse.ArgumentParser(description="Concatenate netCDF files that are in the given directory")
    parser.add_argument('path',help='full path to a folder containing netCDF files')
    parser.add_argument('--out',default='',help='full path to the directory where the output file will be saved, default to same as the "path" argument')
    parser.add_argument('--prefix',default='',help='if given, only use files starting with the given prefix')
    parser.add_argument('--simple-progress',action='store_true',help='Print percentage complete every 250 variables copied instead of the fancy progress bar. (Useful for logging to a file.)')
    parser.add_argument('--test',help='Run tests on a concatenated file, comparing against the files matched by PATH and --prefix.')
    parser.add_argument('--test-verbose',action='store_true',help='Print more information from the testing')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        sys.exit('Invalid path: {}'.format(args.path))

    if not args.out:
        args.out = args.path
    else:
        if not os.path.exists(args.out):
            sys.exit('Invalid path: {}'.format(args.out))

    if args.prefix:
        nc_list = [i for i in os.listdir(args.path) if i.startswith(args.prefix) and i.endswith('.nc')]
    else:
        nc_list = [i for i in os.listdir(args.path) if i.endswith('.nc')]

    file_creation = "Created with Python {}; the library netCDF4 {}; and the code concat_netcdf.py".format(platform.python_version(),netCDF4.__version__)
    history = "Created {} (UTC) from the concatenation of {} netCDF files".format(time.asctime(time.gmtime(time.time())),len(nc_list))

    if len(nc_list) == 0:
        sys.exit('No files to concatenate')
    elif len(nc_list) == 1:
        print('Only one file to concatenate, making a copy instead', file=sys.stderr)
        if args.path == args.out:
            sys.exit("When using on a single file, the output filename is the same as the input file, use a different output directory to avoid overwritting the input file")
        shutil.copy2(os.path.join(args.path, nc_list[0]), args.out)
        new_nc = os.path.join(args.out, nc_list[0]) if os.path.isdir(args.out) else args.out
        with netCDF4.Dataset(new_nc,'r+') as ncout:
            ncout.setncatts({"file_creation":file_creation,"history":history})
        sys.exit(0)

    if args.test is not None:
        print('Running test only!')
        ecode = test_main(nc_list, args.test, verbose=args.test_verbose)
        sys.exit(ecode)

    print('{} files will be concatenated:'.format(len(nc_list)))
    for nc_file in nc_list:
        print('\t',nc_file)

    N_sites = len(set([i[:2] for i in nc_list]))
    if N_sites!=1:
        sys.exit('There are files from {} sites. All netCDF files to concatenate should be from the same site'.format(N_sites))
    site_abbrv = nc_list[0][:2]

    with ExitStack() as stack:
        ncin_list = [stack.enter_context(netCDF4.Dataset(os.path.join(args.path,nc_file),'r')) for nc_file in nc_list]
        # sort the files with increasing starting date
        ncin_list = [ncin_list[i] for i in np.argsort([ncin['time'][0] for ncin in ncin_list])]

        start = datetime.strftime(num2date([ncin_list[0]['time'][0]],ncin_list[0]['time'].units,ncin_list[0]['time'].calendar)[0],'%Y%m%d')
        end = datetime.strftime(num2date([ncin_list[-1]['time'][-1]],ncin_list[-1]['time'].units,ncin_list[-1]['time'].calendar)[0],'%Y%m%d')
        ext_regex = re.compile(r'\.(private|public)(\.qc)?.nc')
        extension = set(ext_regex.search(nc_file).group() for nc_file in nc_list)
        if len(extension) != 1:
            print('WARNING: different types of netCDF file (e.g. public vs. private) detected! Extension defaulting to ".nc".', file=sys.stderr)
        else:
            extension = list(extension)[0]

        # verify that dimensions other than time, prior_time, and specname have the same length across all files,
        # and that all files have the same dimensions. 
        first_file_dim_lengths = {name: len(dim) for name, dim in ncin_list[0].dimensions.items()}
        for ncin in ncin_list[1:]:
            other_file_dim_lengths = {name: len(dim) for name, dim in ncin.dimensions.items()}
            if first_file_dim_lengths.keys() != other_file_dim_lengths.keys():
                print('ERROR: {} and {} have different dimensions!'.format(ncin_list[0].filepath(), ncin.filepath()), file=sys.stderr)
                print_collection_diffs(ncin_list[0].filepath(), first_file_dim_lengths.keys(), ncin.filepath(), other_file_dim_lengths.keys(), stream=sys.stderr)
                sys.exit(1)

            common_dims = set(first_file_dim_lengths.keys()).intersection(other_file_dim_lengths.keys())
            unequal_dims = [d for d in common_dims if d not in ('time', 'prior_time', 'specname') and first_file_dim_lengths[d] != other_file_dim_lengths[d]]
            if len(unequal_dims) > 0:
                print('ERROR: {} and {} have {} common non-time dimensions with different lengths'.format(ncin_list[0].filepath(), ncin.filepath(), len(unequal_dims)), file=sys.stderr)
                print_differing_values(ncin_list[0].filepath(), first_file_dim_lengths, ncin.filepath(), other_file_dim_lengths, common_vals=unequal_dims)
                sys.exit(1)

        # check if global attributes (other than file_creation and history, which are overwritten later) differ
        # this is not a hard fail, but we should warn the user.
        first_file_attrs = {k: v for k, v in ncin_list[0].__dict__.items() if k not in ('history', 'file_creation')}
        warn_different_attrs = False
        for ncin in ncin_list[1:]:
            other_file_attrs = {k: v for k, v in ncin.__dict__.items() if k not in ('history', 'file_creation')}
            if first_file_attrs.keys() != other_file_attrs.keys():
                print('WARNING: {} and {} have different global attributes'.format(ncin_list[0].filepath(), ncin.filepath()), file=sys.stderr)
                print_collection_diffs(ncin_list[0].filepath(), first_file_attrs.keys(), ncin.filepath(), other_file_attrs.keys(), stream=sys.stderr)
                warn_different_attrs = True

            common_attrs = set(first_file_attrs.keys()).intersection(other_file_attrs.keys())
            unequal_attrs = [a for a in common_attrs if first_file_attrs[a] != other_file_attrs[a]]
            if len(unequal_attrs) > 0:
                print('WARNING: {} and {} have different values for global attributes'.format(ncin_list[0].filepath(), ncin.filepath()), file=sys.stderr)
                print_differing_values(ncin_list[0].filepath(), first_file_attrs, ncin.filepath(), other_file_attrs, common_vals=unequal_attrs, stream=sys.stderr)
                warn_different_attrs = True

        if warn_different_attrs:
            print('WARNING: Since at least one .nc file has different global attributes than the first, note that the concatenated file will use the first input file\'s attributes', file=sys.stderr)


        outfile = '{}{}_{}{}'.format(site_abbrv,start,end,extension)
        outpath = os.path.join(args.out,outfile)
        print('Output netCDF file will be saved as',outpath)

        nspec = np.sum([ncin['time'].size for ncin in ncin_list]) # total size of the output time variable
        nprior = np.sum([ncin['prior_time'].size for ncin in ncin_list]) # total size of the output prior_time variable
        print('Output time dimension size:',nspec)
        print('Output prior_time dimension size:',nprior)

        # Figure out the union of variables across all input files, keeping variable order as much as possible.
        # This is necessary if concatenating files containing different detectors, i.e. one is InGaAs+Si and one
        # is InGaAs+InSb
        all_variables = list(ncin_list[0].variables.keys())
        variables_set = set(all_variables)
        for ncin in ncin_list[1:]:
            this_file_variables = list(ncin.variables.keys())
            new_variables = set(this_file_variables).difference(variables_set)
            if len(new_variables) > 0:
                # Keep the new variables in the same order they were in their netCDF file.
                # They'll be added to the end of the concatenated file; no easy way to create
                # the "canonical" order.
                new_variables = sorted(new_variables, key=this_file_variables.index)
                variables_set.update(new_variables)
                all_variables.extend(new_variables)

        ncout = stack.enter_context(netCDF4.Dataset(outpath,'w',nc_format='NETCDF4_CLASSIC'))
        ## copy attributes, dimensions, and variables metadata using the first file
        global_attributes = ncin_list[0].__dict__.copy()
        # update the history and file_creation
        global_attributes['file_creation'] = file_creation
        global_attributes['history'] = history
        ncout.setncatts(global_attributes)

        # Copy dimensions from the first file. We ensured above that all files have the same dimensions
        # and dimensions other than time, prior_time, and specname have the same length, so we can copy
        # from the first file safely. 
        speclength = np.max([ncin.dimensions['specname'].size for ncin in ncin_list])
        for name, dimension in ncin_list[0].dimensions.items():
            if name == 'time':
                ncout.createDimension(name, nspec)
            elif name == 'prior_time':
                ncout.createDimension(name, nprior)
            elif name == 'specname':
                ncout.createDimension(name, speclength)
            else:
                ncout.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))

        varcount = 0
        nvar = len(all_variables)
        for name in all_variables:
            progress(varcount,nvar,word=name,simple=args.simple_progress)
            varcount += 1

            variable = get_first_variable(ncin_list, name)
            ncout.createVariable(name,variable.datatype,variable.dimensions)
            ncout[name].setncatts(variable.__dict__)

            # Set the values of non-time coordinate variables. Should only need to do
            # this from one file; if other 
            is_nontime_dim = (name not in ['prior_time','time']) and (name in list(ncout.dimensions))
            has_no_time_dim = all(d not in variable.dimensions for d in ['prior_time','time'])
            if is_nontime_dim or has_no_time_dim:
                # First verify that the value is the same across all file
                for ncin in ncin_list[1:]:
                    if name in ncin.variables.keys() and not np.ma.allclose(variable[:], ncin[name][:]):
                        if is_nontime_dim:
                            print('ERROR: Dimension coordinate variable "{}" has different values across files'.format(name), file=sys.stderr)
                        elif has_no_time_dim:
                            print('ERROR: Variable without temporal dimension "{}" has different values across files'.format(name), file=sys.stderr)
                        else:
                            print('ERROR: Variable "{}" has different values across files'.format(name), file=sys.stderr)
                        sys.exit(1)

                ncout[name][:] = variable[:]
                continue

            # If this is a character variable, provide a text encoding
            # TODO: handle the GEOS version variables (which won't necessarily be "a32")
            # - both give then the ascii encoding _and_ handle if their second dimension
            # varies from file to file - we'd want to use the largest dimension for the
            # concatenated file.
            if 'a32' in variable.dimensions or 'specname' in variable.dimensions:
                ncout[name]._Encoding = 'ascii'

            spec_count = 0
            prior_count = 0
            for ncin in ncin_list:
                time_size = ncin['time'].size
                prior_time_size = ncin['prior_time'].size

                if name not in ncin.variables.keys():
                    # If concatenating files containing different detectors (e.g. InGaAs+Si or InGaAs+InSb),
                    # there will be some variables absent from some of the files. Don't try to copy in 
                    # that case, but don't skip the whole rest of the loop, because we need to advance spec_count
                    # and prior_count
                    pass
                elif 'prior_time' in ncin[name].dimensions and 'a32' not in ncin[name].dimensions:
                    ncout[name][prior_count:prior_count+prior_time_size] = ncin[name][:]
                elif 'prior_time' in ncin[name].dimensions and 'a32' in ncin[name].dimensions:
                    ncout[name][prior_count:prior_count+prior_time_size] = netCDF4.stringtochar(np.array(ncin[name][:],'S32'))
                elif name == 'prior_index': # special case, need to add prior_count to it to properly sample along the concatenated prior_time dimension
                    ncout[name][spec_count:spec_count+time_size] = ncin[name][:] + prior_count
                elif 'time' in ncin[name].dimensions and 'a32' not in ncin[name].dimensions and 'specname' not in ncin[name].dimensions:
                    ncout[name][spec_count:spec_count+time_size] = ncin[name][:]
                elif 'time' in ncin[name].dimensions and 'a32' in ncin[name].dimensions:
                    ncout[name][spec_count:spec_count+time_size] = netCDF4.stringtochar(np.array(ncin[name][:],'S32'))
                elif 'time' in ncin[name].dimensions and 'specname' in ncin[name].dimensions:
                    ncout[name][spec_count:spec_count+time_size] = netCDF4.stringtochar(np.array(ncin[name][:],'S{}'.format(speclength)))
                
                spec_count += time_size
                prior_count += prior_time_size
            # end of for ncin
        # end of for name,variable
    # end of with statement
    print('\nFinished writing {} {:.2f} MB'.format(outpath,os.path.getsize(outpath)/1e6))

if __name__ == "__main__":
    main()
