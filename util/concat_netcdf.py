import sys
import platform
import os
import numpy as np
import netCDF4
import argparse
from contextlib import ExitStack
import time
from datetime import datetime

"""
Concatenate a set of TCCON netcdf files along the time dimension.

This code assumes that all the files were generated with the same code and using all the same windows.
The only thing that should differ between the netcdf files is the time ranges.
Don't mix .private.nc and .public.nc files.

Following this assumption, there won't be consistency checks for the list of variables etc. the code will just crash.
"""

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


def progress(i,tot,bar_length=20,word=''):
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

def main():
    parser = argparse.ArgumentParser(description="Concatenate netCDF files that are in the given directory")
    parser.add_argument('path',help='full path to a folder containing netCDF files')
    parser.add_argument('--out',default='',help='full path to the directory where the output file will be saved, default to save as "path"')
    parser.add_argument('--prefix',default='',help='if given, only use files starting with the given prefix')
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
    if len(nc_list)<=1:
        sys.exit('No files to concatenate')
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

        outfile = '{}{}_{}.nc'.format(site_abbrv,start,end)
        outpath = os.path.join(args.out,outfile)
        print('Output netCDF file will be saved as',outpath)

        nspec = np.sum([ncin['time'].size for ncin in ncin_list]) # total size of the output time variable
        nprior = np.sum([ncin['prior_time'].size for ncin in ncin_list]) # total size of the output prior_time variable
        print('Output time dimension size:',nspec)
        print('Output prior_time dimension size:',nprior)

        ncout = stack.enter_context(netCDF4.Dataset(outpath,'w',nc_format='NETCDF4_CLASSIC'))
        ## copy attributes, dimensions, and variables metadata using the first file
        global_attributes = ncin_list[0].__dict__.copy()
        # update the history and file_creation
        global_attributes['file_creation'] = "Created with Python {}; the library netCDF4 {}; and the code concat_netcdf.py".format(platform.python_version(),netCDF4.__version__)
        global_attributes['history'] = "Created {} (UTC) from the concatenation of {} netCDF files".format(time.asctime(time.gmtime(time.time())),len(nc_list))
        ncout.setncatts(global_attributes)

        # copy dimensions
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
        nvar = len(ncin_list[0].variables)
        for name,variable in ncin_list[0].variables.items():
            progress(varcount,nvar,word=name)
            varcount += 1
            var = ncout.createVariable(name,variable.datatype,variable.dimensions)
            ncout[name].setncatts(ncin_list[0][name].__dict__)
            if 'a32' in variable.dimensions or 'specname' in variable.dimensions:
                var._Encoding = 'ascii'

            spec_count = 0
            prior_count = 0
            for ncin in ncin_list:
                time_size = ncin['time'].size
                prior_time_size = ncin['prior_time'].size

                if 'prior_time' in ncin[name].dimensions and 'a32' not in ncin[name].dimensions:
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