from __future__ import print_function

"""
Update some attributes in an existing netcdf file produce by write_netcdf.py, using a "site_info" file
"""

import os
import sys
import netCDF4
import pandas as pd
import numpy as np
import argparse
import json

def custom_update(nc_file,info_file):
    """
    Update the netcdf file using a given input file formatted like the site_info.json file hosted on tccon_data.org
    """
    siteID = os.path.basename(nc_file)[:2]
    with open(info_file,'r') as f:
        site_data = json.load(f)[siteID]
    site_data['release_lag'] = '{} days'.format(site_data['release_lag'])

    return site_data

def standard_update(nc_file):
    """
    Update the netcdf file using the site_info.json file hosted on tccon_data.org
    """
    # Will be updated once we figure out how the file will be hosted
    pass

def main():

    def file_choices(choices,file_name):
        """
        Function handler to check file extensions with argparse

        choices: tuple of accepted file extensions
        file_name: path to the file
        """
        ext = os.path.splitext(file_name)[1][1:]
        if ext not in choices:
            parser.error("file doesn't end with one of {}".format(choices))
        return file_name

    description = "Update some attributes in an existing netcdf file using a 'site_info' file, the file will be modified IN PLACE so make sure you have a backup if needed"
    parser = argparse.ArgumentParser(description=description,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('file',type=lambda file_name:file_choices(('nc'),file_name),help='Full path to input TCCON netcdf file, it will be EDITED IN PLACE')
    parser.add_argument('--info-file',help='Full path to a custom site_info.json file formatted as the file hosted at tccon_data.org/site_info.json, if not given the file hosted on tccondata.org will be used')

    args = parser.parse_args()

    nc_file = args.file

    if args.info_file:
        site_data = custom_update(nc_file,args.info_file)
    else:
        site_data = standard_update(nc_file)

    with netCDF4.Dataset(nc_file,'r+') as nc_data:
        print('Updates:')
        for key,val in site_data.items():
            setattr(nc_data,key,val)
            print('{:<20} {}'.format(key,val))

if __name__=='__main__':
    main()
