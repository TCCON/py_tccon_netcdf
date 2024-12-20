from __future__ import print_function

"""
Update some attributes in an existing netcdf file produce by write_netcdf.py, using a "site_info" file
"""

import argparse
import json
import os
import sys
from netrc import netrc
import netCDF4
import pandas as pd
from pathlib import Path
import numpy as np
import requests

class UserError(Exception):
    pass


def custom_update(nc_file,info_file):
    """
    Update the netcdf file using a given input file formatted like the site_info.json file hosted on tccon_data.org
    """
    with open(info_file,'r') as f:
        site_data = json.load(f)
    return get_formatted_site_info(nc_file, site_data)

def standard_update(nc_file):
    """
    Update the netcdf file using the site_info.json file hosted on tccon_data.org
    """
    netrc_file = Path('~/.netrc').expanduser()
    if not netrc_file.exists():
        raise UserError('Must have a ~/.netrc file with TCCON partner login credentials to tccondata.org')
    logins = netrc(netrc_file)
    if 'tccondata.org' not in logins.hosts:
        raise UserError('Must have a ~/.netrc file with TCCON partner login credentials to tccondata.org')
        
    username, _, password = logins.hosts['tccondata.org']
    
    r = requests.get('https://tccondata.org/2b-private-qc/site_info.json', auth=(username, password))
    if r.status_code != 200:
        raise UserError('Unable to retrieve the site_info.json file from tccondata.org')

    site_data = json.loads(r.content)
    return get_formatted_site_info(nc_file, site_data)


def get_formatted_site_info(nc_file, json_dict):
    siteID = os.path.basename(nc_file)[:2]
    site_data = json_dict[siteID]
    site_data['release_lag'] = '{} days'.format(site_data['release_lag'])

    return site_data
    

def driver():

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
    parser.add_argument('--pdb', action='store_true', help='Launch Python debugger')

    args = parser.parse_args()
    if args.pdb:
        import pdb
        pdb.set_trace()

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


def main():
    try:
        driver()
    except Exception as e:
        print('Error:', e.args[0], file=sys.stderr)
        sys.exit(1)


if __name__=='__main__':
    main()
