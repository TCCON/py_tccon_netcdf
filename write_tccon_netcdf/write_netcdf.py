from __future__ import print_function

"""
Use the official output files (.oof) to write the actually official output files (.nc)
"""

import os
import sys
import platform
import subprocess
import netCDF4
import pandas as pd
import numpy as np
import hashlib
import argparse
from collections import OrderedDict
import time
from datetime import datetime, timedelta
import re


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
        sys.stdout.write("\rPercent:[{0}] {1}%".format(hashes + spaces, int(round(percent * 100)))+" "*50+'\n')
    else:
        sys.stdout.write("\rPercent:[{0}] {1}%".format(hashes + spaces, int(round(percent * 100)))+"    "+str(i+1)+"/"+str(tot)+" "+word)
    sys.stdout.flush()


def md5(file_name):
    """
    Reads file_name and get its md5 sum, returns the hexdigest hash string

    file_name: full path to the file
    """
    hash_md5 = hashlib.md5()
    if 'telluric_linelists.md5' not in file_name:
        with open(file_name, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)      
        md5sum = hash_md5.hexdigest()
    else:
        with open(file_name,'r') as f:
            content = f.readlines()
        text = '\n'.join([line.split()[0] for line in content])+'\n'
        md5sum = hashlib.md5(text.encode('utf-8')).hexdigest()

    return md5sum


def checksum(file_name,hexdigest):
    """
    Compare the hash string hexdigest with the md5 sum of the file file_name

    hexdigest: hash string
    file_name: full path to the file
    """

    check = (md5(file_name) == hexdigest)

    if not check:
        print('/!\\ Checksum mismatch for',file_name)
        print('New:',md5(file_name))
        print('Old:',hexdigest)


def file_info(file_name):
    """
    Read the first line of a file and get the number of header lines and number of data columns

    file_name: full path to the file
    """
    with open(file_name,'r') as infile:
        nhead,ncol = [int(i) for i in infile.readline().strip().split()[:2]]
    nhead = nhead-1

    return nhead,ncol


def gravity(gdlat,altit):
    """
    This function is just the GGG/src/comn/gravity.f routine translated to python


    Input Parameters:
        gdlat       GeoDetric Latitude (degrees)
        altit       Geometric Altitude (km)
    
    Output Parameter:
        gravity     Effective Gravitational Acceleration (m/s2)
    
    Computes the effective Earth gravity at a given latitude and altitude.
    This is the sum of the gravitational and centripital accelerations.
    These are based on equation I.2.4-(17) in US Standard Atmosphere 1962
    The Earth is assumed to be an oblate ellipsoid, with a ratio of the
    major to minor axes = sqrt(1+con) where con=.006738
    This eccentricity makes the Earth's gravititational field smaller at
    the poles and larger at the equator than if the Earth were a sphere
    of the same mass. [At the equator, more of the mass is directly
    below, whereas at the poles more is off to the sides). This effect
    also makes the local mid-latitude gravity field not point towards
    the center of mass.
    
    The equation used in this subroutine agrees with the International
    Gravitational Formula of 1967 (Helmert's equation) within 0.005%.
    
    Interestingly, since the centripital effect of the Earth's rotation
    (-ve at equator, 0 at poles) has almost the opposite shape to the
    second order gravitational field (+ve at equator, -ve at poles),
    their sum is almost constant so that the surface gravity could be
    approximated (.07%) by the simple expression g=0.99746*GM/radius^2,
    the latitude variation coming entirely from the variation of surface
    r with latitude. This simple equation is not used in this subroutine.
    """

    d2r=3.14159265/180.0    # Conversion from degrees to radians
    gm=3.9862216e+14        # Gravitational constant times Earth's Mass (m3/s2)
    omega=7.292116E-05      # Earth's angular rotational velocity (radians/s)
    con=0.006738            # (a/b)**2-1 where a & b are equatorial & polar radii
    shc=1.6235e-03          # 2nd harmonic coefficient of Earth's gravity field 
    eqrad=6378178.0         # Equatorial Radius (m)

    gclat=np.arctan(np.tan(d2r*gdlat)/(1.0+con))  # radians

    radius=1000.0*altit+eqrad/np.sqrt(1.0+con*np.sin(gclat)**2)
    ff=(radius/eqrad)**2
    hh=radius*omega**2
    ge=gm/eqrad**2                      # = gravity at Re

    gravity=(ge*(1-shc*(3.0*np.sin(gclat)**2-1)/ff)/ff-hh*np.cos(gclat)**2)*(1+0.5*(np.sin(gclat)*np.cos(gclat)*(hh/ge+2.0*shc/ff**2))**2)

    return gravity


def read_mav(path):
    """
    read .mav files into a dictionary with spectrum filnames as keys (from each "Next spectrum" block in the .mav file)
    values are dataframes with the prior data
    """
    sys.stdout.write('Reading MAV file ...')
    sys.stdout.flush()
    DATA = OrderedDict()

    with open(path,'r') as infile:
        for i in range(3): # line[0] is gsetup version and line[1] is a "next spectrum" line

            line = infile.readline()
            if i == 1:
                spectrum = line.strip().split(':')[1]
        tropalt = float(infile.readline().split(':')[1])
        oblat = float(infile.readline().split(':')[1])
        vmr_time = (datetime.strptime(infile.readline().split()[0].split(os.sep)[-1].split('_')[1][:-1],'%Y%m%d%H')-datetime(1970,1,1)).total_seconds()

    nhead, ncol, nlev = [int(elem) for elem in line.split()]

    d = pd.read_csv(path,skiprows=nhead+1,delim_whitespace=True)
    d.rename(index=str,columns={'Height':'altitude','Temp':'temperature','Pres':'pressure','Density':'density'},inplace=True)

    mav_block = d[:nlev].apply(pd.to_numeric) # turn all the strings into numbers
    DATA[spectrum] = {
                        'data':mav_block[mav_block['altitude']>=0].copy(deep=True), # don't keep cell levels
                        'time':vmr_time,
                        'tropopause_altitude':tropalt,
    }
    DATA[spectrum]['data']['gravity'] = DATA[spectrum]['data']['altitude'].apply(lambda z: gravity(oblat,z))
    
    ispec = 1
    while True:
        block_id = ispec*nlev+(ispec-1)*7
        try:
            spectrum = d['temperature'][block_id].split(':')[1]
        except (KeyError, IndexError) as e:
            break

        tropalt = float(d['pressure'][block_id+2])
        oblat = float(d['pressure'][block_id+3])
        vmr_time = (datetime.strptime(d['altitude'][block_id+4].split(os.sep)[-1].split('_')[1][:-1],'%Y%m%d%H')-datetime(1970,1,1)).total_seconds()
        
        mav_block = d[block_id+7:block_id+7+nlev].apply(pd.to_numeric) # turn all the strings into numbers
        DATA[spectrum] = {
                            'data':mav_block[mav_block['altitude']>=0].copy(deep=True), # don't keep cell levels
                            'time':vmr_time,
                            'tropopause_altitude':tropalt,
        }
        DATA[spectrum]['data']['gravity'] = DATA[spectrum]['data']['altitude'].apply(lambda z: gravity(oblat,z))

        ispec += 1

    nlev = DATA[spectrum]['data']['altitude'].size # get nlev again without the cell levels
    sys.stdout.write(' DONE')
    return DATA, nlev


def main():
    wnc_version = 'write_netcdf.py (Version 1.0; 2019-11-15; SR)\n'
    print(wnc_version, sys.executable)

    try:
        GGGPATH = os.environ['GGGPATH']
    except:
        try:
            GGGPATH = os.environ['gggpath']
        except:
            print('You need to set a GGGPATH (or gggpath) environment variable')
            sys.exit()

    description = wnc_version + "This writes TCCON outputs in a NETCDF file"
    
    parser = argparse.ArgumentParser(description=description,formatter_class=argparse.RawTextHelpFormatter)
    
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
    
    parser.add_argument('file',type=lambda file_name:file_choices(('tav'),file_name),help='The .tav file')
    parser.add_argument('--format',default='NETCDF4_CLASSIC',choices=['NETCDF4_CLASSIC','NETCDF4'],help='the format of the NETCDF files')
    parser.add_argument('-r','--read-only',action='store_true',help="Convenience for python interactive shells; sys.exit() right after reading all the input files")

    args = parser.parse_args()

    nc_format = args.format
    classic = nc_format == 'NETCDF4_CLASSIC'
    print(nc_format,'format')

    # input and output file names
    tav_file = args.file
    mav_file = tav_file.replace('.tav','.mav')
    vav_file = tav_file.replace('.tav','.vav')
    asw_file = tav_file.replace('.tav','.asw')
    ada_file = vav_file+'.ada'
    aia_file = ada_file+'.aia'
    esf_file = aia_file+'.daily_error.out'
    oof_file = aia_file+'.oof'
    
    siteID = tav_file.split(os.sep)[-1][:2] # two letter site abbreviation
    qc_file = os.path.join(GGGPATH,'tccon','{}_qc.dat'.format(siteID))
    header_file = os.path.join(GGGPATH,'tccon','{}_oof_header.dat'.format(siteID))
    preavg_correction_file =  os.path.join(GGGPATH,'tccon','corrections_airmass_preavg.dat')
    postavg_correction_file =  os.path.join(GGGPATH,'tccon','corrections_airmass_postavg.dat')
    insitu_correction_file =  os.path.join(GGGPATH,'tccon','corrections_insitu_postavg.dat')
    lse_file = os.path.join(GGGPATH,'lse','gnd',tav_file.split(os.sep)[-1].replace('.tav','.lse'))
    pth_file = 'extract_pth.out'

    # need to check that the file ends with .col, not just that .col is in it, because
    # otherwise a .col elsewhere in the file name will cause a problem (e.g. if one is
    # open in vi)
    col_file_list = sorted([i for i in os.listdir(os.getcwd()) if i.endswith('.col')])

    if not col_file_list: # [] evaluates to False
        print('No .col files !')
        sys.exit()

    ## read data, I add the file_name to the data dictionaries for some of them

    # read site specific data from the tccon_netcdf repository
    code_dir = os.path.dirname(__file__) # path to the tccon_netcdf repository
    # the .apply and .rename bits just strip the columns from leading and tailing white spaces
    with open(os.path.join(code_dir,'site_info.txt'),'r') as f:
        c = f.read()
    site_data = eval(c)[siteID]
    site_data['release_lag'] = '{} days'.format(site_data['release_lag'])

    # multiggg.sh; use it to get the number of windows fitted and check they all have a .col file
    with open('multiggg.sh','r') as infile:
        content = [line for line in infile.readlines() if line[0]!=':' or line.strip()!=''] # the the file without blank lines or commented out lines starting with ':'
    ncol = len(content)
    if ncol!=len(col_file_list):
        print('/!\\ multiggg.sh has {} command lines but there are {} .col files'.format(ncol,len(col_file_list)))
        sys.exit()

    # averaging kernels
    ak_file_list = os.listdir(os.path.join(code_dir,'lamont_averaging_kernels'))
    ak_data = {} # will have keys as species 'h2o','co2' etc.
    for ak_file in ak_file_list:
        ak_file_path = os.path.join(code_dir,'lamont_averaging_kernels',ak_file)
        nhead,col = file_info(ak_file_path)
        ak_data[ak_file.split('_')[2]] = pd.read_csv(ak_file_path,delim_whitespace=True,skiprows=nhead)
    # check all pressure levels are the same
    check_ak_pres = np.array([(ak_data['co2']['P_hPa']==ak_data[gas]['P_hPa']).all() for gas in ak_data.keys()]).all()
    if not check_ak_pres:
        print('AK files have inconsistent pressure levels !')
        sys.exit()
    nlev_ak = ak_data['co2']['P_hPa'].size
    nsza_ak = ak_data['co2'].columns.size -1 # minus one because of the pressure column

    # read prior data
    prior_data, nlev = read_mav(mav_file)
    nprior = len(prior_data.keys())

    # read pth data
    nhead,ncol = file_info(pth_file)
    pth_data = pd.read_csv(pth_file,delim_whitespace=True,skiprows=nhead)
    pth_data.loc[:,'hout'] = pth_data['hout']*100.0 # convert fractional humidity to percent
    pth_data.loc[:,'hmod'] = pth_data['hmod']*100.0 # convert fractional humidity to percent

    # header file: it contains general information and comments.
    with open(header_file,'r') as infile:
        header_content = infile.read()

    # correction files: it contains the airmass dependent and independent correction factors for main target gases
    # there are three, two for airmass dependent correction, one before and average averaging. And one for airmass independent corrections
    nhead, ncol = file_info(preavg_correction_file)
    preavg_correction_data = pd.read_csv(preavg_correction_file,delim_whitespace=True,skiprows=nhead)

    nhead, ncol = file_info(postavg_correction_file)
    postavg_correction_data = pd.read_csv(postavg_correction_file,delim_whitespace=True,skiprows=nhead)

    nhead, ncol = file_info(insitu_correction_file)
    insitu_correction_data = pd.read_csv(insitu_correction_file,delim_whitespace=True,skiprows=nhead)

    # qc file: it contains information on some variables as well as their flag limits
    nhead, ncol = file_info(qc_file)
    qc_data = pd.read_fwf(qc_file,widths=[15,3,8,7,10,9,10,45],skiprows=nhead+1,names='Variable Output Scale Format Unit Vmin Vmax Description'.split())
    for key in ['Variable','Format','Unit']:
        qc_data[key] = [i.replace('"','') for i in qc_data[key]]

    # error scale factors: 
    nhead, ncol = file_info(esf_file)
    esf_data = pd.read_csv(esf_file,delim_whitespace=True,skiprows=nhead)

    # oof file: 'official output file', it contains data from other files, it isn't directly used here
    nhead, ncol = file_info(oof_file)
    oof_data = pd.read_csv(oof_file,delim_whitespace=True,skiprows=nhead)
    oof_data['file'] = oof_file
    site_info = pd.read_csv(oof_file,delim_whitespace=True,skiprows=lambda x: x in range(nhead-3) or x>=nhead-1) # has keys ['Latitude','Longitude','Altitude','siteID']

    # lse file: contains laser sampling error data
    nhead, ncol = file_info(lse_file)
    lse_data = pd.read_csv(lse_file,delim_whitespace=True,skiprows=nhead)
    lse_data['file'] = lse_file
    lse_data.rename(index=str,columns={'Specname':'spectrum'},inplace=True) # the other files use 'spectrum'

    # tav file: contains VSFs
    with open(tav_file,'r') as infile:
        nhead,ncol,nspec,naux = np.array(infile.readline().split()).astype(int)
    nhead = nhead-1
    tav_data = pd.read_csv(tav_file,delim_whitespace=True,skiprows=nhead)
    tav_data['file'] = tav_file
    nwin = int((ncol-naux)/2)
    speclength = tav_data['spectrum'].map(len).max() # use the longest spectrum file name length for the specname dimension

    # vav file: contains column amounts
    nhead, ncol = file_info(vav_file)
    vav_data = pd.read_csv(vav_file,delim_whitespace=True,skiprows=nhead)
    vav_data['file'] = vav_file

    # ada file: contains column-average dry-air mole fractions
    nhead, ncol = file_info(ada_file)
    ada_data = pd.read_csv(ada_file,delim_whitespace=True,skiprows=nhead)
    ada_data['file'] = ada_file
    
    # aia file: ada file with scale factor applied
    nhead, ncol = file_info(aia_file)
    aia_data = pd.read_csv(aia_file,delim_whitespace=True,skiprows=nhead)
    aia_data['file'] = aia_file

    ## check all files have the same spectrum lists
    check_spec = np.array([(data['spectrum']==vav_data['spectrum']).all() for data in [tav_data,ada_data,aia_data,oof_data]])
    if not check_spec.all():
        print('Files have inconsistent spectrum lists !')
        for data in [tav_data,ada_data,aia_data,oof_data]:
            print(len(data['spectrum']),'spectra in',data['file'][0])
        sys.exit()

    specdates = np.array([datetime(int(aia_data['year'][i]),1,1)+timedelta(days=aia_data['day'][i]-1) for i in range(nspec)])
    start_date = datetime.strftime(specdates[0],'%Y%m%d')
    end_date = datetime.strftime(specdates[-1],'%Y%m%d')

    private_nc_file = '{}{}_{}.private.nc'.format(siteID,start_date,end_date) # the final output file

    # make all the column names consistent between the different files
    for dataframe in [preavg_correction_data,postavg_correction_data,insitu_correction_data,qc_data,esf_data,oof_data,lse_data,vav_data,ada_data,aia_data]:
        dataframe.rename(str.lower,axis='columns',inplace=True) # all lower case
        if 'doy' in dataframe.columns: # all use 'day' and not 'doy'
            dataframe.rename(index=str,columns={'doy':'day'},inplace=True)
        if 'lon' in dataframe.columns:
            dataframe.rename(index=str,columns={'lon':'long'},inplace=True)

    qc_data['rsc'] = qc_data['scale'].copy()

    gas_list = [i for i in tav_data.columns[naux:] if ('_error' not in i) and ('file' not in i)] # gas names

    mchar = 0
    if 'spectrum' in tav_data.columns:
        mchar = 1

    # Let's try to be CF compliant: http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.pdf
    standard_name_dict = {
    'year':'year',
    'run':'run_number',
    'lat':'latitude',
    'long':'longitude',
    'hour':'decimal_hour',
    'azim':'solar_azimuth_angle',
    'asza':'astronomical_solar_zenith_angle',
    'day':'day_of_year',
    'wspd':'wind_speed',
    'wdir':'wind_direction',
    'graw':'spectrum_spectral_point_spacing',
    'tins':'instrument_internal_temperature',
    'tout':'atmospheric_temperature',
    'pins':'instrument_internal_pressure',
    'pout':'atmospheric_pressure',
    'hout':'atmospheric_humidity',
    'tmod':'model_atmospheric_temperature',
    'pmod':'model_atmospheric_pressure',
    'hmod':'model_atmospheric_humidity',
    'sia':'solar_intensity_average',
    'fvsi':'fractional_variation_in_solar_intensity',
    'zobs':'observation_altitude',
    'zmin':'pressure_altitude',
    'osds':'observer_sun_doppler_stretch',
    'gfit_version':'gfit_version',
    'gsetup_version':'gsetup_version',
    'fovi':'internal_field_of_view',
    'opd':'maximum_optical_path_difference',
    'rmsocl':'fit_rms_over_continuum_level',
    'cfampocl':'channel_fringe_amplitude_over_continuum_level',
    'cfperiod':'channel_fringe_period',
    'cfphase':'channel_fringe_phase',
    'nit':'number_of_iterations',
    'cl':'continuum_level',
    'ct':'continuum_tilt',
    'cc':'continuum_curvature',
    'fs':'frequency_shift',
    'sg':'solar_gas_shift',
    'zo':'zero_level_offset',
    'zpres':'pressure_altitude',
    'cbf':'continuum_basis_function_coefficient_{}',
    'ncbf':'number of continuum basis functions',
    'lsf':'laser_sampling_fraction',
    'lse':'laser_sampling_error',
    'lsu':'laser_sampling_error_uncertainty',
    'lst':'laser_sampling_error_correction_type',
    'dip':'dip',
    'mvd':'maximum_velocity_displacement',
    }

    checksum_var_list = ['config','apriori','runlog','levels','mav','ray','isotopologs','windows','telluric_linelists','solar']

    standard_name_dict.update({var+'_checksum':var+'_checksum' for var in checksum_var_list})

    long_name_dict = {key:val.replace('_',' ') for key,val in standard_name_dict.items()} # standard names without underscores
    
    """
    dimensionless and unspecified units will have empty strings
    we could use "1" for dimensionless units instead
    both empty string and 1 are recognized as dimensionless units by udunits
    but using 1 would differentiate actual dimensionless variables and variables with unspecified units
    """
    units_dict = {
    'year':'years',
    'run':'',
    'lat':'degrees_north',
    'long':'degrees_east',
    'hour':'hours',
    'azim':'degrees',
    'asza':'degrees',
    'day':'days',
    'wspd':'m.s-1',
    'wdir':'degrees',
    'graw':'cm-1',
    'tins':'degrees_Celsius',
    'tout':'degrees_Celsius',
    'pins':'hPa',
    'pout':'hPa',
    'hout':'%',
    'sia':'',
    'fvsi':'%',
    'zobs':'km',
    'zmin':'km',
    'osds':'ppm',
    'gfit_version':'',
    'gsetup_version':'',
    'fovi':'radians',
    'opd':'cm',
    'rmsocl':'%',
    'cfampocl':'',
    'cfperiod':'cm-1',
    'cfphase':'radians',
    'nit':'',
    'cl':'',
    'ct':'',
    'cc':'',
    'fs':'mK',
    'sg':'ppm',
    'zo':'%',
    'zpres':'km',
    'cbf':'',
    'ncbf':'',
    'prior_temperature':'degrees_Kelvin',
    'prior_density':'molecules.cm-3',
    'prior_pressure':'atm',
    'prior_altitude':'km',
    'prior_tropopause_altitude':'km',
    'prior_gravity':'m.s-2',
    'prior_h2o':'',
    'prior_hdo':'',
    'prior_co2':'ppm',
    'prior_n2o':'ppb',
    'prior_co':'ppb',
    'prior_ch4':'ppb',
    'prior_hf':'ppt',
    'prior_o2':'',
    }

    special_description_dict = {
        'lco2':' lco2 is the strong CO2 band centered at 4852.87 cm-1 and does not contribute to the xco2 calculation.',
        'wco2':' wco2 is used for the weak CO2 bands centered at 6073.5 and 6500.4 cm-1 and does not contribute to the xco2 calculation.',
        'th2o':' th2o is used for temperature dependent H2O windows and does not contribute to the xh2o calculation.',
        'luft':' luft is used for "dry air"',
    }

    if args.read_only:
        print('\nAll inputs read')
        sys.exit()

    if os.path.exists(private_nc_file):
        os.remove(private_nc_file)

    with netCDF4.Dataset(private_nc_file,'w',format=nc_format) as nc_data:
        
        ## global attributes
        
        # general TCCON
        nc_data.title = "Atmospheric trace gas column-average dry-air mole fractions retrieved from solar absorption spectra measured by ground based Fourier Transform Infrared Spectrometers that are part of the Total Carbon Column Observing Network (TCCON)"
        nc_data.source = "Products retrieved from solar absorption spectra using the GGG2019 software"
        nc_data.data_use_policy = "https://tccon-wiki.caltech.edu/Network_Policy/Data_Use_Policy"
        nc_data.auxiliary_data_description = "https://tccon-wiki.caltech.edu/Network_Policy/Data_Use_Policy/Auxiliary_Data"
        nc_data.description = '\n'+header_content
        nc_data.file_creation = "Created with Python {}; the library netCDF4 {}; and the code {}".format(platform.python_version(),netCDF4.__version__,wnc_version)
        nc_data.flag_info = 'The Vmin and Vmax attributes of the variables indicate the range of valid values.\nThe values comes from the xx_qc.dat file.\n the variable "flag" stores the index of the variable that contains out of range values.\nThe variable "flagged_var_name" stores the name of that variable'
        nc_data.more_information = "https://tccon-wiki.caltech.edu"
        nc_data.tccon_reference = "Wunch, D., G. C. Toon, J.-F. L. Blavier, R. A. Washenfelder, J. Notholt, B. J. Connor, D. W. T. Griffith, V. Sherlock, and P. O. Wennberg (2011), The total carbon column observing network, Philosophical Transactions of the Royal Society - Series A: Mathematical, Physical and Engineering Sciences, 369(1943), 2087-2112, doi:10.1098/rsta.2010.0240. Available from: http://dx.doi.org/10.1098/rsta.2010.0240"
        
        # site specific
        for key,val in site_data.items():
            setattr(nc_data,key,val)

        # other
        nc_data.number_of_spectral_windows = str(len(col_file_list))
       
        if os.path.isdir(os.path.join(GGGPATH,'.hg')): 
            proc = subprocess.Popen(['hg','summary'],cwd=GGGPATH,stdout=subprocess.PIPE)
            out, err = proc.communicate()
            gggtip = out.decode("utf-8")
        else:
            gggtip = "Could not find .hg in the GGG repository"
            print('\n',gggtip)
        nc_data.GGGtip = "The output of 'hg summary' from the GGG repository:\n"+gggtip
        nc_data.history = "Created {} (UTC)".format(time.asctime(time.gmtime(time.time())))

        ## create dimensions
        """
        NOTE: when setting the time dimension as unlimited I get a segmentation fault when writing the prior data
        If the time dimension is fixed the writing of the prior data is MUCH faster and does not lead to segmentation fault.
        This is a known issue from the netCDF4 library (or the C library it's built on) and writing to multidimensional variables with unlimited dimensions
        We can fix this later when they fix the problem.
        As far as I am aware this will only be an issue if people want to concatenate multiple netcdf files along the time dimension, they will have to turn the time dimension to an unlimited dimension first
        """
        nc_data.createDimension('time',nspec)
        nc_data.createDimension('prior_time',nprior)
        nc_data.createDimension('prior_altitude',nlev) # used for the prior profiles
        nc_data.createDimension('ak_pressure',nlev_ak)
        nc_data.createDimension('ak_sza',nsza_ak)

        if classic:
            nc_data.createDimension('specname',speclength)
            nc_data.createDimension('a32',32)

        ## create coordinate variables
        nc_data.createVariable('time',np.float64,('time',))
        nc_data['time'].standard_name = "time"
        nc_data['time'].long_name = "time"
        nc_data['time'].description = 'UTC time'
        nc_data['time'].units = 'seconds since 1970-01-01 00:00:00'
        nc_data['time'].calendar = 'gregorian'

        nc_data.createVariable('prior_time',np.float32,('prior_time'))
        nc_data['prior_time'].standard_name = "prior_time"
        nc_data['prior_time'].long_name = "prior time"
        nc_data['prior_time'].description = 'UTC time for the prior profiles, corresponds to GEOS5 times every 3 hours from 0 to 21'
        nc_data['prior_time'].units = 'seconds since 1970-01-01 00:00:00'
        nc_data['prior_time'].calendar = 'gregorian'

        nc_data.createVariable('prior_altitude',np.float32,('prior_altitude')) # this one doesn't change between priors
        nc_data['prior_altitude'].standard_name = '{}_profile'.format('prior_altitude')
        nc_data['prior_altitude'].long_name = nc_data['prior_altitude'].standard_name.replace('_',' ')
        nc_data['prior_altitude'].units = units_dict['prior_altitude']
        nc_data['prior_altitude'].description = "altitude levels for the prior profiles, these are the same for all the priors"
        nc_data['prior_altitude'][0:nlev] = prior_data[list(prior_data.keys())[0]]['data']['altitude'].values

        nc_data.createVariable('ak_pressure',np.float32,('ak_pressure'))
        nc_data['ak_pressure'].standard_name = "averaging_kernel_pressure_levels"
        nc_data['ak_pressure'].long_name = nc_data['ak_pressure'].standard_name.replace('_',' ')
        nc_data['ak_pressure'].description = "fixed pressure levels for the Lamont (OK, USA) column averaging kernels"
        nc_data['ak_pressure'].units = 'hPa'
        nc_data['ak_pressure'][0:nlev_ak] = ak_data['co2']['P_hPa'].values

        nc_data.createVariable('ak_sza',np.float32,('ak_sza'))
        nc_data['ak_sza'].standard_name = "averaging_kernel_solar_zenith_angles"
        nc_data['ak_sza'].long_name = nc_data['ak_sza'].standard_name.replace('_',' ')
        nc_data['ak_sza'].description = "fixed solar zenith angles for the Lamont (OK, USA) column averaging kernels"
        nc_data['ak_sza'].units = 'degrees'
        nc_data['ak_sza'][0:nsza_ak] = ak_data['co2'].columns[1:].values.astype(np.float32)     

        ## create variables

        # averaging kernels
        for gas in ak_data.keys():
            var = 'ak_{}'.format(gas)
            nc_data.createVariable(var,np.float32,('ak_sza','ak_pressure'))
            nc_data[var].standard_name = '{}_column_averaging_kernel'.format(gas)
            nc_data[var].long_name = nc_data[var].standard_name.replace('_',' ')
            nc_data[var].description = '{} column averaging kernel over Lamont (OK, USA)'.format(gas)
            nc_data[var].units = ''
            # write it now
            for i,sza in enumerate(ak_data[gas].columns[1:]):
                nc_data[var][i,0:nlev_ak] = ak_data[gas][sza].values
 
        # priors
        nc_data.createVariable('prior_index',np.int16,('time',))
        nc_data['prior_index'].standard_name = 'prior_index'
        nc_data['prior_index'].long_name = 'prior index'
        nc_data['prior_index'].units = ''
        nc_data['prior_index'].description = 'Index of the prior profile associated with each measurement, it can be used to sample the prior_ variables along the prior_time dimension'

        for var in ['temperature','pressure','density','gravity','h2o','hdo','co2','n2o','co','ch4','hf','o2']:
            prior_var = 'prior_{}'.format(var)

            nc_data.createVariable(prior_var,np.float32,('prior_time','prior_altitude'))

            nc_data[prior_var].standard_name = '{}_profile'.format(prior_var)
            nc_data[prior_var].long_name = nc_data[prior_var].standard_name.replace('_',' ')
            nc_data[prior_var].description = nc_data[prior_var].long_name
            nc_data[prior_var].units = units_dict[prior_var]
        
        nc_data.createVariable('prior_tropopause_altitude',np.float32,('time'))
        nc_data['prior_tropopause_altitude'].standard_name = 'prior_tropopause_altitude'
        nc_data['prior_tropopause_altitude'].long_name = 'prior tropopause altitude'
        nc_data['prior_tropopause_altitude'].description = 'altitude at which the gradient in the prior temperature profile becomes > -2 degrees per km'
        nc_data['prior_tropopause_altitude'].units = units_dict[prior_var]

        # checksums
        for var in checksum_var_list:
            if classic:
                checksum_var = nc_data.createVariable(var+'_checksum','S1',('time','a32'))
                checksum_var._Encoding = 'ascii'
            else:
                checksum_var = nc_data.createVariable(var+'_checksum',str,('time',))
            checksum_var.standard_name = standard_name_dict[var+'_checksum']
            checksum_var.long_name = long_name_dict[var+'_checksum']
            checksum_var.description = 'hexdigest hash string of the md5 sum of the {} file'.format(var)

        # code versions
        nc_data.createVariable('gfit_version',np.float32,('time',))
        nc_data['gfit_version'].description = "version number of the GFIT code that generated the data"
        nc_data['gfit_version'].standard_name = standard_name_dict['gfit_version']
        nc_data['gfit_version'].long_name_dict = long_name_dict['gfit_version']

        nc_data.createVariable('gsetup_version',np.float32,('time',))
        nc_data['gsetup_version'].description = "version number of the GSETUP code that generated the priors"
        nc_data['gsetup_version'].standard_name = standard_name_dict['gsetup_version']
        nc_data['gsetup_version'].long_name_dict = long_name_dict['gsetup_version']

        # flags
        nc_data.createVariable('flag',np.int16,('time',))
        nc_data['flag'].description = 'data quality flag, 0 = good'
        nc_data['flag'].standard_name = 'quality_flag'
        nc_data['flag'].long_name = 'quality flag'

        if classic:
            v = nc_data.createVariable('flagged_var_name','S1',('time','a32'))
            v._Encoding = 'ascii'
        else:
            nc_data.createVariable('flagged_var_name',str,('time',))
        nc_data['flagged_var_name'].description = 'name of the variable that caused the data to be flagged; empty string = good'
        nc_data['flagged_var_name'].standard_name = 'flagged_variable_name'
        nc_data['flagged_var_name'].long_name = 'flagged variable name'

        # spectrum file names
        if classic:
            v = nc_data.createVariable('spectrum','S1',('time','specname'))
            v._Encoding = 'ascii'
        else:
            nc_data.createVariable('spectrum',str,('time',))
        nc_data['spectrum'].standard_name = 'spectrum_file_name'
        nc_data['spectrum'].long_name = 'spectrum file name'
        nc_data['spectrum'].description = 'spectrum file name'

        for i,specname in enumerate(aia_data['spectrum'].values):
            nc_data['spectrum'][i] = specname        

        # auxiliary variables
        aux_var_list = [tav_data.columns[i] for i in range(1,naux)]
        for var in aux_var_list: 
            qc_id = list(qc_data['variable']).index(var)
            #digit = int(qc_data['format'][qc_id].split('.')[-1])
            if var in ['year','day']:
                var_type = np.int16
            else:
                var_type = np.float32 
            nc_data.createVariable(var,var_type,('time',))#,zlib=True)#,least_significant_digit=digit)
            # set attributes using the qc.dat file
            nc_data[var].description = qc_data['description'][qc_id]
            nc_data[var].units = qc_data['unit'][qc_id].replace('(','').replace(')','').strip()
            nc_data[var].vmin = qc_data['vmin'][qc_id]
            nc_data[var].vmax = qc_data['vmax'][qc_id]
            if var in standard_name_dict.keys():
                nc_data[var].standard_name = standard_name_dict[var]
                nc_data[var].long_name = long_name_dict[var]
                nc_data[var].units = units_dict[var] # reset units here for some of the variables in the qc_file using UDUNITS compatible units

        nc_data['hour'].description = 'Fractional UT hours (zero path difference crossing time)'

        # get model surface values from the output of extract_pth.f
        for key,val in {'tmod':'tout','pmod':'pout','hmod':'hout'}.items(): # use a mapping to the equivalent runlog variables to querry their qc.dat info
            qc_id = list(qc_data['variable']).index(val)
            #digit = int(qc_data['format'][qc_id].split('.')[-1])
            var_type = np.float32 
            nc_data.createVariable(key,var_type,('time'))#,zlib=True)#,least_significant_digit=digit)
            nc_data[key].description = 'model {}'.format(qc_data['description'][qc_id].lower())
            nc_data[key].vmin = qc_data['vmin'][qc_id]
            nc_data[key].vmax = qc_data['vmax'][qc_id]
            if key in standard_name_dict.keys():
                nc_data[key].standard_name = standard_name_dict[key]
                nc_data[key].long_name = long_name_dict[key]
                nc_data[key].units = units_dict[val]

        # averaged variables (from the different windows of each species)
        main_var_list = [tav_data.columns[i] for i in range(naux,len(tav_data.columns)-1)]  # minus 1 because I added the 'file' column
        for var in main_var_list:
            xvar = 'x'+var
            qc_id = list(qc_data['variable']).index(xvar)

            #digit = int(qc_data['format'][qc_id].split('.')[-1])
            nc_data.createVariable(xvar,np.float32,('time',))#,zlib=True)#,least_significant_digit=digit)
            nc_data[xvar].standard_name = xvar
            nc_data[xvar].long_name = xvar.replace('_',' ')
            nc_data[xvar].description = qc_data['description'][qc_id]
            nc_data[xvar].units = qc_data['unit'][qc_id].replace('(','').replace(')','').strip()
            nc_data[xvar].vmin = qc_data['vmin'][qc_id]
            nc_data[xvar].vmax = qc_data['vmax'][qc_id]
            nc_data[xvar].precision = qc_data['format'][qc_id]

            nc_data.createVariable('vsf_'+var,np.float32,('time',))
            nc_data['vsf_'+var].description = var+" Volume Scale Factor."
            nc_data['vsf_'+var][:] = tav_data[var].values
            
            nc_data.createVariable('column_'+var,np.float32,('time',))
            nc_data['column_'+var].description = var+' column average.'
            nc_data['column_'+var].units = 'molecules.m-2'
            nc_data['column_'+var][:] = vav_data[var].values

            nc_data.createVariable('ada_x'+var,np.float32,('time',))
            if 'error' in var:
                nc_data['ada_x'+var].description = 'uncertainty associated with ada_x{}'.format(var.replace('_error',''))
            else:
                nc_data['ada_x'+var].description = var+' column-average dry-air mole fraction computed after airmass dependence is removed, but before scaling to WMO.'
            for key in special_description_dict.keys():
                if key in var:
                    for nc_var in [nc_data[xvar],nc_data['vsf_'+var],nc_data['column_'+var],nc_data['ada_x'+var]]:
                        nc_var.description += special_description_dict[key]
            nc_data['ada_x'+var].units = ""
            nc_data['ada_x'+var][:] = ada_data['x'+var].values

        # lse data
        lse_description = {
                    'lst':'The type of LSE correction applied (0=none; 1=InGaAs (disabled); 2=Si; 3=Dohe et al. (disabled); 4=Other (disabled))',
                    'lse':'Laser sampling error (shift)',
                    'lsu':'Laser sampling error uncertainty',
                    'lsf':'laser sampling fraction',
                    'dip':'A proxy for nonlinearity - the dip at ZPD in the smoothed low-resolution interferogram',
                    'mvd':'Maximum velocity displacement - a measure of how smoothly the scanner is running',
                    }
        common_spec = np.intersect1d(aia_data['spectrum'],lse_data['spectrum'],return_indices=True)[2]
        for var in lse_description.keys():
            nc_data.createVariable(var,np.float32,('time',))
            nc_data[var].standard_name = standard_name_dict[var]
            nc_data[var].long_name = long_name_dict[var]
            nc_data[var].description = lse_description[var]
            nc_data[var][:] = lse_data[var][common_spec].values

        # preavg corrections
        for var in preavg_correction_data['gas']:
            for key in preavg_correction_data.columns[1:]:
                varname = 'preavg_{}_{}'.format(var,key)
                nc_data.createVariable(varname,np.float32,('time',))
                nc_data[varname][:] = preavg_correction_data[key][list(preavg_correction_data['gas']).index(var)] # write directly
        # postavg corrections
        for var in postavg_correction_data['gas']:
            for key in postavg_correction_data.columns[1:]:
                varname = 'postavg_{}_{}'.format(var,key)
                nc_data.createVariable(varname,np.float32,('time',))
                nc_data[varname][:] = postavg_correction_data[key][list(postavg_correction_data['gas']).index(var)] # write directly

        # insitu corrections
        for var in insitu_correction_data['gas']:
            for key in insitu_correction_data.columns[1:]:
                varname = 'postavg_{}_{}'.format(var,key)
                nc_data.createVariable(varname,np.float32,('time',))
                nc_data[varname][:] = insitu_correction_data[key][list(insitu_correction_data['gas']).index(var)] # write directly

        ## write data
        sys.stdout.write('\nWriting prior data ...')
        sys.stdout.flush()
        factor = {'temperature':1.0,'pressure':1.0,'density':1.0,'gravity':1.0,'1h2o':1.0,'1hdo':1.0,'1co2':1e6,'1n2o':1e9,'1co':1e9,'1ch4':1e9,'1hf':1e12,'1o2':1.0}
        prior_spec_list = list(prior_data.keys())
        prior_spec_gen = (spectrum for spectrum in prior_spec_list)
        prior_spectrum = next(prior_spec_gen)
        next_spectrum = next(prior_spec_gen)
        prior_index = 0
        
        spec_list = nc_data['spectrum'][:]
        
        for spec_id,spectrum in enumerate(spec_list):
            if spectrum==next_spectrum:
                prior_spectrum = next_spectrum
                try:
                    next_spectrum = next(prior_spec_gen)
                except StopIteration:
                    pass
                
                prior_index += 1

            nc_data['prior_index'][spec_id] = prior_index

        for prior_spec_id, prior_spectrum in enumerate(prior_spec_list):
            for var in ['temperature','pressure','density','gravity','1h2o','1hdo','1co2','1n2o','1co','1ch4','1hf','1o2']:
                prior_var = 'prior_{}'.format(var.strip('1'))

                nc_data[prior_var][prior_spec_id,0:nlev] = factor[var]*prior_data[prior_spectrum]['data'][var].values

            nc_data['prior_time'][prior_spec_id] = prior_data[prior_spectrum]['time']
            nc_data['prior_tropopause_altitude'][prior_spec_id] = prior_data[prior_spectrum]['tropopause_altitude']

        sys.stdout.write('\rWriting prior data ... DONE')

        # update data with new scale factors and determine flags
        esf_id = 0
        nflag = 0
        for esf_id in range(esf_data['year'].size):
                    
            # indices to slice the data for the concerned spectra
            start = np.sum(esf_data['n'][:esf_id])
            end = start + esf_data['n'][esf_id] 

            """
            If new day, read in the daily error scale factors and compute
            new scale factors (RSC) as weighted averages of the a priori
            ESF factors from the pa_qc.dat file, and the daily values.
            A priori ESF values are the ratio of the xx_error/xxa scale factors
            read in from the pa_qc.dat file, with 100% uncertainties assumed.
            """     
            for gas in gas_list:
                xgas = 'x'+gas
                qc_id = list(qc_data['variable']).index(xgas)
                apesf = qc_data['scale'][qc_id+1]/qc_data['scale'][qc_id]
                qc_data.loc[qc_data.index==qc_id+1,'rsc'] = qc_data['scale'][qc_id]*(1.0/apesf+esf_data[xgas][esf_id]/esf_data[xgas+'_error'][esf_id]**2)/(1.0/apesf**2+1.0/esf_data[xgas+'_error'][esf_id]**2)

            """
            Look within each data record to see if any of the data values are
            outside their VMIN to VMAX range. If so, set eflag to the index of
            the variable that was furthest out of range. Then write out the data.
            """
            eflag = np.zeros(end-start)
            kmax = np.zeros(end-start)
            dmax = np.zeros(end-start)
            for var_id,var in enumerate(aia_data.columns[mchar:-1]):

                if len(aia_data[var][aia_data[var]>=9e29]) >= 1:
                    print('Missing value found (>=9e29) for variable {}.\nEnding Program'.format(var))
                    print('You may need to remove missing .col files from multiggg.sh and rerun post_processing.sh')
                    sys.exit()

                qc_id = list(qc_data['variable']).index(var)
                digit = int(qc_data['format'][qc_id].split('.')[-1])
                
                nc_data[var][start:end] = np.round(aia_data[var][start:end].values*qc_data['rsc'][qc_id],digit)

                dev = abs( (qc_data['rsc'][qc_id]*aia_data[var][start:end].values-qc_data['vmin'][qc_id])/(qc_data['vmax'][qc_id]-qc_data['vmin'][qc_id]) -0.5 )
                
                kmax[dev>dmax] = qc_id+1 # add 1 here, otherwise qc_id starts at 0 for 'year'
                dmax[dev>dmax] = dev[dev>dmax]

            eflag[dmax>0.5] = kmax[dmax>0.5]
            
            # write the flagged variable index
            nc_data['flag'][start:end] = [int(i) for i in eflag]

            # write the flagged variable name
            for i in range(start,end):
                if eflag[i-start] == 0:
                    nc_data['flagged_var_name'][i] = ""
                else:
                    nc_data['flagged_var_name'][i] = qc_data['variable'][eflag[i-start]-1]

        nflag = np.count_nonzero(nc_data['flag'][:])

        # time      
        nc_data['year'][:] = np.round(aia_data['year'][:].values-aia_data['day'][:].values/365.25)
        nc_data['day'][:] = np.round(aia_data['day'][:].values-aia_data['hour'][:].values/24.0)
        nc_data['time'][:] = np.array([elem.total_seconds() for elem in (specdates-datetime(1970,1,1))])

        # write data from .col and .cbf files
        print('\n\nWriting data:')
        for col_id,col_file in enumerate(col_file_list):

            cbf_file = col_file.replace('.col','.cbf')
            with open(cbf_file,'r') as infile:
                content = infile.readlines()
            nhead,ncol = file_info(cbf_file)
            headers = content[nhead].split()
            ncbf = len(headers)-4
            if ncbf>0:
                widths = [speclength+2,8,9,7,12]+[9]*(ncbf-1)
            else:
                widths = [speclength+2,8,9,7]
            cbf_data = pd.read_fwf(cbf_file,widths=widths,names=headers,skiprows=nhead+1)
            cbf_data.rename(str.lower,axis='columns',inplace=True)
            cbf_data.rename(index=str,columns={'cfamp/cl':'cfampocl'},inplace=True)
            cbf_data.rename(index=str,columns={'spectrum_name':'spectrum'},inplace=True)

            gas_XXXX = col_file.split('.')[0] # gas_XXXX, suffix for nc_data variable names corresponding to each .col file (i.e. VSF_h2o from the 6220 co2 window becomes co2_6220_VSF_co2)

            nhead,ncol = file_info(col_file)

            # read col_file headers
            with open(col_file,'r') as infile:
                content = infile.readlines()[1:nhead]
            gfit_version, gsetup_version = content[:2]
            gfit_version = gfit_version.strip().split()[2]
            gsetup_version = gsetup_version.strip().split()[2]

            if col_file == col_file_list[0]:
                # check that the checksums are right for the files listed in the .col file header
                checksum_dict = OrderedDict((key+'_checksum',None) for key in checksum_var_list)
                # If a line begins with a 32-character MD5 hash, then one or more spaces, then
                # a non-whitespace character, verify the checksum. That corresponds to a line like:
                # 
                #   34136d7d03967a662edc7b0c92b984f1  /home/jlaugh/GGG/ggg-my-devel/config/data_part.lst
                #
                # If not, then there's no checksum or no file following it.
                content_lines = [line for line in content if re.match(r'[a-f0-9]{32}\s+[^\s]', line)]
                for i,line in enumerate(content_lines):
                    csum,fpath = line.split()
                    checksum(fpath,csum)

                    checksum_dict[checksum_var_list[i]+'_checksum'] = csum

                nc_data['gfit_version'][:] = gfit_version
                nc_data['gsetup_version'][:] = gsetup_version
                for var in checksum_var_list:
                    checksum_var = var+'_checksum'
                    for i in range(aia_data['spectrum'].size):
                        nc_data[checksum_var][i] = checksum_dict[checksum_var]

            # read col_file data
            with open(col_file,'r') as infile:
                content = infile.readlines()
            ggg_line = content[nhead-1]
            ngas = len(ggg_line.split(':')[-1].split())
            widths = [speclength+1,3,6,6,5,6,6,7,7,8]+[7,11,10,8]*ngas # the fixed widths for each variable so we can read with pandas.read_fwf, because sometimes there is no whitespace between numbers
            headers = content[nhead].split()

            col_data = pd.read_fwf(col_file,widths=widths,names=headers,skiprows=nhead+1)
            col_data.rename(str.lower,axis='columns',inplace=True)
            col_data.rename(index=str,columns={'rms/cl':'rmsocl'},inplace=True)
            if not all(col_data['spectrum'].values == vav_data['spectrum'].values):
                print('\nMismatch between .col file spectra and .vav spectra')
                print('col file:',col_file)
                continue # contine or exit here ? Might not need to exit if we can add in the results from the faulty col file afterwards
            if not all(col_data['spectrum'].values == cbf_data['spectrum'].values) and 'luft' not in col_file: # luft has no cbfs
                print('\nMismatch between .col file spectra and .cbf spectra')
                print('col file:',col_file)
                continue # contine or exit here ? Might not need to exit if we can add in the results from the faulty col file afterwards

            # create window specific variables
            for var in col_data.columns[1:]: # skip the first one ("spectrum")
                varname = '_'.join([gas_XXXX,var])
                nc_data.createVariable(varname,np.float32,('time',))
                if var in standard_name_dict.keys():
                    nc_data[varname].standard_name = standard_name_dict[var]
                    nc_data[varname].long_name = long_name_dict[var]

                nc_data[varname][:] = col_data[var].values
                if '_' in var:
                    nc_data[varname].description = '{} {} retrieved from the {} window centered at {} cm-1.'.format(var.split('_')[1],var.split('_')[0],gas_XXXX.split('_')[0],gas_XXXX.split('_')[1])

                    try:
                        iso = int(var.split('_')[1][0])
                    except:
                        pass
                    else:
                        iso_gas = var.split('_')[1]
                        if iso in [1,2,3]:
                            sup = ['st','nd','rd'][iso-1]
                        else:
                            sup = 'th'
                        nc_data[varname].description += "{} is the {}{} isotopolog of {} as listed in GGG's isotopologs.dat file.".format(iso_gas,iso,sup,iso_gas[1:])
                else:
                    nc_data[varname].description = '{} retrieved from the {} window centered at {} cm-1.'.format(var,gas_XXXX.split('_')[0],gas_XXXX.split('_')[1])
                for key in special_description_dict.keys():
                    if key in varname:
                        nc_data[varname].description += special_description_dict[key]
            
            # add data from the .cbf file
            ncbf_var = '{}_ncbf'.format(gas_XXXX)
            nc_data.createVariable(ncbf_var,np.int32,('time',))
            nc_data[ncbf_var][:] = len(cbf_data.columns)-1 # minus 1 because of the spectrum name column
            for var in cbf_data.columns[1:]: # don't use the 'Spectrum' column
                varname = '_'.join([gas_XXXX,var])
                nc_data.createVariable(varname,np.float32,('time',))
                if '_' in var:
                    nc_data[varname].standard_name = standard_name_dict[var.split('_')[0]].format(var.split('_')[1])
                    nc_data[varname].long_name = long_name_dict[var.split('_')[0]].format(var.split('_')[1])
                else:
                    nc_data[varname].standard_name = standard_name_dict[var]
                    nc_data[varname].long_name = long_name_dict[var]
                    nc_data[varname].units = units_dict[var]
                nc_data[varname][:] = cbf_data[var].values

            progress(col_id,len(col_file_list),word=col_file)


        # read the data from missing_data.txt and update data with fill values to the netCDF4 default fill value
        """
        It is a dictionary with siteID as keys, values are dictionaries of variable:fill_value

        If a site has different null values defined for different time period the key has format siteID_ii_YYYYMMDD_YYYYMMDD
        with ii just the period index (e.g. 01 ) so that they come in order when the keys get sorted
        """
        with open(os.path.join(code_dir,'missing_data.txt'),'r') as f:
            c = f.read()
        missing_data = eval(c)
        missing_data = {key:val for key,val in missing_data.items() if siteID in key}
        if len(missing_data.keys())>1: # if there are different null values for different time periods
            time_period_list = sorted(missing_data.keys())
            for time_period in time_period_list:
                start,end = [(datetime.strptime(elem,'%Y%m%d')-datetime(1970,1,1)).total_seconds for elem in time_period.split('_')[2:]]
                replace_time_ids = set(np.where((start<nc_data['time']) & (nc_data['time']<end))[0])
                for var in missing_data[time_period]:
                    replace_val_ids = set(np.where(nc_data[var]==missing_data[time_period][var])[0])
                    replace_ids = replace_time_ids.intersection(replace_val_ids) # indices for data equal to the fill value in the given time period
                    print('Convert fill value for',var,'from',missing_data[time_period][var],'to',netCDF4.default_fillvals['f4'],'between',str(netCDF4.num2date(start,units=nc_data['time'].units,calendar=nc_data['time'].calendar)),'and',str(netCDF4.num2date(end,units=nc_data['time'].units,calendar=nc_data['time'].calendar)))
                    for id in replace_ids:
                        nc_data[var][id] = netCDF4.default_fillvals['f4'] 
        elif len(missing_data.keys())==1:
            missing_data = missing_data[siteID]
            for var in missing_data:
                replace_ids = list(np.where(nc_data[var]==missing_data[var])[0])
                print('Convert fill value for',var,'to',netCDF4.default_fillvals['f4'])
                for id in replace_ids:
                    nc_data[var][id] = netCDF4.default_fillvals['f4']

    print('\nFinished writing',private_nc_file,'{:.2f}'.format(os.path.getsize(private_nc_file)/1e6),'MB')

    public_nc_file = '{}{}_{}.public.nc'.format(siteID,start_date,end_date)
    with netCDF4.Dataset(private_nc_file,'r') as private_data, netCDF4.Dataset(public_nc_file,'w',format=nc_format) as public_data:
        ## copy all the metadata
        private_attributes = private_data.__dict__
        public_attributes = private_attributes.copy()
        for attr in ['flag_info','release_lag','GGGtip','number_of_spectral_windows']: # remove attributes that are only meant for private files
            public_attributes.pop(attr)

        # update the history to indicate that the public file is a subset of the private file
        public_attributes['history'] = "Created {} (UTC) from the engineering file {}".format(time.asctime(time.gmtime(time.time())),private_nc_file.split(os.sep)[-1])

        public_data.setncatts(public_attributes)

        # get indices of data to copy based on the release_lag
        release_lag = int(private_data.release_lag.split()[0])
        last_public_time = (datetime.utcnow()-datetime(1970,1,1)).total_seconds() - timedelta(days=release_lag).total_seconds()
        release_ids = np.where(private_data['time'][:]<last_public_time)[0]

        # get indices of data with flag = 0
        no_flag_ids = np.where(private_data['flag'][:]==0)[0]
        
        # get the intersection of release_lag and flag constrained indices
        public_ids = list(set(release_ids).intersection(set(no_flag_ids)))

        public_slice = np.array([i in public_ids for i in np.arange(nspec) ]) # boolean array to slice the private variables on the public ids

        nspec_public = len(public_ids)

        ## copy dimensions
        for name, dimension in private_data.dimensions.items():
            if name == 'time':
                public_data.createDimension(name, nspec_public)
            else:
                public_data.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))

        ## copy variables based on the info in public_variables.txt
        with open(os.path.join(code_dir,'public_variables.txt')) as f:
            c = f.read()
        public_variables = eval(c)

        for name,variable in private_data.variables.items():
            
            contain_check = np.array([elem in name for elem in public_variables['contains']]).any()
            startswith_check = np.array([name.startswith(elem) for elem in public_variables['startswith']]).any()
            endswith_check = np.array([name.endswith(elem) for elem in public_variables['endswith']]).any()
            isequalto_check = np.array([name==elem for elem in public_variables['isequalto']]).any()

            excluded = np.array([elem in name for elem in public_variables['exclude']]).any()

            public = np.array([contain_check,isequalto_check,startswith_check,endswith_check]).any() and not excluded
            
            if public:
                if 'time' in variable.dimensions: # only the variables along the 'time' dimension need to be sampled with public_ids
                    public_data.createVariable(name, variable.datatype, variable.dimensions)
                    public_data[name][:] = private_data[name][public_slice]
                else:
                    public_data.createVariable(name, variable.datatype, variable.dimensions)
                    public_data[name][:] = private_data[name][:]
                # copy variable attributes all at once via dictionary
                public_data[name].setncatts(private_data[name].__dict__)
    print('Finished writing',public_nc_file,'{:.2f}'.format(os.path.getsize(public_nc_file)/1e6),'MB')


if __name__=='__main__': # execute only when the code is run by itself, and not when it is imported
    main()