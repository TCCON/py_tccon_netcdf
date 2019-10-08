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
	with open(file_name, "rb") as f:
		for chunk in iter(lambda: f.read(4096), b""):
			hash_md5.update(chunk)
	return hash_md5.hexdigest()

def checksum(file_name,hexdigest):
	"""
	Compare the hash string hexdigest with the md5 sum of the file file_name

	hexdigest: hash string
	file_name: full path to the file
	"""

	check = (md5(file_name) == hexdigest)

	if not check:
		print('/!\\ Checksum mismatch for',fpath)
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

if __name__=='__main__': # execute only when the code is run by itself, and not when it is imported

	wnc_version = 'write_netcdf.py (Version 1.0; 2019-10-07; SR)'
	print(wnc_version)

	try:
		GGGPATH = os.environ['GGGPATH']
	except:
		try:
			GGGPATH = os.environ['gggpath']
		except:
			print('You need to set a GGGPATH (or gggpath) environment variable')
			sys.exit()

	description = wnc_version + "\nThis writes TCCON outputs in a NETCDF file"
	
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

	args = parser.parse_args()

	# input and output file names
	tav_file = args.file
	vav_file = tav_file.replace('.tav','.vav')
	asw_file = tav_file.replace('.tav','.asw')
	ada_file = vav_file+'.ada'
	aia_file = ada_file+'.aia'
	esf_file = aia_file+'.daily_error.out'
	oof_file = aia_file+'.oof'
	
	siteID = tav_file.split(os.sep)[-1][:2] # two letter site abbreviation
	qc_file = os.path.join(GGGPATH,'tccon','{}_qc.dat'.format(siteID))
	header_file = os.path.join(GGGPATH,'tccon','{}_oof_header.dat'.format(siteID))
	correction_file =  os.path.join(GGGPATH,'tccon','corrections.dat')
	lse_file = os.path.join(GGGPATH,'lse','gnd',tav_file.split(os.sep)[-1].replace('.tav','.lse'))
	nc_file = tav_file.replace('.tav','.nc') # the final output file

	col_file_list = sorted([i for i in os.listdir(os.getcwd()) if '.col' in i])
	map_file_list = sorted([i for i in os.listdir(os.getcwd()) if '.map' in i])

	if not col_file_list: # [] evaluates to False
		print('No .col files !')
		sys.exit()
	if not map_file_list:
		print('No .map files !')
		sys.exit()

	## read data, I add the file_name to the data dictionaries for some of them

	# read site specific data from the tccon_netcdf repository
	code_dir = os.path.dirname(__file__) # path to the tccon_netcdf repository
	# the .apply and .rename bits just strip the columns from leading and tailing white spaces
	site_data = pd.read_csv(os.path.join(code_dir,'site_list.txt'),delimiter='|',encoding='latin-1').apply(lambda x: x.str.strip() if x.dtype == "object" else x).rename(columns=lambda x: x.strip()).rename(str.lower,axis='columns')
	site_data = site_data[site_data['id']==siteID].reset_index().loc[0].drop('index') # just keep data for the current site
	site_data['releaselag'] = '{} days'.format(site_data['releaselag'])

	# multiggg.sh
	with open('multiggg.sh','r') as infile:
		content = [line for line in infile.readlines() if line[0]!=':' or line.strip()!=''] # the the file without blank lines or commented out lines starting with ':'
	ncol = len(content)
	if ncol!=len(col_file_list):
		print('/!\\ multiggg.sh has {} command lines but there are {} .col files'.format(ncol,len(col_file_list)))
		sys.exit()

	# header file
	with open(header_file,'r') as infile:
		header_content = infile.read()

	# correction file
	nhead, ncol = file_info(correction_file)
	correction_data = pd.read_csv(correction_file,delim_whitespace=True,skiprows=nhead)

	# qc file
	nhead, ncol = file_info(qc_file)
	qc_data = pd.read_fwf(qc_file,widths=[15,3,8,7,10,9,10,45],skiprows=nhead+1,names='Variable Output Scale Format Unit Vmin Vmax Description'.split())
	for key in ['Variable','Format','Unit']:
		qc_data[key] = [i.replace('"','') for i in qc_data[key]]

	# error scale factors
	nhead, ncol = file_info(esf_file)
	esf_data = pd.read_csv(esf_file,delim_whitespace=True,skiprows=nhead)

	# oof file
	nhead, ncol = file_info(oof_file)
	oof_data = pd.read_csv(oof_file,delim_whitespace=True,skiprows=nhead)
	oof_data['file'] = oof_file
	site_info = pd.read_csv(oof_file,delim_whitespace=True,skiprows=lambda x: x in range(nhead-3) or x>=nhead-1) # has keys ['Latitude','Longitude','Altitude','siteID']

	# lse file
	nhead, ncol = file_info(lse_file)
	lse_data = pd.read_csv(lse_file,delim_whitespace=True,skiprows=nhead)
	lse_data['file'] = lse_file
	lse_data.rename(index=str,columns={'Specname':'spectrum'},inplace=True) # the other files use 'spectrum'

	# tav file
	with open(tav_file,'r') as infile:
		nhead,ncol,nrow,naux = np.array(infile.readline().split()).astype(int)
	nhead = nhead-1
	tav_data = pd.read_csv(tav_file,delim_whitespace=True,skiprows=nhead)
	tav_data['file'] = tav_file
	nwin = int((ncol-naux)/2)

	# vav file
	nhead, ncol = file_info(vav_file)
	vav_data = pd.read_csv(vav_file,delim_whitespace=True,skiprows=nhead)
	vav_data['file'] = vav_file

	# ada file
	nhead, ncol = file_info(ada_file)
	ada_data = pd.read_csv(ada_file,delim_whitespace=True,skiprows=nhead)
	ada_data['file'] = ada_file
	
	# aia file
	nhead, ncol = file_info(aia_file)
	aia_data = pd.read_csv(aia_file,delim_whitespace=True,skiprows=nhead)
	aia_data['file'] = aia_file

	## check all files have the same spectrum list as the vav file
	check_spec = np.array([data['spectrum']==vav_data['spectrum'] for data in [tav_data,ada_data,aia_data,oof_data]]).flatten()
	if False in check_spec:
		print('Files have inconsistent spectrum lists !')
		for data in [vav_data,ada_data,aia_data,oof_data]:
			print(len(data['spectrum']),'spectra in',data['file'][0])
		sys.exit()

	# make all the column names consistent between the different files
	for dataframe in [correction_data,qc_data,esf_data,oof_data,lse_data,vav_data,ada_data,aia_data]:
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

	#sys.exit()

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
	}

	checksum_var_list = ['config','apriori','runlog','levels','mav','ray','isotopologs','windows','telluric_linelists','solar']

	standard_name_dict.update({var+'_checksum':var+'_checksum' for var in checksum_var_list})

	long_name_dict = {key:val.replace('_',' ') for key,val in standard_name_dict.items()} # standard names without underscores

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
	'fvsi':'',
	'zobs':'km',
	'zmin':'km',
	'osds':'ppm',
	'gfit_version':'',
	'gsetup_version':'',
	'fovi':'radians',
	'opd':'cm',
	'rmsocl':'%',
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
	}

	if os.path.exists(nc_file):
		os.remove(nc_file)

	with netCDF4.Dataset(nc_file,'w',format='NETCDF4') as nc_data:
		
		## global attributes
		
		# general TCCON
		nc_data.source = "Atmospheric trace gas concentrations retrieved from solar absorption spectra measured by ground based Fourier Transform Infrared Spectrometers"
		nc_data.data_use_policy = "https://tccon-wiki.caltech.edu/Network_Policy/Data_Use_Policy"
		nc_data.auxiliary_data_description = "https://tccon-wiki.caltech.edu/Network_Policy/Data_Use_Policy/Auxiliary_Data"
		nc_data.description = '\n'+header_content
		nc_data.file_creation = "Created with Python {}; the library netCDF4 {}; and the code {}".format(platform.python_version(),netCDF4.__version__,wnc_version)
		nc_data.flag_info = 'The Vmin and Vmax attributes of some variables indicate the range of values out of which the data would be flagged bad.\n the variable "flag" store the variable index of the flagged variables; the variable "flagged_var_name" store the name of the flagged variables'
		nc_data.more_information = "https://tccon-wiki.caltech.edu"
		nc_data.tccon_reference = "Wunch, D., G. C. Toon, J.-F. L. Blavier, R. A. Washenfelder, J. Notholt, B. J. Connor, D. W. T. Griffith, V. Sherlock, and P. O. Wennberg (2011), The total carbon column observing network, Philosophical Transactions of the Royal Society - Series A: Mathematical, Physical and Engineering Sciences, 369(1943), 2087-2112, doi:10.1098/rsta.2010.0240. Available from: http://dx.doi.org/10.1098/rsta.2010.0240"
		
		# site specific
		for key,val in site_data.items():
			setattr(nc_data,key,val)

		# other
		nc_data.number_of_species = str(nwin)
		nc_data.number_of_spectral_windows = str(len(col_file_list))
		
		proc = subprocess.Popen(['hg','summary'],cwd=GGGPATH,stdout=subprocess.PIPE)
		out, err = proc.communicate()
		nc_data.GGGtip = "The output of 'hg summary' from the GGG repository:\n"+out.decode("utf-8")
		nc_data.history = "Created {} (UTC)".format(time.asctime(time.gmtime(time.time())))

		## create dimensions
		tim = nc_data.createDimension('time',None)

		## create coordinate variables
		nc_data.createVariable('time',np.float64,('time',))
		nc_data['time'].standard_name = "time"
		nc_data['time'].long_name = "time"
		nc_data['time'].description = 'fractional days since 1970-01-01 00:00:00 (UTC)'
		nc_data['time'].units = 'days since 1970-01-01 00:00:00'
		nc_data['time'].calendar = 'gregorian'

		## create variables

		# checksums
		for var in checksum_var_list:
			checksum_var = nc_data.createVariable(var+'_checksum',str,('time',))
			checksum_var.standard_name = standard_name_dict[var+'_checksum']
			checksum_var.long_name = long_name_dict[var+'_checksum']
			checksum_var.description = 'hexdigest hash string of the md5 sum of the {} file'.format(var)

		# code versions
		nc_data.createVariable('gfit_version',np.float64,('time',))
		nc_data['gfit_version'].description = "version number of the GFIT code that generated the data"
		nc_data['gfit_version'].standard_name = standard_name_dict['gfit_version']
		nc_data['gfit_version'].long_name_dict = long_name_dict['gfit_version']

		nc_data.createVariable('gsetup_version',np.float64,('time',))
		nc_data['gsetup_version'].description = "version number of the GSETUP code that generated the priors"
		nc_data['gsetup_version'].standard_name = standard_name_dict['gsetup_version']
		nc_data['gsetup_version'].long_name_dict = long_name_dict['gsetup_version']

		# flags
		nc_data.createVariable('flag',np.float64,('time',))
		nc_data['flag'].description = 'data quality flag, 0 = good'
		nc_data['flag'].standard_name = 'quality_flag'
		nc_data['flag'].long_name = 'quality flag'

		nc_data.createVariable('flagged_var_name',str,('time',))
		nc_data['flagged_var_name'].description = 'name of the variable that caused the data to be flagged; empty string = good'
		nc_data['flagged_var_name'].standard_name = 'flagged_variable_name'
		nc_data['flagged_var_name'].long_name = 'flagged variable name'

		nc_data.createVariable('spectrum',str,('time',))
		nc_data['spectrum'].standard_name = 'spectrum_file_name'
		nc_data['spectrum'].description = 'spectrum file name'
		for i,specname in enumerate(aia_data['spectrum'].values):
			nc_data['spectrum'][i] = specname

		# auxiliary variables
		aux_var_list = [tav_data.columns[i] for i in range(1,naux)]
		for var in aux_var_list: 
			qc_id = list(qc_data['variable']).index(var)
			digit = int(qc_data['format'][qc_id].split('.')[-1])
			nc_data.createVariable(var,np.float64,('time',),zlib=True,least_significant_digit=digit)
			if var in standard_name_dict.keys():
				nc_data[var].standard_name = standard_name_dict[var]
				nc_data[var].long_name = long_name_dict[var]
				nc_data[var].units = units_dict[var]
			# set attributes using the qc.dat file
			nc_data[var].description = qc_data['description'][qc_id]
			nc_data[var].units = qc_data['unit'][qc_id].replace('(','').replace(')','').strip()
			nc_data[var].vmin = qc_data['vmin'][qc_id]
			nc_data[var].vmax = qc_data['vmax'][qc_id]

		nc_data['hour'].description = 'Fractional UT hours (zero path difference crossing time)'

		# averaged variables (from the different windows of each species)
		main_var_list = [tav_data.columns[i] for i in range(naux,len(tav_data.columns)-1)]  # minus 1 because I added the 'file' column
		for var in main_var_list:
			xvar = 'x'+var
			qc_id = list(qc_data['variable']).index(xvar)

			digit = int(qc_data['format'][qc_id].split('.')[-1])
			nc_data.createVariable(xvar,np.float64,('time',),zlib=True,least_significant_digit=digit)
			nc_data[xvar].standard_name = xvar
			nc_data[xvar].long_name = xvar.replace('_',' ')
			nc_data[xvar].description = qc_data['description'][qc_id]
			nc_data[xvar].units = qc_data['unit'][qc_id].replace('(','').replace(')','').strip()
			nc_data[xvar].vmin = qc_data['vmin'][qc_id]
			nc_data[xvar].vmax = qc_data['vmax'][qc_id]
			nc_data[xvar].precision = qc_data['format'][qc_id]

			nc_data.createVariable('vsf_'+var,np.float64,('time',))
			nc_data['vsf_'+var].description = var+" Volume Scale Factor"
			nc_data['vsf_'+var][:] = vav_data[var].values
			
			nc_data.createVariable('column_'+var,np.float64,('time',))
			nc_data['column_'+var].description = var+' molecules per square meter'
			nc_data['column_'+var].units = 'molecules.m-2'
			nc_data['column_'+var][:] = tav_data[var].values

			nc_data.createVariable('ada_x'+var,np.float64,('time',))
			nc_data['ada_x'+var].description = var+' column-average dry-air mole fraction'
			nc_data['ada_x'+var].units = qc_data['unit'][qc_id].replace('(','').replace(')','').strip()
			nc_data['ada_x'+var][:] = ada_data['x'+var].values

		# lse data
		lse_description = {'lst':'Laser sampling T','lse':'Laser sampling error','lsu':'Laser sampling U'}
		common_spec = np.intersect1d(aia_data['spectrum'],lse_data['spectrum'],return_indices=True)[2]
		for var in lse_description.keys():
			nc_data.createVariable(var,np.float64,('time',))
			nc_data[var].description = lse_description[var]
			nc_data[var][:] = lse_data[var][common_spec].values

		# corrections
		for var in correction_data['gas']:
			for key in correction_data.columns[1:]:
				varname = var+'_'+key
				nc_data.createVariable(varname,np.float64,('time',))
				nc_data[varname][:] = correction_data[key][list(correction_data['gas']).index(var)] # write directly

		## write data
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
				
				nc_data[var][start:end] = aia_data[var][start:end].values*qc_data['rsc'][qc_id]

				dev = abs( (qc_data['rsc'][qc_id]*aia_data[var][start:end].values-qc_data['vmin'][qc_id])/(qc_data['vmax'][qc_id]-qc_data['vmin'][qc_id]) -0.5 )
				
				kmax[dev>dmax] = qc_id+1 # add 1 here, otherwise qc_id starts at 0 for 'year'
				dmax[dev>dmax] = dev[dev>dmax]

			eflag[dmax>0.5] = kmax[dmax>0.5]
			
			# write the flagged variable index
			nc_data['flag'][start:end] = eflag

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

		specdate = np.array([datetime(int(aia_data['year'][i]),1,1)+timedelta(days=aia_data['day'][i]-1) for i in range(nrow)])
		nc_data['time'][:] = np.array([elem.total_seconds() for elem in (specdate-datetime(1970,1,1))])/(24.0*3600.0)

		# write data from col files
		for col_id,col_file in enumerate(col_file_list):

			cbf_file = col_file.replace('.col','.cbf')
			nhead,ncol = file_info(cbf_file)
			cbf_data = pd.read_csv(cbf_file,delim_whitespace=True,skiprows=nhead)
			cbf_data.rename(index=str,columns={'Spectrum_Name':'spectrum'},inplace=True)

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
				for i,line in enumerate([line for line in content if len(line.split())==2]):
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
			widths = [21,3,6,6,5,6,6,7,7,8]+[7,11,10,8]*ngas # the fixed widths for each variable so we can read with pandas.read_fwf, because sometimes there is no whitespace between numbers
			headers = content[nhead].split()

			col_data = pd.read_fwf(col_file,widths=widths,names=headers,skiprows=nhead+1)
			col_data.rename(str.lower,axis='columns',inplace=True)
			col_data.rename(index=str,columns={'rms/cl':'rmsocl'},inplace=True)
			if not all(col_data['spectrum'].values == vav_data['spectrum'].values):
				print('\nMismatch between .col file spectra and .vav spectra')
				print('col file:',col_file)
				continue # contine or exit here ? Might not need to exit if we can add in the results from the faulty col file afterwards
			if not all(col_data['spectrum'].values == cbf_data['spectrum'].values):
				print('\nMismatch between .col file spectra and .cbf spectra')
				print('col file:',col_file)
				continue # contine or exit here ? Might not need to exit if we can add in the results from the faulty col file afterwards

			# create window specific variables
			for var in col_data.columns[1:]: # skip the first one ("spectrum")
				varname = '_'.join([gas_XXXX,var])
				nc_data.createVariable(varname,np.float64,('time',))
				if var in standard_name_dict.keys():
					nc_data[varname].standard_name = standard_name_dict[var]
					nc_data[varname].long_name = long_name_dict[var]

				nc_data[varname][:] = col_data[var].values
			
			# add data from the .cbf file
			ncbf_var = '{}_ncbf'.format(gas_XXXX)
			nc_data.createVariable(ncbf_var,np.int32,('time',))
			nc_data[ncbf_var][:] = len(cbf_data.columns)-1 # minus 1 because of the spectrum name column
			for var in cbf_data.columns[1:]: # don't use the 'Spectrum' column
				varname = '_'.join([gas_XXXX,var])
				nc_data.createVariable(varname,np.float64,('time',))
				nc_data[varname].standard_name = standard_name_dict[var.split('_')[0]].format(var.split('_')[1])
				nc_data[varname].long_name = long_name_dict[var.split('_')[0]].format(var.split('_')[1])
				
				nc_data[varname][:] = cbf_data[var].values

			progress(col_id,len(col_file_list),word=col_file)
