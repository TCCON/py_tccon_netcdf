# README #

## write_netcdf.py ##
Code to compile the outputs of GGG in a netCDF file

############
write_netcdf.py (Version 1.0; 2019-10-26; SR)

usage: write_netdcf.py [-h] [-r] file

write_netcdf.py (Version 1.0; 2019-10-18; SR)
This writes TCCON outputs in a NETCDF file

positional arguments:
  file             The .tav file

optional arguments:
  -h, --help       show this help message and exit
  -r, --read-only  Convenience for python interactive shells; sys.exit() right after reading all the input files
############


Run from the directory where the GFIT outputs are saved with:

	python /path/to/write_netcdf.py tavfile

## Dependencies ##

netCDF4 package