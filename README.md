# README #

## Installation ##

Assuming GGG is installed and the GGGPATH environment variable is set.

If you ran GGGPATH/install/master.sh this code is already installed in GGGPATH/src/tccon_netcdf and you do not need to follow the steps below.


It can be installed manually, go to GGGPATH/install and run:
	
	./check_python.sh
	source ~/.bashrc
	./check_environment.sh
	./clone_netcdf_writer.sh

## write_tccon_netcdf/write_netcdf.py ##
Code to compile the outputs of GGG in a netCDF file

############
usage: write_netcdf.py [-h] [--format {NETCDF4_CLASSIC,NETCDF4}] [-r] file

write_netcdf.py (Version 1.0; 2019-11-15; SR)
This writes TCCON outputs in a NETCDF file

positional arguments:
  file                  The .tav file

optional arguments:
  -h, --help            show this help message and exit
  --format {NETCDF4_CLASSIC,NETCDF4}
                        the format of the NETCDF files
  -r, --read-only       Convenience for python interactive shells; sys.exit() right after reading all the input files

############


Run from the directory where the GFIT outputs are saved with:

	python /path/to/write_netcdf.py tavfile

Or if the installation script was use:

	$GGGPATH/bin/write_netcdf tavfile

## Dependencies ##

see setupy.py

        netcdf4>=1.5.0
        numpy>=1.16.0
        pandas>=0.23.0


