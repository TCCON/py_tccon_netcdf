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

>__usage:__ write_netcdf.py [-h]
>
>						[--format {NETCDF4_CLASSIC,NETCDF4}]
>
>						[-r]
>
>						[--eof]
>
>                       [--public]
>
>                       [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
>
>                       [--log-file LOG_FILE]
>
>						[--skip-checksum]
>
>                       [-m MESSGAE]
>
>                       file
>
>write_netcdf.py (Version 1.0; 2019-11-15; SR)
>
>This writes TCCON outputs in a NETCDF file
>
>__positional arguments:__
>
>		file                  The .tav file or private.nc file
>
>__optional arguments:__
>
>		-h, --help            show this help message and exit
>
>		--format {NETCDF4_CLASSIC,NETCDF4}
>                        the format of the NETCDF files
>
>		-r, --read-only       Convenience for python interactive shells; sys.exit() right after reading all the input files
>
>		--eof                 If given, will also write the .eof.csv file
>
>		--public              if given, will write a .public.nc file from the .private.nc
>
>		--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
>                        Log level for the screen (it is always DEBUG for the log file)
>
>		--log-file LOG_FILE   Full path to the log file, by default write_netcdf.log is written to in append mode in the current working directory.
>                             If you want to write the logs of all your write_netcdf.py runs to a signle file, you can use this argument to specify the path.
>
>		--skip-checksum       option to not make a check on the checksums, for example to run the code on outputs generated by someone else or on a different machine
>
>       --m, --message        Add an optional message to be kept in the log file to remember why you ran post-processing e.g. "2020 Eureka R3 processing"

Run from the directory where the GFIT outputs are saved with:

	python /path/to/write_netcdf.py tavfile

Or if the installation script was use:

	$GGGPATH/bin/write_netcdf tavfile

The code can also be used to turn an existing .private.nc file into a .public.nc file if the private file is given as input:

	python /path/to/write_netcdf.py private_nc_file

## Dependencies ##

see setupy.py

        netcdf4>=1.5.0
        numpy>=1.16.0
        pandas>=0.23.0


