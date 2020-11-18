# README #

## concat_netcdf.py ##

Can be used to concatenate netCDF files contained in a folder, the concatenated file name will be:

**xxYYYYMMDD_YYYYMMDD.nc**

where **xx** is the 2-letters site ID and **YYYYMMDD** are the first and last date in the data files.

If you ran the install you can run the code from **$GGGPATH/bin/concat_netcdf**

To run directly with python, activate the python environment of the netCDF writer with:

> conda activate ggg-tccon-default

For usage info of **concat_netcdf.py** run:

> python concat_netcdf.py --help

The code take the full path to a directory containing netCDF files produced by **write_netcdf.py**

> python concat_netcdf.py path

**--out** full path to the **directory** in which the concatenated netCDF file will be saved, default to *path* so be mindful of that when using multiple time

**--prefix** can be given to only select files starting with the given prefix under *path*

