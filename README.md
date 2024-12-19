# README 

## Installation

This package will typically be installed as part of the [GGG](https://github.com/TCCON/GGG/) installation process.
Installation outside of GGG is not officially supported.
If that is needed, the recommended steps are:

1. Create a conda environment with Python 3.10 and the dependencies specified in `pyproject.toml`
2. Activate that environment
3. In this directory, run `pip install -e . --isolated`

This will install the entry point scripts into the `bin` directory of your environment.

## Included programs

- `write_netcdf`: write a TCCON private netCDF file from GGG output or a public TCCON netCDF file from a private one
- `compare_netcdf`: compare two TCCON netCDF files for differences
- `concat_netcdf`: concatenate two TCCON private netCDF files along their time dimensions. Public files are not supported.
- `update_site_info`: update site-related attributes in a TCCON netCDF file
- `subset_netcdf`: subset a private TCCON netCDF file along its time dimensions. Public files are not supported.
- `update_manual_flags`: set manual quality flags in a private TCCON netCDF file.

All programs include command line help, use the `-h` or `--help` flags to see specific options and flags.

## Common usage

Write a private TCCON file - this assumes it is being run in a directory with all of the GGG output and that your GGG
run used the "pa_ggg_benchmark" runlog:

```bash
write_netcdf pa_ggg_benchmark.tav
```

Create a public TCCON file from the private file, `pa20040721_20041222.private.nc`:

```bash
write_netcdf --public pa20040721_20041222.private.nc
```
