from setuptools import setup, find_packages

setup(
    name='write_tccon_netcdf',
    description='Write official TCCON netCDF output file',
    author='Sebastien Roche',
    author_email='sebastien.roche@mail.utoronto.ca',
    version='1.0.0',
    url='',
    install_requires=[
        'netcdf4>=1.5.0',
        'numpy>=1.16.0',
        'pandas>=0.23.0'
        ],
    packages=find_packages(),
    entry_points={'console_scripts': ['write_netcdf=write_tccon_netcdf.write_netcdf:main']}
)
