from setuptools import setup, find_packages

with open("README.md", "r") as infile:
    long_description = infile.read()

setup(
    name='write_tccon_netcdf',
    description='Write official TCCON netCDF output file',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Sebastien Roche',
    author_email='sebastien.roche@mail.utoronto.ca',
    version='1.1.0',
    url='https://github.com/TCCON/py_tccon_netcdf',
    install_requires=[
        'netcdf4>=1.5.0',
        'numpy>=1.16.0',
        'pandas>=0.23.0',
        'requests>=2.28.0',
        'scipy==1.5.2',
        'xarray==0.13.0'
        ],
    packages=find_packages(),
    package_data={'write_tccon_netcdf': ['write_netcdf/*.json', 'write_netcdf/*.nc']},
    classifiers=[
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
    entry_points={'console_scripts': [
        'write_netcdf=write_tccon_netcdf.write_netcdf:main',
        'compare_netcdf=write_tccon_netcdf.write_netcdf:compare_nc_files_command_line',
        'concat_netcdf=util.concat_netcdf:main',
        'update_site_info=util.update_site_info:main',
        'subset_netcdf=util.subset_tccon_netcdf:main',
        'update_manual_flags=write_tccon_netcdf.update_manual_flags:main'
    ]},
    license='MIT',
    python_requires='>=3.7',
)
