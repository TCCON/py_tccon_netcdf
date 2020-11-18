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
    version='1.0.0',
    url='https://bitbucket.org/rocheseb/tccon_netcdf',
    install_requires=[
        'netcdf4>=1.5.0',
        'numpy>=1.16.0',
        'pandas>=0.23.0'
        ],
    packages=find_packages(),
    classifiers=[
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
    entry_points={'console_scripts': [
        'write_netcdf=write_tccon_netcdf.write_netcdf:main',
        'compare_netcdf=write_tccon_netcdf.write_netcdf:compare_nc_files_command_line',
        'concat_netcdf=util.concat_netcdf:main'
    ]},
    license='MIT',
    python_requires='>=3.7',
)
