#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import shutil
import sys
import tomli


def main():
    p = ArgumentParser(description='Copy scripts installed by a Python project to a specific directory')
    p.add_argument('dest', help='Where to copy the scripts to')
    p.add_argument('--pyproject-file', default='./pyproject.toml',
                   help='pyproject.toml file to get the script names from. Default = %(default)s')
    p.add_argument('--env-prefix', help='Prefix for the environment the scripts were installed in, '
                                      'this would be the directory containing the bin directory with the scripts. '
                                      'If not given, the currently active environment is used')
    
    clargs = vars(p.parse_args())
    driver(**clargs)

def driver(dest, pyproject_file='./pyproject.toml', env_prefix=None):
    with open(pyproject_file, 'rb') as f:
        cfg = tomli.load(f)

    env_prefix = env_prefix or sys.prefix
    bin_dir = Path(env_prefix) / 'bin'

    dest = dest
    for script_name in cfg['project']['scripts'].keys():
        script = bin_dir / script_name
        if script.exists():
            shutil.copy2(str(script), str(dest))
            print(f'{script} -> {dest/script_name}')
        else:
            print(f'Warning: {script} not found')


if __name__ == '__main__':
    main()