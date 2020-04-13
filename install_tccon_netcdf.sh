#!/bin/bash -l

if [ $(basename $CONDA_PREFIX) != 'ggg-tccon-default' ]; then
    echo "In $0: must activate ggg-tccon-default environment"
    # should be available because we run this in a login shell
    conda activate ggg-tccon-default
else
    echo "In $0: ggg-tccon-default environment already active"
fi

if [ ! -d scripts ]; then
    mkdir -v scripts
fi

# develop: do not copy to $SITEDIR/, just run from here
# --no-user-cfg: ignore any ~/.pydistutils.cfg file
# --script-dir: write command line scripts to the given directory
python setup.py develop --no-user-cfg --script-dir=./scripts && mv -v ./scripts/write_netcdf $GGGPATH/bin/
