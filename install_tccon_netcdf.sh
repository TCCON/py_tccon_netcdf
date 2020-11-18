#!/bin/bash

if [ $(basename $CONDA_PREFIX) != 'ggg-tccon-default' ]; then
    echo "Error in $0: ggg-tccon-default environment is not active"
    exit 1
fi

if [ ! -d scripts ]; then
    mkdir -v scripts
fi

# develop: do not copy to $SITEDIR/, just run from here
# --no-user-cfg: ignore any ~/.pydistutils.cfg file
# --script-dir: write command line scripts to the given directory
python setup.py --no-user-cfg develop --script-dir=./scripts \
    && mv -v ./scripts/write_netcdf $GGGPATH/bin/ \
    && mv -v ./scripts/compare_netcdf $GGGPATH/bin/ \
    && mv -v ./scripts/concat_netcdf $GGGPATH/bin/
