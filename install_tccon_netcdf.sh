#!/bin/bash
if [ $(which conda | wc -l) == 0 ] ; then
        source $HOME/anaconda3/etc/profile.d/conda.sh
else
        conda_base=$(conda info --base)
        source $conda_base/etc/profile.d/conda.sh
fi

if [ ! -d scripts ]; then
    mkdir -v scripts
fi

python setup.py develop --script-dir=./scripts && mv -v ./scripts/write_netcdf $GGGPATH/bin/
