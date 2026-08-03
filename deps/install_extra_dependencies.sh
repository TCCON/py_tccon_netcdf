#!/bin/bash
GINPUT_REPO="https://github.com/TCCON/py-ginput.git"

cd $(dirname $0)
if [ ! -d ginput ]; then
    git clone -c advice.detachedHead=false -b v1.3.0 $GINPUT_REPO ginput
    (cd ginput && pip install -e)
fi