#!/bin/bash

#export CC=${PREFIX}/bin/gcc
#export CXX=${PREFIX}/bin/g++

# conda provides default values of these on Mac OS X,
# but we don't want them when building with gcc
#export CFLAGS=""
#export CXXFLAGS=""
#export LDFLAGS=""

# Build the python package
$PYTHON setup.py install

# Push examples to anaconda/share/yank/examples/
mkdir $PREFIX/share/yank
cp -r examples $PREFIX/share/yank/
