#!/bin/bash

$PYTHON setup.py install

# Push examples to anaconda/share/yank/examples/
mkdir $PREFIX/share/yank
cp -r examples $PREFIX/share/yank/
