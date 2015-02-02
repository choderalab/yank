#!/bin/bash

cp -r $RECIPE_DIR/../.. $SRC_DIR
$PYTHON setup.py clean
$PYTHON setup.py install

# Push examples to anaconda/share/yank/examples/
mkdir $PREFIX/share/yank
cp -r $RECIPE_DIR/../../examples $PREFIX/share/yank/
