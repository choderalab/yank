#!/bin/bash

cp -r $RECIPE_DIR/../.. $SRC_DIR
$PYTHON setup.py clean
$PYTHON setup.py install

# Eventually we want to push examples to some place like ~/anaconda/share/yank/examples/
#mkdir $PREFIX/share/yank/
#cp -r $RECIPE_DIR/../../examples/ $PREFIX/share/yank/
