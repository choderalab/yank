#!/bin/tcsh

# Set up path for macports-installed pymol
setenv PYTHONPATH /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/
python render_trajectory.py
