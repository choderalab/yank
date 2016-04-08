#!/bin/tcsh

# You need pymol:
# conda install -c mw pymol

# Set up path for macports-installed pymol
#setenv PYTHONPATH /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/

# Render the movie
rm -rf frames
mkdir frames

echo Running PyMOL...
/Applications/MacPyMOL.app/Contents/MacOS/MacPyMOL -qc render_trajectory.py
#/Applications/MacPyMOL.app/Contents/MacOS/MacPyMOL -qcixr render_trajectory.py

# Compile into a movie
ffmpeg -r 30 -i frames/frame%04d.png -r 15 -b:v 5000000 -c:v wmv1 -y movie.wmv
