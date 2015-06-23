#!/bin/tcsh

# Set up path for macports-installed pymol
setenv PYTHONPATH /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/

# Render the movie
rm -rf frames
python render_trajectory.py

# Compile into a movie
ffmpeg -r 30 -i frames/frame%04d.png -r 25 -b:v 5000000 -c:v wmv1 -y movie.wmv

