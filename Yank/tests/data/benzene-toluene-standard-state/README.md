# Benzene/Toluene Standard State Volume Files

These are specially made files to test the restraints of YANK. 
It is 1 benzene molecule (receptor) and 1 toluene molecue (ligand) in 
a cube of volume 1660 A^3 (standard state). 

We use this system to ensure the free energy of turning on restraints 
run by simulation is close to the analytical/numerical estimate for 
that free energy contribution. 

# Manifest
* standard_state_complex.inpcrd - coordinates with a periodic box of 1660 A^3
* standard_state_complex_boxless.inpcrd - coordinates of system without periodic box for vacuum testing
* standard_state_complex_boxless_nan.inpcrd - coordinates of system without periodic where one atom's coordinate is NaN for testing
* standard_state_complex.prmtop - parameter file for benzene/toluene