#!/opt/local/bin/python2.5

#=============================================================================================
# Analyze a set of datafiles produced by YANK.
#=============================================================================================

#=============================================================================================
# REQUIREMENTS
#
# The netcdf4-python module is now used to provide netCDF v4 support:
# http://code.google.com/p/netcdf4-python/
#
# This requires NetCDF with version 4 and multithreading support, as well as HDF5.
#=============================================================================================

#=============================================================================================
# TODO
#=============================================================================================

#=============================================================================================
# CHAGELOG
#=============================================================================================

#=============================================================================================
# VERSION CONTROL INFORMATION
#=============================================================================================

#=============================================================================================
# IMPORTS
#=============================================================================================

import numpy
from numpy import *
#from Scientific.IO import NetCDF # scientific python
#import scipy.io.netcdf as netcdf
import netCDF4 as netcdf # netcdf4-python
import os
import sys
import os.path
import math
import gzip
from pymbar import MBAR # multistate Bennett acceptance ratio
import timeseries # for statistical inefficiency analysis

import simtk.unit as units

#=============================================================================================
# PARAMETERS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA

#=============================================================================================
# SUBROUTINES
#=============================================================================================
def write_file(filename, contents):
    """Write the specified contents to a file.

    ARGUMENTS
      filename (string) - the file to be written
      contents (string) - the contents of the file to be written

    """

    outfile = open(filename, 'w')

    if type(contents) == list:
        for line in contents:
            outfile.write(line)
    elif type(contents) == str:
        outfile.write(contents)
    else:
        raise "Type for 'contents' not supported: " + repr(type(contents))

    outfile.close()

    return

def read_file(filename):
    """Read contents of the specified file.

    ARGUMENTS
      filename (string) - the name of the file to be read

    RETURNS
      lines (list of strings) - the contents of the file, split by line
    """

    infile = open(filename, 'r')
    lines = infile.readlines()
    infile.close()

    return lines

def initialize_netcdf(netcdf_file, title, natoms, is_periodic = False, has_velocities = False):
  """Initialize the given NetCDF file according to the AMBER NetCDF Convention Version 1.0, Revision B.

  ARGUMENTS
    netcdf_file (NetCDFFile object) - the file to initialize global attributes, dimensions, and variables for
    title (string) - the title for the netCDF file
    natoms (integer) - the number of atoms in the trajectories to be written
    is_periodic (boolean) - if True, box coordinates will also be stored
    has_velocities (boolean) - if True, the velocity trajectory variables will also be created

  NOTES
    The AMBER NetCDF convention is defined here:

    http://amber.scripps.edu/netcdf/nctraj.html    

  """

  # Create dimensions.
  netcdf_file.createDimension('frame', 0)        # unlimited number of frames in trajectory
  netcdf_file.createDimension('spatial', 3)      # number of spatial coordinates
  netcdf_file.createDimension('atom', natoms)    # number of atoms in the trajectory
  netcdf_file.createDimension('label', 5)        # label lengths for cell dimensions
  netcdf_file.createDimension('cell_spatial', 3) # naming conventions for cell spatial dimensions
  netcdf_file.createDimension('cell_angular', 3) # naming conventions for cell angular dimensions
  
  # Set attributes.
  setattr(netcdf_file, 'title', title)
  setattr(netcdf_file, 'application', 'AMBER')
  setattr(netcdf_file, 'program', 'sander')
  setattr(netcdf_file, 'programVersion', '8')
  setattr(netcdf_file, 'Conventions', 'AMBER')
  setattr(netcdf_file, 'ConventionVersion', '1.0')

  # Define variables to store unit cell data, if specified.
  if is_periodic:
    cell_spatial = netcdf_file.createVariable('cell_spatial', 'c', ('cell_spatial',))
    cell_angular = netcdf_file.createVariable('cell_angular', 'c', ('cell_spatial', 'label'))
    cell_lengths = netcdf_file.createVariable('cell_lengths', 'd', ('frame', 'cell_spatial'))
    setattr(cell_lengths, 'units', 'angstrom')
    cell_angles = netcdf_file.createVariable('cell_angles', 'd', ('frame', 'cell_angular'))
    setattr(cell_angles, 'units', 'degree')  

    netcdf_file.variables['cell_spatial'][0] = 'x'
    netcdf_file.variables['cell_spatial'][1] = 'y'
    netcdf_file.variables['cell_spatial'][2] = 'z'

    netcdf_file.variables['cell_angular'][0] = 'alpha'
    netcdf_file.variables['cell_angular'][1] = 'beta '
    netcdf_file.variables['cell_angular'][2] = 'gamma'

  # Define variables to store velocity data, if specified.
  if has_velocities:
    velocities = netcdf_file.createVariable('velocities', 'd', ('frame', 'atom', 'spatial'))    
    setattr(velocities, 'units', 'angstrom/picosecond')
    setattr(velocities, 'scale_factor', 20.455)  

  # Define coordinates and snapshot times.
  frame_times = netcdf_file.createVariable('time', 'f', ('frame',))
  setattr(frame_times, 'units', 'picosecond')
  frame_coordinates = netcdf_file.createVariable('coordinates', 'f', ('frame', 'atom', 'spatial'))
  setattr(frame_coordinates, 'units', 'angstrom')

  # Define optional data not specified in the AMBER NetCDF Convention that we will make use of.
  frame_energies = netcdf_file.createVariable('total_energy', 'f', ('frame',))
  setattr(frame_energies, 'units', 'kilocalorie/mole')
  frame_energies = netcdf_file.createVariable('potential_energy', 'f', ('frame',))
  setattr(frame_energies, 'units', 'kilocalorie/mole')  
  
  return

def write_netcdf_frame(netcdf_file, frame_index, time = None, coordinates = None, cell_lengths = None, cell_angles = None, total_energy = None, potential_energy = None):
  """Write a NetCDF frame.

  ARGUMENTS
    netcdf_file (NetCDFFile) - the file to write a frame to
    frame_index (integer) - the frame to be written

  OPTIONAL ARGUMENTS
    time (float) - time of frame (in picoseconds)
    coordinates (natom x nspatial NumPy array) - atomic coordinates (in Angstroms)
    cell_lengths (nspatial NumPy array) - cell lengths (Angstroms)
    cell_angles (nspatial NumPy array) - cell angles (degrees)
    total_energy (float) - total energy (kcal/mol)
    potential_energy (float) - potential energy (kcal/mol)

  """
  if time != None: netcdf_file.variables['time'][frame_index] = time      
  if coordinates != None: netcdf_file.variables['coordinates'][frame_index,:,:] = coordinates
  if cell_lengths != None: netcdf_file.variables['cell_lengths'][frame_index,:] = cell_lengths
  if cell_angles != None: netcdf_file.variables['cell_angles'][frame_index,:] = cell_angles
  if total_energy != None: netcdf_file.variables['total_energy'][frame_index] = total_energy
  if potential_energy != None: netcdf_file.variables['total_energy'][frame_index] = potential_energy
  
  return

def read_amber_energy_frame(infile):
    """Read a frame of energy components from the AMBER energy file.

    ARGUMENTS
      infile (Python file handle) - the file to read from

    RETURNS
      energies (Python dict) -- energies[keyword] contains the energy for the corresponding keyword
    """

    # number of lines per .ene block
    ene_lines_per_block = 10
    
    # energy keys
    energy_keys = [
        'Nsteps', 'time', 'Etot', 'EKinetic', # L0
        'Temp', 'T_solute', 'T_solv', 'Pres_scal_solu', # L1
        'Pres_scal_solv', 'BoxX', 'BoxY', 'BoxZ', # L2
        'volume', 'pres_X', 'pres_Y', 'pres_Z',
        'Pressure', 'EKCoM_x', 'EKCoM_y', 'EKCoM_z',
        'EKComTot', 'VIRIAL_x', 'VIRIAL_y', 'VIRIAL_z',
        'VIRIAL_tot', 'E_pot', 'E_vdw', 'E_el',
        'E_hbon', 'E_bon', 'E_angle', 'E_dih',
        'E_14vdw', 'E_14el', 'E_const', 'E_pol',
        'AV_permMoment', 'AV_indMoment', 'AV_totMoment', 'Density', 'dV/dlambda'
        ]

    # Read energy block.
    energies = dict()
    key_index = 0
    for line_counter in range(ene_lines_per_block):
        line = infile.readline() # read the line
        elements = line.split() # split into elements
        elements.pop(0) # drop the 'L#' initial element
        for element in elements:
            key = energy_keys[key_index] # get the key
            energies[key] = float(element) # store the energy
            key_index += 1 # increment index

    return energies

def write_netcdf_replica_trajectories(directory, prefix, title, ncfile):
    """Write out replica trajectories in AMBER NetCDF format.

    ARGUMENTS
       directory (string) - the directory to write files to
       prefix (string) - prefix for replica trajectory files
       title (string) - the title to give each NetCDF file
       ncfile (NetCDF) - NetCDF file object for input file       
    """
    # Get current dimensions.
    niterations = ncfile.variables['positions'].shape[0]
    nstates = ncfile.variables['positions'].shape[1]
    natoms = ncfile.variables['positions'].shape[2]

    # Write out each replica to a separate file.
    for replica in range(nstates):
        # Create a new replica file.
        output_filename = os.path.join(directory, '%s-%03d.nc' % (prefix, replica))
        ncoutfile = netcdf.Dataset(output_filename, 'w')
        initialize_netcdf(ncoutfile, title + " (replica %d)" % replica, natoms)
        for iteration in range(niterations):
            coordinates = array(ncfile.variables['positions'][iteration,replica,:,:])
            coordinates *= 10.0 # convert nm to angstroms
            write_netcdf_frame(ncoutfile, iteration, time = 1.0 * iteration, coordinates = coordinates)
        ncoutfile.close()

    return

def compute_torsion_trajectories(ncfile, filename):
    """Write out torsion trajectories for Val 111.

    ARGUMENTS
       ncfile (NetCDF) - NetCDF file object for input file
       filename (string) - name of file to be written
    """
    atoms = [1735, 1737, 1739, 1741] # N-CA-CB-CG1 of Val 111        

    # Get current dimensions.
    niterations = ncfile.variables['positions'].shape[0]
    nstates = ncfile.variables['positions'].shape[1]
    natoms = ncfile.variables['positions'].shape[2]

    # Compute torsion angle
    def compute_torsion(positions, atoms):
        # Compute vectors from cross products        
        vBA = positions[atoms[0],:] - positions[atoms[1],:]
        vBC = positions[atoms[2],:] - positions[atoms[1],:]
        vCB = positions[atoms[1],:] - positions[atoms[2],:]
        vCD = positions[atoms[3],:] - positions[atoms[2],:]
        v1 = cross(vBA,vBC)
        v2 = cross(vCB,vCD)
        cos_theta = dot(v1,v2) / sqrt(dot(v1,v1) * dot(v2,v2))
        theta = arccos(cos_theta) * 180.0 / math.pi
        return theta
                
    # Compute torsion angles for each replica
    contents = ""
    for iteration in range(niterations):
        for replica in range(nstates):
            # Compute torsion
            torsion = compute_torsion(array(ncfile.variables['positions'][iteration,replica,:,:]), atoms)
            # Write torsion
            contents += "%8.1f" % torsion
        contents += "\n"

    # Write contents.
    write_file(filename, contents)

    return

def write_pdb_replica_trajectories(basepdb, directory, prefix, title, ncfile):
    """Write out replica trajectories as multi-model PDB files.

    ARGUMENTS
       basepdb (string) - name of PDB file to read atom names and residue information from
       directory (string) - the directory to write files to
       prefix (string) - prefix for replica trajectory files
       title (string) - the title to give each PDB file
       ncfile (NetCDF) - NetCDF file object for input file       
    """

    raise "not implemented yet"
    
    return

def read_pdb(filename):
    """
    Read the contents of a PDB file.

    ARGUMENTS

    filename (string) - name of the file to be read

    RETURNS

    atoms (list of dict) - atoms[index] is a dict of fields for the ATOM residue

    """
    
    # Read the PDB file into memory.
    pdbfile = open(filename, 'r')

    # Extract the ATOM entries.
    # Format described here: http://bmerc-www.bu.edu/needle-doc/latest/atom-format.html
    atoms = list()
    for line in pdbfile:
        if line[0:6] == "ATOM  ":
            # Parse line into fields.
            atom = dict()
            atom["serial"] = line[6:11]
            atom["atom"] = line[12:16]
            atom["altLoc"] = line[16:17]
            atom["resName"] = line[17:20]
            atom["chainID"] = line[21:22]
            atom["Seqno"] = line[22:26]
            atom["iCode"] = line[26:27]
            atom["x"] = line[30:38]
            atom["y"] = line[38:46]
            atom["z"] = line[46:54]
            atom["occupancy"] = line[54:60]
            atom["tempFactor"] = line[60:66]
            atoms.append(atom)
            
    # Close PDB file.
    pdbfile.close()

    # Return dictionary of present residues.
    return atoms

def write_pdb(atoms, filename, iteration, replica, title, ncfile):
    """Write out replica trajectories as multi-model PDB files.

    ARGUMENTS
       atoms (list of dict) - parsed PDB file ATOM entries from read_pdb() - WILL BE CHANGED
       filename (string) - name of PDB file to be written
       title (string) - the title to give each PDB file
       ncfile (NetCDF) - NetCDF file object for input file       
    """

    # Extract coordinates to be written.
    coordinates = array(ncfile.variables['positions'][iteration,replica,:,:])
    coordinates *= 10.0 # convert nm to angstroms

    # Create file.
    outfile = open(filename, 'w')

    # Write ATOM records.
    for (index, atom) in enumerate(atoms):
        atom["x"] = "%8.3f" % coordinates[index,0]
        atom["y"] = "%8.3f" % coordinates[index,1]
        atom["z"] = "%8.3f" % coordinates[index,2]
        outfile.write('ATOM  %(serial)5s %(atom)4s%(altLoc)c%(resName)3s %(chainID)c%(Seqno)5s   %(x)8s%(y)8s%(z)8s\n' % atom)
        
    # Close file.
    outfile.close()

    return

def write_crd(filename, iteration, replica, title, ncfile):
    """
    Write out AMBER format CRD file.

    """
    # Extract coordinates to be written.
    coordinates = array(ncfile.variables['positions'][iteration,replica,:,:])
    coordinates *= 10.0 # convert nm to angstroms

    # Create file.
    outfile = open(filename, 'w')

    # Write title.
    outfile.write(title + '\n')

    # Write number of atoms.
    natoms = ncfile.variables['positions'].shape[2]
    outfile.write('%6d\n' % natoms)

    # Write coordinates.
    for index in range(natoms):
        outfile.write('%12.7f%12.7f%12.7f' % (coordinates[index,0], coordinates[index,1], coordinates[index,2]))
        if ((index+1) % 2 == 0): outfile.write('\n')
        
    # Close file.
    outfile.close()
    
def show_mixing_statistics(ncfile, cutoff=0.05, nequil=0):
    """
    Print summary of mixing statistics.

    ARGUMENTS

    ncfile (netCDF4.Dataset) - NetCDF file
    
    OPTIONAL ARGUMENTS

    cutoff (float) - only transition probabilities above 'cutoff' will be printed (default: 0.05)
    nequil (int) - if specified, only samples nequil:end will be used in analysis (default: 0)
    
    """
    
    # Get dimensions.
    niterations = ncfile.variables['states'].shape[0]
    nstates = ncfile.variables['states'].shape[1]

    # Compute statistics of transitions.
    Nij = numpy.zeros([nstates,nstates], numpy.float64)
    for iteration in range(nequil, niterations-1):
        for ireplica in range(nstates):
            istate = ncfile.variables['states'][iteration,ireplica]
            jstate = ncfile.variables['states'][iteration+1,ireplica]
            Nij[istate,jstate] += 0.5
            Nij[jstate,istate] += 0.5
    Tij = numpy.zeros([nstates,nstates], numpy.float64)
    for istate in range(nstates):
        Tij[istate,:] = Nij[istate,:] / Nij[istate,:].sum()

    # Print observed transition probabilities.
    print "Cumulative symmetrized state mixing transition matrix:"
    print "%6s" % "",
    for jstate in range(nstates):
        print "%6d" % jstate,
    print ""
    for istate in range(nstates):
        print "%-6d" % istate,
        for jstate in range(nstates):
            P = Tij[istate,jstate]
            if (P >= cutoff):
                print "%6.3f" % P,
            else:
                print "%6s" % "",
        print ""

    # Estimate second eigenvalue and equilibration time.
    mu = numpy.linalg.eigvals(Tij)
    mu = -numpy.sort(-mu) # sort in descending order
    if (mu[1] >= 1):
        print "Perron eigenvalue is unity; Markov chain is decomposable."
    else:
        print "Perron eigenvalue is %9.5f; state equilibration timescale is ~ %.1f iterations" % (mu[1], 1.0 / (1.0 - mu[1]))
        
    return

def show_mixing_statistics(ncfile, cutoff=0.05, nequil=0):
    """
    Print summary of mixing statistics.

    ARGUMENTS

    ncfile (netCDF4.Dataset) - NetCDF file
    
    OPTIONAL ARGUMENTS

    cutoff (float) - only transition probabilities above 'cutoff' will be printed (default: 0.05)
    nequil (int) - if specified, only samples nequil:end will be used in analysis (default: 0)
    
    """
    
    # Get dimensions.
    niterations = ncfile.variables['states'].shape[0]
    nstates = ncfile.variables['states'].shape[1]

    # Compute statistics of transitions.
    Nij = numpy.zeros([nstates,nstates], numpy.float64)
    for iteration in range(nequil, niterations-1):
        for ireplica in range(nstates):
            istate = ncfile.variables['states'][iteration,ireplica]
            jstate = ncfile.variables['states'][iteration+1,ireplica]
            Nij[istate,jstate] += 0.5
            Nij[jstate,istate] += 0.5
    Tij = numpy.zeros([nstates,nstates], numpy.float64)
    for istate in range(nstates):
        Tij[istate,:] = Nij[istate,:] / Nij[istate,:].sum()

    # Print observed transition probabilities.
    print "Cumulative symmetrized state mixing transition matrix:"
    print "%6s" % "",
    for jstate in range(nstates):
        print "%6d" % jstate,
    print ""
    for istate in range(nstates):
        print "%-6d" % istate,
        for jstate in range(nstates):
            P = Tij[istate,jstate]
            if (P >= cutoff):
                print "%6.3f" % P,
            else:
                print "%6s" % "",
        print ""

    # Estimate second eigenvalue and equilibration time.
    mu = numpy.linalg.eigvals(Tij)
    mu = -numpy.sort(-mu) # sort in descending order
    if (mu[1] >= 1):
        print "Perron eigenvalue is unity; Markov chain is decomposable."
    else:
        print "Perron eigenvalue is %9.5f; state equilibration timescale is ~ %.1f iterations" % (mu[1], 1.0 / (1.0 - mu[1]))
        
    return

def analyze_acceptance_probabilities(ncfile, cutoff = 0.4):
    """Analyze acceptance probabilities.

    ARGUMENTS
       ncfile (NetCDF) - NetCDF file to be analyzed.

    OPTIONAL ARGUMENTS
       cutoff (float) - cutoff for showing acceptance probabilities as blank (default: 0.4)
    """

    # Get current dimensions.
    niterations = ncfile.variables['mixing'].shape[0]
    nstates = ncfile.variables['mixing'].shape[1]

    # Compute mean.
    print "Computing mean of mixing probabilities..."
    mixing = ncfile.variables['mixing'][:,:,:]
    Pij = mean(mixing, 0)

    # Write title.
    print "Average state-to-state acceptance probabilities"
    print "(Probabilities less than %(cutoff)f shown as blank.)" % vars()
    print ""

    # Write header.
    print "%4s" % "",
    for j in range(nstates):
        print "%6d" % j,
    print ""

    # Write rows.
    for i in range(nstates):
        print "%4d" % i, 
        for j in range(nstates):
            if Pij[i,j] > cutoff:
                print "%6.3f" % Pij[i,j],
            else:
                print "%6s" % "",
            
        print ""

    return

def check_energies(ncfile, atoms):
    """
    Examine energy history for signs of instability (nans).

    ARGUMENTS
       ncfile (NetCDF) - input YANK netcdf file
    """

    # Get current dimensions.
    niterations = ncfile.variables['energies'].shape[0]
    nstates = ncfile.variables['energies'].shape[1]

    # Extract energies.
    print "Reading energies..."
    energies = ncfile.variables['energies']
    u_kln_replica = zeros([nstates, nstates, niterations], float64)
    for n in range(niterations):
        u_kln_replica[:,:,n] = energies[n,:,:]
    print "Done."

    # Deconvolute replicas
    print "Deconvoluting replicas..."
    u_kln = zeros([nstates, nstates, niterations], float64)
    for iteration in range(niterations):
        state_indices = ncfile.variables['states'][iteration,:]
        u_kln[state_indices,:,iteration] = energies[iteration,:,:]
    print "Done."

    # Show all self-energies
    show_self_energies = False
    if (show_self_energies):
        print 'all self-energies for all replicas'
        for iteration in range(niterations):
            for replica in range(nstates):
                state = int(ncfile.variables['states'][iteration,replica])
                print '%12.1f' % energies[iteration, replica, state],
            print ''

    # If no energies are 'nan', we're clean.
    if not any(isnan(energies[:,:,:])):
        return

    # There are some energies that are 'nan', so check if the first iteration has nans in their *own* energies:
    u_k = diag(energies[0,:,:])
    if any(isnan(u_k)):
        print "First iteration has exploded replicas.  Check to make sure structures are minimized before dynamics"
        print "Energies for all replicas after equilibration:"
        print u_k
        sys.exit(1)

    # There are some energies that are 'nan' past the first iteration.  Find the first instances for each replica and write PDB files.
    first_nan_k = zeros([nstates], int32)
    for iteration in range(niterations):
        for k in range(nstates):
            if isnan(energies[iteration,k,k]) and first_nan_k[k]==0:
                first_nan_k[k] = iteration
    if not all(first_nan_k == 0):
        print "Some replicas exploded during the simulation."
        print "Iterations where explosions were detected for each replica:"
        print first_nan_k
        print "Writing PDB files immediately before explosions were detected..."
        for replica in range(nstates):            
            if (first_nan_k[replica] > 0):
                state = ncfile.variables['states'][iteration,replica]
                iteration = first_nan_k[replica] - 1
                filename = 'replica-%d-before-explosion.pdb' % replica
                title = 'replica %d state %d iteration %d' % (replica, state, iteration)
                write_pdb(atoms, filename, iteration, replica, title, ncfile)
                filename = 'replica-%d-before-explosion.crd' % replica                
                write_crd(filename, iteration, replica, title, ncfile)
        sys.exit(1)

    # There are some energies that are 'nan', but these are energies at foreign lambdas.  We'll just have to be careful with MBAR.
    # Raise a warning.
    print "WARNING: Some energies at foreign lambdas are 'nan'.  This is recoverable."
        
    return

def check_positions(ncfile):
    """Make sure no positions have gone 'nan'.

    ARGUMENTS
       ncfile (NetCDF) - NetCDF file object for input file
    """

    # Get current dimensions.
    niterations = ncfile.variables['positions'].shape[0]
    nstates = ncfile.variables['positions'].shape[1]
    natoms = ncfile.variables['positions'].shape[2]

    # Compute torsion angles for each replica
    for iteration in range(niterations):
        for replica in range(nstates):
            # Extract positions
            positions = array(ncfile.variables['positions'][iteration,replica,:,:])
            # Check for nan
            if any(isnan(positions)):
                # Nan found -- raise error
                print "Iteration %d, state %d - nan found in positions." % (iteration, replica)
                # Report coordinates
                for atom_index in range(natoms):
                    print "%16.3f %16.3f %16.3f" % (positions[atom_index,0], positions[atom_index,1], positions[atom_index,2])
                    if any(isnan(positions[atom_index,:])):
                        raise "nan detected in positions"

    return

def estimate_free_energies(ncfile, ndiscard = 0, nuse = None):
    """Estimate free energies of all alchemical states.

    ARGUMENTS
       ncfile (NetCDF) - input YANK netcdf file

    OPTIONAL ARGUMENTS
       ndiscard (int) - number of iterations to discard to equilibration
       nuse (int) - maximum number of iterations to use (after discarding)

    TODO: Automatically determine 'ndiscard'.
    """

    # Get current dimensions.
    niterations = ncfile.variables['energies'].shape[0]
    nstates = ncfile.variables['energies'].shape[1]
    natoms = ncfile.variables['energies'].shape[2]

    # Extract energies.
    print "Reading energies..."
    energies = ncfile.variables['energies']
    u_kln_replica = zeros([nstates, nstates, niterations], float64)
    for n in range(niterations):
        u_kln_replica[:,:,n] = energies[n,:,:]
    print "Done."

    # Deconvolute replicas
    print "Deconvoluting replicas..."
    u_kln = zeros([nstates, nstates, niterations], float64)
    for iteration in range(niterations):
        state_indices = ncfile.variables['states'][iteration,:]
        u_kln[state_indices,:,iteration] = energies[iteration,:,:]
    print "Done."

    # Compute total negative log probability over all iterations.
    u_n = zeros([niterations], float64)
    for iteration in range(niterations):
        u_n[iteration] = sum(diagonal(u_kln[:,:,iteration]))
    #print u_n

    # DEBUG
    outfile = open('u_n.out', 'w')
    for iteration in range(niterations):
        outfile.write("%8d %24.3f\n" % (iteration, u_n[iteration]))
    outfile.close()

    # Discard initial data to equilibration.
    u_kln_replica = u_kln_replica[:,:,ndiscard:]
    u_kln = u_kln[:,:,ndiscard:]
    u_n = u_n[ndiscard:]

    # Truncate to number of specified conforamtions to use
    if (nuse):
        u_kln_replica = u_kln_replica[:,:,0:nuse]
        u_kln = u_kln[:,:,0:nuse]
        u_n = u_n[0:nuse]
    
    # Subsample data to obtain uncorrelated samples
    N_k = zeros(nstates, int32)
    indices = timeseries.subsampleCorrelatedData(u_n) # indices of uncorrelated samples
    #indices = range(0,u_n.size) # DEBUG - assume samples are uncorrelated
    N = len(indices) # number of uncorrelated samples
    N_k[:] = N      
    u_kln[:,:,0:N] = u_kln[:,:,indices]
    print "number of uncorrelated samples:"
    print N_k
    print ""

    #===================================================================================================
    # Estimate free energy difference with MBAR.
    #===================================================================================================   
   
    # Initialize MBAR (computing free energy estimates, which may take a while)
    print "Computing free energy differences..."
    mbar = MBAR(u_kln, N_k, verbose = False, method = 'adaptive', maximum_iterations = 50000) # use slow self-consistent-iteration (the default)
    #mbar = MBAR(u_kln, N_k, verbose = False, method = 'self-consistent-iteration', maximum_iterations = 50000) # use slow self-consistent-iteration (the default)
    #mbar = MBAR(u_kln, N_k, verbose = True, method = 'Newton-Raphson') # use faster Newton-Raphson solver

    # Get matrix of dimensionless free energy differences and uncertainty estimate.
    print "Computing covariance matrix..."
    (Deltaf_ij, dDeltaf_ij) = mbar.getFreeEnergyDifferences(uncertainty_method='svd-ew')
   
#    # Matrix of free energy differences
    print "Deltaf_ij:"
    for i in range(nstates):
        for j in range(nstates):
            print "%8.3f" % Deltaf_ij[i,j],
        print ""        
    
#    print Deltaf_ij
#    # Matrix of uncertainties in free energy difference (expectations standard deviations of the estimator about the true free energy)
    print "dDeltaf_ij:"
    for i in range(nstates):
        for j in range(nstates):
            print "%8.3f" % dDeltaf_ij[i,j],
        print ""        

    # Return free energy differences and an estimate of the covariance.
    return (Deltaf_ij, dDeltaf_ij)

def estimate_enthalpies(ncfile, ndiscard = 0, nuse = None):
    """Estimate enthalpies of all alchemical states.

    ARGUMENTS
       ncfile (NetCDF) - input YANK netcdf file

    OPTIONAL ARGUMENTS
       ndiscard (int) - number of iterations to discard to equilibration
       nuse (int) - number of iterations to use (after discarding) 

    TODO: Automatically determine 'ndiscard'.
    TODO: Combine some functions with estimate_free_energies.
    """

    # Get current dimensions.
    niterations = ncfile.variables['energies'].shape[0]
    nstates = ncfile.variables['energies'].shape[1]
    natoms = ncfile.variables['energies'].shape[2]

    # Extract energies.
    print "Reading energies..."
    energies = ncfile.variables['energies']
    u_kln_replica = zeros([nstates, nstates, niterations], float64)
    for n in range(niterations):
        u_kln_replica[:,:,n] = energies[n,:,:]
    print "Done."

    # Deconvolute replicas
    print "Deconvoluting replicas..."
    u_kln = zeros([nstates, nstates, niterations], float64)
    for iteration in range(niterations):
        state_indices = ncfile.variables['states'][iteration,:]
        u_kln[state_indices,:,iteration] = energies[iteration,:,:]
    print "Done."

    # Compute total negative log probability over all iterations.
    u_n = zeros([niterations], float64)
    for iteration in range(niterations):
        u_n[iteration] = sum(diagonal(u_kln[:,:,iteration]))
    #print u_n

    # DEBUG
#    outfile = open('u_n.out', 'w')
#    for iteration in range(niterations):
#        outfile.write("%8d %24.3f\n" % (iteration, u_n[iteration]))
#    outfile.close()

    # Discard initial data to equilibration.
    u_kln_replica = u_kln_replica[:,:,ndiscard:]
    u_kln = u_kln[:,:,ndiscard:]
    u_n = u_n[ndiscard:]
    
    # Truncate to number of specified conformations to use
    if (nuse):
        u_kln_replica = u_kln_replica[:,:,0:nuse]
        u_kln = u_kln[:,:,0:nuse]
        u_n = u_n[0:nuse]

    # Subsample data to obtain uncorrelated samples
    N_k = zeros(nstates, int32)
    indices = timeseries.subsampleCorrelatedData(u_n) # indices of uncorrelated samples
    #indices = range(0,u_n.size) # DEBUG - assume samples are uncorrelated
    N = len(indices) # number of uncorrelated samples
    N_k[:] = N      
    u_kln[:,:,0:N] = u_kln[:,:,indices]
    print "number of uncorrelated samples:"
    print N_k
    print ""

    # Compute average enthalpies.
    H_k = zeros([nstates], float64) # H_i[i] is estimated enthalpy of state i
    dH_k = zeros([nstates], float64)
    for k in range(nstates):
        H_k[k] = u_kln[k,k,:].mean()
        dH_k[k] = u_kln[k,k,:].std() / sqrt(N)

    return (H_k, dH_k)

def extract_u_n(ncfile):
    """
    Extract timeseries of u_n = - log q(x_n)

    """

    # Get current dimensions.
    niterations = ncfile.variables['energies'].shape[0]
    nstates = ncfile.variables['energies'].shape[1]
    natoms = ncfile.variables['energies'].shape[2]

    # Extract energies.
    print "Reading energies..."
    energies = ncfile.variables['energies']
    u_kln_replica = numpy.zeros([nstates, nstates, niterations], numpy.float64)
    for n in range(niterations):
        u_kln_replica[:,:,n] = energies[n,:,:]
    print "Done."

    # Deconvolute replicas
    print "Deconvoluting replicas..."
    u_kln = numpy.zeros([nstates, nstates, niterations], numpy.float64)
    for iteration in range(niterations):
        state_indices = ncfile.variables['states'][iteration,:]
        u_kln[state_indices,:,iteration] = energies[iteration,:,:]
    print "Done."

    # Compute total negative log probability over all iterations.
    u_n = numpy.zeros([niterations], numpy.float64)
    for iteration in range(niterations):
        u_n[iteration] = numpy.sum(numpy.diagonal(u_kln[:,:,iteration]))

    return u_n

def detect_equilibration(A_t):
    """
    Automatically detect equilibrated region.

    ARGUMENTS

    A_t (numpy.array) - timeseries

    RETURNS

    t (int) - start of equilibrated data
    g (float) - statistical inefficiency of equilibrated data
    Neff_max (float) - number of uncorrelated samples   
    
    """
    T = A_t.size

    # Special case if timeseries is constant.
    if A_t.std() == 0.0:
        return (0, 1, T)
    
    g_t = numpy.ones([T-1], numpy.float32)
    Neff_t = numpy.ones([T-1], numpy.float32)
    for t in range(T-1):
        g_t[t] = timeseries.statisticalInefficiency(A_t[t:T])
        Neff_t[t] = (T-t+1) / g_t[t]
    
    Neff_max = Neff_t.max()
    t = Neff_t.argmax()
    g = g_t[t]
    
    return (t, g, Neff_max)

#=============================================================================================
# MAIN
#=============================================================================================

#data_directory = '/Users/yank/data-from-lincoln/T4-lysozyme-L99A/amber-gbsa/amber-gbsa/' # directory containing datafiles
#data_directory = '/Users/yank/data-from-lincoln/FKBP/amber-gbsa/' # directory containing datafiles
#data_directory = '/Users/yank/data-from-lincoln/FKBP/amber-gbvi/' # directory containing datafiles
#data_directory = '/uf/ac/jchodera/code/yank/test-systems/T4-lysozyme-L99A/amber-gbsa/amber-gbsa/' # directory containing datafiles
#data_directory = '/scratch/users/jchodera/yank/test-systems/T4-lysozyme-L99A/amber-gbsa/amber-gbsa/' # directory containing datafiles
data_directory = 'examples/p-xylene' # directory containing datafiles

# Store molecule data.
molecule_data = dict()

# Generate list of files in this directory.
import commands
molecules = commands.getoutput('ls -1 %s' % data_directory).split()
for molecule in molecules:
    source_directory = os.path.join(data_directory, molecule)
    print source_directory

    # Storage for different phases.
    data = dict()

    phases = ['vacuum', 'solvent', 'complex']    

    # Process each netcdf file.
    for phase in phases:
        # Construct full path to NetCDF file.
        fullpath = os.path.join(source_directory, phase + '.nc')

        # Skip if the file doesn't exist.
        if (not os.path.exists(fullpath)): continue

        # Open NetCDF file for reading.
        print "Opening NetCDF trajectory file '%(fullpath)s' for reading..." % vars()
        ncfile = netcdf.Dataset(fullpath, 'r')

        # DEBUG
        print "dimensions:"
        for dimension_name in ncfile.dimensions.keys():
            print "%16s %8d" % (dimension_name, len(ncfile.dimensions[dimension_name]))
    
        # Read dimensions.
        niterations = ncfile.variables['positions'].shape[0]
        nstates = ncfile.variables['positions'].shape[1]
        natoms = ncfile.variables['positions'].shape[2]
        print "Read %(niterations)d iterations, %(nstates)d states" % vars()

#        # Compute torsion trajectories
#        if phase in ['complex', 'receptor']:
#            print "Computing torsions..."
#            compute_torsion_trajectories(ncfile, os.path.join(source_directory, phase + ".val111"))

#        # Write out replica trajectories
#        print "Writing replica trajectories...\n"
#        title = 'Source %(source_directory)s phase %(phase)s' % vars()        
#        write_netcdf_replica_trajectories(source_directory, phase, title, ncfile)

        # Read reference PDB file.
        if phase in ['vacuum', 'solvent']:
            reference_pdb_filename = os.path.join(source_directory, "ligand.pdb")
        else:
            reference_pdb_filename = os.path.join(source_directory, "complex.pdb")
        atoms = read_pdb(reference_pdb_filename)

        # Write replica trajectories.
        #title = 'title'
        #write_pdb_replica_trajectories(reference_pdb_filename, source_directory, phase, title, ncfile, trajectory_by_state=False)
        
        # Check to make sure no self-energies go nan.
        check_energies(ncfile, atoms)

        # Check to make sure no positions are nan
        check_positions(ncfile)

        # Choose number of samples to discard to equilibration
        #nequil = 50
        #if phase == 'complex':
        #    nequil = 2000 # discard 2 ns of complex simulations
        u_n = extract_u_n(ncfile)
        [nequil, g_t, Neff_max] = detect_equilibration(u_n)
        print [nequil, Neff_max]
 
        # Examine acceptance probabilities.
        show_mixing_statistics(ncfile, cutoff=0.05, nequil=nequil)

        # Estimate free energies.
        (Deltaf_ij, dDeltaf_ij) = estimate_free_energies(ncfile, ndiscard = nequil)
    
        # Estimate average enthalpies
        (DeltaH_i, dDeltaH_i) = estimate_enthalpies(ncfile, ndiscard = nequil)
    
        # Accumulate free energy differences
        entry = dict()
        entry['DeltaF'] = Deltaf_ij[0,nstates-1] 
        entry['dDeltaF'] = dDeltaf_ij[0,nstates-1]
        entry['DeltaH'] = DeltaH_i[nstates-1] - DeltaH_i[0]
        entry['dDeltaH'] = numpy.sqrt(dDeltaH_i[0]**2 + dDeltaH_i[nstates-1]**2)
        data[phase] = entry

        # Get temperatures.
        ncvar = ncfile.groups['thermodynamic_states'].variables['temperatures']
        temperature = ncvar[0] * units.kelvin
        kT = kB * temperature

        # Close input NetCDF file.
        ncfile.close()

    # Skip if we have no data.
    if not ('vacuum' in data) or ('solvent' in data) or ('complex' in data): continue
    
    if (data.haskey('vacuum') and data.haskey('solvent')):
        # Compute hydration free energy (free energy of transfer from vacuum to water)
        DeltaF = data['vacuum']['DeltaF'] - data['solvent']['DeltaF']
        dDeltaF = numpy.sqrt(data['vacuum']['dDeltaF']**2 + data['solvent']['dDeltaF']**2)
        print "Hydration free energy: %.3f +- %.3f kT (%.3f +- %.3f kcal/mol)" % (DeltaF, dDeltaF, DeltaF * kT / units.kilocalories_per_mole, dDeltaF * kT / units.kilocalories_per_mole)

        # Compute enthalpy of transfer from vacuum to water
        DeltaH = data['vacuum']['DeltaH'] - data['solvent']['DeltaH']
        dDeltaH = numpy.sqrt(data['vacuum']['dDeltaH']**2 + data['solvent']['dDeltaH']**2)
        print "Enthalpy of hydration: %.3f +- %.3f kT (%.3f +- %.3f kcal/mol)" % (DeltaH, dDeltaH, DeltaH * kT / units.kilocalories_per_mole, dDeltaH * kT / units.kilocalories_per_mole)

    # Read standard state correction free energy.
    DeltaF_restraints = 0.0
    phase = 'complex'
    fullpath = os.path.join(source_directory, phase + '.nc')
    ncfile = netcdf.Dataset(fullpath, 'r')
    DeltaF_restraints = ncfile.groups['metadata'].variables['standard_state_correction'][0]
    ncfile.close()
    
    # Compute binding free energy (free energy of transfer from vacuum to water)
    DeltaF = data['solvent']['DeltaF'] - DeltaF_restraints - data['complex']['DeltaF']
    dDeltaF = numpy.sqrt(data['solvent']['dDeltaF']**2 + data['complex']['dDeltaF']**2)
    print ""
    print "Binding free energy : %16.3f +- %.3f kT (%16.3f +- %.3f kcal/mol)" % (DeltaF, dDeltaF, DeltaF * kT / units.kilocalories_per_mole, dDeltaF * kT / units.kilocalories_per_mole)
    print ""
    print "DeltaG vacuum       : %16.3f +- %.3f kT" % (data['vacuum']['DeltaF'], data['vacuum']['dDeltaF'])
    print "DeltaG solvent      : %16.3f +- %.3f kT" % (data['solvent']['DeltaF'], data['solvent']['dDeltaF'])
    print "DeltaG complex      : %16.3f +- %.3f kT" % (data['complex']['DeltaF'], data['complex']['dDeltaF'])
    print "DeltaG restraint    : %16.3f          kT" % DeltaF_restraints
    print ""

    # Compute binding enthalpy
    DeltaH = data['solvent']['DeltaH'] - DeltaF_restraints - data['complex']['DeltaH'] 
    dDeltaH = numpy.sqrt(data['solvent']['dDeltaH']**2 + data['complex']['dDeltaH']**2)
    print "Binding enthalpy    : %16.3f +- %.3f kT (%16.3f +- %.3f kcal/mol)" % (DeltaH, dDeltaH, DeltaH * kT / units.kilocalories_per_mole, dDeltaH * kT / units.kilocalories_per_mole)

    # Store molecule data.
    molecule_data[molecule] = data

# Extract sorted binding affinities.
sorted_molecules = ['1-methylpyrrole',
                    '1,2-dichlorobenzene',
                    '2-fluorobenzaldehyde',
                    '2,3-benzofuran',
                    'benzene',
                    'ethylbenzene',
                    'indene',
                    'indole',
                    'isobutylbenzene',
                    'n-butylbenzene',
                    'N-methylaniline',
                    'n-propylbenzene',
                    'o-xylene',
                    'p-xylene',
                    'phenol',
                    'toluene']

print ""
print "DeltaG"                                                                           
for molecule in sorted_molecules:
    try:
        DeltaF = molecule_data[molecule]['solvent']['DeltaF'] - molecule_data[molecule]['DeltaF_restraints'] - molecule_data[molecule]['complex']['DeltaF']
        dDeltaF = sqrt(molecule_data[molecule]['solvent']['dDeltaF']**2 + molecule_data[molecule]['complex']['dDeltaF']**2)
        print "%8.3f %8.3f %% %s" % (DeltaF, dDeltaF, molecule)
    except:
        print "%8.3f %8.3f %% %s" % (0.0, 0.0, molecule)        
        pass

print ""
print "DeltaH"                                                                           
for molecule in sorted_molecules:
    try:
        DeltaH = molecule_data[molecule]['solvent']['DeltaH'] - molecule_data[molecule]['complex']['DeltaH']
        dDeltaH = sqrt(molecule_data[molecule]['solvent']['dDeltaH']**2 + molecule_data[molecule]['complex']['dDeltaH']**2)
        print "%8.3f %8.3f %% %s" % (DeltaH, dDeltaH, molecule)
    except:
        print "%8.3f %8.3f %% %s" % (0.0, 0.0, molecule)                
        pass

