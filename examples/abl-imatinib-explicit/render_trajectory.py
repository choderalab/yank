#!/opt/local/bin/python2.5

#=============================================================================================
# Render a replica trajectory in PyMOL
#=============================================================================================

#=============================================================================================
# REQUIREMENTS
#
# This code requires the NetCDF module, available either as Scientific.IO.NetCDF or standalone through pynetcdf:
# http://pypi.python.org/pypi/pynetcdf/
# http://sourceforge.net/project/showfiles.php?group_id=1315&package_id=185504
#=============================================================================================

#=============================================================================================
# TODO
#=============================================================================================

#=============================================================================================
# CHAGELOG
#=============================================================================================

#=============================================================================================
# VERSION CONTROL INFORMATION
# * 2009-08-01 JDC
# Created file.
#=============================================================================================

#=============================================================================================
# IMPORTS
#=============================================================================================

import numpy
from numpy import *
#import Scientific.IO.NetCDF
import netCDF4 as NetCDF
import os
import os.path
from pymol import cmd
from pymol import util

# try to keep PyMOL quiet
#cmd.set('internal_gui', 0)
#cmd.feedback("disable","all","actions")
#cmd.feedback("disable","all","results")

#=============================================================================================
# PARAMETERS
#=============================================================================================

#=============================================================================================
# SUBROUTINES
#=============================================================================================

def readAtomsFromPDB(pdbfilename):
    """Read atom records from the PDB and return them in a list.

    present_sequence = getPresentSequence(pdbfilename, chain=' ')
    contents of protein.seqfile
    REQUIRED ARGUMENTS
      pdbfilename - the filename of the PDB file to import from

    OPTIONAL ARGUMENTS
      chain - the one-character chain ID of the chain to import (default ' ')

    RETURN VALUES
      atoms - a list of atom{} dictionaries

    The ATOM records are read, and the sequence for which there are atomic coordinates is stored.

    """

    # Read the PDB file into memory.
    pdbfile = open(pdbfilename, 'r')
    lines = pdbfile.readlines()
    pdbfile.close()


    # Read atoms.
    atoms = []
    for line in lines:
        if line[0:5] == "ATOM ":
            # Parse line into fields.
            atom = { }
            atom["serial"] = int(line[5:11])
            atom["name"] = line[12:16]
            atom["altLoc"] = line[16:17]
            atom["resName"] = line[17:21]
            atom["chainID"] = line[21:22]
            atom["resSeq"] = int(line[22:26])
            atom["iCode"] = line[26:27]
            atom["x"] = float(line[30:38])
            atom["y"] = float(line[38:46])
            atom["z"] = float(line[46:54])

            atom["occupancy"] = 1.0
            if (line[54:60].strip() != ''):
              atom["occupancy"] = float(line[54:60])

            atom["tempFactor"] = 0.0
            if (line[60:66].strip() != ''):
              atom["tempFactor"] = float(line[60:66])

            atom["segID"] = line[72:76]
            atom["element"] = line[76:78]
            atom["charge"] = line[78:80]

            # Mangle resSeq|iCode:
            if atom['iCode'] != ' ':
                atom['resSeq'] = str(atom['resSeq']) + atom['iCode']
                atom['iCode'] = ' '

            atoms.append(atom)

    # Return list of atoms.
    return atoms

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
        #ncoutfile = NetCDF.NetCDFFile(output_filename, 'w')
        ncoutfile = NetCDF.Dataset(output_filename, 'w')
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

#=============================================================================================
# MAIN
#=============================================================================================

#import __main__
#__main__.pymol_argv = [ 'pymol', '-qc']
#import pymol
#pymol.finish_launching()

# DEBUG: ANALYSIS PATH IS HARD-CODED FOR NOW
source_directory = 'experiments'

reference_pdbfile = 'setup/systems/Abl_STI_rf/complex.pdb'
phase = 'complex-explicit'
replica = 0 # replica index to render
#replica = 15 # replica index to render

# Load PDB file.
cmd.rewind()
cmd.delete('all')
cmd.reset()
cmd.load(reference_pdbfile, 'complex')
cmd.remove('resn WAT') # remove waters
cmd.select('receptor', '(not resn MOL) and (not resn WAT) and (not hydrogen)')
cmd.select('ligand', 'resn MOL and not hydrogen')
cmd.select('ions', 'resn Na\+ or resn Cl\-')
cmd.deselect()
cmd.hide('all')
cmd.show('cartoon', 'receptor')
cmd.show('spheres', 'ligand')
cmd.show('spheres', 'ions')
util.cbay('ligand')
cmd.color('green', 'receptor')

# speed up builds
cmd.set('defer_builds_mode', 3)
cmd.set('cache_frames', 0)

cmd.set('ray_transparency_contrast', 3.0)
cmd.set('ray_transparency_shadows', 0)

model = cmd.get_model('complex')
#for atom in model.atom:
#    print "%8d %4s %3s %5d %8.3f %8.3f %8.3f" % (atom.index, atom.name, atom.resn, int(atom.resi), atom.coord[0], atom.coord[1], atom.coord[2])

#pymol.finish_launching()

# Read atoms from PDB
pdbatoms = readAtomsFromPDB(reference_pdbfile)

# Build mappings.
pdb_indices = dict()
for (index, atom) in enumerate(pdbatoms):
    if atom['chainID'] == ' ': atom['chainID'] = ''
    #if atom['resName'] == 'WAT': continue
    key = (atom['chainID'], int(atom['resSeq']), atom['name'].strip())
    value = index
    pdb_indices[key] = value
print "pdb_indices has %d entries" % len(pdb_indices.keys())

model_indices = dict()
for (index, atom) in enumerate(model.atom):
    #if atom.resn == 'WAT': continue
    key = (atom.chain, int(atom.resi), atom.name)
    value = index
    model_indices[key] = value
print "model_indices has %d entries" % len(model_indices.keys())

#model_mapping = list()
#for (pdb_index, atom) in enumerate(pdbatoms):
#    #if atom['resName'] == 'WAT': continue
#    key = (atom['chainID'], int(atom['resSeq']), atom['name'].strip())
#    model_mapping.append(model_indices[key])

# Omit waters.
pdb_mapping = list()
for (index, atom) in enumerate(model.atom):
    #if atom.resn == 'WAT': continue
    key = (atom.chain, int(atom.resi), atom.name)
    pdb_mapping.append(pdb_indices[key])
print pdb_mapping

# Construct full path to NetCDF file.
fullpath = os.path.join(source_directory, phase + '.nc')

# Open NetCDF file for reading.
print "Opening NetCDF trajectory file '%(fullpath)s' for reading..." % vars()
#ncfile = Scientific.IO.NetCDF.NetCDFFile(fullpath, 'r')
ncfile = NetCDF.Dataset(fullpath, 'r')

# DEBUG
print "dimensions:"
print ncfile.dimensions

# Read dimensions.
[niterations,nstates,natoms,ndim] = ncfile.variables['positions'].shape
print "Read %(niterations)d iterations, %(nstates)d states" % vars()

cmd.viewport(640,480)
#niterations = 10 # DEBUG

# Load frames
cmd.set('all_states', 0)
print "Loading frames..."
for iteration in range(niterations):
    # Set coordinates
    print "iteration %8d / %8d" % (iteration, niterations)
    positions = (10.0 * ncfile.variables['positions'][iteration, replica, :, :]).squeeze()
    positions = positions[pdb_mapping,:]
    xyz = positions.tolist()
    xyz_iter = iter(xyz)
    #cmd.load_model(model, 'complex', state=iteration+1)
    #cmd.frame(iteration+1)
    #model = cmd.get_model('complex', state=1)
    #cmd.load_model(model, 'complex', state=iteration+1)
    cmd.create('complex', 'complex', 1, iteration+1)
    cmd.alter_state(iteration+1, 'complex', '(x,y,z) = xyz_iter.next()', space=locals())

    #for pdb_index in range(natoms):
        #if (pdb_index % 100)==0: print pdb_index
        #model_index = model_mapping[pdb_index]
        #model.atom[model_index].coord = (10 * ncfile.variables['positions'][iteration, replica, pdb_index, :]).squeeze().tolist()
        #for k in range(3):
        #    model.atom[model_index].coord[k] = float(ncfile.variables['positions'][iteration, replica, pdb_index, k]) * 10.0 # convert to angstroms
    #cmd.load_model(model, 'complex', state=iteration+1)
    #cmd.load_model(model, 'complex')

print "done"

# Align all states
cmd.intra_fit('all')

cmd.hide('all')
#cmd.rewind()


cmd.select('receptor', '(not resn MOL) and (not resn WAT) and (not hydrogen)')
cmd.select('ligand', 'resn MOL and not hydrogen')
cmd.select('ions', 'resn Na\+ or resn Cl\-')
cmd.deselect()
cmd.hide('all')
cmd.show('cartoon', 'receptor')
cmd.show('spheres', 'ligand')
cmd.show('spheres', 'ions')
util.cbay('ligand')
cmd.color('green', 'receptor')

cmd.show('surface', 'receptor')

cmd.set('transparency', 0.65)
cmd.set('surface_mode', 3)
cmd.set('surface_color', 'white')


# Create one-to-one mapping between states and frames.
cmd.mset("1 -%d" % cmd.count_states())

# Zoom viewport
cmd.zoom('complex')
#cmd.orient('complex')

#cmd.zoom('ligand')
#cmd.orient('ligand')
#cmd.turn('x', -90)

# Render movie
frame_prefix = 'frames/frame'
cmd.set('ray_trace_frames', 1)
cmd.set('ray_trace_frames', 0) # DEBUG
for iteration in range(niterations):
    print "rendering frame %04d / %04d" % (iteration+1, niterations)
    cmd.frame(iteration+1)
    cmd.rotate([0,0,1], 1, 'all')
    state_index = int(ncfile.variables['states'][iteration, replica])
    cmd.set('sphere_transparency', float(state_index) / float(nstates-1))
    cmd.png(frame_prefix + '%04d.png' % (iteration), ray=True)
    #cmd.mpng(frame_prefix, iteration+1, iteration+1)
    #cmd.load_model(model, 'complex')

cmd.set('ray_trace_frames', 0)

# Close file
ncfile.close()
