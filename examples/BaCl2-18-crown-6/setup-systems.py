#=============================================================================================
# Example illustrating use of 'moltools' in computing absolute hydration free energies of small molecules.
#
# PROTOCOL
#
# * Construct molecule from IUPAC name (protonation and tautomeric states are heuristically guessed) [OpenEye OEMol]
# * Generate multiple likely conformations to use for replicates [OpenEye Omega]
# * Parameterize the molecule with GAFF [AmberTools Antechamber]
# * Set up the solvated system [AmberTools LEaP]
# * Convert the system to gromacs topology and coordinates
# * Set up vacuum and solvated runs with gromacs_dg
# 
# Written by John D. Chodera <jchodera@gmail.com> 2008-01-23
#=============================================================================================

#=============================================================================================
# Imports
#=============================================================================================

import commands
import os    

from mmtools.moltools.ligandtools import *

from numpy import *

#=============================================================================================
# CHANGELOG
#============================================================================================
# 2008-04-19 JDC
# * Both vacuum and solvated simulations now have counterions added for charged solutes.
#=============================================================================================

#=============================================================================================
# KNOWN BUGS
#============================================================================================
# * A vacuum simulation should not be set up for decoupling simulations.
# * Multiple solute conformations should be used to see different replicates and the solute should start in the decoupled state.
#=============================================================================================

#=============================================================================================
# PARAMETERS
#=============================================================================================

clearance = 5.0 # clearance around solute for box construction, in Angstroms

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
#=============================================================================================
def setupSystems(solute, verbose = True, jobname = "hydration"):
    """Set up an absolute alchemical hydration free energy calculation for the given molecule.

    ARGUMENTS
      solute (OEMol) - the molecule for which hydration free energy is to be computed (with fully explicit hydrogens) in the desired protonation state.

    OPTIONAL ARGUMENTS
      verbose (boolean) - if True, extra debug information will be printed (default: True)
      jobname (string) - string to use for job name (default: "hydration")

    NOTES
      A directory will be created 'molecules/[molecule name]' as obtained from molecule.GetTitle().
    
    """

    # get current directory
    current_path = os.getcwd()

    # Center the solute molecule.
    OECenter(solute)

    # get molecule name
    solute_name = molecule.GetTitle()
    if verbose: print solute_name

    # create molecule path/directory
    work_path = os.path.abspath(os.path.join('molecules', solute_name))
    os.makedirs(work_path)

    # SET UP SOLUTE TOPOLOGY

    if verbose: print "\nCONSTRUCTING SOLUTE TOPOLOGY"

    # Get formal charge of ligand.
    solute_charge = formalCharge(solute)
    if verbose: print "solute formal charge is %d" % solute_charge
    
    # Write molecule with explicit hydrogens to mol2 file.
    print "Writing solute mol2 file..."
    solute_mol2_filename = os.path.abspath(os.path.join(work_path, 'solute.mol2'))
    writeMolecule(solute, solute_mol2_filename)

    # Set substructure name (which will become residue name).
    print "Modifying molecule name..."
    modifySubstructureName(solute_mol2_filename, 'MOL')

    # Run antechamber to assign GAFF atom types.
    print "Running antechamber..."
    os.chdir(work_path)
    gaff_mol2_filename = os.path.join(work_path, 'solute.gaff.mol2')
    charge_model = 'bcc'
    command = 'antechamber -i %(solute_mol2_filename)s -fi mol2 -o solute.gaff.mol2 -fo mol2 -c %(charge_model)s -nc %(solute_charge)d > antechamber.out' % vars()
    if verbose: print command
    output = commands.getoutput(command)
    if verbose: print output
    os.chdir(current_path)

    # Generate frcmod file for additional GAFF parameters.
    solute_frcmod_filename = os.path.join(work_path, 'frcmod.solute')
    command = 'parmchk -i %(gaff_mol2_filename)s -f mol2 -o %(solute_frcmod_filename)s' % vars()
    if verbose: print command
    output = commands.getoutput(command)
    if verbose: print output

    # Run LEaP to generate topology / coordinates.
    solute_prmtop_filename = os.path.join(work_path,'solute.prmtop')
    solute_crd_filename = os.path.join(work_path,'solute.crd')
    solute_off_filename = os.path.join(work_path, 'solute.off')
    
    tleap_input_filename = os.path.join(work_path, 'setup-solute.leap.in')
    tleap_output_filename = os.path.join(work_path, 'setup-solute.leap.out')
    contents = """
# Load GAFF parameters.
source leaprc.gaff

# load antechamber-generated additional parameters
mods = loadAmberParams %(solute_frcmod_filename)s

# load solute
solute = loadMol2 %(gaff_mol2_filename)s

# check the solute
check solute

# report net charge
charge solute

# save AMBER parameters
saveAmberParm solute %(solute_prmtop_filename)s %(solute_crd_filename)s

# write .off file
saveOff solute %(solute_off_filename)s

# exit
quit
""" % vars()
    write_file(tleap_input_filename, contents)
    command = 'tleap -f %(tleap_input_filename)s > %(tleap_output_filename)s' % vars()
    output = commands.getoutput(command)

    # extract total charge
    solute_charge = commands.getoutput('grep "Total unperturbed charge" %(tleap_output_filename)s | cut -c 27-' % vars())
    solute_charge = int(round(float(solute_charge))) # round to nearest whole charge
    if verbose: print "solute charge is %d" % solute_charge    

    # PREPARE SOLVATED SOLUTE

    print "\nPREPARING SOLVATED SOLUTE"

    # create the directory if it doesn't exist
    solvent_path = os.path.join(work_path, 'solvent')
    if not os.path.exists(solvent_path):
        os.makedirs(solvent_path)

    # solvate the solute
    print "Solvating the solute with tleap..."
    system_prmtop_filename = os.path.join(solvent_path,'system.prmtop')
    system_crd_filename = os.path.join(solvent_path,'system.crd')
    tleap_input_filename = os.path.join(solvent_path, 'setup-system.leap.in')
    tleap_output_filename = os.path.join(solvent_path, 'setup-system.leap.out')
    clearance = globals()['clearance'] # clearance around solute (in A)
    contents = """
source leaprc.ff99
source leaprc.gaff

# load antechamber-generated additional parameters
mods = loadAmberParams %(solute_frcmod_filename)s

# Load solute.
loadOff %(solute_off_filename)s

# Create system.
system = combine { solute }
""" % vars()
    # add counterions
    if (solute_charge != 0):
        nions = abs(solute_charge)
        if solute_charge < 0: iontype = 'Na+'
        if solute_charge > 0: iontype = 'Cl-'
        contents += """
# Add counterions.
addions system %(iontype)s %(nions)d
""" % vars()
    #
    contents += """
# Solvate in truncated octahedral box.
solvateBox system TIP3PBOX %(clearance)f iso

# Check the system
check system

# Write the system
saveamberparm system %(system_prmtop_filename)s %(system_crd_filename)s
""" % vars()    
    write_file(tleap_input_filename, contents)
    command = 'tleap -f %(tleap_input_filename)s > %(tleap_output_filename)s' % vars()
    output = commands.getoutput(command)    

    # SET UP VACUUM SIMULATION

    # construct pathname for vacuum simulations
    vacuum_path = os.path.join(work_path, 'vacuum')
    if not os.path.exists(vacuum_path):
        os.makedirs(vacuum_path)

    # solvate the solute
    print "Preparing vacuum solute with tleap..."
    system_prmtop_filename = os.path.join(vacuum_path,'system.prmtop')
    system_crd_filename = os.path.join(vacuum_path,'system.crd')
    tleap_input_filename = os.path.join(vacuum_path, 'setup-system.leap.in')
    tleap_output_filename = os.path.join(vacuum_path, 'setup-system.leap.out')
    clearance = 50.0 # clearance in A
    contents = """
source leaprc.ff99
source leaprc.gaff

# load antechamber-generated additional parameters
mods = loadAmberParams %(solute_frcmod_filename)s

# Load solute.
loadOff %(solute_off_filename)s

# Create system.
system = combine { solute }
""" % vars()
    # add counterions
    if (solute_charge != 0):
        nions = abs(solute_charge)
        if solute_charge < 0: iontype = 'Na+'
        if solute_charge > 0: iontype = 'Cl-'
        contents += """
# Add counterions.
addions system %(iontype)s %(nions)d
""" % vars()
    #
    contents += """

# Create big box.
setBox system centers %(clearance)f iso

# Check the system
check system

# Write the system
saveamberparm system %(system_prmtop_filename)s %(system_crd_filename)s
""" % vars()    
    write_file(tleap_input_filename, contents)
    command = 'tleap -f %(tleap_input_filename)s > %(tleap_output_filename)s' % vars()
    output = commands.getoutput(command)    
    
    # convert to gromacs
    print "Converting to gromacs..."    
    amb2gmx = os.path.join(os.getenv('MMTOOLSPATH'), 'converters', 'amb2gmx.pl')
    os.chdir(vacuum_path)    
    command = '%(amb2gmx)s --prmtop system.prmtop --crd system.crd --outname system' % vars()
    print command
    output = commands.getoutput(command)
    print output
    os.chdir(current_path)    

    # make a PDB file for checking
    print "Converting system to PDB..."
    os.chdir(vacuum_path)
    command = 'cat system.crd | ambpdb -p system.prmtop > system.pdb' % vars()
    output = commands.getoutput(command)
    print output
    os.chdir(current_path)


    return


#=============================================================================================
# MAIN
#=============================================================================================

# Create a molecule.
# molecule = createMoleculeFromIUPAC('phenol')
# molecule = createMoleculeFromIUPAC('biphenyl')
# molecule = createMoleculeFromIUPAC('4-(2-Methoxyphenoxy) benzenesulfonyl chloride')

# molecule = readMolecule('/Users/jchodera/projects/CUP/SAMPL-2008/jnk3/data-from-openeye/jnk.aff/jnk.aff-1.sdf', normalize = True)

# List of tuples describing molecules to construct:
# ('common name to use for directory', 'IUPAC name', formal_charge to select)
molecules = [
#    ('phenol', 'phenol', 0), # phenol
#    ('N-methylacetamide', 'N-methylacetamide', 0),
#    ('E-oct-2-enal', 'E-oct-2-enal', 0),
#    ('catechol', 'catechol', 0),
#    ('trimethylphosphate', 'trimethylphosphate', 0),
#    ('trimethylphosphate', 'triacetylglycerol', 0),
#    ('imatinib', '4-[(4-methylpiperazin-1-yl)methyl]-N-[4-methyl-3-[(4-pyridin-3-ylpyrimidin-2-yl)amino]-phenyl]-benzamide', 0), # imatinib
#    ('NanoKid', '2-(2,5-bis(3,3-dimethylbut-1-ynyl)-4-(2-(3,5-di(pent-1-ynyl)phenyl)ethynyl)phenyl)-1,3-dioxolane', 0), # NanoKid - http://en.wikipedia.org/wiki/Nanoputian
#    ('lipitor', '[R-(R*,R*)]-2-(4-fluorophenyl)-beta,delta-dihydroxy-5-(1-methylethyl)-3-phenyl-4-[(phenylamino)carbonyl]-1H-pyrrole-1-heptanoic acid', 0), # broken

    # trypsin inhibitors from Alan E. Mark study JACS 125:10570, 2003.
#    ('benzamidine', 'benzamidine', 1), # benzamidine
#    ('p-methylbenzamidine', 'p-methylbenzamidine', 1), # R = Me
#    ('p-ethylbenzamidine', 'p-ethylbenzamidine', 1), # R = Et
#    ('p-n-propylbenzamidine', 'p-n-propylbenzamidine', 1), # R = n-Pr
#    ('p-isopropylbenzamidine', 'p-isopropylbenzamidine', 1), # R = i-Pr
#    ('p-n-butylbenzamidine', 'p-n-butylbenzamidine', 1), # R = n-Bu
#    ('p-t-butylbenzamidine', 'p-t-butylbenzamidine', 1), # R = t-Bu
#    ('p-n-pentylbenzamidine', 'p-n-pentylbenzamidine', 1), # R = n-Pent
#    ('p-n-hexylbenzamidine', 'p-n-hexylbenzamidine', 1), # R = n-Hex
    ('18-crown-6', '18-crown-6', 0), # crown ether
    ]

for (molecule_name, molecule_iupac_name, formal_charge) in molecules:
    print "Setting up %s" % molecule_name

    # Create the molecule from the common name with the desired formal charge.
    molecule = createMoleculeFromIUPAC(molecule_iupac_name, charge = formal_charge)
    if not molecule:
        print "Could not build '%(molecule_name)s' from IUPAC name '%(molecule_iupac_name)s', skipping..." % vars()
        continue
    print "Net charge is %d" % formalCharge(molecule)

    # Replace the title with the common name
    molecule.SetTitle(molecule_name)

    # Expand set of conformations so we have multiple conformations to start from.
    expandConformations(molecule)
    
    # Set up systems.
    setupSystems(molecule, jobname = molecule_name)

    print "\n\n"

