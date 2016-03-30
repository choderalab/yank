#!/usr/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Test YAML functions.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import time
import shutil
import textwrap
import unittest
import tempfile
import itertools

from nose.tools import raises

from yank.yamlbuild import *
from mdtraj.formats.mol2 import mol2_to_dataframes

#=============================================================================================
# SUBROUTINES FOR TESTING
#=============================================================================================

standard_protocol = """
        absolute-binding:
            phases:
                complex:
                    alchemical_path:
                        lambda_electrostatics: [1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
                        lambda_sterics: [1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
                solvent:
                    alchemical_path:
                        lambda_electrostatics: [1.0, 0.8, 0.6, 0.3, 0.0]
                        lambda_sterics: [1.0, 0.8, 0.6, 0.3, 0.0]"""

def indent(str):
    """Put 4 extra spaces in front of every line."""
    return '\n    '.join(str.split('\n'))


def examples_paths():
    """Return the absolute path to the Yank examples relevant to tests."""
    paths = {}
    examples_dir = utils.get_data_filename(os.path.join('..', 'examples'))
    p_xylene_dir = os.path.join(examples_dir, 'p-xylene-implicit', 'setup')
    ben_tol_dir = os.path.join(examples_dir, 'benzene-toluene-explicit', 'setup')
    abl_imatinib_dir = os.path.join(examples_dir, 'abl-imatinib-explicit', 'setup')
    paths['lysozyme'] = os.path.join(p_xylene_dir, 'receptor.pdbfixer.pdb')
    paths['p-xylene'] = os.path.join(p_xylene_dir, 'ligand.tripos.mol2')
    paths['benzene'] = os.path.join(ben_tol_dir, 'benzene.tripos.mol2')
    paths['toluene'] = os.path.join(ben_tol_dir, 'toluene.tripos.mol2')
    paths['abl'] = os.path.join(abl_imatinib_dir, '2HYY-pdbfixer.pdb')
    return paths

#=============================================================================================
# UNIT TESTS
#=============================================================================================

def test_compute_min_dist():
    """Test computation of minimum distance between two molecules"""
    mol1_pos = np.array([[-1, -1, -1], [1, 1, 1]], np.float)
    mol2_pos = np.array([[3, 3, 3], [3, 4, 5]], np.float)
    mol3_pos = np.array([[2, 2, 2], [2, 4, 5]], np.float)
    assert compute_min_dist(mol1_pos, mol2_pos, mol3_pos) == np.sqrt(3)

def test_compute_dist_bound():
    """Test compute_dist_bound() function."""
    mol1_pos = np.array([[-1, -1, -1], [1, 1, 1]])
    mol2_pos = np.array([[2, 2, 2], [2, 4, 5]])  # determine min dist
    mol3_pos = np.array([[3, 3, 3], [3, 4, 5]])  # determine max dist
    min_dist, max_dist = compute_dist_bound(mol1_pos, mol2_pos, mol3_pos)
    assert min_dist == np.linalg.norm(mol1_pos[1] - mol2_pos[0])
    assert max_dist == np.linalg.norm(mol1_pos[1] - mol3_pos[1])

def test_remove_overlap():
    """Test function remove_overlap()."""
    mol1_pos = np.array([[-1, -1, -1], [1, 1, 1]], np.float)
    mol2_pos = np.array([[1, 1, 1], [3, 4, 5]], np.float)
    mol3_pos = np.array([[2, 2, 2], [2, 4, 5]], np.float)
    assert compute_min_dist(mol1_pos, mol2_pos, mol3_pos) < 0.1
    mol1_pos = remove_overlap(mol1_pos, mol2_pos, mol3_pos, min_distance=0.1, sigma=2.0)
    assert compute_min_dist(mol1_pos, mol2_pos, mol3_pos) >= 0.1

def test_pull_close():
    """Test function pull_close()."""
    mol1_pos = np.array([[-1, -1, -1], [1, 1, 1]], np.float)
    mol2_pos = np.array([[-1, -1, -1], [1, 1, 1]], np.float)
    mol3_pos = np.array([[10, 10, 10], [13, 14, 15]], np.float)
    translation2 = pull_close(mol1_pos, mol2_pos, 1.5, 5)
    translation3 = pull_close(mol1_pos, mol3_pos, 1.5, 5)
    assert isinstance(translation2, np.ndarray)
    assert 1.5 <= compute_min_dist(mol1_pos, mol2_pos + translation2) <= 5
    assert 1.5 <= compute_min_dist(mol1_pos, mol3_pos + translation3) <= 5

def test_pack_transformation():
    """Test function pack_transformation()."""
    BOX_SIZE = 5
    CLASH_DIST = 1

    mol1 = np.array([[-1, -1, -1], [1, 1, 1]], np.float)
    mols = [np.copy(mol1),  # distance = 0
            mol1 + 2 * BOX_SIZE]  # distance > box
    mols_affine = [np.append(mol, np.ones((2, 1)), axis=1) for mol in mols]

    transformations = [pack_transformation(mol1, mol2, CLASH_DIST, BOX_SIZE) for mol2 in mols]
    for mol, transf in zip(mols_affine, transformations):
        assert isinstance(transf, np.ndarray)
        mol2 = mol.dot(transf.T)[:, :3]  # transform and "de-affine"
        min_dist, max_dist = compute_dist_bound(mol1, mol2)
        assert CLASH_DIST <= min_dist and max_dist <= BOX_SIZE

def test_yaml_parsing():
    """Check that YAML file is parsed correctly."""

    # Parser handles no options
    yaml_content = """
    ---
    test: 2
    """
    yaml_builder = YamlBuilder(textwrap.dedent(yaml_content))
    assert len(yaml_builder.options) == len(yaml_builder.DEFAULT_OPTIONS)
    assert len(yaml_builder.yank_options) == 0

    # Correct parsing
    yaml_content = """
    ---
    metadata:
        title: Test YANK YAML YAY!

    options:
        verbose: true
        mpi: yes
        resume_setup: true
        resume_simulation: true
        output_dir: /path/to/output/
        setup_dir: /path/to/output/setup/
        experiments_dir: /path/to/output/experiments/
        pack: no
        temperature: 300*kelvin
        pressure: 1*atmosphere
        constraints: AllBonds
        hydrogen_mass: 2*amus
        restraint_type: harmonic
        randomize_ligand: yes
        randomize_ligand_sigma_multiplier: 2.0
        randomize_ligand_close_cutoff: 1.5 * angstrom
        mc_displacement_sigma: 10.0 * angstroms
        collision_rate: 5.0 / picosecond
        constraint_tolerance: 1.0e-6
        timestep: 2.0 * femtosecond
        nsteps_per_iteration: 2500
        number_of_iterations: 1000.999
        equilibration_timestep: 1.0 * femtosecond
        number_of_equilibration_iterations: 100
        minimize: False
        minimize_tolerance: 1.0 * kilojoules_per_mole / nanometers
        minimize_max_iterations: 0
        replica_mixing_scheme: swap-all
        online_analysis: no
        online_analysis_min_iterations: 20
        show_energies: True
        show_mixing_statistics: yes
        annihilate_sterics: no
        annihilate_electrostatics: true
    """

    yaml_builder = YamlBuilder(textwrap.dedent(yaml_content))
    assert len(yaml_builder.options) == 35
    assert len(yaml_builder.yank_options) == 23

    # Check correct types
    assert yaml_builder.options['constraints'] == openmm.app.AllBonds
    assert yaml_builder.yank_options['replica_mixing_scheme'] == 'swap-all'
    assert yaml_builder.yank_options['timestep'] == 2.0 * unit.femtoseconds
    assert yaml_builder.yank_options['constraint_tolerance'] == 1.0e-6
    assert yaml_builder.yank_options['nsteps_per_iteration'] == 2500
    assert type(yaml_builder.yank_options['nsteps_per_iteration']) is int
    assert yaml_builder.yank_options['number_of_iterations'] == 1000
    assert type(yaml_builder.yank_options['number_of_iterations']) is int
    assert yaml_builder.yank_options['minimize'] is False
    assert yaml_builder.yank_options['show_mixing_statistics'] is True

@raises(YamlParseError)
def test_yaml_unknown_options():
    """Check that YamlBuilder raises an exception when an unknown option is found."""
    # Raise exception on unknown options
    yaml_content = """
    ---
    options:
        wrong_option: 3
    """
    YamlBuilder(textwrap.dedent(yaml_content))

@raises(YamlParseError)
def test_yaml_wrong_option_value():
    """Check that YamlBuilder raises an exception when option type is wrong."""
    yaml_content = """
    ---
    options:
        minimize: 100
    """
    YamlBuilder(textwrap.dedent(yaml_content))

@raises(YamlParseError)
def test_no_molecule_source():
    """Test that an exception is raised if there's no source for molecule."""
    yaml_content = """
    ---
    molecules:
        moltest:
            parameters: antechamber
    """
    YamlBuilder(textwrap.dedent(yaml_content))

@raises(YamlParseError)
def test_no_molecule_parameters():
    """Test that an exception is raised if there are no parameters for molecule."""
    yaml_content = """
    ---
    molecules:
        moltest:
            filepath: moltest.pdb
    """
    YamlBuilder(textwrap.dedent(yaml_content))

@raises(YamlParseError)
def test_multiple_molecule_source():
    """An exception is raised if there are multiple sources for a molecule."""
    yaml_content = """
    ---
    molecules:
        moltest:
            filepath: moltest.mol2
            name: moltest
            parameters: antechamber
    """
    YamlBuilder(textwrap.dedent(yaml_content))

@raises(YamlParseError)
def test_unsupported_molecule_file():
    """An exception is raised with unsupported file formats."""
    yaml_content = """
    ---
    molecules:
        moltest:
            filepath: moltest.bla
            parameters: antechamber
    """
    YamlBuilder(textwrap.dedent(yaml_content))

@raises(YamlParseError)
def test_unknown_molecule_option():
    """An exception is raised if there are unknown molecule options."""
    yaml_content = """
    ---
    molecules:
        moltest:
            filepath: moltest.mol2
            blabla: moltest
            parameters: antechamber
    """
    YamlBuilder(textwrap.dedent(yaml_content))

@raises(YamlParseError)
def test_wrong_molecule_option():
    """An exception is raised if a molecule option have wrong type."""
    yaml_content = """
    ---
    molecules:
        moltest:
            filepath: 3
            parameters: antechamber
    """
    YamlBuilder(textwrap.dedent(yaml_content))

@raises(YamlParseError)
def test_no_nonbonded_method():
    """An exception is raised when a solvent doesn't specify nonbonded method."""
    yaml_content = """
    ---
    solvents:
        solvtest:
            nonbonded_cutoff: 3*nanometers
    """
    YamlBuilder(textwrap.dedent(yaml_content))

@raises(YamlParseError)
def test_implicit_solvent_consistence():
    """An exception is raised with NoCutoff and nonbonded_cutoff."""
    yaml_content = """
    ---
    solvents:
        solvtest:
            nonbonded_method: NoCutoff
            nonbonded_cutoff: 3*nanometers
    """
    YamlBuilder(textwrap.dedent(yaml_content))

@raises(YamlParseError)
def test_explicit_solvent_consistence():
    """An exception is raised with explicit nonbonded method and implicit_solvent."""
    yaml_content = """
    ---
    solvents:
        solvtest:
            nonbonded_method: PME
            implicit_solvent: OBC2
    """
    YamlBuilder(textwrap.dedent(yaml_content))

@raises(YamlParseError)
def test_unknown_solvent_option():
    """An exception is raised when there are unknown solvent options."""
    yaml_content = """
    ---
    solvents:
        solvtest:
            nonbonded_method: NoCutoff
            blabla: 3*nanometers
    """
    YamlBuilder(textwrap.dedent(yaml_content))

@raises(YamlParseError)
def test_wrong_solvent_option():
    """An exception is raised when a solvent option has the wrong type."""
    yaml_content = """
    ---
    solvents:
        solvtest:
            nonbonded_method: NoCutoff
            implicit_solvent: OBX2
    """
    YamlBuilder(textwrap.dedent(yaml_content))

def test_alchemical_path():
    """Check that conversion to list of AlchemicalStates is correct."""
    yaml_content = """
    ---
    protocols:{}
    """.format(standard_protocol)
    yaml_builder = YamlBuilder(textwrap.dedent(yaml_content))
    alchemical_paths = yaml_builder._get_alchemical_paths('absolute-binding')

    assert len(alchemical_paths) == 2
    assert 'complex' in alchemical_paths
    assert 'solvent' in alchemical_paths

    complex_path = alchemical_paths['complex']
    assert isinstance(complex_path[0], AlchemicalState)
    assert complex_path[3]['lambda_electrostatics'] == 0.6
    assert complex_path[4]['lambda_sterics'] == 0.4
    assert complex_path[5]['lambda_restraints'] == 0.0
    assert len(complex_path) == 7

@raises(YamlParseError)
def test_mandatory_alchemical_path():
    """An exception is raised if alchemical path is not specified."""
    yaml_content = """
    ---
    protocols:
        absolute-binding:
            phases:
                complex:
                    alchemical_path:
                        lambda_electrostatics: [1.0, 0.8, 0.6, 0.3, 0.0]
                        lambda_sterics: [1.0, 0.8, 0.6, 0.3, 0.0]
                solvent:
                    alchemical_path:
                        lambda_electrostatics: [1.0, 0.8, 0.6, 0.3, 0.0]
    """
    YamlBuilder(textwrap.dedent(yaml_content))

def test_exp_sequence():
    """Test all experiments in a sequence are parsed."""
    yaml_content = """
    ---
    molecules:
        rec:
            filepath: rec.pdb
            parameters: oldff/leaprc.ff99SBildn
        lig:
            name: lig
            parameters: antechamber
    solvents:
        solv1:
            nonbonded_method: NoCutoff
        solv2:
            nonbonded_method: PME
            nonbonded_cutoff: 1*nanometer
            clearance: 10*angstroms
    protocols:{}
    experiment1:
        components:
            receptor: rec
            ligand: lig
            solvent: [solv1, solv2]
        protocol: absolute-binding
    experiment2:
        components:
            receptor: rec
            ligand: lig
            solvent: solv1
        protocol: absolute-binding
    experiments: [experiment1, experiment2]
    """.format(standard_protocol)
    yaml_builder = YamlBuilder(textwrap.dedent(yaml_content))
    assert len(yaml_builder._experiments) == 2


@raises(YamlParseError)
def test_unkown_component():
    """An exception is thrown if we cannot identify components."""
    yaml_content = """
    ---
    molecules:
        rec:
            filepath: rec.pdb
            parameters: oldff/leaprc.ff99SBildn
        lig:
            name: lig
            parameters: antechamber
    solvents:
        solv1:
            nonbonded_method: NoCutoff
    protocols:{}
    experiments:
        components:
            receptor: rec
            ligand: lig
            solvent: [solv1, solv2]
        protocol: absolute-bindings
    """.format(standard_protocol)
    YamlBuilder(textwrap.dedent(yaml_content))

@raises(YamlParseError)
def test_no_component():
    """An exception is thrown there are no components."""
    yaml_content = """
    ---
    protocols:{}
    experiments:
        protocols: absolute-bindings
    """.format(standard_protocol)
    YamlBuilder(textwrap.dedent(yaml_content))

@raises(YamlParseError)
def test_no_protocol():
    """An exception is thrown there is no protocol."""
    yaml_content = """
    ---
    molecules:
        mol:
            filepath: mol.pdb
            parameters: oldff/leaprc.ff99SBildn
    solvents:
        solv1:
            nonbonded_method: NoCutoff
    protocols:{}
    experiments:
        components:
            receptor: mol
            ligand: mol
            solvent: solv1
    """.format(standard_protocol)
    YamlBuilder(textwrap.dedent(yaml_content))

def test_yaml_mol2_antechamber():
    """Test antechamber setup of molecule files."""
    with utils.temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            output_dir: {}
            setup_dir: .
        molecules:
            benzene:
                filepath: {}
                parameters: antechamber
        """.format(tmp_dir, examples_paths()['benzene'])

        yaml_builder = YamlBuilder(textwrap.dedent(yaml_content))
        yaml_builder._db._setup_molecules('benzene')

        output_dir = os.path.join(tmp_dir, SetupDatabase.MOLECULES_DIR, 'benzene')
        gaff_path = os.path.join(output_dir, 'benzene.gaff.mol2')
        frcmod_path = os.path.join(output_dir, 'benzene.frcmod')

        # Get last modified time
        last_touched_gaff = os.stat(gaff_path).st_mtime
        last_touched_frcmod = os.stat(frcmod_path).st_mtime

        # Check that output files have been created
        assert os.path.exists(gaff_path)
        assert os.path.exists(frcmod_path)
        assert os.path.getsize(gaff_path) > 0
        assert os.path.getsize(frcmod_path) > 0

        # Check that setup_molecules do not recreate molecule files
        time.sleep(0.5)  # st_mtime doesn't have much precision
        yaml_builder._db._setup_molecules('benzene')
        assert last_touched_gaff == os.stat(gaff_path).st_mtime
        assert last_touched_frcmod == os.stat(frcmod_path).st_mtime


@unittest.skipIf(not utils.is_openeye_installed(), 'This test requires OpenEye installed.')
def test_setup_name_smiles_openeye_charges():
    """Setup molecule from name and SMILES with openeye charges and gaff."""
    with utils.temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            output_dir: {}
            setup_dir: .
        molecules:
            p-xylene:
                name: p-xylene
                parameters: openeye:am1bcc-gaff
            toluene:
                smiles: Cc1ccccc1
                parameters: antechamber
        """.format(tmp_dir)

        yaml_builder = YamlBuilder(textwrap.dedent(yaml_content))
        yaml_builder._db._setup_molecules('toluene', 'p-xylene')

        for mol in ['toluene', 'p-xylene']:
            output_basepath = os.path.join(tmp_dir, SetupDatabase.MOLECULES_DIR, mol, mol)

            # Check that all the files have been created
            assert os.path.exists(output_basepath + '.mol2')
            assert os.path.exists(output_basepath + '.gaff.mol2')
            assert os.path.exists(output_basepath + '.frcmod')
            assert os.path.getsize(output_basepath + '.mol2') > 0
            assert os.path.getsize(output_basepath + '.gaff.mol2') > 0
            assert os.path.getsize(output_basepath + '.frcmod') > 0

            atoms_frame, _ = mol2_to_dataframes(output_basepath + '.mol2')
            input_charges = atoms_frame['charge']
            atoms_frame, _ = mol2_to_dataframes(output_basepath + '.gaff.mol2')
            output_charges = atoms_frame['charge']

            # With openeye:am1bcc charges, the final charges should be unaltered
            if mol == 'p-xylene':
                assert input_charges.equals(output_charges)
            else:  # With antechamber, sqm should alter the charges a little
                assert not input_charges.equals(output_charges)


@unittest.skipIf(not utils.is_openeye_installed(), 'This test requires OpenEye installed.')
def test_clashing_atoms():
    """Check that clashing atoms are resolved."""
    clearance = 10.0
    benzene_path = examples_paths()['benzene']
    toluene_path = examples_paths()['toluene']
    with utils.temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            output_dir: {}
            setup_dir: .
        molecules:
            benzene:
                filepath: {}
                parameters: antechamber
            toluene:
                filepath: {}
                parameters: antechamber
        solvents:
            vacuum:
                nonbonded_method: NoCutoff
            PME:
                nonbonded_method: PME
                nonbonded_cutoff: 1*nanometer
                clearance: {}*angstroms
        """.format(tmp_dir, benzene_path, toluene_path, clearance)

        # Sanity check: at the beginning molecules clash
        toluene_pos = utils.get_oe_mol_positions(utils.read_oe_molecule(toluene_path))
        benzene_pos = utils.get_oe_mol_positions(utils.read_oe_molecule(benzene_path))
        assert compute_min_dist(toluene_pos, benzene_pos) < SetupDatabase.CLASH_THRESHOLD

        yaml_builder = YamlBuilder(textwrap.dedent(yaml_content))

        components = {'receptor': 'benzene', 'ligand': 'toluene'}
        for solvent in ['vacuum', 'PME']:
            components['solvent'] = solvent
            system_dir = yaml_builder._db.get_system(components)

            # Get positions of molecules in the final system
            prmtop = openmm.app.AmberPrmtopFile(os.path.join(system_dir, 'complex.prmtop'))
            inpcrd = openmm.app.AmberInpcrdFile(os.path.join(system_dir, 'complex.inpcrd'))
            positions = inpcrd.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            atom_indices = pipeline.find_components(prmtop.topology, 'resname TOL')
            benzene_pos2 = positions.take(atom_indices['receptor'], axis=0)
            toluene_pos2 = positions.take(atom_indices['ligand'], axis=0)

            # Test that clashes are resolved in the system
            min_dist, max_dist = compute_dist_bound(toluene_pos2, benzene_pos2)
            assert min_dist >= SetupDatabase.CLASH_THRESHOLD

            # For solvent we check that molecule is within the box
            if solvent == 'PME':
                assert max_dist <= clearance


@unittest.skipIf(not omt.schrodinger.is_schrodinger_suite_installed(),
                 "This test requires Schrodinger's suite")
def test_epik_enumeration():
    """Test that epik protonation state enumeration."""
    with utils.temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            output_dir: {}
            setup_dir: .
        molecules:
            benzene:
                filepath: {}
                epik: 0
                parameters: antechamber
        """.format(tmp_dir, examples_paths()['benzene'])

        yaml_builder = YamlBuilder(textwrap.dedent(yaml_content))
        yaml_builder._db._setup_molecules('benzene')

        output_basename = os.path.join(tmp_dir, SetupDatabase.MOLECULES_DIR, 'benzene',
                                       'benzene-epik.')
        assert os.path.exists(output_basename + 'mol2')
        assert os.path.getsize(output_basename + 'mol2') > 0
        assert os.path.exists(output_basename + 'sdf')
        assert os.path.getsize(output_basename + 'sdf') > 0


def test_strip_protons():
    """Test that protons are stripped correctly for tleap."""
    abl_path = examples_paths()['abl']
    with utils.temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            output_dir: {}
            setup_dir: .
        molecules:
            abl:
                filepath: {}
                parameters: leaprc.ff14SB
        """.format(tmp_dir, abl_path)

        # Safety check: protein must have protons
        has_hydrogen = False
        with open(abl_path, 'r') as f:
            for line in f:
                if line[:6] == 'ATOM  ' and (line[12] == 'H' or line[13] == 'H'):
                    has_hydrogen = True
                    break
        assert has_hydrogen

        yaml_builder = YamlBuilder(textwrap.dedent(yaml_content))
        output_path = os.path.join(tmp_dir, SetupDatabase.MOLECULES_DIR, 'abl', 'abl.pdb')

        # We haven't set the strip_protons options, so this shouldn't do anything
        yaml_builder._db._setup_molecules('abl')
        assert not os.path.exists(output_path)

        # Now we set the strip_protons options and repeat
        yaml_builder._db.molecules['abl']['strip_protons'] = True
        yaml_builder._db._setup_molecules('abl')
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # The new pdb does not have hydrogen atoms
        has_hydrogen = False
        with open(output_path, 'r') as f:
            for line in f:
                if line[:6] == 'ATOM  ' and (line[12] == 'H' or line[13] == 'H'):
                    has_hydrogen = True
                    break
        assert not has_hydrogen


class TestMultiMoleculeFiles():

    @classmethod
    def setup_class(cls):
        """Create a 2-frame PDB file in pdb_path. The second frame has same positions
        of the first one but with inversed z-coordinate."""
        # Creating a temporary directory and generating paths for output files
        cls.tmp_dir = tempfile.mkdtemp()
        cls.pdb_path = os.path.join(cls.tmp_dir, 'multi.pdb')
        cls.smiles_path = os.path.join(cls.tmp_dir, 'multi.smiles')
        cls.sdf_path = os.path.join(cls.tmp_dir, 'multi.sdf')
        cls.mol2_path = os.path.join(cls.tmp_dir, 'multi.mol2')

        # Rotation matrix to invert z-coordinate, i.e. flip molecule w.r.t. x-y plane
        rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

        # Create 2-frame PDB file. First frame is lysozyme, second is lysozyme with inverted z
        lysozyme_path = examples_paths()['lysozyme']
        lysozyme = PDBFile(lysozyme_path)

        # Rotate positions to invert z for the second frame
        symmetric_pos = lysozyme.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        symmetric_pos = symmetric_pos.dot(rot) * unit.angstrom

        with open(cls.pdb_path, 'w') as f:
            PDBFile.writeHeader(lysozyme.topology, file=f)
            PDBFile.writeModel(lysozyme.topology, lysozyme.positions, file=f, modelIndex=0)
            PDBFile.writeModel(lysozyme.topology, symmetric_pos, file=f, modelIndex=1)

        # Create 2-molecule SMILES file
        with open(cls.smiles_path, 'w') as f:
            f.write('benzene,c1ccccc1\n')
            f.write('toluene,Cc1ccccc1\n')

        # Create 2-molecule sdf and mol2 with OpenEye
        if utils.is_openeye_installed():
            from openeye import oechem
            oe_benzene = utils.read_oe_molecule(examples_paths()['benzene'])
            oe_benzene_pos = utils.get_oe_mol_positions(oe_benzene).dot(rot)
            oe_benzene.NewConf(oechem.OEFloatArray(oe_benzene_pos.flatten()))

            # Save 2-conformer benzene in sdf and mol2 format
            utils.write_oe_molecule(oe_benzene, cls.sdf_path)
            utils.write_oe_molecule(oe_benzene, cls.mol2_path, mol2_resname='MOL')


    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_expand_molecules(self):
        """Check that combinatorial molecules are handled correctly."""
        yaml_content = """
        ---
        molecules:
            rec:
                filepath: [conf1.pdb, conf2.pdb]
                parameters: oldff/leaprc.ff99SBildn
            lig:
                name: [iupac1, iupac2]
                parameters: antechamber
                epik: [0, 2]
            multi:
                filepath: {}
                parameters: leaprc.ff14SB
                select: all
            smiles:
                filepath: {}
                parameters: antechamber
                select: all
            sdf:
                filepath: {}
                parameters: antechamber
                select: all
            mol2:
                filepath: {}
                parameters: antechamber
                select: all
        solvents:
            solv1:
                nonbonded_method: NoCutoff
            solv2:
                nonbonded_method: PME
                nonbonded_cutoff: 1*nanometer
                clearance: 10*angstroms
        protocols:{}
        experiments:
            components:
                receptor: [rec, multi]
                ligand: lig
                solvent: [solv1, solv2]
            protocol: absolute-binding
        """.format(self.pdb_path, self.smiles_path, self.sdf_path,
                   self.mol2_path, indent(indent(standard_protocol)))
        yaml_content = textwrap.dedent(yaml_content)

        expected_content = """
        ---
        molecules:
            rec_conf1pdb:
                filepath: conf1.pdb
                parameters: oldff/leaprc.ff99SBildn
            rec_conf2pdb:
                filepath: conf2.pdb
                parameters: oldff/leaprc.ff99SBildn
            lig_0_iupac1:
                name: iupac1
                parameters: antechamber
                epik: 0
            lig_2_iupac1:
                name: iupac1
                parameters: antechamber
                epik: 2
            lig_0_iupac2:
                name: iupac2
                parameters: antechamber
                epik: 0
            lig_2_iupac2:
                name: iupac2
                parameters: antechamber
                epik: 2
            multi_0:
                filepath: {}
                parameters: leaprc.ff14SB
                select: 0
            multi_1:
                filepath: {}
                parameters: leaprc.ff14SB
                select: 1
            smiles_0:
                filepath: {}
                parameters: antechamber
                select: 0
            smiles_1:
                filepath: {}
                parameters: antechamber
                select: 1
            sdf_0:
                filepath: {}
                parameters: antechamber
                select: 0
            sdf_1:
                filepath: {}
                parameters: antechamber
                select: 1
            mol2_0:
                filepath: {}
                parameters: antechamber
                select: 0
            mol2_1:
                filepath: {}
                parameters: antechamber
                select: 1
        solvents:
            solv1:
                nonbonded_method: NoCutoff
            solv2:
                nonbonded_method: PME
                nonbonded_cutoff: 1*nanometer
                clearance: 10*angstroms
        protocols:{}
        experiments:
            components:
                receptor: [rec_conf1pdb, rec_conf2pdb, multi_1, multi_0]
                ligand: [lig_0_iupac2, lig_2_iupac1, lig_2_iupac2, lig_0_iupac1]
                solvent: [solv1, solv2]
            protocol: absolute-binding
        """.format(self.pdb_path, self.pdb_path, self.smiles_path, self.smiles_path,
                   self.sdf_path, self.sdf_path, self.mol2_path, self.mol2_path,
                   indent(standard_protocol))
        expected_content = textwrap.dedent(expected_content)

        raw = yaml.load(yaml_content)
        expanded = YamlBuilder(yaml_content)._expand_molecules(raw)
        assert expanded == yaml.load(expected_content)

    def test_select_pdb_conformation(self):
        """Check that frame selection in multi-model PDB files works."""
        with utils.temporary_directory() as tmp_dir:
            yaml_content = """
            ---
            options:
                output_dir: {}
                setup_dir: .
            molecules:
                selected:
                    filepath: {}
                    parameters: leaprc.ff14SB
                    select: 1
            """.format(tmp_dir, self.pdb_path)
            yaml_content = textwrap.dedent(yaml_content)
            yaml_builder = YamlBuilder(yaml_content)

            # The molecule now is neither set up nor processed
            is_setup, is_processed = yaml_builder._db.is_molecule_setup('selected')
            assert is_setup is False
            assert is_processed is False

            # The setup of the molecule must isolate the frame in a single-frame PDB
            yaml_builder._db._setup_molecules('selected')
            selected_pdb_path = os.path.join(tmp_dir, SetupDatabase.MOLECULES_DIR,
                                             'selected', 'selected.pdb')
            assert os.path.exists(os.path.join(selected_pdb_path))
            assert os.path.getsize(os.path.join(selected_pdb_path)) > 0

            # The positions must be the ones of the second frame
            selected_pdb = PDBFile(selected_pdb_path)
            selected_pos = selected_pdb.getPositions(asNumpy=True)
            second_pos = PDBFile(self.pdb_path).getPositions(asNumpy=True, frame=1)
            assert selected_pdb.getNumFrames() == 1
            assert (selected_pos == second_pos).all()

            # The description of the molecule is now updated
            assert os.path.normpath(yaml_builder._db.molecules['selected']['filepath']) == selected_pdb_path

            # The molecule now both set up and processed
            is_setup, is_processed = yaml_builder._db.is_molecule_setup('selected')
            assert is_setup is True
            assert is_processed is True

            # A new instance of YamlBuilder is able to resume with correct molecule
            yaml_builder = YamlBuilder(yaml_content)
            is_setup, is_processed = yaml_builder._db.is_molecule_setup('selected')
            assert is_setup is True
            assert is_processed is True

    @unittest.skipIf(not utils.is_openeye_installed(), 'This test requires OpenEye installed.')
    def test_setup_smiles(self):
        """Check that setup molecule from SMILES files works."""
        from openeye.oechem import OEMolToSmiles

        with utils.temporary_directory() as tmp_dir:
            yaml_content = """
            ---
            options:
                output_dir: {}
                setup_dir: .
            molecules:
                take-first:
                    filepath: {}
                    parameters: antechamber
                select-second:
                    filepath: {}
                    parameters: antechamber
                    select: 1
            """.format(tmp_dir, self.smiles_path, self.smiles_path)
            yaml_content = textwrap.dedent(yaml_content)
            yaml_builder = YamlBuilder(yaml_content)

            for i, mol_id in enumerate(['take-first', 'select-second']):
                # The molecule now is neither set up nor processed
                is_setup, is_processed = yaml_builder._db.is_molecule_setup(mol_id)
                assert is_setup is False
                assert is_processed is False

                # The single SMILES has been converted to mol2 file
                yaml_builder._db._setup_molecules(mol_id)
                mol2_path = os.path.join(tmp_dir, SetupDatabase.MOLECULES_DIR, mol_id, mol_id + '.mol2')
                assert os.path.exists(os.path.join(mol2_path))
                assert os.path.getsize(os.path.join(mol2_path)) > 0

                # The mol2 represents the right molecule
                csv_smiles_str = (open(self.smiles_path, 'r').readlines()[i]).strip().split(',')[1]
                mol2_smiles_str = OEMolToSmiles(utils.read_oe_molecule(mol2_path))
                assert mol2_smiles_str == csv_smiles_str

                # The molecule now both set up and processed
                is_setup, is_processed = yaml_builder._db.is_molecule_setup(mol_id)
                assert is_setup is True
                assert is_processed is True

                # A new instance of YamlBuilder is able to resume with correct molecule
                yaml_builder = YamlBuilder(yaml_content)
                is_setup, is_processed = yaml_builder._db.is_molecule_setup(mol_id)
                assert is_setup is True
                assert is_processed is True

    @unittest.skipIf(not utils.is_openeye_installed(), 'This test requires OpenEye installed.')
    def test_select_sdf_mol2(self):
        """Check that selection in sdf and mol2 files works."""
        with utils.temporary_directory() as tmp_dir:
            yaml_content = """
            ---
            options:
                output_dir: {}
                setup_dir: .
            molecules:
                sdf_0:
                    filepath: {}
                    parameters: antechamber
                    select: 0
                sdf_1:
                    filepath: {}
                    parameters: antechamber
                    select: 1
                mol2_0:
                    filepath: {}
                    parameters: antechamber
                    select: 0
                mol2_1:
                    filepath: {}
                    parameters: antechamber
                    select: 1
            """.format(tmp_dir, self.sdf_path, self.sdf_path, self.mol2_path, self.mol2_path)
            yaml_content = textwrap.dedent(yaml_content)
            yaml_builder = YamlBuilder(yaml_content)

            for extension in ['sdf', 'mol2']:
                multi_path = getattr(self, extension + '_path')
                for model_idx in [0, 1]:
                    mol_id = extension + '_' + str(model_idx)

                    # The molecule now is neither set up nor processed
                    is_setup, is_processed = yaml_builder._db.is_molecule_setup(mol_id)
                    assert is_setup is False
                    assert is_processed is False

                    yaml_builder._db._setup_molecules(mol_id)

                    # The setup of the molecule must isolate the frame in a single-frame PDB
                    single_mol_path = os.path.join(tmp_dir, SetupDatabase.MOLECULES_DIR,
                                                   mol_id, mol_id + '.' + extension)
                    assert os.path.exists(os.path.join(single_mol_path))
                    assert os.path.getsize(os.path.join(single_mol_path)) > 0

                    # sdf files must be converted to mol2 to be fed to antechamber
                    if extension == 'sdf':
                        single_mol_path = os.path.join(tmp_dir, SetupDatabase.MOLECULES_DIR,
                                                       mol_id, mol_id + '.mol2')
                        assert os.path.exists(os.path.join(single_mol_path))
                        assert os.path.getsize(os.path.join(single_mol_path)) > 0

                    # Check antechamber parametrization
                    single_mol_path = os.path.join(tmp_dir, SetupDatabase.MOLECULES_DIR,
                                                   mol_id, mol_id + '.gaff.mol2')
                    assert os.path.exists(os.path.join(single_mol_path))
                    assert os.path.getsize(os.path.join(single_mol_path)) > 0

                    # The positions must be approximately correct (antechamber move the molecule)
                    selected_oe_mol = utils.read_oe_molecule(single_mol_path)
                    selected_pos = utils.get_oe_mol_positions(selected_oe_mol)
                    second_oe_mol = utils.read_oe_molecule(multi_path, conformer_idx=model_idx)
                    second_pos = utils.get_oe_mol_positions(second_oe_mol)
                    assert selected_oe_mol.NumConfs() == 1
                    assert np.allclose(selected_pos, second_pos, atol=1e-1)

                    # The molecule now both set up and processed
                    is_setup, is_processed = yaml_builder._db.is_molecule_setup(mol_id)
                    assert is_setup is True
                    assert is_processed is True

                    # A new instance of YamlBuilder is able to resume with correct molecule
                    yaml_builder = YamlBuilder(yaml_content)
                    is_setup, is_processed = yaml_builder._db.is_molecule_setup(mol_id)
                    assert is_setup is True
                    assert is_processed is True


def test_setup_implicit_system_leap():
    """Create prmtop and inpcrd for implicit solvent protein-ligand system."""
    with utils.temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            output_dir: {}
            resume_setup: yes
        molecules:
            T4lysozyme:
                filepath: {}
                parameters: oldff/leaprc.ff99SBildn
            p-xylene:
                filepath: {}
                parameters: antechamber
        solvents:
            GBSA-OBC2:
                nonbonded_method: NoCutoff
                implicit_solvent: OBC2
        """.format(tmp_dir, examples_paths()['lysozyme'], examples_paths()['p-xylene'])

        yaml_builder = YamlBuilder(textwrap.dedent(yaml_content))
        components = {'receptor': 'T4lysozyme',
                      'ligand': 'p-xylene',
                      'solvent': 'GBSA-OBC2'}

        output_dir = yaml_builder._db.get_system(components)
        last_modified_path = os.path.join(output_dir, 'complex.prmtop')
        last_modified = os.stat(last_modified_path).st_mtime

        # Test that output files exist and there is no water
        for phase in ['complex', 'solvent']:
            found_resnames = set()
            pdb_path = os.path.join(output_dir, phase + '.pdb')
            prmtop_path = os.path.join(output_dir, phase + '.prmtop')
            inpcrd_path = os.path.join(output_dir, phase + '.inpcrd')

            with open(pdb_path, 'r') as f:
                for line in f:
                    if len(line) > 10:
                        found_resnames.add(line[17:20])

            assert os.path.exists(prmtop_path)
            assert os.path.exists(inpcrd_path)
            assert os.path.getsize(prmtop_path) > 0
            assert os.path.getsize(inpcrd_path) > 0
            assert 'MOL' in found_resnames
            assert 'WAT' not in found_resnames

        # Test that another call do not regenerate the system
        time.sleep(0.5)  # st_mtime doesn't have much precision
        yaml_builder._db.get_system(components)
        assert last_modified == os.stat(last_modified_path).st_mtime

@unittest.skipIf(not utils.is_openeye_installed(), 'This test requires OpenEye installed.')
def test_setup_explicit_system_leap():
    """Create prmtop and inpcrd protein-ligand system in explicit solvent."""
    with utils.temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            output_dir: {}
        molecules:
            benzene:
                filepath: {}
                parameters: antechamber
            toluene:
                name: toluene
                parameters: antechamber
        solvents:
            PMEtip3p:
                nonbonded_method: PME
                clearance: 10*angstroms
        """.format(tmp_dir, examples_paths()['benzene'])

        yaml_builder = YamlBuilder(textwrap.dedent(yaml_content))
        components = {'receptor': 'benzene',
                      'ligand': 'toluene',
                      'solvent': 'PMEtip3p'}

        output_dir = yaml_builder._db.get_system(components)

        # Test that output file exists and that there is water
        expected_resnames = {'complex': set(['BEN', 'TOL', 'WAT']),
                             'solvent': set(['TOL', 'WAT'])}
        for phase in expected_resnames:
            found_resnames = set()
            pdb_path = os.path.join(output_dir, phase + '.pdb')
            prmtop_path = os.path.join(output_dir, phase + '.prmtop')
            inpcrd_path = os.path.join(output_dir, phase + '.inpcrd')

            with open(pdb_path, 'r') as f:
                for line in f:
                    if len(line) > 10:
                        found_resnames.add(line[17:20])

            assert os.path.exists(prmtop_path)
            assert os.path.exists(inpcrd_path)
            assert os.path.getsize(prmtop_path) > 0
            assert os.path.getsize(inpcrd_path) > 0
            assert found_resnames == expected_resnames[phase]

def test_neutralize_system():
    """Test whether the system charge is neutralized correctly."""
    with utils.temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            output_dir: {}
        molecules:
            receptor:
                filepath: {}
                parameters: leaprc.ff14SB
            ligand:
                filepath: {}
                parameters: antechamber
        solvents:
            PMEtip3p:
                nonbonded_method: PME
                clearance: 10*angstroms
                positive_ion: K+
                negative_ion: Cl-
        """.format(tmp_dir, examples_paths()['lysozyme'], examples_paths()['p-xylene'])

        yaml_builder = YamlBuilder(textwrap.dedent(yaml_content))
        components = {'receptor': 'receptor',
                      'ligand': 'ligand',
                      'solvent': 'PMEtip3p'}

        output_dir = yaml_builder._db.get_system(components)

        # Test that output file exists and that there are ions
        found_resnames = set()
        with open(os.path.join(output_dir, 'complex.pdb'), 'r') as f:
            for line in f:
                if len(line) > 10:
                    found_resnames.add(line[17:20])
        assert set(['MOL', 'WAT', 'Cl-']) <= found_resnames

        # Check that parameter files exist
        prmtop_path = os.path.join(output_dir, 'complex.prmtop')
        inpcrd_path = os.path.join(output_dir, 'complex.inpcrd')
        assert os.path.exists(prmtop_path)
        assert os.path.exists(inpcrd_path)


@unittest.skipIf(not utils.is_openeye_installed(), "This test requires OpenEye toolkit")
def test_charged_ligand():
    """Check that there are alchemical counterions for charged ligands."""
    with utils.temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            output_dir: {}
            resume_setup: yes
        molecules:
            aspartic-acid:
                name: "(3S)-3-amino-4-hydroxy-4-oxo-butanoate"
                parameters: antechamber
            lysine:
                name: "[(5S)-5-amino-6-hydroxy-6-oxo-hexyl]azanium"
                parameters: antechamber
        solvents:
            PMEtip3p:
                nonbonded_method: PME
                clearance: 10*angstroms
                positive_ion: Na+
                negative_ion: Cl-
        """.format(tmp_dir)

        yaml_builder = YamlBuilder(textwrap.dedent(yaml_content))
        components = {'receptor': 'lysine',
                      'ligand': 'aspartic-acid',
                      'solvent': 'PMEtip3p'}
        output_dir = yaml_builder._db.get_system(components)

        # Safety check: Asp is negatively charged, Lys is positively charged
        asp_path = os.path.join(yaml_builder._db.get_molecule_dir('aspartic-acid'),
                                'aspartic-acid.gaff.mol2')
        lys_path = os.path.join(yaml_builder._db.get_molecule_dir('lysine'),
                                'lysine.gaff.mol2')
        assert utils.get_mol2_net_charge(asp_path) == -1
        assert utils.get_mol2_net_charge(lys_path) == 1

        # Even if the system is globally neutral, there should be ions
        # because there must be an alchemical ion for the charged ligand
        found_resnames = set()
        with open(os.path.join(output_dir, 'complex.pdb'), 'r') as f:
            for line in f:
                if len(line) > 10:
                    found_resnames.add(line[17:20])
        assert set(['Na+', 'Cl-']) <= found_resnames

        # Test that 'ligand_counterions' component contain one anion
        _, systems, _, atom_indices = pipeline.prepare_amber(output_dir, 'resname ASP',
                                                {'nonbondedMethod': openmm.app.PME}, -1)
        for phase in ['complex', 'solvent']:
            prmtop_file_path = os.path.join(output_dir, '{}.prmtop'.format(phase))
            topology = openmm.app.AmberPrmtopFile(prmtop_file_path).topology
            phase += '-explicit'

            assert len(atom_indices[phase]['ligand_counterions']) == 1
            ion_idx = atom_indices[phase]['ligand_counterions'][0]
            ion_atom = next(itertools.islice(topology.atoms(), ion_idx, None))
            assert '+' in ion_atom.residue.name


def test_yaml_creation():
    """Test the content of generated single experiment YAML files."""
    ligand_path = examples_paths()['p-xylene']
    with utils.temporary_directory() as tmp_dir:
        molecules = """
            T4lysozyme:
                filepath: {}
                parameters: oldff/leaprc.ff99SBildn""".format(examples_paths()['lysozyme'])
        solvent = """
            vacuum:
                nonbonded_method: NoCutoff"""
        protocol = indent(standard_protocol)
        experiment = """
            components:
                ligand: p-xylene
                receptor: T4lysozyme
                solvent: vacuum
            protocol: absolute-binding"""

        yaml_content = """
        ---
        options:
            output_dir: {}
        molecules:{}
            p-xylene:
                filepath: {}
                parameters: antechamber
            benzene:
                filepath: benzene.mol2
                parameters: antechamber
        solvents:{}
            GBSA-OBC2:
                nonbonded_method: NoCutoff
                implicit_solvent: OBC2
        protocols:{}
        experiments:{}
        """.format(os.path.relpath(tmp_dir), molecules,
                   os.path.relpath(ligand_path), solvent,
                   protocol, experiment)

        # We need to check whether the relative paths to the output directory and
        # for p-xylene are handled correctly while absolute paths (T4lysozyme) are
        # left untouched
        expected_yaml_content = textwrap.dedent("""
        ---
        options:
            experiments_dir: .
            output_dir: .
        molecules:{}
            p-xylene:
                filepath: {}
                parameters: antechamber
        solvents:{}
        protocols:{}
        experiments:{}
        """.format(molecules, os.path.relpath(ligand_path, tmp_dir),
                   solvent, protocol, experiment))
        expected_yaml_content = expected_yaml_content[1:]  # remove first '\n'

        yaml_builder = YamlBuilder(textwrap.dedent(yaml_content))

        # during setup we can modify molecule's fields, so we need
        # to check that it doesn't affect the YAML file exported
        experiment_dict = yaml.load(experiment)
        yaml_builder._db.get_system(experiment_dict['components'])

        yaml_builder._generate_yaml(experiment_dict, os.path.join(tmp_dir, 'experiment.yaml'))
        with open(os.path.join(tmp_dir, 'experiment.yaml'), 'r') as f:
            for line, expected in zip(f, expected_yaml_content.split('\n')):
                assert line[:-1] == expected  # without final '\n'

def test_run_experiment():
    with utils.temporary_directory() as tmp_dir:
        yaml_content = """
        ---
        options:
            resume_setup: no
            resume_simulation: no
            number_of_iterations: 1
            output_dir: overwritten
            annihilate_sterics: yes
        molecules:
            T4lysozyme:
                filepath: {}
                parameters: oldff/leaprc.ff99SBildn
                select: 0
            p-xylene:
                filepath: {}
                parameters: antechamber
        solvents:
            vacuum:
                nonbonded_method: NoCutoff
            GBSA-OBC2:
                nonbonded_method: NoCutoff
                implicit_solvent: OBC2
        protocols:{}
        experiments:
            components:
                receptor: T4lysozyme
                ligand: p-xylene
                solvent: [vacuum, GBSA-OBC2]
            options:
                output_dir: {}
                setup_dir: ''
                experiments_dir: ''
            protocol: absolute-binding
        """.format(examples_paths()['lysozyme'], examples_paths()['p-xylene'],
                   indent(standard_protocol), tmp_dir)

        yaml_builder = YamlBuilder(textwrap.dedent(yaml_content))

        # Now check_setup_resume should not raise exceptions
        yaml_builder._check_resume()

        # We setup a molecule and with resume_setup: now we can't do the experiment
        err_msg = ''
        yaml_builder._db._setup_molecules('p-xylene')
        try:
            yaml_builder.build_experiment()
        except YamlParseError as e:
            err_msg = str(e)
        assert 'molecule' in err_msg

        # Same thing with a system
        err_msg = ''
        system_dir = yaml_builder._db.get_system({'receptor': 'T4lysozyme',
                                                  'ligand': 'p-xylene',
                                                  'solvent': 'vacuum'})
        try:
            yaml_builder.build_experiment()
        except YamlParseError as e:
            err_msg = str(e)
        assert 'system' in err_msg

        # Now we set resume_setup to True and things work
        yaml_builder.options['resume_setup'] = True
        ligand_dir = yaml_builder._db.get_molecule_dir('p-xylene')
        frcmod_file = os.path.join(ligand_dir, 'p-xylene.frcmod')
        prmtop_file = os.path.join(system_dir, 'complex.prmtop')
        molecule_last_touched = os.stat(frcmod_file).st_mtime
        system_last_touched = os.stat(prmtop_file).st_mtime
        yaml_builder.build_experiment()

        # Neither the system nor the molecule has been processed again
        assert molecule_last_touched == os.stat(frcmod_file).st_mtime
        assert system_last_touched == os.stat(prmtop_file).st_mtime

        # The experiments folders are correctly named and positioned
        for exp_name in ['vacuum', 'GBSAOBC2']:
            # The output directory must be the one in the experiment section
            output_dir = os.path.join(tmp_dir, exp_name)
            assert os.path.isdir(output_dir)
            assert os.path.isfile(os.path.join(output_dir, 'complex-implicit.nc'))
            assert os.path.isfile(os.path.join(output_dir, 'solvent-implicit.nc'))
            assert os.path.isfile(os.path.join(output_dir, exp_name + '.yaml'))
            assert os.path.isfile(os.path.join(output_dir, exp_name + '.log'))

        # Now we can't run the experiment again with resume_simulation: no
        try:
            yaml_builder.build_experiment()
        except YamlParseError as e:
            err_msg = str(e)
        assert 'experiment' in err_msg

        # We set resume_simulation: yes and now things work
        yaml_builder.options['resume_simulation'] = True
        yaml_builder.build_experiment()

# TODO start from prmtop and inpcrd files
# TODO start form gro and top files

# TODO default solvents?
# TODO default protocol?
