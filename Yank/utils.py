import os
import copy
import shutil
import logging
import tempfile
import itertools
import contextlib
import subprocess
import collections

from pkg_resources import resource_filename

from simtk import unit
from mdtraj.utils import enter_temp_directory

#========================================================================================
# Logging functions
#========================================================================================

def is_terminal_verbose():
    """Check whether the logging on the terminal is configured to be verbose.

    This is useful in case one wants to occasionally print something that is not really
    relevant to yank's log (e.g. external library verbose, citations, etc.).

    Returns
    is_verbose : bool
        True if the terminal is configured to be verbose, False otherwise.
    """

    # If logging.root has no handlers this will ensure that False is returned
    is_verbose = False

    for handler in logging.root.handlers:
        # logging.FileHandler is a subclass of logging.StreamHandler so
        # isinstance and issubclass do not work in this case
        if type(handler) is logging.StreamHandler and handler.level <= logging.DEBUG:
            is_verbose = True
            break

    return is_verbose

def config_root_logger(verbose, log_file_path=None, mpicomm=None):
    """Setup the the root logger's configuration.

     The log messages are printed in the terminal and saved in the file specified
     by log_file_path (if not None) and printed. Note that logging use sys.stdout
     to print logging.INFO messages, and stderr for the others. The root logger's
     configuration is inherited by the loggers created by logging.getLogger(name).

     Different formats are used to display messages on the terminal and on the log
     file. For example, in the log file every entry has a timestamp which does not
     appear in the terminal. Moreover, the log file always shows the module that
     generate the message, while in the terminal this happens only for messages
     of level WARNING and higher.

    Parameters
    ----------
    verbose : bool
        Control the verbosity of the messages printed in the terminal. The logger
        displays messages of level logging.INFO and higher when verbose=False.
        Otherwise those of level logging.DEBUG and higher are printed.
    log_file_path : str, optional, default = None
        If not None, this is the path where all the logger's messages of level
        logging.DEBUG or higher are saved.
    mpicomm : mpi4py.MPI.COMM communicator, optional, default=None
        If specified, this communicator will be used to determine node rank.

    """

    class TerminalFormatter(logging.Formatter):
        """
        Simplified format for INFO and DEBUG level log messages.

        This allows to keep the logging.info() and debug() format separated from
        the other levels where more information may be needed. For example, for
        warning and error messages it is convenient to know also the module that
        generates them.
        """

        # This is the cleanest way I found to make the code compatible with both
        # Python 2 and Python 3
        simple_fmt = logging.Formatter('%(asctime)-15s: %(message)s')
        default_fmt = logging.Formatter('%(asctime)-15s: %(levelname)s - %(name)s - %(message)s')

        def format(self, record):
            if record.levelno <= logging.INFO:
                return self.simple_fmt.format(record)
            else:
                return self.default_fmt.format(record)

    # Check if root logger is already configured
    n_handlers = len(logging.root.handlers)
    if n_handlers > 0:
        root_logger = logging.root
        for i in xrange(n_handlers):
            root_logger.removeHandler(root_logger.handlers[0])

    # If this is a worker node, don't save any log file
    if mpicomm:
        rank = mpicomm.rank
    else:
        rank = 0

    if rank != 0:
        log_file_path = None

    # Add handler for stdout and stderr messages
    terminal_handler = logging.StreamHandler()
    terminal_handler.setFormatter(TerminalFormatter())
    if rank != 0:
        terminal_handler.setLevel(logging.WARNING)
    elif verbose:
        terminal_handler.setLevel(logging.DEBUG)
    else:
        terminal_handler.setLevel(logging.INFO)
    logging.root.addHandler(terminal_handler)

    # Add file handler to root logger
    if log_file_path is not None:
        file_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(file_format))
        logging.root.addHandler(file_handler)

    # Do not handle logging.DEBUG at all if unnecessary
    if log_file_path is not None:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(terminal_handler.level)

#========================================================================================
# Combinatorial tree
#========================================================================================

class CombinatorialTree(collections.MutableMapping):
    """A tree that can be expanded in a combinatorial fashion.

    Each tree node with its subnodes is represented as a nested dictionary. Nodes can be
    accessed through their specific "path" (i.e. the list of the nested dictionary keys
    that lead to the node value).

    Values of a leaf nodes that are list-like objects can be expanded combinatorially in
    the sense that it is possible to iterate over all possible combinations of trees that
    are generated by taking leaf node list and create a sequence of trees, each one
    defining only one of the single values in those lists per leaf node (see Examples).

    Examples
    --------
    Set an arbitrary nested path
    >>> tree = CombinatorialTree({'a': {'b': 2}})
    >>> path = ('a', 'b')
    >>> tree[path]
    2
    >>> tree[path] = 3
    >>> tree[path]
    3

    Paths can be accessed also with the usual dict syntax
    >>> tree['a']['b']
    3

    Deletion of a node leave an empty dict!
    >>> del tree[path]
    >>> print tree
    {'a': {}}

    Expand all possible combinations of a tree. The iterator return a dict, not another
    CombinatorialTree object.
    >>> import pprint  # pprint sort the dictionary by key before printing
    >>> tree = CombinatorialTree({'a': 1, 'b': [1, 2], 'c': {'d': [3, 4]}})
    >>> for t in tree:
    ...     pprint.pprint(t)
    {'a': 1, 'b': 1, 'c': {'d': 3}}
    {'a': 1, 'b': 2, 'c': {'d': 3}}
    {'a': 1, 'b': 1, 'c': {'d': 4}}
    {'a': 1, 'b': 2, 'c': {'d': 4}}

    """
    def __init__(self, dictionary):
        """Build a combinatorial tree from the given dictionary."""
        self._d = copy.deepcopy(dictionary)

    def __getitem__(self, path):
        return reduce(lambda d,k: d[k], path, self._d)

    def __setitem__(self, path, value):
        d_node = self.__getitem__(path[:-1])
        d_node[path[-1]] = value

    def __delitem__(self, path):
        d_node = self.__getitem__(path[:-1])
        del d_node[path[-1]]

    def __len__(self):
        return len(self._d)

    def __str__(self):
        return str(self._d)

    def __eq__(self, other):
        return self._d == other

    def __iter__(self):
        """Iterate over all possible combinations of trees.

        The iterator returns dict objects, not other CombinatorialTrees.

        """
        template_tree = CombinatorialTree(self._d)
        leaf_paths, leaf_vals = template_tree._find_leaves()

        # itertools.product takes only iterables so we need to convert single values
        for i, leaf_val in enumerate(leaf_vals):
            if not is_iterable_container(leaf_val):
                leaf_vals[i] = [leaf_val]

        # generating all combinations
        for combination in itertools.product(*leaf_vals):
            # update values of template tree
            for leaf_path, leaf_val in zip(leaf_paths, combination):
                template_tree[leaf_path] = leaf_val
            yield copy.deepcopy(template_tree._d)

    def _find_leaves(self):
        """Traverse a dict tree and find the leaf nodes.

        Returns:
        --------
        A tuple containing two lists. The first one is a list of paths to the leaf
        nodes in a tuple format (e.g. the path to node['a']['b'] is ('a', 'b') while
        the second one is a list of all the values of those leaf nodes.

        Examples:
        ---------
        >>> simple_tree = CombinatorialTree({'simple': {'scalar': 1,
        ...                                             'vector': [2, 3, 4],
        ...                                             'nested': {
        ...                                                 'leaf': ['a', 'b', 'c']}}})
        >>> leaf_paths, leaf_vals = simple_tree._find_leaves()
        >>> leaf_paths
        [('simple', 'scalar'), ('simple', 'vector'), ('simple', 'nested', 'leaf')]
        >>> leaf_vals
        [1, [2, 3, 4], ['a', 'b', 'c']]

        """
        def recursive_find_leaves(node):
            leaf_paths = []
            leaf_vals = []
            for child_key, child_val in node.items():
                if isinstance(child_val, collections.Mapping):
                    subleaf_paths, subleaf_vals = recursive_find_leaves(child_val)
                    # prepend child key to path
                    leaf_paths.extend([(child_key,) + subleaf for subleaf in subleaf_paths])
                    leaf_vals.extend(subleaf_vals)
                else:
                    leaf_paths.append((child_key,))
                    leaf_vals.append(child_val)
            return leaf_paths, leaf_vals

        return recursive_find_leaves(self._d)

#========================================================================================
# Yank configuration
#========================================================================================

class YankOptions(collections.MutableMapping):
    """Helper class to manage Yank configuration.

    This class provide a single point of entry to read Yank options specified by command
    line, YAML or determined at runtime (i.e. the ones hardcoded). When the same option
    is specified multiple times the priority is runtime > command line > YAML > default.

    Attributes
    ----------
    cli : dict
        The options from the command line interface.
    yaml : dict
        The options from the YAML configuration file.
    default : dict
        The default options.

    Examples
    --------
    Command line options have priority over YAML

    >>> cl_opt = {'option1': 1}
    >>> yaml_opt = {'option1': 2}
    >>> options = YankOptions(cl_opt=cl_opt, yaml_opt=yaml_opt)
    >>> options['option1']
    1

    Modify specific priority level

    >>> options.default = {'option2': -1}
    >>> options['option2']
    -1

    Modify options at runtime and restore them

    >>> options['option1'] = 0
    >>> options['option1']
    0
    >>> del options['option1']
    >>> options['option1']
    1
    >>> options['hardcoded'] = 'test'
    >>> options['hardcoded']
    'test'

    """

    def __init__(self, cl_opt={}, yaml_opt={}, default_opt={}):
        """Constructor.

        Parameters
        ----------
        cl_opt : dict, optional, default {}
            The options from the command line.
        yaml_opt : dict, optional, default {}
            The options from the YAML configuration file.
        default_opt : dict, optional, default {}
            Default options. They have the lowest priority.

        """
        self._runtime_opt = {}
        self.cli = cl_opt
        self.yaml = yaml_opt
        self.default = default_opt

    def __getitem__(self, option):
        try:
            return self._runtime_opt[option]
        except KeyError:
            try:
                return self.cli[option]
            except KeyError:
                try:
                    return self.yaml[option]
                except KeyError:
                    return self.default[option]

    def __setitem__(self, option, value):
        self._runtime_opt[option] = value

    def __delitem__(self, option):
        del self._runtime_opt[option]

    def __iter__(self):
        """Iterate over options keeping into account priorities."""

        found_options = set()
        for opt_set in (self._runtime_opt, self.cli, self.yaml, self.default):
            for opt in opt_set:
                if opt not in found_options:
                    found_options.add(opt)
                    yield opt

    def __len__(self):
        return sum(1 for _ in self)


#========================================================================================
# Miscellaneous functions
#========================================================================================

def get_data_filename(relative_path):
    """Get the full path to one of the reference files shipped for testing

    In the source distribution, these files are in ``examples/*/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.

    Parameters
    ----------
    name : str
        Name of the file to load, with respect to the yank egg folder which
        is typically located at something like
        ~/anaconda/lib/python2.7/site-packages/yank-*.egg/examples/
    """

    fn = resource_filename('yank', relative_path)

    if not os.path.exists(fn):
        raise ValueError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)

    return fn

def is_iterable_container(value):
    """Check whether the given value is a list-like object or not.

    Returns
    -------
    Flase if value is a string or not iterable, True otherwise.

    """
    # strings are iterable too so we have to treat them as a special case
    return not isinstance(value, str) and isinstance(value, collections.Iterable)

@contextlib.contextmanager
def temporary_directory():
    """Context for safe creation of temporary directories."""
    tmp_dir = tempfile.mkdtemp()
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir)

@contextlib.contextmanager
def temporary_cd(dir_path):
    """Context to temporary change the working directory."""
    prev_dir = os.getcwd()
    os.chdir(os.path.abspath(dir_path))
    try:
        yield
    finally:
        os.chdir(prev_dir)

#========================================================================================
# Conversion utilities
#========================================================================================

def typename(atype):
    """Convert a type object into a fully qualified typename.

    Parameters
    ----------
    atype : type
        The type to convert

    Returns
    -------
    typename : str
        The string typename.

    For example,

    >>> typename(type(1))
    'int'

    >>> import numpy
    >>> x = numpy.array([1,2,3], numpy.float32)
    >>> typename(type(x))
    'numpy.ndarray'

    """
    if not isinstance(atype, type):
        raise Exception('Argument is not a type')

    modulename = atype.__module__
    typename = atype.__name__

    if modulename != '__builtin__':
        typename = modulename + '.' + typename

    return typename

def process_unit_bearing_str(quantity_str, compatible_units):
    """
    Process a unit-bearing string to produce a Quantity.

    Parameters
    ----------
    quantity_str : str
        A string containing a value with a unit of measure.
    compatible_units : simtk.unit.Unit
       The result will be checked for compatibility with specified units, and an
       exception raised if not compatible.

    Returns
    -------
    quantity : simtk.unit.Quantity
       The specified string, returned as a Quantity.

    Raises
    ------
    TypeError
        If quantity_str does not contains units.
    ValueError
        If the units attached to quantity_str are incompatible with compatible_units

    Examples
    --------
    >>> process_unit_bearing_str('1.0*micrometers', unit.nanometers)
    Quantity(value=1.0, unit=micrometer)

    """

    # WARNING: This is dangerous!
    # See: http://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
    # TODO: Can we use a safer form of (or alternative to) 'eval' here?
    quantity = eval(quantity_str, unit.__dict__)
    # Unpack quantity if it was surrounded by quotes.
    if isinstance(quantity, str):
        quantity = eval(quantity, unit.__dict__)
    # Check to make sure units are compatible with expected units.
    try:
        quantity.unit.is_compatible(compatible_units)
    except:
        raise TypeError("String %s does not have units attached." % quantity_str)
    # Check that units are compatible with what we expect.
    if not quantity.unit.is_compatible(compatible_units):
        raise ValueError("Units of %s must be compatible with %s" % (quantity_str,
                                                                     str(compatible_units)))
    # Return unit-bearing quantity.
    return quantity

def validate_parameters(parameters, template_parameters, check_unknown=False,
                        process_units_str=False, float_to_int=False):

    # Create validated parameters
    validated_par = {par: parameters[par] for par in parameters
                     if par in template_parameters}

    # Check for unknown parameters
    if check_unknown and len(validated_par) < len(parameters):
        diff = set(parameters) - set(template_parameters)
        raise TypeError("found unknown parameter {}".format(', '.join(diff)))

    for par, value in validated_par.items():
        templ_value = template_parameters[par]

        # Convert requested types
        # bool inherits from int in Python so we can't simply use isinstance
        if float_to_int and type(templ_value) is int:
            validated_par[par] = int(value)
        elif process_units_str and isinstance(templ_value, unit.Quantity):
            validated_par[par] = process_unit_bearing_str(value, templ_value.unit)

        # Check for incompatible types
        if type(validated_par[par]) != type(templ_value) and templ_value is not None:
            raise ValueError("parameter {}={} is incompatible with {}".format(
                par, validated_par[par], template_parameters[par]))

    return validated_par

#=============================================================================================
# Stuff to move to openmoltools when they'll be stable
#=============================================================================================

def is_openeye_installed():
    try:
        from openeye import oechem
        from openeye import oequacpac
        from openeye import oeiupac
        from openeye import oeomega

        if not (oechem.OEChemIsLicensed() and oequacpac.OEQuacPacIsLicensed()
                and oeiupac.OEIUPACIsLicensed() and oeomega.OEOmegaIsLicensed()):
            raise ImportError
    except ImportError:
        return False
    return True

def read_oe_molecule(file_path):
    from openeye import oechem
    molecule = oechem.OEGraphMol()
    ifs = oechem.oemolistream()
    ifs.open(file_path)
    oechem.OEReadMolecule(ifs, molecule)
    ifs.close()
    return molecule

def get_oe_mol_positions(molecule):
    import numpy as np
    from openeye import oechem
    oe_coords = oechem.OEFloatArray(3)
    molecule_pos = np.zeros((molecule.NumAtoms(), 3))
    for i, atom in enumerate(molecule.GetAtoms()):
        molecule.GetCoords(atom, oe_coords)
        molecule_pos[i] = oe_coords
    return molecule_pos

def set_oe_mol_positions(molecule, positions):
    for i, atom in enumerate(molecule.GetAtoms()):
        molecule.SetCoords(atom, positions[i])

class TLeap:
    """Programmatic interface to write and run tLeap scripts."""

    @property
    def script(self):
        return self._script.format(**(self._file_paths)) + '\nquit\n'

    def __init__(self):
        self._script = ''
        self._file_paths = {}  # paths of input/output files to copy in/from temp dir
        # self._input_paths = []  # paths of input files to copy in temp dir
        # self._output_paths = []  # paths of output files to copy from temp dir
        #self._local_input = []  # input file names in temp dir
        #self._local_output = []  # output file names in temp dir

    def add_commands(self, *args):
        for command in args:
            self._script += command + '\n'

    def load_parameters(self, *args):
        for par_file in args:
            extension = os.path.splitext(par_file)[1]
            if extension == '.frcmod':
                local_name = 'moli{}'.format(len(self._file_paths))
                self.add_commands('loadAmberParams {{{}}}'.format(local_name))

                # Update list of input files to copy in temporary folder before run
                self._file_paths[local_name] = par_file
            else:
                self.add_commands('source ' + par_file)

    def load_group(self, name, file_path):
        extension = os.path.splitext(file_path)[1]
        if extension == '.mol2':
            load_command = 'loadMol2'
        elif extension == '.pdb':
            load_command = 'loadPdb'
        else:
            raise ValueError('cannot load format {} in tLeap'.format(extension))
        local_name = 'moli{}'.format(len(self._file_paths))
        self.add_commands('{} = {} {{{}}}'.format(name, load_command, local_name))

        # Update list of input files to copy in temporary folder before run
        self._file_paths[local_name] = file_path

    def combine(self, group, *args):
        components = ' '.join(args)
        self.add_commands('{} = combine {{{{ {} }}}}'.format(group, components))

    def solvate(self, group, water_model, clearance):
        self.add_commands('solvateBox {} {} {} iso'.format(group, water_model,
                                                           str(clearance)))

    def save_group(self, group, output_path):
        file_name = os.path.basename(output_path)
        file_name, extension = os.path.splitext(file_name)
        local_name = 'molo{}'.format(len(self._file_paths))

        # Update list of output files to copy from temporary folder after run
        self._file_paths[local_name] = output_path

        # Add command
        if extension == '.prmtop' or extension == '.inpcrd':
            local_name2 = 'molo{}'.format(len(self._file_paths))
            self.add_commands('saveAmberParm {} {{{}}} {{{}}}'.format(group, local_name,
                                                                      local_name2))
            # Update list of output files with the one not explicit
            if extension == '.inpcrd':
                extension2 = '.prmtop'
            else:
                extension2 = '.inpcrd'
            output_path2 = os.path.join(os.path.dirname(output_path), file_name + extension2)
            self._file_paths[local_name2] = output_path2
        elif extension == '.pdb':
            self.add_commands('savePDB {} {{{}}}'.format(group, local_name))
        else:
            raise ValueError('cannot export format {} from tLeap'.format(extension[1:]))

    def new_section(self, comment):
        self.add_commands('\n# ' + comment)

    def export_script(self, file_path):
        with open(file_path, 'w') as f:
            f.write(self.script)

    def run(self):
        # Transform paths in absolute paths since we'll change the working directory
        input_files = {local + os.path.splitext(path)[1]: os.path.abspath(path)
                       for local, path in self._file_paths.items() if 'moli' in local}
        output_files = {local + os.path.splitext(path)[1]: os.path.abspath(path)
                        for local, path in self._file_paths.items() if 'molo' in local}

        # Resolve all the names in the script
        local_files = {local: local + os.path.splitext(path)[1]
                       for local, path in self._file_paths.items()}
        script = self._script.format(**local_files) + 'quit\n'

        with enter_temp_directory():
            # Copy input files
            for local_file, file_path in input_files.items():
                shutil.copy(file_path, local_file)

            # Save script and run tleap
            with open('leap.in', 'w') as f:
                f.write(script)
            subprocess.check_output(['tleap', '-f', 'leap.in'])

            #Copy back output files
            for local_file, file_path in output_files.items():
                shutil.copy(local_file, file_path)


#=============================================================================================
# Main and tests
#=============================================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()
