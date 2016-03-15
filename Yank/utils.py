import os
import re
import csv
import copy
import shutil
import inspect
import logging
import tempfile
import itertools
import contextlib
import subprocess
import collections

from pkg_resources import resource_filename

import mdtraj
import numpy as np
from simtk import unit

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

    # Create different log files for each MPI process
    if rank != 0 and log_file_path is not None:
        basepath, ext = os.path.splitext(log_file_path)
        log_file_path = '{}_{}{}'.format(basepath, rank, ext)

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

    Expand all possible combinations and assign unique names
    >>> for name, t in tree.named_combinations(separator='_', max_name_length=5):
    ...     print name
    3_1
    3_2
    4_1
    4_2

    """
    def __init__(self, dictionary):
        """Build a combinatorial tree from the given dictionary."""
        self._d = copy.deepcopy(dictionary)

    def __getitem__(self, path):
        return self._resolve_path(self._d, path)

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
        leaf_paths, leaf_vals = self._find_leaves()
        return self._combinations_generator(leaf_paths, leaf_vals)

    def named_combinations(self, separator, max_name_length):
        """Iterate over all possible combinations of trees and assign them unique names.

        The names are generated by gluing together the first letters of the values of
        the combinatorial leaves only, separated by the given separator. If the values
        contain special characters, they are ignored. Only letters, numbers and the
        separator are found in the generated names.

        The iterator return tuples of (name, dict), not other CombinatorialTrees. If
        there is only a single combination, an empty string is returned for the name.

        Parameters
        ----------
        separator : str
            The string used to separate the words in the name.
        max_length : int
            The maximum length of the generated names, excluding disambiguation number.

        """
        leaf_paths, leaf_vals = self._find_leaves()
        generated_names = {}  # name: count, how many times we have generated the same name

        # Find set of paths to combinatorial leaves
        comb_paths = [path for path, val in zip(leaf_paths, leaf_vals)
                      if is_iterable_container(val)]

        # Compile regular expression used during filtering
        filter = re.compile('[^A-Za-z\d]+')

        # Iterate over combinations
        for combination in self._combinations_generator(leaf_paths, leaf_vals):
            # Retrieve values of combinatorial leaves and filter special
            # characters in values that we don't use for names
            comb_vals = [str(self._resolve_path(combination, path)) for path in comb_paths]
            comb_vals = [filter.sub('', val) for val in comb_vals]

            # Generate name
            if len(comb_vals) == 0:
                name = ''
            if len(comb_vals) == 1:
                name = comb_vals[0][:max_name_length]
            else:
                name = separator.join(comb_vals)
                original_vals = comb_vals[:]
                while len(name) > max_name_length:
                    # Sort the strings by descending length, if two values have the
                    # same length put first the one whose original value is the shortest
                    sorted_vals = sorted(enumerate(comb_vals), reverse=True,
                                         key=lambda x: (len(x[1]), -len(original_vals[x[0]])))

                    # Find how many strings have the maximum length
                    max_val_length = len(sorted_vals[0][1])
                    n_max_vals = len([x for x in sorted_vals if len(x[1]) == max_val_length])

                    # We trim the longest str by the necessary number of characters
                    # to reach max_name_length or the second longest value
                    length_diff = len(name) - max_name_length

                    if n_max_vals < len(comb_vals):
                        second_max_val_length = len(sorted_vals[n_max_vals][1])
                        length_diff = min(length_diff, max_val_length - second_max_val_length)

                    # Trim all the longest strings by few characters
                    for i in range(n_max_vals - 1, -1, -1):
                        # Division truncation ensures that we trim more the
                        # ones whose original value is the shortest
                        char_per_str = length_diff / (i + 1)
                        if char_per_str != 0:
                            idx = sorted_vals[i][0]
                            comb_vals[idx] = comb_vals[idx][:-char_per_str]
                        length_diff -= char_per_str

                    name = separator.join(comb_vals)

            if name in generated_names:
                generated_names[name] += 1
                name += separator + str(generated_names[name])
            else:
                generated_names[name] = 1
            yield name, combination

    @staticmethod
    def _resolve_path(d, path):
        """Retrieve the value of a nested key in a dictionary.

        Parameters
        ----------
        d : dict
            The nested dictionary.
        path : iterable of keys
            The "path" to the node of the dictionary.

        Return
        ------
        val :
            The value contained in the node pointed by the path.

        """
        return reduce(lambda d,k: d[k], path, d)

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

    def _combinations_generator(self, leaf_paths, leaf_vals):
        """Generate all possible combinations of experiments.

        The iterator returns dict objects, not other CombinatorialTrees.

        Parameters
        ----------
        leaf_paths : list of tuples of strings
            The list of paths as returned by _find_leaves().
        leaf_vals : list
            The list of the correspondent values as returned by _find_leaves().
        """
        template_tree = CombinatorialTree(self._d)

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

def underscore_to_camelcase(underscore_str):
    """Convert the given string from underscore_case to camelCase.

    Underscores at the beginning or at the end of the string are ignored. All
    underscores in the middle of the string are removed.

    Parameters
    ----------
    underscore_str : str
        String in underscore_case to convert to camelCase style.

    Returns
    -------
    camelcase_str : str
        String in camelCase style.

    Examples
    --------
    >>> underscore_to_camelcase('__my___variable_')
    '__myVariable_'

    """
    # Count leading and trailing '_' characters
    n_leading = re.search(r'[^_]', underscore_str)
    if n_leading is None:  # this is empty or contains only '_'s
        return underscore_str
    n_leading = n_leading.start()
    n_trailing = re.search(r'[^_]', underscore_str[::-1]).start()

    # Remove all underscores, join and capitalize
    words = underscore_str.split('_')
    camelcase_str = '_' * n_leading + words[n_leading]
    camelcase_str += ''.join(str.capitalize(word) for word in words[n_leading + 1:])
    camelcase_str += '_' * n_trailing

    return camelcase_str

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

def get_keyword_args(function):
    """Inspect function signature and return keyword args with their default values.

    Parameters
    ----------
    function : function
        The function to interrogate.

    Returns
    -------
    kwargs : dict
        A dictionary 'keyword argument' -> 'default value'. The arguments of the
        function that do not have a default value will not be included.

    """
    argspec = inspect.getargspec(function)
    kwargs = argspec.args[len(argspec.args) - len(argspec.defaults):]
    kwargs = {arg: value for arg, value in zip(kwargs, argspec.defaults)}
    return kwargs

def validate_parameters(parameters, template_parameters, check_unknown=False,
                        process_units_str=False, float_to_int=False,
                        ignore_none=True, special_conversions={}):
    """Utility function for parameters and options validation.

    Use the given template to filter the given parameters and infer their expected
    types. Perform various automatic conversions when requested. If the template is
    None, the parameter to validate is not checked for type compatibility.

    Parameters
    ----------
    parameters : dict
        The parameters to validate.
    template_parameters : dict
        The template used to filter the parameters and infer the types.
    check_unknown : bool
        If True, an exception is raised when parameters contain a key that is not
        contained in template_parameters.
    process_units_str: bool
        If True, the function will attempt to convert the strings whose template
        type is simtk.unit.Quantity.
    float_to_int : bool
        If True, floats in parameters whose template type is int are truncated.
    ignore_none : bool
        If True, the function do not process parameters whose value is None.
    special_conversions : dict
        Contains a coverter function with signature convert(arg) that must be
        applied to the parameters specified by the dictionary key.

    Returns
    -------
    validate_par : dict
        The converted parameters that are contained both in parameters and
        template_parameters.

    Raises
    ------
    TypeError
        If check_unknown is True and there are parameters not in template_parameters.
    ValueError
        If a parameter has an incompatible type with its template parameter.

    Examples
    --------
    Create the template parameters
    >>> template_pars = dict()
    >>> template_pars['bool'] = True
    >>> template_pars['int'] = 2
    >>> template_pars['unspecified'] = None  # this won't be checked for type compatibility
    >>> template_pars['to_be_converted'] = [1, 2, 3]
    >>> template_pars['length'] = 2.0 * unit.nanometers

    Now the parameters to validate
    >>> input_pars = dict()
    >>> input_pars['bool'] = None  # this will be skipped with ignore_none=True
    >>> input_pars['int'] = 4.3  # this will be truncated to 4 with float_to_int=True
    >>> input_pars['unspecified'] = 'input'  # this can be of any type since the template is None
    >>> input_pars['to_be_converted'] = {'key': 3}
    >>> input_pars['length'] = '1.0*nanometers'
    >>> input_pars['unknown'] = 'test'  # this will be silently filtered if check_unkown=False

    Validate the parameters
    >>> valid = validate_parameters(input_pars, template_pars, process_units_str=True,
    ...                             float_to_int=True, special_conversions={'to_be_converted': list})
    >>> import pprint
    >>> pprint.pprint(valid)
    {'bool': None,
     'int': 4,
     'length': Quantity(value=1.0, unit=nanometer),
     'to_be_converted': ['key'],
     'unspecified': 'input'}

    """

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
        if ignore_none and value is None:
            continue

        # Special conversions have priority
        if par in special_conversions:
            converter_func = special_conversions[par]
            validated_par[par] = converter_func(value)
        else: # Automatic conversions and type checking
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


def get_mol2_net_charge(mol2_file_path):
    """Compute the sum of the partial charges for a molecule in a mol2 file.

    Note that the mol2 file must indicated the charge of each atom consistently.
    The function works only for single-structure mol2 files.

    Parameters
    ----------
    mol2_file_path : str
        Path to the mol2 file containing the molecule(s).

    Returns
    -------
    net_charge : int
        The molecule net charge calculated as the sum of its partial charges

    """
    atoms_frame, _ = mdtraj.formats.mol2.mol2_to_dataframes(mol2_file_path)
    net_charge = atoms_frame['charge'].sum()
    return int(round(net_charge))


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


def read_oe_molecule(file_path, conformer_idx=None):
    from openeye import oechem

    # Open input file stream
    ifs = oechem.oemolistream()
    if not ifs.open(file_path):
        oechem.OEThrow.Fatal('Unable to open {}'.format(file_path))

    # Read all conformations
    for mol in ifs.GetOEMols():
        try:
            molecule.NewConf(mol)
        except UnboundLocalError:
            molecule = oechem.OEMol(mol)

    # Select conformation of interest
    if conformer_idx is not None:
        if molecule.NumConfs() <= conformer_idx:
            raise ValueError('conformer_idx {} out of range'.format(conformer_idx))
        molecule = oechem.OEGraphMol(molecule.GetConf(oechem.OEHasConfIdx(conformer_idx)))

    return molecule


def write_oe_molecule(oe_mol, file_path, mol2_resname=None):
    """Write all conformations in a file and automatically detects format."""
    from openeye import oechem

    # Get correct OpenEye format
    extension = os.path.splitext(file_path)[1][1:]  # remove dot
    oe_format = getattr(oechem, 'OEFormat_' + extension.upper())

    # Open stream and write molecule
    ofs = oechem.oemolostream()
    ofs.SetFormat(oe_format)
    if not ofs.open(file_path):
        oechem.OEThrow.Fatal('Unable to create {}'.format(file_path))
    oechem.OEWriteMolecule(ofs, oe_mol)
    ofs.close()

    # If this is a mol2 file, we need to replace the resname
    # TODO when you merge to openmoltools, incapsulate this and add to molecule_to_mol2()
    if mol2_resname is not None and extension == 'mol2':
        with open(file_path, 'r') as f:
            lines = f.readlines()
        lines = [line.replace('<0>', mol2_resname) for line in lines]
        with open(file_path, 'w') as f:
            f.writelines(lines)


def get_oe_mol_positions(molecule, conformer_idx=0):
    from openeye import oechem
    # Extract correct conformer
    if conformer_idx > 0:
        try:
            if molecule.NumConfs() <= conformer_idx:
                raise UnboundLocalError  # same error message
            molecule = oechem.OEGraphMol(molecule.GetConf(oechem.OEHasConfIdx(conformer_idx)))
        except UnboundLocalError:
            raise ValueError('conformer_idx {} out of range'.format(conformer_idx))
    # Extract positions
    oe_coords = oechem.OEFloatArray(3)
    molecule_pos = np.zeros((molecule.NumAtoms(), 3))
    for i, atom in enumerate(molecule.GetAtoms()):
        molecule.GetCoords(atom, oe_coords)
        molecule_pos[i] = oe_coords
    return molecule_pos

def set_oe_mol_positions(molecule, positions):
    for i, atom in enumerate(molecule.GetAtoms()):
        molecule.SetCoords(atom, positions[i])

def get_mol2_resname(file_path):
    """Find resname of first atom in tripos mol2 file."""
    with open(file_path, 'r') as f:
        atom_found = False
        for line in f:
            fields = line.split()
            if atom_found:
                try:
                    return fields[7]
                except IndexError:
                    return None
            elif len(fields) > 0 and fields[0] == '@<TRIPOS>ATOM':
                atom_found = True


class TLeap:
    """Programmatic interface to write and run tLeap scripts.

    To avoid problems with special characters in file paths, the class run the
    tleap script in a temporary folder with hardcoded names for files and then
    copy the output files in their respective folders.

    """

    @property
    def script(self):
        return self._script.format(**(self._file_paths)) + '\nquit\n'

    def __init__(self):
        self._script = ''
        self._file_paths = {}  # paths of input/output files to copy in/from temp dir
        self._loaded_parameters = set()  # parameter files already loaded

    def add_commands(self, *args):
        for command in args:
            self._script += command + '\n'

    def load_parameters(self, *args):
        for par_file in args:
            # Check that this is not already loaded
            if par_file in self._loaded_parameters:
                continue

            # use loadAmberParams if this is a frcmod file and source otherwise
            extension = os.path.splitext(par_file)[1]
            if extension == '.frcmod':
                local_name = 'moli{}'.format(len(self._file_paths))
                self.add_commands('loadAmberParams {{{}}}'.format(local_name))

                # Update list of input files to copy in temporary folder before run
                self._file_paths[local_name] = par_file
            else:
                self.add_commands('source ' + par_file)

            # Update loaded parameters cache
            self._loaded_parameters.add(par_file)

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

    def add_ions(self, unit, ion, num_ions=0):
        self.add_commands('addIons2 {} {} {}'.format(unit, ion, num_ions))

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

    def transform(self, unit, transformation):
        """Transformation is an array-like representing the affine transformation matrix."""
        command = 'transform {} {}'.format(unit, transformation)
        command = command.replace(r'[', '{{').replace(r']', '}}')
        command = command.replace('\n', '').replace('  ', ' ')
        self.add_commands(command)

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

        with mdtraj.utils.enter_temp_directory():
            # Copy input files
            for local_file, file_path in input_files.items():
                shutil.copy(file_path, local_file)

            # Save script and run tleap
            with open('leap.in', 'w') as f:
                f.write(script)
            subprocess.check_output(['tleap', '-f', 'leap.in'])

            # Save leap.log in directory of first output file
            if len(output_files) > 0:
                log_path = os.path.join(os.path.dirname(output_files.values()[0]), 'leap.log')
                shutil.copy('leap.log', log_path)

            #Copy back output files
            for local_file, file_path in output_files.items():
                shutil.copy(local_file, file_path)


#=============================================================================================
# Main and tests
#=============================================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()
