import os
import re
import sys
import copy
import glob
import json
import shutil
import signal
import pandas
import inspect
import logging
import itertools
import subprocess
import collections
from contextlib import contextmanager

from pkg_resources import resource_filename

import mdtraj
import parmed
import numpy as np
from simtk import unit
from schema import Optional, Use

from openmoltools.utils import wraps_py2, unwrap_py2  # Shortcuts for other modules

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
        for i in range(n_handlers):
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


# =======================================================================================
# MPI utility functions
# =======================================================================================

def initialize_mpi():
    """Initialize and configure MPI to handle correctly terminate.

    Returns
    -------
    mpicomm : mpi4py communicator
        The communicator for this node.

    """
    # Check for environment variables set by mpirun. Variables are from
    # http://docs.roguewave.com/threadspotter/2012.1/linux/manual_html/apas03.html
    variables = ['PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'OMPI_MCA_ns_nds_vpid',
                 'PMI_ID', 'SLURM_PROCID', 'LAMRANK', 'MPI_RANKID',
                 'MP_CHILD', 'MP_RANK', 'MPIRUN_RANK']
    use_mpi = False
    for var in variables:
        if var in os.environ:
            use_mpi = True
            break
    if not use_mpi:
        return None

    # Initialize MPI
    from mpi4py import MPI
    MPI.COMM_WORLD.barrier()
    mpicomm = MPI.COMM_WORLD

    # Override sys.excepthook to abort MPI on exception
    def mpi_excepthook(type, value, traceback):
        sys.__excepthook__(type, value, traceback)
        sys.stdout.flush()
        sys.stderr.flush()
        if mpicomm.size > 1:
            mpicomm.Abort(1)
    # Use our eception handler
    sys.excepthook = mpi_excepthook

    # Catch sigterm signals
    def handle_signal(signal, frame):
        if mpicomm.size > 1:
            mpicomm.Abort(1)
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGABRT]:
        signal.signal(sig, handle_signal)

    return mpicomm

@contextmanager
def delay_termination():
    """Context manager to delay handling of termination signals."""
    signals_to_catch = [signal.SIGINT, signal.SIGTERM, signal.SIGABRT]
    old_handlers = {signum: signal.getsignal(signum) for signum in signals_to_catch}
    signals_received = {signum: None for signum in signals_to_catch}

    def delay_handler(signum, frame):
        signals_received[signum] = (signum, frame)

    # Set handlers fot delay
    for signum in signals_to_catch:
        signal.signal(signum, delay_handler)

    yield  # Resume program

    # Restore old handlers
    for signum, handler in listitems(old_handlers):
        signal.signal(signum, handler)

    # Fire delayed signals
    for signum, s in listitems(signals_received):
        if s is not None:
            old_handlers[signum](*s)


def delayed_termination(func):
    """Decorator to delay handling of termination signals during function execution."""
    @wraps_py2(func)
    def _delayed_termination(*args, **kwargs):
        with delay_termination():
            return func(*args, **kwargs)
    return _delayed_termination


# =======================================================================================
# Combinatorial tree
# =======================================================================================

class CombinatorialLeaf(list):
    """List type that can be expanded combinatorially in CombinatorialTree."""
    def __repr__(self):
        return "Combinatorial({})".format(super(CombinatorialLeaf, self).__repr__())


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
    >>> print(tree)
    {'a': {}}

    Expand all possible combinations of a tree. The iterator return a dict, not another
    CombinatorialTree object.
    >>> import pprint  # pprint sort the dictionary by key before printing
    >>> tree = CombinatorialTree({'a': 1, 'b': CombinatorialLeaf([1, 2]),
    ...                           'c': {'d': CombinatorialLeaf([3, 4])}})
    >>> for t in tree:
    ...     pprint.pprint(t)
    {'a': 1, 'b': 1, 'c': {'d': 3}}
    {'a': 1, 'b': 2, 'c': {'d': 3}}
    {'a': 1, 'b': 1, 'c': {'d': 4}}
    {'a': 1, 'b': 2, 'c': {'d': 4}}

    Expand all possible combinations and assign unique names
    >>> for name, t in tree.named_combinations(separator='_', max_name_length=5):
    ...     print(name)
    3_1
    3_2
    4_1
    4_2

    """
    def __init__(self, dictionary):
        """Build a combinatorial tree from the given dictionary."""
        self._d = copy.deepcopy(dictionary)

    def __getitem__(self, path):
        try:
            return self._d[path]
        except KeyError:
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
        leaf_paths, leaf_vals = self._find_combinatorial_leaves()
        return self._combinations_generator(leaf_paths, leaf_vals)

    def named_combinations(self, separator, max_name_length):
        """Iterate over all possible combinations of trees and assign them unique names.

        The names are generated by gluing together the first letters of the values of
        the combinatorial leaves only, separated by the given separator. If the values
        contain special characters, they are ignored. Only letters, numbers and the
        separator are found in the generated names. Values representing paths to
        existing files contribute to the name only with they file name without extensions.

        The iterator yields tuples of (name, dict), not other CombinatorialTrees. If
        there is only a single combination, an empty string is returned for the name.

        Parameters
        ----------
        separator : str
            The string used to separate the words in the name.
        max_length : int
            The maximum length of the generated names, excluding disambiguation number.

        """
        leaf_paths, leaf_vals = self._find_combinatorial_leaves()
        generated_names = {}  # name: count, how many times we have generated the same name

        # Compile regular expression used to discard special characters
        filter = re.compile('[^A-Za-z\d]+')

        # Iterate over combinations
        for combination in self._combinations_generator(leaf_paths, leaf_vals):
            # Retrieve single values of combinatorial leaves
            filtered_vals = [str(self._resolve_path(combination, path)) for path in leaf_paths]

            # Strip down file paths to only the file name without extensions
            for i, val in enumerate(filtered_vals):
                if os.path.exists(val):
                    filtered_vals[i] = os.path.basename(val).split(os.extsep)[0]

            # Filter special characters in values that we don't use for names
            filtered_vals = [filter.sub('', val) for val in filtered_vals]

            # Generate name
            if len(filtered_vals) == 0:
                name = ''
            elif len(filtered_vals) == 1:
                name = filtered_vals[0][:max_name_length]
            else:
                name = separator.join(filtered_vals)
                original_vals = filtered_vals[:]
                while len(name) > max_name_length:
                    # Sort the strings by descending length, if two values have the
                    # same length put first the one whose original value is the shortest
                    sorted_vals = sorted(enumerate(filtered_vals), reverse=True,
                                         key=lambda x: (len(x[1]), -len(original_vals[x[0]])))

                    # Find how many strings have the maximum length
                    max_val_length = len(sorted_vals[0][1])
                    n_max_vals = len([x for x in sorted_vals if len(x[1]) == max_val_length])

                    # We trim the longest str by the necessary number of characters
                    # to reach max_name_length or the second longest value
                    length_diff = len(name) - max_name_length

                    if n_max_vals < len(filtered_vals):
                        second_max_val_length = len(sorted_vals[n_max_vals][1])
                        length_diff = min(length_diff, max_val_length - second_max_val_length)

                    # Trim all the longest strings by few characters
                    for i in range(n_max_vals - 1, -1, -1):
                        # Division truncation ensures that we trim more the
                        # ones whose original value is the shortest
                        char_per_str = int(length_diff / (i + 1))
                        if char_per_str != 0:
                            idx = sorted_vals[i][0]
                            filtered_vals[idx] = filtered_vals[idx][:-char_per_str]
                        length_diff -= char_per_str

                    name = separator.join(filtered_vals)

            if name in generated_names:
                generated_names[name] += 1
                name += separator + str(generated_names[name])
            else:
                generated_names[name] = 1
            yield name, combination

    def expand_id_nodes(self, id_nodes_path, update_nodes_paths):
        """Return a new CombinatorialTree with id-bearing nodes expanded
        and updated in the rest of the script.

        Parameters
        ----------
        id_nodes_path : tuple of str
            The path to the parent node containing ids.
        update_nodes_paths : list of tuple of str
            A list of all the paths referring to the ids expanded. The string '*'
            means every node.

        Returns
        -------
        expanded_tree : CombinatorialTree
            The tree with id nodes expanded.

        Examples
        --------
        >>> d = {'molecules':
        ...          {'mol1': {'mol_value': CombinatorialLeaf([1, 2])}},
        ...      'systems':
        ...          {'sys1': {'molecules': 'mol1'},
        ...           'sys2': {'prmtopfile': 'mysystem.prmtop'}}}
        >>> update_nodes_paths = [('systems', '*', 'molecules')]
        >>> t = CombinatorialTree(d).expand_id_nodes('molecules', update_nodes_paths)
        >>> t['molecules'] == {'mol1_1': {'mol_value': 1}, 'mol1_2': {'mol_value': 2}}
        True
        >>> t['systems'] == {'sys1': {'molecules': CombinatorialLeaf(['mol1_2', 'mol1_1'])},
        ...                  'sys2': {'prmtopfile': 'mysystem.prmtop'}}
        True

        """
        expanded_tree = copy.deepcopy(self)
        combinatorial_id_nodes = {}  # map combinatorial_id -> list of combination_ids

        for id_node_key, id_node_val in self.__getitem__(id_nodes_path).items():
            # Find all combinations and expand them
            id_node_val = CombinatorialTree(id_node_val)
            combinations = {id_node_key + '_' + name: comb for name, comb
                            in id_node_val.named_combinations(separator='_', max_name_length=30)}

            if len(combinations) > 1:
                # Substitute combinatorial node with all combinations
                del expanded_tree[id_nodes_path][id_node_key]
                expanded_tree[id_nodes_path].update(combinations)
                # We need the combinatorial_id_nodes substituted to an id_node_key
                # to have a deterministic value or MPI parallel processes will
                # iterate over combinations in different orders
                combinatorial_id_nodes[id_node_key] = sorted(combinations.keys())

        # Update ids in the rest of the tree
        for update_path in update_nodes_paths:
            for update_node_key, update_node_val in self._resolve_paths(self._d, update_path):
                # Check if the value is a collection or a scalar
                if isinstance(update_node_val, list):
                    for v in update_node_val:
                        if v in combinatorial_id_nodes:
                            i = expanded_tree[update_node_key].index(v)
                            expanded_tree[update_node_key][i:i+1] = combinatorial_id_nodes[v]
                elif update_node_val in combinatorial_id_nodes:
                    comb_leaf = CombinatorialLeaf(combinatorial_id_nodes[update_node_val])
                    expanded_tree[update_node_key] = comb_leaf

        return expanded_tree

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
        The value contained in the node pointed by the path.

        """
        accum_value = d
        for node_key in path:
            accum_value = accum_value[node_key]
        return accum_value

    @staticmethod
    def _resolve_paths(d, path):
        """Retrieve all the values of a nested key in a dictionary.

        Paths containing the string '*' are interpreted as any node and
        are yielded one by one.

        Parameters
        ----------
        d : dict
            The nested dictionary.
        path : iterable of str
            The "path" to the node of the dictionary. The character '*'
            means any node.

        Examples
        --------
        >>> d = {'nested': {'correct1': {'a': 1}, 'correct2': {'a': 2}, 'wrong': {'b': 3}}}
        >>> p = [x for x in CombinatorialTree._resolve_paths(d, ('nested', '*', 'a'))]
        >>> print(sorted(p))
        [(('nested', 'correct1', 'a'), 1), (('nested', 'correct2', 'a'), 2)]

        """
        try:
            if len(path) == 0:
                yield (), d
            elif len(path) == 1:
                yield (path[0],), d[path[0]]
            else:
                if path[0] == '*':
                    keys = d.keys()
                else:
                    keys = [path[0]]
                for key in keys:
                    for p, v in CombinatorialTree._resolve_paths(d[key], path[1:]):
                        if v is not None:
                            yield (key,) + p, v
        except KeyError:
            yield None, None

    def _find_leaves(self):
        """Traverse a dict tree and find the leaf nodes.

        Returns
        -------
        A tuple containing two lists. The first one is a list of paths to the leaf
        nodes in a tuple format (e.g. the path to node['a']['b'] is ('a', 'b')) while
        the second one is a list of all the values of those leaf nodes.

        Examples
        --------
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
            for child_key, child_val in listitems(node):
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

    def _find_combinatorial_leaves(self):
        """Traverse a dict tree and find CombinatorialLeaf nodes.

        Returns
        -------
        combinatorial_leaf_paths, combinatorial_leaf_vals : tuple of tuples
            combinatorial_leaf_paths is a tuple of paths to combinatorial leaf
            nodes in tuple format (e.g. the path to node['a']['b'] is ('a', 'b'))
            while combinatorial_leaf_vals is the tuple of the values of those nodes.
            The list of paths is guaranteed to be sorted by alphabetical order.

        """
        leaf_paths, leaf_vals = self._find_leaves()

        # Filter leaves that are not combinatorial
        combinatorial_ids = [i for i, val in enumerate(leaf_vals) if isinstance(val, CombinatorialLeaf)]
        combinatorial_leaf_paths = [leaf_paths[i] for i in combinatorial_ids]
        combinatorial_leaf_vals = [leaf_vals[i] for i in combinatorial_ids]

        # Sort leaves by alphabetical order of the path
        if len(combinatorial_leaf_paths) > 0:
            combinatorial_leaf_paths, combinatorial_leaf_vals = zip(*sorted(zip(combinatorial_leaf_paths,
                                                                                combinatorial_leaf_vals)))
        return combinatorial_leaf_paths, combinatorial_leaf_vals

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

        # All leaf values must be CombinatorialLeafs at this point
        assert all(isinstance(leaf_val, CombinatorialLeaf) for leaf_val in leaf_vals)

        # generating all combinations
        for combination in itertools.product(*leaf_vals):
            # update values of template tree
            for leaf_path, leaf_val in zip(leaf_paths, combination):
                template_tree[leaf_path] = leaf_val
            yield copy.deepcopy(template_tree._d)


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


def find_phases_in_store_directory(store_directory):
    """Build a list of phases in the store directory.

    Parameters
    ----------
    store_directory : str
       The directory to examine for stored phase NetCDF data files.

    Returns
    -------
    phases : dict of str
       A dictionary phase_name -> file_path that maps phase names to its NetCDF
       file path.

    """
    full_paths = glob.glob(os.path.join(store_directory, '*.nc'))

    phases = {}
    for full_path in full_paths:
        file_name = os.path.basename(full_path)
        short_name, _ = os.path.splitext(file_name)
        phases[short_name] = full_path

    if len(phases) == 0:
        raise RuntimeError("Could not find any valid YANK store (*.nc) files in "
                           "store directory: {}".format(store_directory))
    return phases


def is_iterable_container(value):
    """Check whether the given value is a list-like object or not.

    Returns
    -------
    Flase if value is a string or not iterable, True otherwise.

    """
    # strings are iterable too so we have to treat them as a special case
    return not isinstance(value, str) and isinstance(value, collections.Iterable)


# ==============================================================================
# Conversion utilities
# ==============================================================================

def serialize_topology(topology):
    """Serialize topology to string.

    Parameters
    ----------
    topology : mdtraj.Topology, simtk.openmm.app.Topology
        The topology object to serialize.

    Returns
    -------
    serialized_topology : str
        String obtained by jsonizing the return value of to_dataframe() of the
        mdtraj Topology object.

    """
    # Check if we need to convert the topology to mdtraj
    if isinstance(topology, mdtraj.Topology):
        mdtraj_top = topology
    else:
        mdtraj_top = mdtraj.Topology.from_openmm(topology)

    atoms, bonds = mdtraj_top.to_dataframe()
    separators = (',', ':')  # don't dump whitespaces to save space
    serialized_topology = json.dumps({'atoms': atoms.to_json(orient='records'),
                                      'bonds': bonds.tolist()},
                                     separators=separators)
    return serialized_topology


def deserialize_topology(serialized_topology):
    """Serialize topology to string.

    Parameters
    ----------
    serialized_topology : str
        Serialized topology as returned by serialize_topology().

    Returns
    -------
    topology : mdtraj.Topology
        The deserialized topology object.

    """
    topology_dict = json.loads(serialized_topology)
    atoms = pandas.read_json(topology_dict['atoms'], orient='records')
    bonds = np.array(topology_dict['bonds'])
    topology = mdtraj.Topology.from_dataframe(atoms, bonds)
    return topology


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

    # Python 2/3 builtins check
    if modulename != '__builtin__' and modulename != 'builtins':
        typename = modulename + '.' + typename

    return typename


def merge_dict(dict1, dict2):
    """Return the union of two dictionaries.

    In Python 3.5 there is a syntax to do this {**dict1, **dict2} but
    in Python 2 you need to go through update().

    """
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict


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


def camelcase_to_underscore(camelcase_str):
    """Convert the given string from camelCase to underscore_case.

    Parameters
    ----------
    camelcase_str : str
        String in camelCase to convert to underscore style.

    Returns
    -------
    underscore_str : str
        String in underscore style.

    Examples
    --------
    >>> camelcase_to_underscore('myVariable')
    'my_variable'
    >>> camelcase_to_underscore('__my_Variable_')
    '__my__variable_'

    """
    underscore_str = re.sub(r'([A-Z])', '_\g<1>', camelcase_str)
    return underscore_str.lower()


def quantity_from_string(quantity_str):
    """
    Generate a simtk.unit.Quantity object from a string of arbitrary nested strings

    Parameters
    ----------
    quantity_str : string
        A string containing a value with a unit of measure

    Returns
    -------
    quantity : simtk.unit.Quantity
        The specified string, returned as a Quantity

    Raises
    ------
    AttributeError
        If quantity_str does not contain any parsable data
    TypeError
        If quantity_str does not contain units

    Examples
    --------
    >>> quantity_from_string("1*atmosphere")
    Quantity(value=1.0, unit=atmosphere)

    >>> quantity_from_string("'1 * joule / second'")
    Quanity(value=1, unit=joule/second)

    """

    # Strip out (possible) surrounding quotes
    quote_pattern = '[^\'"]+'
    try:
        quantity_str = re.search(quote_pattern, quantity_str).group()
    except AttributeError as e:
        raise AttributeError("Please pass a quantity in format of '#*unit'. e.g. '1*atmosphere'")
    # Parse String
    operators = ['(', ')', '*', '/']
    def find_operator(passed_str):
        # Process the current string until the next operator
        for i, char in enumerate(passed_str):
           if char in operators:
               break
        return i

    def nested_string(passed_str):
        def exponent_unit(passed_str):
            # Attempt to cast argument as an exponent
            future_operator_loc = find_operator(passed_str)
            future_operator = passed_str[future_operator_loc]
            if future_operator == '(': # This catches things like x**(3*2), rare, but it could happen
                exponent, exponent_type, exp_count_indices = nested_string(passed_str[future_operator_loc+1:])
            elif future_operator_loc == 0:
                # No more operators
                exponent = passed_str
                future_operator_loc = len(passed_str)
                exp_count_indices = future_operator_loc + 2 # +2 to skip the **
            else:
                exponent = passed_str[:future_operator_loc]
                exp_count_indices = future_operator_loc + 2 # +2 to skip the **
            exponent = float(exponent)  # These should only ever be numbers, not quantities, error occurs if they aren't
            if exponent.is_integer():  # Method of float
                exponent = int(exponent)
            return exponent, exp_count_indices
        # Loop through a given string level, returns how many indices of the string it got through
        last_char_loop = 0
        number_pass_string = len(passed_str)
        last_operator = None
        final_quantity = None
        # Close Parenthesis flag
        paren_closed = False
        while last_char_loop < number_pass_string:
            next_char_loop = find_operator(passed_str[last_char_loop:]) + last_char_loop
            next_char = passed_str[next_char_loop]
            # Figure out what the operator is
            if (next_char_loop == number_pass_string - 1 and (next_char != ')')) or (next_char_loop == 0 and next_char != '(' and next_char != ')'):
                # Case of no new operators found
                argument = passed_str[last_char_loop:]
            else:
                argument = passed_str[last_char_loop:next_char_loop]
            # Strip leading/trailing spaces
            argument = argument.strip(' ')
            # Determine if argument is a unit
            try:
                arg_unit = getattr(unit, argument)
                arg_type = 'unit'
            except Exception as e:
                # Assume its float
                try:
                    arg_unit = float(argument)
                    arg_type = 'float'
                except:  # Usually empty string
                    if argument == '':
                        arg_unit = None
                        arg_type = 'None'
                    else:
                        raise e  # Raise the syntax error
            # See if we are at the end
            augment = None
            count_indices = 1 # How much to offset by to move past operator
            if next_char_loop != number_pass_string:
                next_operator = passed_str[next_char_loop]
                if next_operator == '*':
                    try: # Exponent
                        if passed_str[next_char_loop+1] == '*':
                            exponent, exponent_offset = exponent_unit(passed_str[next_char_loop+2:])
                            try:
                                next_char_loop += exponent_offset
                                # Set the actual next operator (Does not handle nested **)
                                next_operator = passed_str[next_char_loop]
                            except IndexError:
                                # End of string
                                next_operator = None
                            # Apply exponent
                            arg_unit **= exponent
                    except:
                        pass
                # Check for parentheses
                if next_operator == '(':
                    augment, augment_type, count_indices  = nested_string(passed_str[next_char_loop+1:])
                    count_indices += 1 # add 1 more to offset the '(' itself
                elif next_operator == ')':
                    paren_closed = True
            else:
                # Case of no found operators
                next_operator = None
            # Handle the conditions
            if last_operator is None:
                if (final_quantity is None) and (arg_type is 'None') and (augment is None):
                    raise TypeError("Given Quantity could not be interpreted as presented")
                elif (final_quantity is None) and (augment is None):
                    final_quantity = arg_unit
                    final_type = arg_type
                elif (final_quantity is None) and (arg_type is 'None'):
                    final_quantity = augment
                    final_type = augment_type
            else:
                if augment is None:
                    augment = arg_unit
                    augment_type = arg_type
                if last_operator == '*':
                    final_quantity *= augment
                elif last_operator == '/':
                    final_quantity /= augment
                # Assign type
                if augment_type == 'unit':
                    final_type = 'unit'
                elif augment_type == 'float':
                    final_type = 'float'
            last_operator = next_operator
            last_char_loop = next_char_loop + count_indices # Set the new position here skipping over processed terms
            if paren_closed:
                # Determine if the next term is a ** to exponentiate augment
                try:
                    if passed_str[last_char_loop:last_char_loop+2] == '**':
                        exponent, exponent_offset = exponent_unit(passed_str[last_char_loop+2:])
                        final_quantity **= exponent
                        last_char_loop += exponent_offset
                except:
                    pass
                break
        return final_quantity, final_type, last_char_loop

    quantity, final_type, x = nested_string(quantity_str)
    return quantity


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

    # Convert string of a Quantity to actual Quantity
    quantity = quantity_from_string(quantity_str)
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


def to_unit_validator(compatible_units):
    """Function generator to test unit bearing strings with Schema."""
    def _to_unit_validator(quantity_str):
        return process_unit_bearing_str(quantity_str, compatible_units)
    return _to_unit_validator


def generate_signature_schema(func, update_keys=None, exclude_keys=frozenset()):
    """Generate a dictionary to test function signatures with Schema.

    Parameters
    ----------
    func : function
        The function used to build the schema.
    update_keys : dict
        Keys in here have priority over automatic generation. It can be
        used to make an argument mandatory, or to use a specific validator.
    exclude_keys : list-like
        Keys in here are ignored and not included in the schema.

    Returns
    -------
    func_schema : dict
        The dictionary to be used as Schema type. Contains all keyword
        variables in the function signature as optional argument with
        the default type as validator. Unit bearing strings are converted.
        Argument with default None are always accepted. Camel case
        parameters in the function are converted to underscore style.

    Examples
    --------
    >>> from schema import Schema
    >>> def f(a, b, camelCase=True, none=None, quantity=3.0*unit.angstroms):
    ...     pass
    >>> f_dict = generate_signature_schema(f, exclude_keys=['quantity'])
    >>> print(isinstance(f_dict, dict))
    True
    >>> # Print (key, values) in the correct order
    >>> print(sorted(listitems(f_dict), key=lambda x: x[1]))
    [(Optional('camel_case'), <type 'bool'>), (Optional('none'), <type 'object'>)]
    >>> f_schema = Schema(generate_signature_schema(f))
    >>> f_schema.validate({'quantity': '1.0*nanometer'})
    {'quantity': Quantity(value=1.0, unit=nanometer)}

    """
    if update_keys is None:
        update_keys = {}

    func_schema = {}
    args, _, _, defaults = inspect.getargspec(unwrap_py2(func))

    # Check keys that must be excluded from first pass
    exclude_keys = set(exclude_keys)
    exclude_keys.update(update_keys)
    exclude_keys.update({k._schema for k in update_keys if isinstance(k, Optional)})

    # Transform camelCase to underscore
    # TODO: Make sure this is working from the Py3.X conversion
    args = [camelcase_to_underscore(arg) for arg in args ]

    # Build schema
    for arg, default_value in zip(args[-len(defaults):], defaults):
        if arg not in exclude_keys:  # User defined keys are added later
            if default_value is None:  # None defaults are always accepted
                validator = object
            elif isinstance(default_value, unit.Quantity):  # Convert unit strings
                validator = Use(to_unit_validator(default_value.unit))
            else:
                validator = type(default_value)
            func_schema[Optional(arg)] = validator

    # Add special user keys
    func_schema.update(update_keys)

    return func_schema


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
                        ignore_none=True, special_conversions=None):
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
    if special_conversions is None:
        special_conversions = {}

    # Create validated parameters
    validated_par = {par: parameters[par] for par in parameters
                     if par in template_parameters}

    # Check for unknown parameters
    if check_unknown and len(validated_par) < len(parameters):
        diff = set(parameters) - set(template_parameters)
        raise TypeError("found unknown parameter {}".format(', '.join(diff)))

    for par, value in listitems(validated_par):
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


# ==============================================================================
# Stuff to move to openmoltools/ParmEd when they'll be stable
# ==============================================================================

class Mol2File(object):
    """Wrapper of ParmEd mol2 parser for easy manipulation of mol2 files.

    This is not efficient as every operation access the file. The purpose
    of this class is simply to provide a shortcut to read and write the mol2
    file with a one-liner. If you need to do multiple operations before
    saving the file, use ParmEd directly.

    This works only for single-structure mol2 files.

    """

    def __init__(self, file_path):
        """Constructor.

        Parameters
        -----------
        file_path : str
            Path to the mol2 path.

        """
        self._file_path = file_path

    @property
    def resname(self):
        residue = parmed.load_file(self._file_path)
        return residue.name

    @resname.setter
    def resname(self, value):
        residue = parmed.load_file(self._file_path)
        residue.name = value
        parmed.formats.Mol2File.write(residue, self._file_path)

    @property
    def net_charge(self):
        residue = parmed.load_file(self._file_path)
        return sum(a.charge for a in residue.atoms)

    @net_charge.setter
    def net_charge(self, value):
        residue = parmed.load_file(self._file_path)
        residue.fix_charges(to=value, precision=6)
        parmed.formats.Mol2File.write(residue, self._file_path)


# OpenEye functions
# ------------------
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


class TLeap:
    """Programmatic interface to write and run tLeap scripts.

    To avoid problems with special characters in file paths, the class run the
    tleap script in a temporary folder with hardcoded names for files and then
    copy the output files in their respective folders.

    """

    @property
    def script(self):
        return self._script.format(**self._file_paths) + '\nquit\n'

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

            # Check whether this is a user file or a tleap file, and
            # update list of input files to copy in temporary folder before run
            if os.path.isfile(par_file):
                local_name = 'moli{}'.format(len(self._file_paths))
                self._file_paths[local_name] = par_file
                local_name = '{' + local_name + '}'
            else:  # tleap file
                local_name = par_file

            # use loadAmberParams if this is a frcmod file and source otherwise
            base_name = os.path.basename(par_file)
            extension = os.path.splitext(base_name)[1]
            if 'frcmod' in base_name:
                self.add_commands('loadAmberParams ' + local_name)
            elif extension == '.off' or extension == '.lib':
                self.add_commands('loadOff ' + local_name)
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
            command = 'saveAmberParm ' + group + ' {{{}}} {{{}}}'

            # Update list of output files with the one not explicit
            if extension == '.inpcrd':
                extension2 = '.prmtop'
                command = command.format(local_name2, local_name)
            else:
                extension2 = '.inpcrd'
                command = command.format(local_name, local_name2)
            output_path2 = os.path.join(os.path.dirname(output_path), file_name + extension2)
            self._file_paths[local_name2] = output_path2

            self.add_commands(command)
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
        """Run script and return warning messages in leap log file."""
        # Transform paths in absolute paths since we'll change the working directory
        input_files = {local + os.path.splitext(path)[1]: os.path.abspath(path)
                       for local, path in listitems(self._file_paths) if 'moli' in local}
        output_files = {local + os.path.splitext(path)[1]: os.path.abspath(path)
                        for local, path in listitems(self._file_paths) if 'molo' in local}

        # Resolve all the names in the script
        local_files = {local: local + os.path.splitext(path)[1]
                       for local, path in listitems(self._file_paths)}
        script = self._script.format(**local_files) + 'quit\n'

        with mdtraj.utils.enter_temp_directory():
            # Copy input files
            for local_file, file_path in listitems(input_files):
                shutil.copy(file_path, local_file)

            # Save script and run tleap
            with open('leap.in', 'w') as f:
                f.write(script)
            leap_output = subprocess.check_output(['tleap', '-f', 'leap.in']).decode()

            # Save leap.log in directory of first output file
            if len(output_files) > 0:
                #Get first output path in Py 3.X way that is also thread-safe
                for val in listvalues(output_files):
                    first_output_path = val
                    break
                first_output_name = os.path.basename(first_output_path).split('.')[0]
                first_output_dir = os.path.dirname(first_output_path)
                log_path = os.path.join(first_output_dir, first_output_name + '.leap.log')
                shutil.copy('leap.log', log_path)

            # Copy back output files. If something goes wrong, some files may not exist
            error_msg = ''
            try:
                for local_file, file_path in listitems(output_files):
                    shutil.copy(local_file, file_path)
            except IOError:
                error_msg = "Could not create one of the system files."

            # Look for errors in log that don't raise CalledProcessError
            error_patterns = ['Argument #\d+ is type \S+ must be of type: \S+']
            for pattern in error_patterns:
                m = re.search(pattern, leap_output)
                if m is not None:
                    error_msg = m.group(0)
                    break

            if error_msg != '':
                raise RuntimeError(error_msg + ' Check log file {}'.format(log_path))

            # Check for and return warnings
            return re.findall('WARNING: (.+)', leap_output)


#=============================================================================================
# Python 2/3 compatability
#=============================================================================================

"""
Generate same behavior for dict.item in both versions of Python
Avoids external dependancies on future.utils or six

"""
try:
    dict.iteritems
except AttributeError:
    # Python 3
    def listvalues(d):
        return list(d.values())
    def listitems(d):
        return list(d.items())
else:
    # Python 2
    def listvalues(d):
        return d.values()
    def listitems(d):
        return d.items()

#=============================================================================================
# Main and tests
#=============================================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()
