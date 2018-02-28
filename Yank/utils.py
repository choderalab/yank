#!/usr/local/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""
Utils
=====

Utilities for the YANK modules

Provides many helper functions and common operations used by the various YANK suites

"""


# ==============================================================================
# GLOBAL IMPORTS
# ==============================================================================

import os
import re
import copy
import glob
import shutil
import inspect
import logging
import importlib
import functools
import itertools
import contextlib
import subprocess
import collections

from pkg_resources import resource_filename

import mdtraj
import parmed
import numpy as np
from simtk import unit

import openmmtools as mmtools

from . import mpi


# ========================================================================================
# Logging functions
# ========================================================================================

def is_terminal_verbose():
    """Check whether the logging on the terminal is configured to be verbose.

    This is useful in case one wants to occasionally print something that is not really
    relevant to yank's log (e.g. external library verbose, citations, etc.).

    Returns
    -------
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


def config_root_logger(verbose, log_file_path=None):
    """
    Setup the the root logger's configuration.

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
    mpicomm = mpi.get_mpicomm()
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
    file_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    if log_file_path is not None:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(file_format))
        logging.root.addHandler(file_handler)

    # Do not handle logging.DEBUG at all if unnecessary
    if log_file_path is not None:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(terminal_handler.level)

    # Setup critical logger file if a logfile is specified
    # No need to worry about MPI due to it already being set above
    if log_file_path is not None:
        basepath, ext = os.path.splitext(log_file_path)
        critical_log_path = basepath + "_CRITICAL" + ext
        # Create the critical file handler to only create the file IF a critical message is sent
        critical_file_handler = logging.FileHandler(critical_log_path, delay=True)
        critical_file_handler.setLevel(logging.CRITICAL)
        # Add blank lines to space out critical errors
        critical_file_format = file_format + "\n\n\n"
        critical_file_handler.setFormatter(logging.Formatter(critical_file_format))
        logging.root.addHandler(critical_file_handler)


# =======================================================================================
# Profiling Functions
# This is a series of functions and wrappers used for debugging, hence their private nature
# =======================================================================================

def _profile_block_separator_string(message):
    """Write a simple block spacing separator"""
    import time
    time_format = '%d %b %Y %H:%M:%S'
    current_time = time.strftime(time_format)
    spacing_min = 50
    spacing = max(len(current_time), len(message), spacing_min)
    filler = '{' + '0: ^{}'.format(spacing) + '}'
    separator = '#' * (spacing + 2)
    output_string = ''
    output_string += separator + '\n'
    output_string += '#' + filler.format(current_time) + '#\n'
    output_string += '#' + filler.format(message) + '#\n'
    output_string += separator + '\n'
    return output_string


@contextlib.contextmanager
def _profile(output_file='profile.log'):
    """
    Function that allows a ``with _profile():`` to wrap around a calls

    Parameters
    ----------
    output_file: str, Default: 'profile.log'
        Name of the profile you want to write to

    """
    # Imports only used for debugging, not making this part of the name space

    import pstats
    import cProfile
    start_string = _profile_block_separator_string('START PROFILE')
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    end_string = _profile_block_separator_string('END PROFILE')
    sort_by = ['filename', 'cumulative']
    with open(output_file, 'a+') as s:
        s.write(start_string)
        ps = pstats.Stats(pr, stream=s).sort_stats(*sort_by)
        ps.print_stats()
        s.write(end_string)


def _with_profile(output_file='profile.log'):
    """Decorator that profiles the full function wrapper to :func:`_profile`

    Parameters
    ----------
    output_file: str, Default: 'profile.log'
        Name of the profile you want to write to
    """

    def __with_profile(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            with _profile(output_file):
                return func(*args, **kwargs)
        return _wrapper
    return __with_profile


# =======================================================================================
# Method dispatch wrapping to better handle cyclomatic complexity with input types
# Relies on functools.singledispatch (Python 3.5+ only)
# https://stackoverflow.com/questions/24601722/how-can-i-use-functools-singledispatch-with-instance-methods/24602374#24602374
# And the comment to use update_wrapper(wrapper, dispatcher) instead
# =======================================================================================

def methoddispatch(func):
    dispatcher = functools.singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    functools.update_wrapper(wrapper, dispatcher)

    return wrapper


# =======================================================================================
# Combinatorial tree
# =======================================================================================

class CombinatorialLeaf(list):
    """List type that can be expanded combinatorially in :class:`CombinatorialTree`."""
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
        """Generator to iterate over all possible combinations of trees and assign them unique names.

        The names are generated by gluing together the first letters of the values of
        the combinatorial leaves only, separated by the given separator. If the values
        contain special characters, they are ignored. Only letters, numbers and the
        separator are found in the generated names. Values representing paths to
        existing files contribute to the name only with they file name without extensions.

        The iterator yields tuples of ``(name, dict)``, not other :class:`CombinatorialTree`'s. If
        there is only a single combination, an empty string is returned for the name.

        Parameters
        ----------
        separator : str
            The string used to separate the words in the name.
        max_name_length : int
            The maximum length of the generated names, excluding disambiguation number.

        Yields
        ------
        name : str
            Unique name of the combination. Empty string returned if there is only one combination
        combination : dict
            Combination of leafs that was used to create the name

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
        """Return a new :class:`CombinatorialTree` with id-bearing nodes expanded
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
        nodes in a tuple format (e.g. the path to ``node['a']['b']`` is ``('a', 'b')``) while
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

    def _find_combinatorial_leaves(self):
        """Traverse a dict tree and find CombinatorialLeaf nodes.

        Returns
        -------
        combinatorial_leaf_paths, combinatorial_leaf_vals : tuple of tuples
            ``combinatorial_leaf_paths`` is a tuple of paths to combinatorial leaf
            nodes in tuple format (e.g. the path to ``node['a']['b']`` is ``('a', 'b')``)
            while ``combinatorial_leaf_vals`` is the tuple of the values of those nodes.
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

        The iterator returns dict objects, not other :class:`CombinatorialTree`s.

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


# ========================================================================================
# Miscellaneous functions
# ========================================================================================

def get_data_filename(relative_path):
    """Get the full path to one of the reference files shipped for testing

    In the source distribution, these files are in ``examples/*/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.

    Parameters
    ----------
    relative_path : str
        Name of the file to load, with respect to the yank egg folder which
        is typically located at something like
        ``~/anaconda/lib/python3.6/site-packages/yank-*.egg/examples/``

    Returns
    -------
    fn : str
        Resource Filename
    """

    fn = resource_filename('yank', relative_path)

    if not os.path.exists(fn):
        raise ValueError("Sorry! {} does not exist. If you just added it, you'll have to re-install".format(fn))

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


def update_nested_dict(original, updated):
    """
    Return a copy of a (possibly) nested dict of arbitrary depth

    Parameters
    ----------
    original : dict
        Original dict which we want to update, can contain nested dicts
    updated : dict
        Dictionary of updated values to place in original

    Returns
    -------
    new : dict
        Copy of original with values updated from updated
    """
    new = original.copy()
    for key, value in updated.items():
        if isinstance(value, collections.Mapping):
            replacement = update_nested_dict(new.get(key, {}), value)
            new[key] = replacement
        else:
            new[key] = updated[key]
    return new


# ==============================================================================
# Conversion utilities
# ==============================================================================

def find_all_subclasses(parent_cls, discard_abstract=False):
    """Return a set of all the classes inheriting from ``parent_cls``.

    The functions handle multiple inheritance and discard the same classes.

    Parameters
    ----------
    parent_cls : type
        The parent class.
    discard_abstract : bool, optional
        If True, abstract classes are not returned (default is False).

    Returns
    -------
    subclasses : set of type
        The set of all the classes inheriting from ``parent_cls``.

    """
    subclasses = set()
    for subcls in parent_cls.__subclasses__():
        if not (discard_abstract and inspect.isabstract(subcls)):
            subclasses.add(subcls)
        subclasses.update(find_all_subclasses(subcls, discard_abstract))
    return subclasses


def find_subclass(parent_cls, subcls_name):
    """Return the class called ``subcls_name`` inheriting from ``parent_cls``.

    Parameters
    ----------
    parent_cls : type
        The parent class.
    subcls_name : str
        The name of the class inheriting from ``parent_cls``.

    Returns
    -------
    subcls : type
        The class inheriting from ``parent_cls`` called ``subcls_name``.

    Raises
    ------
    ValueError
        If there is no class or there are multiple classes called ``subcls_name``
        that inherit from ``parent_cls``.
    """
    subclasses = []
    for subcls in find_all_subclasses(parent_cls):
        if subcls.__name__ == subcls_name:
            subclasses.append(subcls)
    if len(subclasses) == 0:
        raise ValueError('Could not found class {} inheriting from {}'
                         ''.format(subcls_name, parent_cls))
    if len(subclasses) > 1:
        raise ValueError('Found multiple classes inheriting from {}: {}'
                         ''.format(parent_cls, subclasses))
    return subclasses[0]


def underscore_to_camelcase(underscore_str):
    """Convert the given string from ``underscore_case`` to ``camelCase``.

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
    """Convert the given string from ``camelCase`` to ``underscore_case``.

    Underscores at the beginning and end of the string are preserved. All capital letters are cast to lower case.

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


def quantity_from_string(expression, compatible_units=None):
    """Create a Quantity object from a string expression.

    All the functions in the standard module math are available together
    with most of the methods inside the ``simtk.unit`` module.

    Parameters
    ----------
    expression : str
        The mathematical expression to rebuild a Quantity as a string.
    compatible_units : simtk.unit.Unit, optional
       If given, the result is checked for compatibility against the
       specified units, and an exception raised if not compatible.

       `Note`: The output is not converted to ``compatible_units``, they
       are only used as a unit to validate the input.

    Returns
    -------
    quantity
        The result of the evaluated expression.

    Raises
    ------
    TypeError
        If ``compatible_units`` is given and the quantity in expression is
        either unit-less or has incompatible units.

    Examples
    --------
    >>> expr = '4 * kilojoules / mole'
    >>> quantity_from_string(expr)
    Quantity(value=4.000000000000002, unit=kilojoule/mole)

    >>> expr = '1.0*second'
    >>> quantity_from_string(expr, compatible_units=unit.femtosecond)
    Quantity(value=1.0, unit=second)

    """
    # Retrieve units from unit module.
    if not hasattr(quantity_from_string, '_units'):
        units_tuples = inspect.getmembers(unit, lambda x: isinstance(x, unit.Unit))
        quantity_from_string._units = dict(units_tuples)

    # Eliminate nested quotes and excess whitespace
    expression = expression.strip('\'" ')

    # Handle a special case of the unit when it is just "inverse unit",
    # e.g. Hz == /second
    if expression[0] == '/':
        expression = '(' + expression[1:] + ')**(-1)'

    # Evaluate expressions.
    quantity = mmtools.utils.math_eval(expression, variables=quantity_from_string._units)

    # Check to make sure units are compatible with expected units.
    if compatible_units is not None:
        try:
            is_compatible = quantity.unit.is_compatible(compatible_units)
        except AttributeError:
            raise TypeError("String {} does not have units attached.".format(expression))
        if not is_compatible:
            raise TypeError("Units of {} must be compatible with {}"
                            "".format(expression, str(compatible_units)))

    return quantity


def get_keyword_args(function, try_mro_from_class=None):
    """Inspect function signature and return keyword args with their default values.

    Parameters
    ----------
    function : callable
        The function to interrogate.
    try_mro_from_class : any Class or None
        Try and trace the method resolution order (MRO) of the ``function_to_inspect`` by inferring a method stack from
        the supplied class.
        The signature of the function is checked in every MRO up the stack so long as there exists as
        **kwargs in the method call. This is setting will yield expected results in every case, for instance, if
        the method does not call `super()`, or the Super class has a different function name.
        In the case of conflicting keywords, the lower MRO function is preferred.

    Returns
    -------
    kwargs : dict
        A dictionary ``{'keyword argument': 'default value'}``. The arguments of the
        function that do not have a default value will not be included.

    """

    def extract_kwargs(input_argspec):
        defaults = input_argspec.defaults
        if defaults is None:
            defaults = []
        n_defaults = len(defaults)
        n_args = len(input_argspec.args)
        # Cycle through the kwargs only
        cycle_kwargs = input_argspec.args[n_args - n_defaults:]
        cycle_kwargs = {arg: value for arg, value in zip(cycle_kwargs, defaults)}
        # Handle the kwonlyargs for calls with `def F(a,b *args, x=True, **kwargs)
        if input_argspec.kwonlydefaults is not None:
            cycle_kwargs = {**cycle_kwargs, **input_argspec.kwonlydefaults}
        return cycle_kwargs

    agspec = inspect.getfullargspec(function)
    kwargs = extract_kwargs(agspec)
    if try_mro_from_class is not None and agspec.varkw is not None:
        try:
            mro = inspect.getmro(try_mro_from_class)
        except AttributeError:
            # No MRO
            mro = [try_mro_from_class]
        for cls in mro[1:]:
            try:
                parent_function = getattr(cls, function.__name__)
            except AttributeError:
                # Class does not have a method name
                pass
            else:
                inner_argspec = inspect.getfullargspec(parent_function)
                kwargs = {**extract_kwargs(inner_argspec), **kwargs}
    return kwargs


def validate_parameters(parameters, template_parameters, check_unknown=False,
                        process_units_str=False, float_to_int=False,
                        ignore_none=True, special_conversions=None):
    """
    Utility function for parameters and options validation.

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
        contained in ``template_parameters``.
    process_units_str: bool
        If True, the function will attempt to convert the strings whose template
        type is simtk.unit.Quantity.
    float_to_int : bool
        If True, floats in parameters whose template type is int are truncated.
    ignore_none : bool
        If True, the function do not process parameters whose value is None.
    special_conversions : dict
        Contains a converter function with signature convert(arg) that must be
        applied to the parameters specified by the dictionary key.

    Returns
    -------
    validate_par : dict
        The converted parameters that are contained both in parameters and
        ``template_parameters``.

    Raises
    ------
    TypeError
        If ``check_unknown`` is True and there are parameters not in ``template_parameters``.
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
    >>> input_pars['unknown'] = 'test'  # this will be silently filtered if check_unknown=False


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
                validated_par[par] = quantity_from_string(value, templ_value.unit)

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

    Parameters
    -----------
    file_path : str
        Path to the mol2 path.

    Attributes
    ----------
    resname
    resnames
    net_charge

    """

    def __init__(self, file_path):
        """Constructor."""
        self._file_path = file_path

    @property
    def resname(self):
        """The residue name of the first molecule found in the mol2 file.

        This assumes that each molecule in the mol2 file has a single residue name.

        """
        return next(self.resnames)

    @property
    def resnames(self):
        """Iterate over the names of all the molecules in the file (read-only).

        This assumes that each molecule in the mol2 file has a single residue name.

        """
        new_resname = False
        with open(self._file_path, 'r') as f:
            for line in f:
                # If the previous line was the ATOM directive, yield the resname.
                if new_resname:
                    # The residue name is the 8th word in the line.
                    yield line.split()[7]
                    new_resname = False
                # Go on until you find an atom.
                elif line.startswith('@<TRIPOS>'):
                    section = line[9:].strip()
                    if section == 'ATOM':
                        new_resname = True

    @property
    def net_charge(self):
        """Net charge of the file as a float (read-only)."""
        structure = parmed.load_file(self._file_path, structure=True)
        return self._compute_net_charge(structure)

    def round_charge(self):
        """Round the net charge to the nearest integer to 6-digit precision.

        Raises
        ------
        RuntimeError
            If the total net charge is far from the nearest integer by more
            than 0.05.

        """
        precision = 6

        # Load mol2 file. We load as structure as residues are buggy (see ParmEd#898).
        structure = parmed.load_file(self._file_path, structure=True)
        old_net_charge = self._compute_net_charge(structure)

        # We don't rewrite the mol2 file with ParmEd if the
        # net charge is already within precision.
        expected_net_charge = round(old_net_charge)
        if abs(expected_net_charge - old_net_charge) < 10**(-precision):
            return

        # Convert to residue to use the fix_charges method.
        residue_container = parmed.modeller.ResidueTemplateContainer.from_structure(structure)
        if len(residue_container) > 1:
            logging.warning("Found mol2 file with multiple residues. The charge of "
                            "each residue will be rounded to the nearest integer.")

        # Round the net charge.
        residue_container.fix_charges(precision=precision)

        # Compute new net charge.
        new_net_charge = self._compute_net_charge(residue_container)
        logging.debug('Fixing net charge from {} to {}'.format(old_net_charge, new_net_charge))

        # Something is wrong if the new rounded net charge is very different.
        if abs(old_net_charge - new_net_charge) > 0.05:
            raise RuntimeError('The rounded net charge is too different from the original one.')

        # Copy new charges to structure.
        for structure_residue, residue in zip(structure.residues, residue_container):
            for structure_atom, atom in zip(structure_residue.atoms, residue.atoms):
                structure_atom.charge = atom.charge

        # Rewrite charges.
        parmed.formats.Mol2File.write(structure, self._file_path)

    @staticmethod
    def _compute_net_charge(residue):
        try:
            tot_charge = sum(a.charge for a in residue.atoms)
        except AttributeError:  # residue is a ResidueTemplateContainer
            tot_charge = 0.0
            for res in residue:
                tot_charge += sum(a.charge for a in res.atoms)
        return tot_charge


# -----------------
# OpenEye functions
# -----------------

def is_openeye_installed(oetools=('oechem', 'oequacpac', 'oeiupac', 'oeomega')):
    """
    Check if a given OpenEye tool is installed and Licensed

    If the OpenEye toolkit is not installed, returns False

    Parameters
    ----------
    oetools : str or iterable of strings, Optional, Default: ('oechem', 'oequacpac', 'oeiupac', 'oeomega')
        Set of tools to check by their string name. Defaults to the complete set that YANK *could* use, depending on
        feature requested.

        Only checks the subset of tools if passed. Also accepts a single tool to check as a string instead of an
        iterable of length 1.

    Returns
    -------
    all_installed : bool
        True if all tools in ``oetools`` are installed and licensed, False otherwise
    """
    # Complete list of module: License check
    tools_license = {'oechem': 'OEChemIsLicensed',
                     'oequacpac': 'OEQuacPacIsLicensed',
                     'oeiupac': 'OEIUPACIsLicensed',
                     'oeomega': 'OEOmegaIsLicensed'}
    tool_keys = tools_license.keys()
    # Cast oetools to tuple if its a single string
    if type(oetools) is str:
        oetools = (oetools,)
    tool_set = set(oetools)
    valid_tool_set = set(tool_keys)
    if tool_set & valid_tool_set == set():
        # Check for empty set intersection
        raise ValueError("Expected OpenEye tools to have at least of the following {}, "
                         "but instead got {}".format(tool_keys, oetools))
    try:
        for tool in oetools:
            if tool in tool_keys:
                # Try loading the module
                try:
                    module = importlib.import_module('openeye', tool)
                except SystemError: # Python 3.4 relative import fix
                    module = importlib.import_module('openeye.' + tool)
                # Check that we have the license
                if not getattr(module, tools_license[tool])():
                    raise ImportError
    except ImportError:
        return False
    return True


def load_oe_molecules(file_path, molecule_idx=None):
    """Read one or more molecules from a file.

    Requires OpenEye Toolkit. Several formats are supported (including
    mol2, sdf and pdb).

    Parameters
    ----------
    file_path : str
        Complete path to the file on disk.
    molecule_idx : None or int, optional, default: None
        Index of the molecule on the file. If None, all of them are
        returned.

    Returns
    -------
    molecule : openeye.oechem.OEMol or list of openeye.oechem.OEMol
        The molecules stored in the file. If molecule_idx is specified
        only one molecule is returned, otherwise a list (even if the
        file contain only 1 molecule).

    """
    from openeye import oechem
    extension = os.path.splitext(file_path)[1][1:]  # Remove dot.

    # Open input file stream
    ifs = oechem.oemolistream()
    if extension == 'mol2':
        mol2_flavor = (oechem.OEIFlavor_Generic_Default |
                       oechem.OEIFlavor_MOL2_Default |
                       oechem.OEIFlavor_MOL2_Forcefield)
        ifs.SetFlavor(oechem.OEFormat_MOL2, mol2_flavor)
    if not ifs.open(file_path):
        oechem.OEThrow.Fatal('Unable to open {}'.format(file_path))

    # Read all molecules.
    molecules = []
    for mol in ifs.GetOEMols():
        molecules.append(oechem.OEMol(mol))

    # Select conformation of interest
    if molecule_idx is not None:
        return molecules[molecule_idx]

    return molecules


def write_oe_molecule(oe_mol, file_path, mol2_resname=None):
    """Write all conformations in a file and automatically detects format.

    Requires OpenEye Toolkit

    Parameters
    ----------
    oe_mol : OpenEye Molecule
        Molecule to write to file
    file_path : str
        Complete path to file with filename and extension
    mol2_resname : None or str, Optional, Default: None
        Name to replace the residue name if the file is a .mol2 file
        Requires ``file_path`` to match ``*mol2``
    """
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
    """
    Get the molecule positions from an OpenEye Molecule

    Requires OpenEye Toolkit

    Parameters
    ----------
    molecule : OpenEye Molecule
        Molecule to extract coordinates from
    conformer_idx : int, Optional, Default: 0
        Index of the conformer on the file, leave as 0 to not use
    """
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


def _sanitize_tleap_unit_name(func):
    """Decorator version of TLeap._sanitize_unit_name.

    This takes as unit name a keyword argument called "unit_name" or the
    second sequential argument (skipping self).
    """
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        try:
            kwargs['unit_name'] = TLeap._sanitize_unit_name(kwargs['unit_name'])
        except KeyError:
            # Tuples are immutable so we need to use concatenation.
            args = args[:1] + (TLeap._sanitize_unit_name(args[1]), ) + args[2:]
        func(*args, **kwargs)
    return _wrapper


class TLeap:
    """
    Programmatic interface to write and run AmberTools' ``tLEaP`` scripts.

    To avoid problems with special characters in file paths, the class run the
    tleap script in a temporary folder with hardcoded names for files and then
    copy the output files in their respective folders.

    Attributes
    ----------
    script
    """

    @property
    def script(self):
        """
        Complete and return the finalized script string

        Adds a ``quit`` command to the end of the script.
        """
        return self._script.format(**self._file_paths) + '\nquit\n'

    def __init__(self):
        self._script = ''
        self._file_paths = {}  # paths of input/output files to copy in/from temp dir
        self._loaded_parameters = set()  # parameter files already loaded

    def add_commands(self, *args):
        """
        Append commands to the script

        Parameters
        ----------
        args : iterable of strings
            Individual commands to add to the script written in full as strings.
            Newline characters are added after each command
        """
        for command in args:
            self._script += command + '\n'

    def load_parameters(self, *args):
        """
        Load the LEaP parameters into the working TLEaP script if not already loaded

        This adds to the script

        Uses ``loadAmberParams`` for ``frcmod.*`` files

        Uses ``loadOff`` for ``*.off`` and ``*.lib`` files

        Uses ``source`` for other files.

        Parameters
        ----------
        args : iterable of strings
            File names for each type of leap file that can be loaded.
            Method to load them is automatically determined from file extension or base name
        """
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
            if 'frcmod' in base_name or extension == '.dat':
                self.add_commands('loadAmberParams ' + local_name)
            elif extension == '.off' or extension == '.lib':
                self.add_commands('loadOff ' + local_name)
            else:
                self.add_commands('source ' + local_name)

            # Update loaded parameters cache
            self._loaded_parameters.add(par_file)

    @_sanitize_tleap_unit_name
    def load_unit(self, unit_name, file_path):
        """
        Load a Unit into LEaP, this is typically a molecule or small complex.

        This adds to the script

        Accepts ``*.mol2`` or ``*.pdb`` files

        Parameters
        ----------
        unit_name : str
            Name of the unit as it should be represented in LEaP
        file_path : str
            Full file path with extension of the file to read into LEaP as a new unit
        """
        extension = os.path.splitext(file_path)[1]
        if extension == '.mol2':
            load_command = 'loadMol2'
        elif extension == '.pdb':
            load_command = 'loadPdb'
        else:
            raise ValueError('cannot load format {} in tLeap'.format(extension))
        local_name = 'moli{}'.format(len(self._file_paths))
        self.add_commands('{} = {} {{{}}}'.format(unit_name, load_command, local_name))

        # Update list of input files to copy in temporary folder before run
        self._file_paths[local_name] = file_path

    @_sanitize_tleap_unit_name
    def combine(self, unit_name, *args):
        """
        Combine units in LEaP

        This adds to the script

        Parameters
        ----------
        unit_name : str
            Name of LEaP unit to assign the combination to
        args : iterable of strings
            Name of LEaP units to combine into a single unit called leap_name

        """
        # Sanitize unit names.
        args = [self._sanitize_unit_name(arg) for arg in args]
        components = ' '.join(args)
        self.add_commands('{} = combine {{{{ {} }}}}'.format(unit_name, components))

    @_sanitize_tleap_unit_name
    def add_ions(self, unit_name, ion, num_ions=0, replace_solvent=False):
        """
        Add ions to a unit in LEaP

        This adds to the script

        Parameters
        ----------
        unit_name : str
            Name of the existing LEaP unit which Ions will be added into
        ion : str
            LEaP recognized name of ion to add
        num_ions : int, optional
            Number of ions of type ion to add to unit_name. If 0, the unit
            is neutralized (default is 0).
        replace_solvent : bool, optional
            If True, ions will replace solvent molecules rather than being
            added.
        """
        if replace_solvent:
            self.add_commands('addIonsRand {} {} {}'.format(unit_name, ion, num_ions))
        else:
            self.add_commands('addIons2 {} {} {}'.format(unit_name, ion, num_ions))

    @_sanitize_tleap_unit_name
    def solvate(self, unit_name, solvent_model, clearance):
        """
        Solvate a unit in LEaP isometrically

        This adds to the script

        Parameters
        ----------
        unit_name : str
            Name of the existing LEaP unit which will be solvated
        solvent_model : str
            LEaP recognized name of the solvent model to use, e.g. "TIP3PBOX"
        clearance : float
            Add solvent up to clearance distance away from the unit_name (radial)
        """
        self.add_commands('solvateBox {} {} {} iso'.format(unit_name, solvent_model,
                                                           str(clearance)))

    @_sanitize_tleap_unit_name
    def save_unit(self, unit_name, output_path):
        """
        Write a LEaP unit to file.

        Accepts either ``*.prmtop``, ``*.inpcrd``, or ``*.pdb`` files

        This adds to the script

        Parameters
        ----------
        unit_name : str
            Name of the unit to save
        output_path : str
            Full file path with extension to save.
            Outputs with multiple files (e.g. Amber Parameters) have their names derived from this instead
        """
        file_name = os.path.basename(output_path)
        file_name, extension = os.path.splitext(file_name)
        local_name = 'molo{}'.format(len(self._file_paths))

        # Update list of output files to copy from temporary folder after run
        self._file_paths[local_name] = output_path

        # Add command
        if extension == '.prmtop' or extension == '.inpcrd':
            local_name2 = 'molo{}'.format(len(self._file_paths))
            command = 'saveAmberParm ' + unit_name + ' {{{}}} {{{}}}'

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
            self.add_commands('savePDB {} {{{}}}'.format(unit_name, local_name))
        else:
            raise ValueError('cannot export format {} from tLeap'.format(extension[1:]))

    @_sanitize_tleap_unit_name
    def transform(self, unit_name, transformation):
        """Transformation is an array-like representing the affine transformation matrix."""
        command = 'transform {} {}'.format(unit_name, transformation)
        command = command.replace(r'[', '{{').replace(r']', '}}')
        command = command.replace('\n', '').replace('  ', ' ')
        self.add_commands(command)

    def new_section(self, comment):
        """Adds a comment line to the script"""
        self.add_commands('\n# ' + comment)

    def export_script(self, file_path):
        """
        Write script to file

        Parameters
        ----------
        file_path : str
            Full file path with extension of the script to save
        """
        with open(file_path, 'w') as f:
            f.write(self.script)

    def run(self):
        """Run script and return warning messages in leap log file."""

        def create_dirs_and_copy(path_to_copy, copied_path):
            """Create directories before copying the file."""
            output_dir_path = os.path.dirname(copied_path)
            if not os.path.isdir(output_dir_path):
                os.makedirs(output_dir_path)
            shutil.copy(path_to_copy, copied_path)

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
            leap_output = subprocess.check_output(['tleap', '-f', 'leap.in']).decode()

            # Save leap.log in directory of first output file
            log_path = ''
            if len(output_files) > 0:
                # Get first output path in Py 3.X way that is also thread-safe
                for val in output_files.values():
                    first_output_path = val
                    break
                first_output_name = os.path.basename(first_output_path).split('.')[0]
                first_output_dir = os.path.dirname(first_output_path)
                log_path = os.path.join(first_output_dir, first_output_name + '.leap.log')
                create_dirs_and_copy('leap.log', log_path)

            # Copy back output files. If something goes wrong, some files may not exist
            known_error_msg = []
            try:
                for local_file, file_path in output_files.items():
                    create_dirs_and_copy(local_file, file_path)
            except IOError:
                known_error_msg.append("Could not create one of the system files.")

            # Look for errors in log that don't raise CalledProcessError
            error_patterns = ['Argument #\d+ is type \S+ must be of type: \S+']
            for pattern in error_patterns:
                m = re.search(pattern, leap_output)
                if m is not None:
                    known_error_msg.append(m.group(0))
                    break

            # Analyze log file for water mismatch
            m = re.search("Could not find bond parameter for: EP - \w+W", leap_output)
            if m is not None:
                # Found mismatch water and missing parameters
                known_error_msg.append('It looks like the water used has virtual sites, but '
                                       'missing parameters.\nMake sure your leap parameters '
                                       'use the correct water model as specified by '
                                       'solvent_model.')

            if len(known_error_msg) > 0:
                final_error = ('Some things went wrong with LEaP\nWe caught a few but their may be more.\n'
                               'Please see the log file for LEaP for more info:\n{}\n============\n{}')
                raise RuntimeError(final_error.format(log_path, '\n---------\n'.join(known_error_msg)))

            # Check for and return warnings
            return re.findall('WARNING: (.+)', leap_output)

    @staticmethod
    def _sanitize_unit_name(unit_name):
        """Sanitize tleap unit names.

        Leap doesn't like names that start with digits so, in this case, we
        prepend an arbitrary character.

        This takes as unit name a keyword argument called "unit_name" or the
        second sequential argument (skipping self).
        """
        if unit_name[0].isdigit():
            unit_name = 'M' + unit_name
        return unit_name


# =============================================================================================
# Main and tests
# =============================================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()
