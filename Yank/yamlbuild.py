#!/usr/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Tools to build Yank experiments from a YAML configuration file.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import yaml
import copy
import logging
import itertools
import collections
logger = logging.getLogger(__name__)

from simtk import unit

import utils


#=============================================================================================
# UTILITY FUNCTIONS
#=============================================================================================

def set_dict_path(d, path, val):
    """Set the value of a dictionary by indicating a path as a tuple.

    Parameters
    ----------
    d : dict
        The dictionary to be modified.
    path : iterable of keys
        A list-like object containing the nested keys of the value to modify.
    val :
        The value used to set the dictionary node.

    Examples
    --------
    >>> d = {'a': {'b': 2}}
    >>> set_dict_path(d, ('a', 'b'), 3)
    >>> d
    {'a': {'b': 3}}

    """
    d_node = reduce(lambda d,k: d[k], path[:-1], d)
    d_node[path[-1]] = val

def not_iterable_container(val):
    """Check whether the given value is a list-like object or not.

    Returns
    -------
    True if val is a string or not iterable, False otherwise.

    """
    # strings are iterable too so we have to treat them as a special case
    return isinstance(val, str) or not isinstance(val, collections.Iterable)

def find_leaves(node):
    """Recursive function to traverse a dict tree and find the leaf nodes.

    Parameters:
    node : dict
        The root of the tree in a dict format.

    Returns:
    --------
    A tuple containing two lists. The first one is a list of paths to the leaf
    nodes in a tuple format (e.g. the path to node['a']['b'] is ('a', 'b') while
    the second one is a list of all the values of those leaf nodes.

    Examples:
    ---------
    >>> simple_tree = {'simple': {'scalar': 1,
    ...                           'vector': [2, 3, 4],
    ...                           'nested': {'leaf': ['a', 'b', 'c']}}}
    >>> leaf_paths, leaf_vals = find_leaves(simple_tree)
    >>> leaf_paths
    [('simple', 'scalar'), ('simple', 'vector'), ('simple', 'nested', 'leaf')]
    >>> leaf_vals
    [1, [2, 3, 4], ['a', 'b', 'c']]

    """
    leaf_paths = []
    leaf_vals = []
    for child_key, child_val in node.items():
        if isinstance(child_val, collections.Mapping):
            subleaf_paths, subleaf_vals = find_leaves(child_val)
            # prepend child key to path
            leaf_paths.extend([(child_key,) + subleaf for subleaf in subleaf_paths])
            leaf_vals.extend(subleaf_vals)
        else:
            leaf_paths.append((child_key,))
            leaf_vals.append(child_val)
    return leaf_paths, leaf_vals

def expand_tree(tree):
    """Iterate over all possible combinations of trees.

    The tree is in the format of a nested dictionary. If the function finds leaf nodes
    containing list-like objects, it generates all possible combinations of trees, each
    one defining only one of the single values in those lists per leaf node.

    Parameters
    ----------
    tree : dict
        A nested dict representing a tree. All the list-like objects in the leaf nodes
        are expanded in a combinatorial sense.

    Returns
    -------
    Iterator that loops over all the possible combinations of trees.

    Examples
    --------
    >>> import pprint
    >>> tree = {'a': 1, 'b': [1, 2], 'c': {'d': [3, 4]}}
    >>> for t in expand_tree(tree):
    ...     pprint.pprint(t)  # pprint sort the dictionary before printing
    {'a': 1, 'b': 1, 'c': {'d': 3}}
    {'a': 1, 'b': 2, 'c': {'d': 3}}
    {'a': 1, 'b': 1, 'c': {'d': 4}}
    {'a': 1, 'b': 2, 'c': {'d': 4}}

    """
    template_tree = copy.deepcopy(tree)
    leaf_paths, leaf_vals = find_leaves(tree)

    # itertools.product takes only iterables so we need to convert single values
    for i, leaf_val in enumerate(leaf_vals):
        if not_iterable_container(leaf_val):
            leaf_vals[i] = [leaf_val]

    # generating all combinations
    for combination in itertools.product(*leaf_vals):
        # update values of template tree
        for leaf_path, leaf_val in zip(leaf_paths, combination):
            set_dict_path(template_tree, leaf_path, leaf_val)
        yield copy.deepcopy(template_tree)

def process_second_compatible_quantity(quantity_str):
    """
    Shortcut to process a string containing a quantity compatible with seconds.

    Parameters
    ----------
    quantity_str : str
         A string containing a value with a unit of measure for time.

    Returns
    -------
    quantity : simtk.unit.Quantity
       The specified string, returned as a Quantity.

    See Also
    --------
    utils.process_unit_bearing_str : the function used for the actual conversion.

    """
    return utils.process_unit_bearing_str(quantity_str, unit.seconds)

def process_bool(bool_val):
    """Raise ValueError if this an ambiguous representation of a bool.

    PyYAML load a boolean the following words: true, True, yes, Yes, false, False,
    no and No, but in Python strings and numbers can be used as booleans and create
    subtle errors. This function ensure that the value is a true boolean.

    Returns
    -------
    bool
        bool_val only if it is a boolean.

    Raises
    ------
    ValueError
        If bool_var is not a boolean.

    """
    if isinstance(bool_val, bool):
        return bool_val
    else:
        raise ValueError('The value must be true, yes, false, or no.')

#=============================================================================================
# BUILDER CLASS
#=============================================================================================

class YamlParseError(Exception):
    """Represent errors occurring during parsing of Yank YAML file."""
    pass

class YamlBuilder:
    """Parse YAML configuration file and build the experiment.

    Properties
    ----------
    options : dict
        The options specified in the parsed YAML file.

    """

    _accepted_options = frozenset(['title',
                                   'timestep', 'nsteps_per_iteration', 'number_of_iterations',
                                   'minimize','equilibrate', 'equilibration_timestep',
                                   'number_of_equilibration_iterations'])
    _expected_options_types = (('timestep', process_second_compatible_quantity),
                               ('nsteps_per_iteration', int),
                               ('number_of_iterations', int),
                               ('minimize', process_bool),
                               ('equilibrate', process_bool),
                               ('equilibration_timestep', process_second_compatible_quantity),
                               ('number_of_equilibration_iterations', int))

    @property
    def options(self):
        return self._options

    def __init__(self, yaml_file):
        """Parse the given YAML configuration file.

        This does not build the actual experiment but simply checks that the syntax
        is correct and loads the configuration into memory.

        Parameters
        ----------
        yaml_file : str
            A relative or absolute path to the YAML configuration file.

        """

        # TODO check version of yank-yaml language
        # TODO what if there are multiple streams in the YAML file?
        with open(yaml_file, 'r') as f:
            yaml_config = yaml.load(f)

        if yaml_config is None:
            error_msg = 'The YAML file is empty!'
            logger.error(error_msg)
            raise YamlParseError(error_msg)

        # Find and merge options and metadata
        try:
            opts = yaml_config['options']
        except KeyError:
            opts = {}
            logger.warning('No YAML options found.')
        try:
            opts.update(yaml_config['metadata'])
        except KeyError:
            pass

        # Set only accepted options
        self._options = {x: opts[x] for x in opts if x in YamlBuilder._accepted_options}
        if len(self._options) != len(opts):
            unknown_opts = {x for x in opts if x not in YamlBuilder._accepted_options}
            error_msg = 'YAML configuration contains unidentifiable options: '
            error_msg += ', '.join(unknown_opts)
            logger.error(error_msg)
            raise YamlParseError(error_msg)

        # Enforce types that are not automatically recognized by yaml
        for special_opt, casting_func in YamlBuilder._expected_options_types:
            if special_opt in self._options:
                try:
                    self._options[special_opt] = casting_func(self._options[special_opt])
                except (TypeError, ValueError) as e:
                    error_msg = 'YAML option %s: %s' % (special_opt, str(e))
                    logger.error(error_msg)
                    raise YamlParseError(error_msg)

    def build_experiment(self):
        """Build the Yank experiment (TO BE IMPLEMENTED)."""
        raise NotImplemented

if __name__ == "__main__":
    import doctest
    doctest.testmod()
