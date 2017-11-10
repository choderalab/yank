import cerberus
from openmmtools.utils import typename
from datetime import date
from collections.abc import Mapping, Sequence


class YANKCerberusUtilsError(Exception):
    """Error class to handle all base errors raise by the utilities in this module"""
    pass


def generate_unknown_type_validator(a_type):
    """
    Helper function to :func:`type_to_cerberus_map` to convert unknown types into a validator

    Parameters
    ----------
    a_type : type
        Any valid type, although this works for any type, its preferred to let :func:`type_to_cerberus_map`
        to invoke this method

    Returns
    -------
    validator_map : dict
        Map dictionary of form ``{'validator': <function>}`` where ``'validator'`` is literal and
        ``<function>`` is a validator function of type checker.
    """
    def nonstandard_type_validator(field, value, error):
        if type(value) is not a_type:
            error(field, 'Must be of type {}'.format(typename(a_type)))
    validator_dict = {'validator': nonstandard_type_validator}
    return validator_dict


def type_to_cerberus_map(a_type):
    """
    Convert a provided type to a valid Cerberus internal type dict.

    Parameters
    ----------
    a_type : type
        Type you want the converter to built-in Cerberus strings

    Returns
    -------
    type_map : dict
        Type map dictionary of form ``{'type': 'TYPE'}`` where ``'type'`` is literal and ``TYPE`` is the mapped
        type in Cerberus

        OR

         Map dictionary of form ``{'validator': <function>}`` where ``'validator'`` is literal and
        ``<function>`` is a validator function of type checker for non-standard

    Raises
    ------
    YANKCerberusUtilsError
        When the type has no known map
    """
    known_types = {bool, 'boolean',
                   bytes, 'binary',
                   bytearray, 'binary',
                   date, 'date',
                   Mapping, 'dict',
                   dict, 'dict',
                   float, 'float',
                   int, 'integer',
                   tuple, 'list',
                   list, 'list',
                   Sequence, 'list',
                   str, 'string',
                   set, 'set'}
    try:
        type_map = {'type': known_types[a_type]}
    except KeyError:
        type_map = generate_unknown_type_validator(a_type)
    return type_map


