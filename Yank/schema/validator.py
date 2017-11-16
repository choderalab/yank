import os
import logging
import cerberus
import simtk.unit as unit

from datetime import date
from openmmtools.utils import typename
from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)


class YANKCerberusValidator(cerberus.Validator):
    """
    Custom cerberus.Validator class for YANK extending the Validator to include all the methods needed
    by YANK
    """
    # ====================================================
    # DATA COERCION
    # ====================================================
    def _normalize_coerce_single_str_to_list(self, value):
        """Cast a single string to a list of string"""
        return [value] if isinstance(value, str) else value

    def _normalize_coerce_sort_alphabetically_by_extension(self, files):
        provided_extensions = [os.path.splitext(filepath)[1][1:] for filepath in files]
        return [filepath for (ext, filepath) in sorted(zip(provided_extensions, files))]

    # ====================================================
    # DATA VALIDATORS
    # ====================================================
    def _validator_file_exists(self, field, filepath):
        """Assert that the file is in fact, a file!"""
        if not os.path.isfile(filepath):
            self._error(field, 'File path {} does not exist.'.format(filepath))

    def _validator_directory_exists(self, field, directory_path):
        """Assert that the file is in fact, a file!"""
        if not os.path.isdir(directory_path):
            self._error(field, 'Directory {} does not exist.'.format(directory_path))

    def _validator_is_peptide(self, field, filepath):
        """Input file is a peptide."""
        extension = os.path.splitext(filepath)[1]
        if extension != '.pdb':
            self._error(field, "Not a .pdb file")

    def _validator_is_small_molecule(self, field, filepath):
        """Input file is a small molecule."""
        file_formats = frozenset(['mol2', 'sdf', 'smiles', 'csv'])
        extension = os.path.splitext(filepath)[1][1:]
        if extension not in file_formats:
            self._error(field, 'File is not one of {}'.format(file_formats))

    def _validator_positive_int_list(self, field, value):
        for p in value:
            if not isinstance(p, int) or not p >= 0:
                self._error(field, "{} of must be a positive integer!".format(p))

    def _validator_int_or_all_string(self, field, value):
        if value != 'all' and not isinstance(value, int):
            self._error(field, "{} must be an int or the string 'all'".format(value))

    def _validator_is_system_files_matching_phase1(self, field, phase2):
        """Ensure the phase2_filepaths match the type of the phase1 paths"""
        phase1 = self.document.get('phase1_path')
        # Handle non-existant
        if phase1 is None:
            self._error(field, "phase1_path must be present to use phase2_path")
            return
        # Handle the not iterable type
        try:
            _ = [f for f in phase1]
            _ = [f for f in phase2]
        except TypeError:
            self._error(field, 'phase1_path and phase2_path must be a list of file paths')
            return
        # Now process
        provided_phase1_extensions = [os.path.splitext(filepath)[1][1:] for filepath in phase1]
        provided_phase2_extensions = [os.path.splitext(filepath)[1][1:] for filepath in phase2]
        # Check phase1 extensions
        phase1_type = None
        valid_extensions = None
        expected_extensions = {
            'amber': ['inpcrd', 'prmtop'],
            'gromacs': ['gro', 'top'],
            'openmm': ['pdb', 'xml']
        }
        for extension_type, valid_extensions in expected_extensions.items():
            if sorted(provided_phase1_extensions) == sorted(valid_extensions):
                phase1_type = extension_type
                break
        if phase1_type is None:
            self._error(field, 'phase1_path must have file extensions matching one of the following types: '
                               '{}'.format(expected_extensions))
            return
        # Ensure phase 2 is of the same type
        if sorted(provided_phase2_extensions) != sorted(valid_extensions):
            self._error(field, 'phase2_path must have files of the same extensions as phase1_path ({}) to ensure '
                               'phases came from the same setup pipeline.'.format(phase1_type))
            return

        logger.debug('Correctly recognized phase1_path files ({}) and phase2_path files ({}) '
                     'as {} files'.format(phase1, phase2, phase1_type))

    # ====================================================
    # DATA TYPES
    # ====================================================

    def _validate_type_quantity(self, value):
        if isinstance(value, unit.Quantity):
            return True


# ====================================================
# UTILITY FUNCTIONS
# ====================================================

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
    known_types = {bool: 'boolean',
                   bytes: 'binary',
                   bytearray: 'binary',
                   date: 'date',
                   Mapping: 'dict',
                   dict: 'dict',
                   float: 'float',
                   int: 'integer',
                   tuple: 'list',
                   list: 'list',
                   Sequence: 'list',
                   str: 'string',
                   set: 'set'}
    try:
        type_map = {'type': known_types[a_type]}
    except KeyError:
        type_map = generate_unknown_type_validator(a_type)
    return type_map
