import os
import logging
import cerberus
import simtk.unit as unit

logger = logging.getLogger(__name__)


class YANKCerberusValidationError(Exception):
    """Base error class for YANK Cerberus Validation"""
    pass


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