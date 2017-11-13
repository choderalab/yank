import cerberus
import os


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

    # ====================================================
    # DATA VALIDATORS
    # ====================================================
    def _validator_is_peptide(self, field, filepath):
        """Input file is a peptide."""
        if not os.path.isfile(filepath):
            self._error(field, 'File path does not exist.')
        extension = os.path.splitext(filepath)[1]
        if extension != '.pdb':
            self._error(field, "Not a .pdb file")

    def _validator_is_small_molecule(self, field, filepath):
        """Input file is a small molecule."""
        file_formats = frozenset(['mol2', 'sdf', 'smiles', 'csv'])
        if not os.path.isfile(filepath):
            self._error(field, 'File path does not exist.')
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

