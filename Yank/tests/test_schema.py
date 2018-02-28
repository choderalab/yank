#!/usr/local/bin/env python

"""
Test Cerberus schema validation.

"""

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

from yank.schema.validator import *


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def test_generate_signature_schema():
    """Test generate_signature_schema() function."""
    def f(a, b, camelCase=True, none=None, quantity=3.0*unit.angstroms):
        pass

    f_schema = generate_signature_schema(f)
    assert len(f_schema) == 3
    for v in f_schema.values():
        assert isinstance(v, dict)
        assert v['required'] is False

    # Remove Optional() marker for comparison
    assert f_schema['camel_case']['type'] == 'boolean'
    assert f_schema['none']['nullable'] is True
    assert hasattr(f_schema['quantity']['coerce'], '__call__')  # Callable validator

    # Check conversion
    f_validator = YANKCerberusValidator(f_schema)
    assert f_validator.validated({'quantity': '5*angstrom'}) == {'quantity': 5*unit.angstrom}

    # Check update
    optional_instance = {'camel_case': {'type': 'integer'}, 'none': {'type': 'float'}}
    updated_schema = generate_signature_schema(f, update_keys=optional_instance,
                                               exclude_keys={'quantity'})
    assert len(updated_schema) == 2
    assert updated_schema['none'].get('nullable') is None
    assert updated_schema['none'].get('type') == 'float'
    assert updated_schema['camel_case'].get('type') == 'integer'
