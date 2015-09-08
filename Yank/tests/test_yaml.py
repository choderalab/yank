#!/usr/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Test YAML functions.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import tempfile
import textwrap

from yank.yamlbuild import YamlBuilder

#=============================================================================================
# UNIT TESTS
#=============================================================================================

def test_yaml_parse():
    """Check that YAML file is parsed correctly."""

    yaml_text = """
    ---
    options:
        wrong_option: 3
        timestep: 1.0 * femtoseconds
        nsteps_per_iteration: 2500
    """

    yaml_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        # Check that handles no options
        yaml_file.write('---\ntest: 2')
        yaml_file.close()
        yaml_builder = YamlBuilder(yaml_file.name)
        assert len(yaml_builder.options) == 0

        # Check that handle unknown options
        with open(yaml_file.name, 'w') as f:
            f.write(textwrap.dedent(yaml_text))
            f.close()
        yaml_builder = YamlBuilder(yaml_file.name)
        assert len(yaml_builder.options) == 2
        assert 'timestep' in yaml_builder.options
        assert yaml_builder.options['nsteps_per_iteration'] == 2500
    finally:
        os.remove(yaml_file.name)

