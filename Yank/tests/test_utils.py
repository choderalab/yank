#!/usr/local/bin/env python

"""
Test various utility functions.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

from yank.utils import YankOptions

#=============================================================================================
# TESTING FUNCTIONS
#=============================================================================================

def test_yank_options():
    """Test option priorities and handling."""

    cl_opt = {'option1': 1}
    yaml_opt = {'option1': 2, 'option2': 'test'}
    default_yank_opt = YankOptions()
    yank_opt = YankOptions(cl_opt=cl_opt, yaml_opt=yaml_opt)

    assert yank_opt['option2'] == 'test'
    assert yank_opt['option1'] == 1  # command line > yaml
    assert len(yank_opt) == len(defaul_yank_opt) + 2, "Excepted two additional options beyond default, found: %s" % str([x for x in yank_opt])

    # runtime > command line
    yank_opt['option1'] = 0
    assert yank_opt['option1'] == 0

    # restore old option when deleted at runtime
    del yank_opt['option1']
    assert yank_opt['option1'] == 1

    # modify specific priority level
    yank_opt.default = {'option3': -2}
    assert len(yank_opt) == 3
    assert yank_opt['option3'] == -2

    # test iteration interface
    assert yank_opt.items() == [('option1', 1), ('option2', 'test'), ('option3', -2)]
    assert yank_opt.keys() == ['option1', 'option2', 'option3']

