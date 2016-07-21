#!/usr/bin/python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Test examples.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import shutil
import subprocess

import yaml
import openmoltools as omt

from nose.plugins.attrib import attr


#=============================================================================================
# UNIT TESTS
#=============================================================================================

def remove_files(*args):
    for arg in args:
        if os.path.isdir(arg):
            shutil.rmtree(arg)
        elif os.path.isfile(arg):
            os.remove(arg)


def run_example(path, example):
    n_iterations = 2
    example_dir = os.path.join(path, example)
    runsh_path = os.path.join(example_dir, 'run.sh')
    yaml_path = os.path.join(example_dir, 'yank.yaml')

    # Skip folders that do not contain working examples (e.g. yank-yaml-cookbook)
    if not os.path.exists(runsh_path):
        return

    # check if this is a yaml or a cli example
    if os.path.exists(yaml_path):
        # temporary files to delete
        testoutput_dir = os.path.join(example_dir, 'testoutput')
        testsh_path = os.path.join(example_dir, 'test.sh')
        testyaml_path = os.path.join(example_dir, 'yank_test.yaml')

        # clean up previous unsuccessful simulation
        remove_files(testoutput_dir, testsh_path, testyaml_path)

        # adjust yaml simulation parameters
        with open(yaml_path, 'r') as f:
            test_yaml_script = yaml.load(f)
        test_yaml_script['options']['number_of_iterations'] = n_iterations
        test_yaml_script['options']['output_dir'] = testoutput_dir
        with open(testyaml_path, 'w') as f:
            f.write(yaml.dump(test_yaml_script))

        # adjust run.sh to use yank_test.yaml
        with open(runsh_path, 'r') as f:
            testsh_script = f.read()
        testsh_script = testsh_script.replace('yank.yaml', 'yank_test.yaml')
        testsh_script = testsh_script.replace('experiments',
                                              os.path.join(testoutput_dir, 'experiments'))
        with open(testsh_path, 'w') as f:
            f.write(testsh_script)

        # call test.sh
        command = 'bash test.sh'
    else:
        # set number of iterations through environment variable
        command = 'NITERATIONS={} ./run.sh'.format(n_iterations)

    # Change to example directory.
    with omt.utils.temporary_cd(example_dir):
        # Execute n_iterations of the example
        returncode = subprocess.call(command, shell=True, executable='/bin/bash')

    if returncode:
        raise Exception('Example %s returned exit code %d' % (example, returncode))

    # Clean up temporary files AFTER raising eventual exception so that
    # we can dig into the log to understand the reasons behind the failure
    if os.path.exists(yaml_path):
        remove_files(testoutput_dir, testsh_path, testyaml_path)

    return

def get_immediate_subdirectories(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and os.path.exists(os.path.join(path, name, 'run.sh'))]

@attr('slow') # Skip on Travis-CI
def test_examples():
    # Get example path.
    from pkg_resources import resource_filename
    path = resource_filename('yank', '../examples')

    # Get list of examples.
    directories = get_immediate_subdirectories(path)

    # Test examples
    for directory in directories:
        run_example(path, directory)

