import os
import sys
import logging

from pkg_resources import resource_filename

#========================================================================================
# Exceptions
#========================================================================================

class OverwriteLogException(Exception):
    """
    Exception denoting an attempt to overwrite logging's configuration.

    """
    pass

#========================================================================================
# Utility functions
#========================================================================================

def config_root_logger(verbose, log_file_path=None, overwrite=False):
    """Setup the the root logger's configuration.

     The log messages are saved in the file specified by log_file_path (if not
     None) and printed. Note that logging use sys.stdout to print logging.INFO
     messages, and stderr for the others. The root logger's configuration is
     inherited by the others module loggers created by logging.getLogger(name).

     In order to overwrite the root logger's configuration one must set
     overwrite=True or an Exception will be raised. This helps to control if
     parts of the code attempt to modify the user's configuration.

    Parameters
    ----------
    verbose : bool
        Control the verbosity of the messages printed on stdout. The logger
        displays messages of level logging.INFO and higher when verbose=False.
        Otherwise those of level logging.DEBUG and higher are printed.
    log_file_path : str, optional, default = None
        If not None, this is the path where all the logger's messages of level
        logging.DEBUG or higher are saved.
    overwrite : bool, optional, default = False
        Set to True to force overwriting an eventual existing configuration. An
        attempt to do so when overwrite=False raises an Exception.

    """
    # Check if root logger is already configured
    n_handlers = len(logging.getLogger().handlers)
    if n_handlers > 0:
        if overwrite:
            root_logger = logging.getLogger()
            for i in xrange(n_handlers):
                root_logger.removeHandler(root_logger.handlers[0])
        else:
            raise OverwriteLogException('Attempted to overwrite logging configuration.')

    # Configure verbosity of stdout and stderr messages
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Add file handler to root logger
    if log_file_path is not None:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(file_handler)

def get_data_filename(relative_path):
    """Get the full path to one of the reference files shipped for testing

    In the source distribution, these files are in ``examples/*/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.

    Parameters
    ----------
    name : str
        Name of the file to load, with respect to the yank egg folder which
        is typically located at something like
        ~/anaconda/lib/python2.7/site-packages/yank-*.egg/examples/
    """

    fn = resource_filename('yank', relative_path)

    if not os.path.exists(fn):
        raise ValueError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)

    return fn
