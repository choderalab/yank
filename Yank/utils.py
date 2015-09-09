import os
import logging

from pkg_resources import resource_filename

#========================================================================================
# Utility functions
#========================================================================================

def typename(atype):
    """Convert a type object into a fully qualified typename.

    Parameters
    ----------
    atype : type
        The type to convert
    
    Returns
    -------
    typename : str
        The string typename.
    
    For example,

    >>> typename(type(1))
    'int'

    >>> import numpy
    >>> x = numpy.array([1,2,3], numpy.float32)
    >>> typename(type(x))
    'numpy.ndarray'

    """
    if not isinstance(atype, type):
        raise Exception('Argument is not a type')

    modulename = atype.__module__
    typename = atype.__name__

    if modulename != '__builtin__':
        typename = modulename + '.' + typename

    return typename

def is_terminal_verbose():
    """Check whether the logging on the terminal is configured to be verbose.

    This is useful in case one wants to occasionally print something that is not really
    relevant to yank's log (e.g. external library verbose, citations, etc.).

    Returns
    is_verbose : bool
        True if the terminal is configured to be verbose, False otherwise.
    """

    # If logging.root has no handlers this will ensure that False is returned
    is_verbose = False

    for handler in logging.root.handlers:
        # logging.FileHandler is a subclass of logging.StreamHandler so
        # isinstance and issubclass do not work in this case
        if type(handler) is logging.StreamHandler and handler.level <= logging.DEBUG:
            is_verbose = True
            break

    return is_verbose

def config_root_logger(verbose, log_file_path=None, mpicomm=None):
    """Setup the the root logger's configuration.

     The log messages are printed in the terminal and saved in the file specified
     by log_file_path (if not None) and printed. Note that logging use sys.stdout
     to print logging.INFO messages, and stderr for the others. The root logger's
     configuration is inherited by the loggers created by logging.getLogger(name).

     Different formats are used to display messages on the terminal and on the log
     file. For example, in the log file every entry has a timestamp which does not
     appear in the terminal. Moreover, the log file always shows the module that
     generate the message, while in the terminal this happens only for messages
     of level WARNING and higher.

    Parameters
    ----------
    verbose : bool
        Control the verbosity of the messages printed in the terminal. The logger
        displays messages of level logging.INFO and higher when verbose=False.
        Otherwise those of level logging.DEBUG and higher are printed.
    log_file_path : str, optional, default = None
        If not None, this is the path where all the logger's messages of level
        logging.DEBUG or higher are saved.
    mpicomm : mpi4py.MPI.COMM communicator, optional, default=None
        If specified, this communicator will be used to determine node rank.

    """

    class TerminalFormatter(logging.Formatter):
        """
        Simplified format for INFO and DEBUG level log messages.

        This allows to keep the logging.info() and debug() format separated from
        the other levels where more information may be needed. For example, for
        warning and error messages it is convenient to know also the module that
        generates them.
        """

        # This is the cleanest way I found to make the code compatible with both
        # Python 2 and Python 3
        simple_fmt = logging.Formatter('%(message)s')
        default_fmt = logging.Formatter('%(levelname)s - %(name)s - %(message)s')

        def format(self, record):
            if record.levelno <= logging.INFO:
                return self.simple_fmt.format(record)
            else:
                return self.default_fmt.format(record)

    # Check if root logger is already configured
    n_handlers = len(logging.root.handlers)
    if n_handlers > 0:
        root_logger = logging.root
        for i in xrange(n_handlers):
            root_logger.removeHandler(root_logger.handlers[0])

    # If this is a worker node, don't save any log file
    if mpicomm:
        rank = mpicomm.rank
    else:
        rank = 0

    if rank != 0:
        log_file_path = None

    # Add handler for stdout and stderr messages
    terminal_handler = logging.StreamHandler()
    terminal_handler.setFormatter(TerminalFormatter())
    if rank != 0:
        terminal_handler.setLevel(logging.WARNING)
    elif verbose:
        terminal_handler.setLevel(logging.DEBUG)
    else:
        terminal_handler.setLevel(logging.INFO)
    logging.root.addHandler(terminal_handler)

    # Add file handler to root logger
    if log_file_path is not None:
        file_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(file_format))
        logging.root.addHandler(file_handler)

    # Do not handle logging.DEBUG at all if unnecessary
    if log_file_path is not None:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(terminal_handler.level)

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
