.. _building-docs:

Building the documentation
==========================

The YANK documentation system is modeled after the `MDTraj <http://mdtraj.org>`_ documentation system and runs on
Sphinx with Read The Docs styling.

The YANK documentation is built using `sphinx <http://sphinx-doc.org/>`_ and requires a few dependencies like
`Jupyter <http://jupyter.org/>`_ and `matplotlib <http://matplotlib.org/>`_ that you probably already have installed.
We use `travis-ci <https://travis-ci.org/>`_ for continuous integration (running the tests), and also for building the
documentation, which is built and pushed directly to Amazon S3 after every successful build.

Although `readthedocs <https://readthedocs.org/>`_ is a great tool, it doesn't have the flexibility we need for this
project. We use sphinx's autodoc feature to generate documentation from docstrings formated from NumPy docstring
processing. We also have custom formatting for our detailed YAML options lists.

If you'd like to build the docs on your machine, you'll first need to install sphinx and numpydoc:

.. code-block:: bash

    $ conda install sphinx numpydoc sphinx_rtd_theme runipy

You may also need Jupyter notebooks and matplotlib:

.. code-block:: bash

    $ conda install jupyter matplotlib
  
Now, go back to the docs subdirectory in the main repository. The documentation will be built in the ``docs/_build``
subdirectory.

.. code-block:: bash

    $ cd docs
    $ make html

To view the output of your build, go into the ``docs/_build`` directory and open up ``index.html`` to see how it
rendered in your local browser.