####
YANK
####

.. TODO: Standardize this tagline

**A GPU-accelerated Python framework for exploring algorithms for alchemical free energy calculations**

.. TODO: Refine this description

.. note:: |IntroNoteText|

.. |IntroNoteText| replace::
    YANK is now in Early Access for its 1.0 release!
    YAML syntax should be fully operational while we prepare the underlying Python API.
    The program has not yet been extensively validated. Use at your own risk!

Features
--------
* Modular Python framework to facilitate development and testing of new algorithms
* GPU-accelerated via the `OpenMM toolkit <http://openmm.org>`_ (see `benchmarks <http://openmm.org/about.html#benchmarks>`_)
* `Alchemical free energy calculations <http://alchemistry.org>`_ in both **explicit** (PME, reaction field) and **implicit** (GBSA) solvent
* Multi-state sampling schemes, including replica exchange with `Gibbs sampling <http://dx.doi.org/10.1063/1.3660669>`_ and `self-adjusted mixture sampling (SAMS) <https://doi.org/10.1080/10618600.2015.1113975>`_
* Extensible `Markov chain Monte Carlo <http://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ framework for exploring enhanced sampling methods
* Built-in [automated equilibration detection](http://dx.doi.org/10.1021/acs.jctc.5b00784) and convergence diagnostics
* Support for reading Amber, gromacs, and CHARMM input files
* Support for absolute binding free energy calculations and transfer free energies (such as hydration, partition, or relative solvation free energies)

Get going
---------
* `Get YANK up and running now <quickstart.html>`_

Get involved
------------
* Download the code or fork the repo on `GitHub <https://github.com/choderalab/yank>`_
* See YANK in `action <examples/index.html>`_
* Report a bug or request a feature on the `GitHub issue tracker <https://github.com/choderalab/yank/issues>`_

--------------------------------------------------------------------------------

.. Special TOC so people can always get home

Getting YANK Up and Running
---------------------------

.. toctree::
   :maxdepth: 3

   quickstart
   installation
   running
   yamlpages/index
   yank-yaml-cookbook/index
   whatsnew
   examples/index
   faq

The Science Behind YANK
-----------------------

.. toctree::
   :maxdepth: 3

   theory
   algorithms
   analysis
   benchmarks

Citations and Thanks
--------------------

.. toctree::
   :maxdepth: 1

   references
   acknowledgments

API Reference
-------------

.. toctree::
   :maxdepth: 1

   api/index

Developing
----------

.. toctree::
   :maxdepth: 1

   store_format
   building_docs

--------------------------------------------------------------------------------

.. raw:: html

   </div>

License
-------
YANK is licensed under the MIT License.
See the `LICENSE` file distributed with YANK for more details.
