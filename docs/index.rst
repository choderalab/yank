####
YANK
####

.. TODO: Standardize this tagline

**A GPU-accelerated Python framework for exploring algorithms for alchemical free energy calculations**

.. TODO: Refine this description

Features
--------
* Modular Python framework for easily exploring new algorithms
* GPU-accelerated via the `OpenMM toolkit <http://openmm.org>`_
* `Alchemical free energy calculations <http://alchemistry.org>`_ in both **explicit** and **implicit** solvent
* Hamiltonian exchange among alchemical intermediates with Gibbs sampling framework
* General `Markov chain Monte Carlo <http://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ framework for exploring enhanced sampling methods
* Built-in equilibration detection and convergence diagnostics
* Support for AMBER prmtop/inpcrd files
* Support for absolute binding free energy calculations
* Support for transfer free energies (such as hydration free energies)

Get involved
------------
* Download the code or fork the repo on `GitHub <https://github.com/choderalab/yank>`_
* See YANK `action <examples/index.html>`_
* Report a bug or request a feature on the `GitHub issue tracker <https://github.com/choderalab/yank/issues>`_

--------------------------------------------------------------------------------

Documentation
-------------

.. toctree::
   :maxdepth: 1

   quickstart
   installation
   running
   theory
   algorithms
   analysis
   benchmarks
   acknowledgments
   examples/index
   whatsnew
   faq

API Reference
-------------

.. toctree::
   :maxdepth: 1

   yank

Developing
----------

.. toctree::
   :maxdepth: 1

   store_format
   building_docs
   style

--------------------------------------------------------------------------------

.. raw:: html

   </div>

License
-------
YANK is licensed under the Lesser GNU General Public License (LGPL v2.1+).
See the `LICENSE` file distributed with YANK for more details.
