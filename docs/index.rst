####
YANK
####

.. TODO: Standardize this tagline
**A GPU-accelerated Python framework for exploring algorithms for alchemical free energy calculations**

.. TODO: Refine this description

Features
--------
* modular Python framework for easily exploring new algorithms
* GPU-accelerated via the `OpenMM toolkit <http://openmm.org>`_
* `alchemical free energy calculations <http://alchemistry.org>`_ in both **explicit** and **implicit** solvent
* Hamiltonian exchange among alchemical intermediates with Gibbs sampling framework
* general `Markov chain Monte Carlo <http://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ framework for exploring enhanced sampling methods
* built-in equilibration detection and convergence diagnostics
* support for AMBER prmtop/inpcrd files
* support for absolute binding free energy calculations
* support for transfer free energies (such as hydration free energies)

Get involved
------------
* `Download the code <https://github.com/choderalab/yank>`_
* `See YANK in action <examples/index.html>`_
* `Report a bug or request a feature <https://github.com/choderalab/yank/issues>`_
.. TODO: * `Get help with YANK <http://discourse.choderalab.org>`_

--------------------------------------------------------------------------------

Documentation
-------------

.. toctree::
   :maxdepth: 1

   quickstart
   installation
   getting_started
   theory
   algorithms
   analysis
   examples
   benchmarks
   acknowledgments
   <!---
   examples/index
   whatsnew
   faq
   Discussion Forums <http://discourse.choderalab.org>
   -->

API Reference
-------------

.. toctree::
   :maxdepth: 1

   yank

Developing
----------

.. toctree::
   :maxdepth: 1

   <!--- store_format -->
   building_docs
   style

--------------------------------------------------------------------------------

.. raw:: html

   </div>

License
-------
YANK is licensed under the Lesser GNU General Public License (LGPL v2.1+).
See the `LICENSE` file distributed with YANK for more details.