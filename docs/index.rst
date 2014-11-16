YANK
====

*GPU-accelerated alchemical free energy calculations made simple.*

YANK uses a sophisticated set of algorithms to rigorously compute (within a classical statistical mechanical framework) biomolecular ligand binding free energies. This is accomplished by performing an alchemical free energy calculation in either implicit or explicit solvent, in which the interactions between a ligand (usually a small molecule or peptide) are decoupled in a number of alchemical intermediates whose interactions with the environment are modified, creating an alternative thermodynamic cycle to the direct dissociation of ligand and target biomolecule. The free energy of decoupling the ligand from the environment is computed both in the presence and absence of the biomolecular target, yielding the overall binding free energy (in the strong single-binding-mode framework of Gilson et al.) once a standard state correction is applied to correct for the restraint added between ligand and biomolecule in the complex.

Computation of the free energy for each leg of the thermodynamic cycle utilized a modified replica-exchange simulations in which exchanges between alchemical states are permitted (Hamiltonian exchange) and faster mixing is achieved by using a Gibbs sampling framework.

<!---
.. raw:: html

  <div>
      <h2 style="display: inline; float:left; margin-left:5em">
          <a href="https://github.com/choderalab/yank/releases/latest">
          Download the Code</a>
      </h2>
      <h2 style="display: inline; float:right; margin-right:5em">
          <a href="examples/index.html">
          See it in Action</a>
      </h2>
      <div style="clear:both"></div>
      <h2 style="display: inline; float:left; margin-left:7em">
          <a href="http://discourse.choderalab.org/">
          Get Help</a>
      </h2>
      <h2 style="display: inline; float:right; margin-right:7em">
          <a href="https://github.com/choderalab/yank/issues">
          Get Involved</a>
      </h2>
  </div>
  <br/>

.. raw:: html

   <div style="display:none">
--->

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

   <!---
   yank
   -->

Developing
----------

.. toctree::
   :maxdepth: 1

   <!--- netcdf_format -->
   building_docs
   style

--------------------------------------------------------------------------------

.. raw:: html

   </div>

License
-------
YANK is licensed under the Lesser GNU General Public License (LGPL v2.1+).
