.. _analysis:

.. currentmodule:: yank

Analysis
========

The YANK analysis tools are built around bringing in modern analysis techniques to help the user better understand
the final free energy output that you get. In theory, we could just print out
"Free energy of Binding =  XX.XXX +- Y.YYY kcal/mol," but that alone does not tell you anything about the quality of
your free energy estimate. We try to provide guides and built-in analysis to help understand the numbers YANK provides.

The analysis framework comes three ways to help the user analyze their data (use these links to **jump to usage**):

* :ref:`Command line automatic analysis <auto_analyze>`
* :ref:`Visually with Jupyter Notebooks<visual_analyze>`
* :ref:`Programmatic Python API <programmatic_analyze>`


Analysis Theory
---------------

We run YANK simulations through multiple data processing tools improve the estimate in free energy and other
thermodynamic properties. We document them in
the :doc:`algorithms` page, but summarize them here, and link to their detailed explanation in the appropriate page.
When running :ref:`automatically <auto_analyze>`, these steps are run in order to yield the final free energy estimate.
The analysis procedure is carried out for each phase of the YANK simulation.


Equilibration and Decorrelation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We first discard non-equilibrated samples and then subsample to remove correlation effects between the remaining
samples. The :ref:`timeseries to carry out this analysis from replica exchange <repex_timeseries>` is chosen as the sum
of potential energies from each sample, evaluated in state it was drawn from. The timeseries is run through the
``detectEquilibration`` routine of the
`timeseries module of PyMBAR <https://pymbar.readthedocs.io/en/master/timeseries.html>`_ to determine the correlation
times. The :ref:`decorrelation rate is analyzed at each point in the timeseries <autocorrelate_algorithm>` by assuming
all samples before that point are part of the equilibration time, then the number of remaining samples are computed
by subsampling at the decorrelation rate. The point in the timeseries which preserves the maximum number of samples
is treated as the sample at which equilibration is complete, and the remaining subsampled data are passed onto MBAR
for free energy estimate.


Replica Mixing Convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^

Converged free energy estimates require a sufficient mixing of replicas which
`indicated good phase space overlap <http://www.alchemistry.org/wiki/Free_Energy_Fundamentals#Facts_from_the_Free_Energy_Difference_Definition>`_.
Replica exchange simulations enhance sampling by allowing configurations to swap into new states, reducing energetic or steric barriers.
However, if no or few swaps occur, then the replica exchange provides little additional benefit over single simulations.
Furthermore, there is a good chance you have low phase space overlap between states the replicas can sample, which may
yield not only an incorrect free energy estimate, but also underestimate the error in the free energy as well.

.. note::

    Terminology Note: The "Replica" is the time-continuous trajectory of particles and box vectors. A (thermodynamic)
    state is the collection of forces and rules which dictate how particles interact with each other and the surrounding
    bulk environment, including temperature, pressure, and Hamiltonian. In YANK, the Hamiltonian Replica Exchange is
    carried out by proposing each replica swap their states, so each replica has a time-discontinuous state associated
    with it as well.

YANK provides a visual and quantitative guide for how well the replicas mixed. The first guide is the transition
state matrix, which is the measure how frequently states swapped replicas with each other. In
:ref:`automatic <auto_analyze>` or :ref:`programmatic <programmatic_analyze>`, the transition state matrix is shown as a
table indexed by ``i, j`` where each entry shows the number of times state ``i`` (row) swapped with state ``j`` (column).
In :ref:`visual <visual_analyze>` mode, a heat map shows the same data where darker shading indicates a denser exchange
of states. Note that if a state does not exchange replicas, it is counted as "exchanging with itself." You want a
state to exchange with at least one other state.

The subdominant eigenvalue is computed as a quantitative metric of how well mixed the replicas are. This is computed as
the :ref:`second eigenvalue of the transition state matrix <sim_hp_report>`. This quantity is the provides the estimate
for how decomposable the transition state matrix is and how many iterations it would take for one state to swap to all
replicas. The lower this number is, the better mixing has occurred.

The :ref:`visual analyze<visual_analyze>` mode provides an additional qualitative guide for how well individual replicas
sampled each state. The state each replica is in is plotted as a function of time to see if any particular replicas
are getting stuck in a single state.


Free Energy Estimation
^^^^^^^^^^^^^^^^^^^^^^

The free energy difference and its error are estimated through the Multistate Bennet Acceptance Ratio implemented in
PyMBAR. The equilibrated and decorrelated data :ref:`are analyzed by MBAR <mbar_algorithm>` with default values for the
initial guess. If there are unsampled states, such as when
:ref:`anisotropic-dispersion corrections are accounted for <ansiotropic_algorithm>`, then the free energy difference to
those states are also estimated using one-sided exponential reweighting. The final
:doc:`free energy difference of scientific interest <theory>`, such as
binding or hydration, is taken when multiple phases' free energies are added together, along with any
:ref:`standard state corrections <standard_state_algorithm>`.


.. _analyze_usage:

Analysis Usage
--------------

.. _auto_analyze:

Automatic Analysis
^^^^^^^^^^^^^^^^^^

YANK looks at a simulation output directory and makes automatic choices based on best practices and procedures. All
the steps taken here are outlined above and in order. You can use this ``yank analyze`` facility, even when YANK is
running.

.. code-block:: bash

  $ yank analyze --store={experiments}

Replace ``{experiments}`` with the output directory for your simulation. This minimal form of the command has a few
limitations which can be corrected with more complex flags: it only outputs to terminal, it can only target single
directories. Both of these can be corrected with more complex invocation of the command:

.. code-block:: bash

  $ yank analyze --yaml={Some YAML file which ran with ``yank script``}

This form with the ``-y YAML`` or ``--yaml=YAML`` flag tells the analysis to look inside of a YAML file which was read
to run the experiment(s) you want to analyze. In this form, the experiment paths and names are determined from the YAML
file and can analyze any number of :ref:`combinatorial experiments <yaml_combinatorial_head>` found within the file.
The ``-y`` flag is exclusive from ``-s``. This will still output to terminal though, so we need another flag to
save the data to disk.

.. code-block:: bash

  $ yank analyze -y YAML -e {serial Pickle file}

The ``-e SERIAL`` (alternately ``--serial=SERIAL``) flag tells YANK analyze to save the analysis data to the ``SERIAL``
file in Pickle format. This saves not only the final free energy, but also all intermediate values in between. See the
docs for the :ref:`ExperimentAnalyzer class <API_analysis>` for how the output data is formatted and what values
you can extract. The ``-e`` option can be set with either the ``-s`` or the ``-y`` flags.

.. note::

    We know Pickle has problems when importing across Python versions and are working on a solution to carry the
    data in a more universal way. The data are composed of simple numbers, NumPy arrays, and SimTK Quantities for
    unit handling, so we are exploring options to represent them in a transferable way which does not require intamate
    knowledge of what unit system YANK outputs in.


.. _visual_analyze:

Visual Analysis
^^^^^^^^^^^^^^^

YANK can create `Jupyter Notebooks <http://jupyter.org/>`_ to analyze your simulations to help visually see more than
just walls of text and numbers. These notebooks behave similar to the automatic analysis in that they follow the
set of data processing and analysis methods, but will render graphic representation of the same data.

.. note::

    Rendering these notebooks requires both the ``juptyer`` and ``matplotlib`` packages and their dependencies. These
    are not required to run YANK itself, and will not be installed by default if you installed YANK through conda,
    pip, or setup.py. You can still create the notebook as without these packages.

To generate the notebook, use the following command:

.. code-block:: bash

   $ yank analyze report --store={experiments} --output={mynotebook.ipynb}

Replace ``{experiments}`` with the output directory for your simulation and ``{mynotebook.ipynb}`` a filename to save
the notebook as. This only generates the notebook, you will have to open and run the notebook yourself.

If you want to generate static, auto-rendered notebooks, change the extension on your ``--output`` flag to either
``.pdf`` or ``.html``. This will generate, render, and export the notebook to the corresponding file type. Note that
additional packages or external programs may be required to use this feature (e.g. ``.pdf`` requires a ``xelatex``
binary to be on the current system path).


.. _programmatic_analyze:

Programmatic Analysis
^^^^^^^^^^^^^^^^^^^^^

The full :doc:`analyze module's API <api/multistate_api/analyzers_api>` provides extensible, granular access to the analysis suite.
This is helpful if you want to add new analysis methods, manipulate the data yourself, or integrate the analysis tools
into your own code. Simply ``import yank.analyze`` into your code and use the
:doc:`API <api/multistate_api/analyzers_api>` to your own desire. Should you find that you want your changes permanently
added to YANK, feel free to
`open a pull request on GitHub <https://github.com/choderalab/yank/pulls>`_ to start the conversation and consideration!
