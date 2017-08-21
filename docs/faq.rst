.. _faq:

Frequently Asked Questions (FAQ)
================================

#. Can you restart a simulation once its finished (or interrupted)?

   Yes! Either specify the :ref:`yaml_options_resume_simulation` in the YAML file, or use the
   :func:`yank.yank.AlchemicalPhase.from_storage` function to resume from existing file.

#. Which MPI program should I use?

   We recommend the ``mpich`` and ``mpi4py`` packages that come from the ``conda-forge`` channel of ``conda``.
   These should be installed automatically if you installed YANK through conda correctly. If you use conda's MPI package,
   do *not* use your local cluster's MPI module if it has one. If you wish to use your own MPI package, make sure the
   ``mpi4py`` you install is compiled against your local MPI, otherwise you may see difficult to diagnose errors.

#. How do I add a new state to the ``protocols`` section of a YAML file?

   You can allow YANK to fully specify the alchemical protocol for you with the ``auto`` argument for the
   ``alchemical path``, instead of adding each state by hand. You can see its
   :ref:`full usage documentation <yaml_protocols_auto>` or its
   :ref:`implementation information <algorithm_auto_protocol>`

   If you want to add states by hand, follow these instructions:

   #. Find the phase you want to change in the YAML file
   #. Add an entry in each list of values for a given parameter at the index of that list where you wish to insert the state

   See the :ref:`protocols documentation <yaml_protocols_alchemical_path>` for more information or
   `watch our video guide to see how to add a state <https://youtu.be/nVVl6if6g0w?t=2m46s>`_

#. I found a bug or want a feature. Where do I tell you?

   Head to `YANK's GitHub page <https://github.com/choderalab/yank>`_ and post an issue.

#. I want to contribute to YANK. What is the procedure for doing so?

   #. :ref:`Install YANK from source <install_from_source>` or make a fork.
   #. Make your changes on a new branch
   #. Open a `Pull Request on GitHub <https://github.com/choderalab/yank/pulls>`_
   #. We'll discuss in on GitHub now as though it were an issue, request changes, and get your changes in!

#. What is the airspeed velocity of an unladen swallow?

   `African or European? <http://style.org/unladenswallow/>`_

