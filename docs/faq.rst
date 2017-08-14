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

#. I found a bug or want a feature. Where do I tell you?

   Head to `YANK's GitHub page <https://github.com/choderalab/yank>`_ and post an issue.

#. I want to contribute to YANK. What is the procedure for doing so?

   #. :ref:`Install YANK from source <install_from_source>` or make a fork.
   #. Make your changes on a new branch
   #. Open a `Pull Request on GitHub <https://github.com/choderalab/yank/pulls>`_
   #. We'll discuss in on GitHub now as though it were an issue, request changes, and get your changes in!

#. What is the airspeed velocity of an unladen swallow?

   `African or European? <http://style.org/unladenswallow/>`_

