.. _API:

#############
API Reference
#############

YANK's API allows users to setup and run their own simulations in programmatic ways. If you would rather run YANK with
YAML input files, please see the command line interface and the YAML options.

.. toctree::
   :maxdepth: 2

   analysis_api
   experiment_api
   mpi_api
   multistate_api/index
   pipeline_api
   restraints_api
   utils_api
   yank_api


UML class diagram
=================

This UML diagram is a minimal representation of YANK's structure. This is not meant to be a comprehensive and accurate
representation of the API, but only to provide a visual overview of the relationships between the available classes
available, their role, and the way the interact with the OpenMMTools framework.

.. figure:: site-resources/images/YANKUMLClassDiagram.png
