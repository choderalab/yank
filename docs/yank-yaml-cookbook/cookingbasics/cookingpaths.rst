.. _yaml_cookbook_basics_paths:

Basic Path Manipulation Recipe
******************************

Sometimes you may want to output your setup and simulation results in another place. This can happen on clusters and
shared directories where there may be a common repository of directories where the YAML script lives, but the work
directory or drive is located elsewhere.

This recipe manipulates the output directories for a YANK setup and simulation. Normally, the default options are
chosen and these don't have to be set.

There are two flavors here. The first assumes the YAML script is in a directory which can be written to and has
the space to store the YANK experiments. The second is a very uncommon recipe where a central molecule setup directory
is used. This is extremely uncommon but for similar experiments may be helpful. Because the default settings of these
options exist, in most cases, you will likely never use these recipes.

.. note::

    The ``output_dir`` path is *relative* to the *YAML file location*, but the other paths are relative to
    ``output_dir`` Keep this in mind when setting paths.
    Absolute paths, however, are always respected.


.. literalinclude:: raw_cooking_basics/rawpathscommon.yaml
   :language: yaml

.. literalinclude:: raw_cooking_basics/rawpathsrare.yaml
   :language: yaml