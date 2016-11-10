.. _yaml_version_head:

Version Header for YAML Files
*****************************

The ``version`` header is an **optional** section where we define the language specification version of our YAML files.

The header provides a means to ensure that your YAML file will be supported by the version of YANK you are running on.

The only valid version right now is "1.0" but as YANK develops, more options will become available.

.. code-block:: yaml

    version: 1.0

Valid Options: [1.0]