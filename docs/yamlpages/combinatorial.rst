.. _yaml_combinatorial_head:

Combinatorial Options in YAML
-----------------------------
YANK's YAML inputs also support running experiments combinatorially, instead of individually running them one at a time. 
YANK will automatically set up each combination of options you specify with the special 
``!Combinatorial [itemA, itemB, ...]`` structure and run them back to back for you.

**NOTE:** Every combinatorial option is itself a complete experiment, so total wall clock time will increase... combinatorially! Take this into account when planning your experiments.

Every option which can be specified in the YAML file can be given the `!Combinatorial` flag. 
**Exceptions** to this rule are as follows:

* All settings in the ``protocols`` header.
* All settings in the ``solvents`` header.

This means all ``options``, ``molecules``, ``systems``, and ``experiments`` can be specified with ``!Combinatorial`` options.

For options which are themselves lists (such as many of the file path options), a nested list is expected.
e.g. ``!Combinatorial[[complexA.gro, complexA.top], [complexB.gro, complexB.top]]``.
