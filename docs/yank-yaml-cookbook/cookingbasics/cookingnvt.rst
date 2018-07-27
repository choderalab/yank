.. _yaml_cookbook_basics_nvt:

Basic NVT Simulation Recipe
***************************

This recipe shows a stock NVT setup in the :ref:`yaml-options-head`. We will run the simulations at 300 Kelvin.

NVT simulations by their nature are NOT periodic, so the the molecules can drift in an infinite medium. The coordinates
for this systems are often re-centered to prevent the numerical value of their positions from drifting into infinite.
However, this process takes the center of mass of the whole system and shifts it. If you have multiple molecules (e.g.
waters) which can drift apart, you can quickly find your system drifting into vapor phase with molecules so far
away that no useful statistics can be gathered. Because of this, only implicit solvent should be simulated under
these conditions.

For this recipe, we choose the Generalized Born (GB) model augmented with the hydrophobic solvent accessible surface
area (SA) model; GBSA for short. This model has several implementations and we have selected the
Onufriev-Bashford-Case variant 2 model (``OBC2``).

.. literalinclude:: raw_cooking_basics/rawnvt.yaml
   :language: yaml