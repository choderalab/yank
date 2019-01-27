# Cyclodextrin

Host-guest set composed by alpha and beta cyclodextrin and 43 ligands (22 for alpha-CD and 21 for beta-CD). The
ITC experimental measurements can be found in [1]. Attach-pull-release (APR) calculations for the whole set are
described in [2]. The input files here were originally created for the latter work, and they were uploaded to
`[benchmarksets](https://github.com/MobleyLab/benchmarksets/)`.


## Manifest

- `input/`: the mol2 input files for the YANK pipeline.
- `cyclodextrin.yaml`: YANK script for setup and running the calculations.
- `gaff17.dat`: leap parameters for GAFF 1.7.
- `run-lsf.sh`: LSF script to run YANK with MPI.


## References

- [1] Rekharsky, M. V., Mayhew, M. P., Goldberg, R. N., Ross, P. D., Yamashoji, Y., & Inoue, Y. (1997). Thermodynamic
and nuclear magnetic resonance study of the reactions of α-and β-cyclodextrin with acids, aliphatic amines, and cyclic
alcohols. The Journal of Physical Chemistry B, 101(1), 87-100.
- [2] Henriksen, N. M., & Gilson, M. K. (2017). Evaluating force field performance in thermodynamic calculations of
cyclodextrin host-guest binding: water models, partial charges, and host force field parameters. Journal of Chemical
Theory and Computation.
