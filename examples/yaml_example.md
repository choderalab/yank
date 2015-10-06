Supported combinatorics
* receptors
* ligands
* mutations
* solvent method
* alchemy
* options (per phase)

Things that can't be done with this format:
* sequences of experiments that are not combinatorial, for example gbsa + 2000 iterations and then pme + 1000 iterations, or abl+imatinib and then p38+p38a_2n.

Random observations
* a general combinatorial syntax like the one we discussed in the meeting (the ```matrix```) is very cool but
  * I'm afraid that either the code will become too complicated in the attempt of supporting all kinds of combinatorial setups, or that it won't be clear what kinds of combinatorics is supported and what it's not.
  * there are some things that needs to go together, for example ```solvate: PME``` needs ```watermodel```, ```alchemy_protocol``` needs electrostatic, retraint, sterics lambda values etc. The only way I could find to solve this problem is to specify solvents in a group with names.

Questions
* How should we handle files with multiple molecules specified? Take all of them? only the first one?
* How can the syntax handle relative/absolute energy calculations? Is there a way we can define the ```phases``` in a general way to do that?

```yaml
#yank-yaml0.1
---
metadata:
  title:
  email:

options:
  yank:
    minimize: yes
    output_directory: /path/to/output/
  repex:
    timestep: 1.0 * femtoseconds
    nsteps_per_iteration: 2500
    niterations: 1000
  alchemy:
    path_discovery: yes

molecules:
  abl-2hyy:
    rcsbid: 2hyy
    select: "chain A"
    mutations: "ASP-137-ASH"
    parameters: ['amber99sbildn.xml', 'ions.xml']
    loopoptions: stuff for loop refinement
  imatinib:
    filename: gleevec.smiles
    epik: 0  # take 0th state from epik for protonation
    parameters: antechamber/paramchem
  bosutinib:
    prmtop: bosu.prmtop      or      charmmpsf: bosu.psf
    inpcrd: bosu.inpcrd      or      charmmpdb: bosu.pdb

  p38-schrodinger:
    filename: p38_protein.pdb
  p38-ligands:
    filename: p32_ligands.sdf
    select: [p38a_2n, p38a_3flz, p38a_2h]  # how do we handle multi-molecules files?

solvents:
  PMEtip3p:
    solvate: PME
    watermodel: tip3p
    salt:
      NaCl: 150*millimolar
      MgCl2: 10*millimolar
    clearance: 10*angstroms
  RFtip3p:
    solvate: RF
    watermodel: tip3p
    salt:
      NaCl: 150*millimolar
      MgCl2: 10*millimolar
    clearance: 10*angstroms
  PMEtip4p:
    solvate: PME
    watermodel: tip4p
    salt:
      KCl: 150*millimolar
    clearance: 10*angstroms
  GBSAobc1:
    solvate: gbsa
    gbsamodel: obc1
  GBSAobc2:
    solvate: gbsa
    gbsamodel: obc2

alchemy:
  standard_protocol:
    lambda_electrostatics: [1.0, 0.9, 0.8, ...]
    lambda_sterics: [1.0, 0.9, 0.8, ...]
  20states_protocol:
    lambda_electrostatics: [1.0, 0.95, 0.9, ...]
    lambda_sterics: [1.0, 0.95, 0.9, ...]

# everything that is in lists is combinatorial
experiments:
  receptors: [abl-2hyy, ...]
  ligands: [imatinib, afatinib, bosutinib]
  mutations:  # both receptors and eventual peptide ligands
    abl-2hyy: [VAL31ALA, ...]  # these mutations are on top of molecules.abl-2hyy.mutations
  solvents: [PMEtip3p, RFtip3p, PMEtip4p, GBSAobc1, GBSAobc2]
  hmr: [yes, no]
  phases:
    complex:
      alchemy: [standard_protocol, 20states_protocol]
      phase_options:
        repex.niterations: [1000, 2000]
    solvent:
      alchemy: standard_protocol
      phase_options:
        repex.niterations: 100
```
