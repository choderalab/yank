#!/usr/local/bin/env python

#=============================================================================================
# Analyze datafiles produced by YANK.
#=============================================================================================

#=============================================================================================
# REQUIREMENTS
#
# The netcdf4-python module is now used to provide netCDF v4 support:
# http://code.google.com/p/netcdf4-python/
#
# This requires NetCDF with version 4 and multithreading support, as well as HDF5.
#=============================================================================================

import os
import os.path
import sys
import math

import numpy as np

import netCDF4 as netcdf # netcdf4-python

from pymbar import MBAR # multistate Bennett acceptance ratio
from pymbar import timeseries # for statistical inefficiency analysis

import simtk.unit as units

import logging
logger = logging.getLogger(__name__)

#=============================================================================================
# PARAMETERS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA

#=============================================================================================
# SUBROUTINES
#=============================================================================================


def show_mixing_statistics(ncfile, cutoff=0.05, nequil=0):
    """
    Print summary of mixing statistics.

    Parameters
    ----------

    ncfile : netCDF4.Dataset
       NetCDF file
    cutoff : float, optional, default=0.05
       Only transition probabilities above 'cutoff' will be printed    
    nequil : int, optional, default=0
       If specified, only samples nequil:end will be used in analysis

    """

    # Get dimensions.
    niterations = ncfile.variables['states'].shape[0]
    nstates = ncfile.variables['states'].shape[1]

    # Compute empirical transition count matrix.
    Nij = np.zeros([nstates,nstates], np.float64)
    for iteration in range(nequil, niterations-1):
        for ireplica in range(nstates):
            istate = ncfile.variables['states'][iteration,ireplica]
            jstate = ncfile.variables['states'][iteration+1,ireplica]
            Nij[istate,jstate] += 1

    # Compute transition matrix estimate.
    # TODO: Replace with maximum likelihood reversible count estimator from msmbuilder or pyemma.
    Tij = np.zeros([nstates,nstates], np.float64)
    for istate in range(nstates):
        denom = (Nij[istate,:].sum() + Nij[:,istate].sum())
        if (denom > 0):
            for jstate in range(nstates):
                Tij[istate,jstate] = (Nij[istate,jstate] + Nij[jstate,istate]) / denom
        else:
            Tij[istate,istate] = 1.0

    # Print observed transition probabilities.
    logger.info("Cumulative symmetrized state mixing transition matrix:")
    str_row = "%6s" % ""
    for jstate in range(nstates):
        str_row += "%6d" % jstate
    logger.info(str_row)

    for istate in range(nstates):
        str_row = ""
        str_row += "%-6d" % istate
        for jstate in range(nstates):
            P = Tij[istate,jstate]
            if (P >= cutoff):
                str_row += "%6.3f" % P
            else:
                str_row += "%6s" % ""
        logger.info(str_row)

    # Estimate second eigenvalue and equilibration time.
    mu = np.linalg.eigvals(Tij)
    mu = -np.sort(-mu) # sort in descending order
    if (mu[1] >= 1):
        logger.info("Perron eigenvalue is unity; Markov chain is decomposable.")
    else:
        logger.info("Perron eigenvalue is %9.5f; state equilibration timescale is ~ %.1f iterations" % (mu[1], 1.0 / (1.0 - mu[1])))

    return


def estimate_free_energies(ncfile, ndiscard=0, nuse=None, g=None):
    """
    Estimate free energies of all alchemical states.

    Parameters
    ----------
    ncfile : NetCDF
       Input YANK netcdf file
    ndiscard : int, optional, default=0
       Number of iterations to discard to equilibration
    nuse : int, optional, default=None
       Maximum number of iterations to use (after discarding)
    g : int, optional, default=None
       Statistical inefficiency to use if desired; if None, will be computed.

    TODO
    ----
    * Automatically determine 'ndiscard'.

    """

    # Get current dimensions.
    niterations = ncfile.variables['energies'].shape[0]
    nstates = ncfile.variables['energies'].shape[1]
    natoms = ncfile.variables['energies'].shape[2]

    # Extract energies.
    logger.info("Reading energies...")
    energies = ncfile.variables['energies']
    u_kln_replica = np.zeros([nstates, nstates, niterations], np.float64)
    for n in range(niterations):
        u_kln_replica[:,:,n] = energies[n,:,:]
    logger.info("Done.")

    # Deconvolute replicas
    logger.info("Deconvoluting replicas...")
    u_kln = np.zeros([nstates, nstates, niterations], np.float64)
    for iteration in range(niterations):
        state_indices = ncfile.variables['states'][iteration,:]
        u_kln[state_indices,:,iteration] = energies[iteration,:,:]
    logger.info("Done.")

    # Compute total negative log probability over all iterations.
    u_n = np.zeros([niterations], np.float64)
    for iteration in range(niterations):
        u_n[iteration] = np.sum(np.diagonal(u_kln[:,:,iteration]))
    #logger.info(u_n

    # DEBUG
    outfile = open('u_n.out', 'w')
    for iteration in range(niterations):
        outfile.write("%8d %24.3f\n" % (iteration, u_n[iteration]))
    outfile.close()

    # Discard initial data to equilibration.
    u_kln_replica = u_kln_replica[:,:,ndiscard:]
    u_kln = u_kln[:,:,ndiscard:]
    u_n = u_n[ndiscard:]

    # Truncate to number of specified conforamtions to use
    if (nuse):
        u_kln_replica = u_kln_replica[:,:,0:nuse]
        u_kln = u_kln[:,:,0:nuse]
        u_n = u_n[0:nuse]

    # Subsample data to obtain uncorrelated samples
    N_k = np.zeros(nstates, np.int32)
    indices = timeseries.subsampleCorrelatedData(u_n, g=g) # indices of uncorrelated samples
    #print u_n # DEBUG
    #indices = range(0,u_n.size) # DEBUG - assume samples are uncorrelated
    N = len(indices) # number of uncorrelated samples
    N_k[:] = N
    u_kln[:,:,0:N] = u_kln[:,:,indices]
    logger.info("number of uncorrelated samples:")
    logger.info(N_k)
    logger.info("")

    #===================================================================================================
    # Estimate free energy difference with MBAR.
    #===================================================================================================

    # Initialize MBAR (computing free energy estimates, which may take a while)
    logger.info("Computing free energy differences...")
    mbar = MBAR(u_kln, N_k)

    # Get matrix of dimensionless free energy differences and uncertainty estimate.
    logger.info("Computing covariance matrix...")

    try:
        # pymbar 2
        (Deltaf_ij, dDeltaf_ij) = mbar.getFreeEnergyDifferences()
    except ValueError:
        # pymbar 3
        (Deltaf_ij, dDeltaf_ij, theta_ij) = mbar.getFreeEnergyDifferences()

#    # Matrix of free energy differences
    logger.info("Deltaf_ij:")
    for i in range(nstates):
        str_row = ""
        for j in range(nstates):
            str_row += "%8.3f" % Deltaf_ij[i, j]
        logger.info(str_row)

#    print Deltaf_ij
#    # Matrix of uncertainties in free energy difference (expectations standard deviations of the estimator about the true free energy)
    logger.info("dDeltaf_ij:")
    for i in range(nstates):
        str_row = ""
        for j in range(nstates):
            str_row += "%8.3f" % dDeltaf_ij[i, j]
        logger.info(str_row)

    # Return free energy differences and an estimate of the covariance.
    return (Deltaf_ij, dDeltaf_ij)

def estimate_enthalpies(ncfile, ndiscard=0, nuse=None, g=None):
    """
    Estimate enthalpies of all alchemical states.

    Parameters
    ----------
    ncfile : NetCDF
       Input YANK netcdf file
    ndiscard : int, optional, default=0
       Number of iterations to discard to equilibration
    nuse : int, optional, default=None
       Number of iterations to use (after discarding)
    g : int, optional, default=None
       Statistical inefficiency to use if desired; if None, will be computed.

    TODO
    ----
    * Automatically determine 'ndiscard'.
    * Combine some functions with estimate_free_energies.

    """

    # Get current dimensions.
    niterations = ncfile.variables['energies'].shape[0]
    nstates = ncfile.variables['energies'].shape[1]
    natoms = ncfile.variables['energies'].shape[2]

    # Extract energies.
    logger.info("Reading energies...")
    energies = ncfile.variables['energies']
    u_kln_replica = np.zeros([nstates, nstates, niterations], np.float64)
    for n in range(niterations):
        u_kln_replica[:,:,n] = energies[n,:,:]
    logger.info("Done.")

    # Deconvolute replicas
    logger.info("Deconvoluting replicas...")
    u_kln = np.zeros([nstates, nstates, niterations], np.float64)
    for iteration in range(niterations):
        state_indices = ncfile.variables['states'][iteration,:]
        u_kln[state_indices,:,iteration] = energies[iteration,:,:]
    logger.info("Done.")

    # Compute total negative log probability over all iterations.
    u_n = np.zeros([niterations], np.float64)
    for iteration in range(niterations):
        u_n[iteration] = np.sum(np.diagonal(u_kln[:,:,iteration]))
    #print u_n

    # DEBUG
    outfile = open('u_n.out', 'w')
    for iteration in range(niterations):
        outfile.write("%8d %24.3f\n" % (iteration, u_n[iteration]))
    outfile.close()

    # Discard initial data to equilibration.
    u_kln_replica = u_kln_replica[:,:,ndiscard:]
    u_kln = u_kln[:,:,ndiscard:]
    u_n = u_n[ndiscard:]

    # Truncate to number of specified conformations to use
    if (nuse):
        u_kln_replica = u_kln_replica[:,:,0:nuse]
        u_kln = u_kln[:,:,0:nuse]
        u_n = u_n[0:nuse]

    # Subsample data to obtain uncorrelated samples
    N_k = np.zeros(nstates, np.int32)
    indices = timeseries.subsampleCorrelatedData(u_n, g=g) # indices of uncorrelated samples
    #print u_n # DEBUG
    #indices = range(0,u_n.size) # DEBUG - assume samples are uncorrelated
    N = len(indices) # number of uncorrelated samples
    N_k[:] = N
    u_kln[:,:,0:N] = u_kln[:,:,indices]
    logger.info("number of uncorrelated samples:")
    logger.info(N_k)
    logger.info("")

    # Compute average enthalpies.
    H_k = np.zeros([nstates], np.float64) # H_i[i] is estimated enthalpy of state i
    dH_k = np.zeros([nstates], np.float64)
    for k in range(nstates):
        H_k[k] = u_kln[k,k,:].mean()
        dH_k[k] = u_kln[k,k,:].std() / np.sqrt(N)

    return (H_k, dH_k)

def extract_u_n(ncfile):
    """
    Extract timeseries of u_n = - log q(X_n) from store file

    where q(X_n) = \pi_{k=1}^K u_{s_{nk}}(x_{nk})

    with X_n = [x_{n1}, ..., x_{nK}] is the current collection of replica configurations
    s_{nk} is the current state of replica k at iteration n
    u_k(x) is the kth reduced potential

    Parameters
    ----------
    ncfile : str
       The filename of the repex NetCDF file.

    Returns
    -------
    u_n : numpy array of numpy.float64
       u_n[n] is -log q(X_n)

    TODO
    ----
    Move this to repex.

    """

    # Get current dimensions.
    niterations = ncfile.variables['energies'].shape[0]
    nstates = ncfile.variables['energies'].shape[1]
    natoms = ncfile.variables['energies'].shape[2]

    # Extract energies.
    logger.info("Reading energies...")
    energies = ncfile.variables['energies']
    u_kln_replica = np.zeros([nstates, nstates, niterations], np.float64)
    for n in range(niterations):
        u_kln_replica[:,:,n] = energies[n,:,:]
    logger.info("Done.")

    # Deconvolute replicas
    logger.info("Deconvoluting replicas...")
    u_kln = np.zeros([nstates, nstates, niterations], np.float64)
    for iteration in range(niterations):
        state_indices = ncfile.variables['states'][iteration,:]
        u_kln[state_indices,:,iteration] = energies[iteration,:,:]
    logger.info("Done.")

    # Compute total negative log probability over all iterations.
    u_n = np.zeros([niterations], np.float64)
    for iteration in range(niterations):
        u_n[iteration] = np.sum(np.diagonal(u_kln[:,:,iteration]))

    return u_n

#=============================================================================================
# SHOW STATUS OF STORE FILES
#=============================================================================================

def print_status(store_directory):
    """
    Print a quick summary of simulation progress.

    Parameters
    ----------
    store_directory : string
       The location of the NetCDF simulation output files.

    Returns
    -------
    success : bool
       True is returned on success; False if some files could not be read.

    """

    # Process each netcdf file.
    phases = ['solvent', 'complex']
    for phase in phases:
        # Construct full path to NetCDF file.
        fullpath = os.path.join(store_directory, phase + '.nc')

        # Check that the file exists.
        if (not os.path.exists(fullpath)):
            # Report failure.
            logger.info("File %s not found." % fullpath)
            logger.info("Check to make sure the right directory was specified, and 'yank setup' has been run.")
            return False

        # Open NetCDF file for reading.
        logger.debug("Opening NetCDF trajectory file '%(fullpath)s' for reading..." % vars())
        ncfile = netcdf.Dataset(fullpath, 'r')

        # Read dimensions.
        niterations = ncfile.variables['positions'].shape[0]
        nstates = ncfile.variables['positions'].shape[1]
        natoms = ncfile.variables['positions'].shape[2]

        # Print summary.
        logger.info("%s" % phase)
        logger.info("  %8d iterations completed" % niterations)
        logger.info("  %8d alchemical states" % nstates)
        logger.info("  %8d atoms" % natoms)

        # TODO: Print average ns/day and estimated completion time.

        # Close file.
        ncfile.close()

    return True

#=============================================================================================
# ANALYZE STORE FILES
#=============================================================================================

def analyze(source_directory):
    """
    Analyze contents of store files to compute free energy differences.

    Parameters
    ----------
    source_directory : string
       The location of the NetCDF simulation storage files.

    """

    # Storage for different phases.
    data = dict()

    phase_prefixes = ['solvent', 'complex']
    suffixes = ['explicit', 'implicit']

    DeltaF_restraints = None

    # Process each netcdf file.
    netcdf_files_found = 0
    for phase in phase_prefixes:
        # Read reference PDB file.
        #from simtk.openmm import app
        #reference_pdb_filename = os.path.join(source_directory, phase + '.pdb')
        #reference_pdb = app.PDBFile(reference_pdb_filename)
            #if phase in ['vacuum', 'solvent']:
            #    reference_pdb_filename = os.path.join(source_directory, "ligand.pdb")
            #else:
            #    reference_pdb_filename = os.path.join(source_directory, "complex.pdb")
            #atoms = read_pdb(reference_pdb_filename)

        for suffix in suffixes:
            # Construct full path to NetCDF file.
            fullpath = os.path.join(source_directory, '%s-%s.nc' % (phase, suffix))
            logger.debug("Attempting to open %s..." % fullpath)

            # Skip if the file doesn't exist.
            if (not os.path.exists(fullpath)): continue

            # Open NetCDF file for reading.
            logger.info("Opening NetCDF trajectory file '%(fullpath)s' for reading..." % vars())
            try:
                ncfile = netcdf.Dataset(fullpath, 'r')
            except Exception as e:
                logger.error(e.message)
                raise Exception("Error opening NetCDF trajectory file '%(fullpath)s' for reading..." % vars())

            # DEBUG
            logger.info("dimensions:")
            for dimension_name in ncfile.dimensions.keys():
                logger.info("%16s %8d" % (dimension_name, len(ncfile.dimensions[dimension_name])))

            # Read dimensions.
            niterations = ncfile.variables['positions'].shape[0]
            nstates = ncfile.variables['positions'].shape[1]
            natoms = ncfile.variables['positions'].shape[2]
            logger.info("Read %(niterations)d iterations, %(nstates)d states" % vars())

            # Increment number of netcdf files found.
            netcdf_files_found += 1

            # Read standard state correction free energy.
            if phase == 'complex':
                DeltaF_restraints = ncfile.groups['metadata'].variables['standard_state_correction'][0]

            # Read reference PDB file.
            #if phase in ['vacuum', 'solvent']:
            #    reference_pdb_filename = os.path.join(source_directory, "ligand.pdb")
            #else:
            #    reference_pdb_filename = os.path.join(source_directory, "complex.pdb")
            #atoms = read_pdb(reference_pdb_filename)

            # Check to make sure no self-energies go nan.
            #check_energies(ncfile, atoms)

            # Check to make sure no positions are nan
            #check_positions(ncfile)

            # Choose number of samples to discard to equilibration
            MIN_ITERATIONS = 10 # minimum number of iterations to use automatic detection
            if niterations > MIN_ITERATIONS:
                from pymbar import timeseries
                u_n = extract_u_n(ncfile)
                u_n = u_n[1:] # discard initial frame of zero energies TODO: Get rid of initial frame of zero energies
                [nequil, g_t, Neff_max] = timeseries.detectEquilibration(u_n)
                nequil += 1 # account for initial frame of zero energies
                logger.info([nequil, Neff_max])
            else:
                nequil = 1 # discard first frame
                g_t = 1
                Neff_max = niterations

            # Examine acceptance probabilities.
            show_mixing_statistics(ncfile, cutoff=0.05, nequil=nequil)

            # Estimate free energies.
            (Deltaf_ij, dDeltaf_ij) = estimate_free_energies(ncfile, ndiscard = nequil, g=g_t)

            # Estimate average enthalpies
            (DeltaH_i, dDeltaH_i) = estimate_enthalpies(ncfile, ndiscard = nequil, g=g_t)

            # Accumulate free energy differences
            entry = dict()
            entry['DeltaF'] = Deltaf_ij[0,nstates-1]
            entry['dDeltaF'] = dDeltaf_ij[0,nstates-1]
            entry['DeltaH'] = DeltaH_i[nstates-1] - DeltaH_i[0]
            entry['dDeltaH'] = np.sqrt(dDeltaH_i[0]**2 + dDeltaH_i[nstates-1]**2)
            data[phase] = entry

            # Get temperatures.
            ncvar = ncfile.groups['thermodynamic_states'].variables['temperatures']
            temperature = ncvar[0] * units.kelvin
            kT = kB * temperature

            # Close input NetCDF file.
            ncfile.close()

    # Give the user a useful warning if no NetCDF files found.
    if netcdf_files_found == 0:
        raise Exception("No YANK output files were found in the specified store directory (%s)" % source_directory)

    # Compute hydration free energy (free energy of transfer from vacuum to water)
    #DeltaF = data['vacuum']['DeltaF'] - data['solvent']['DeltaF']
    #dDeltaF = numpy.sqrt(data['vacuum']['dDeltaF']**2 + data['solvent']['dDeltaF']**2)
    #print "Hydration free energy: %.3f +- %.3f kT (%.3f +- %.3f kcal/mol)" % (DeltaF, dDeltaF, DeltaF * kT / units.kilocalories_per_mole, dDeltaF * kT / units.kilocalories_per_mole)

    # Compute enthalpy of transfer from vacuum to water
    #DeltaH = data['vacuum']['DeltaH'] - data['solvent']['DeltaH']
    #dDeltaH = numpy.sqrt(data['vacuum']['dDeltaH']**2 + data['solvent']['dDeltaH']**2)
    #print "Enthalpy of hydration: %.3f +- %.3f kT (%.3f +- %.3f kcal/mol)" % (DeltaH, dDeltaH, DeltaH * kT / units.kilocalories_per_mole, dDeltaH * kT / units.kilocalories_per_mole)

    if DeltaF_restraints is None:
        raise Exception("DeltaF_restraints not found.")

    # Compute binding free energy.
    DeltaF = data['solvent']['DeltaF'] - DeltaF_restraints - data['complex']['DeltaF']
    dDeltaF = np.sqrt(data['solvent']['dDeltaF']**2 + data['complex']['dDeltaF']**2)
    logger.info("")
    logger.info("Binding free energy : %16.3f +- %.3f kT (%16.3f +- %.3f kcal/mol)" % (DeltaF, dDeltaF, DeltaF * kT / units.kilocalories_per_mole, dDeltaF * kT / units.kilocalories_per_mole))
    logger.info("")
    #logger.info("DeltaG vacuum       : %16.3f +- %.3f kT" % (data['vacuum']['DeltaF'], data['vacuum']['dDeltaF']))
    logger.info("DeltaG solvent      : %16.3f +- %.3f kT" % (data['solvent']['DeltaF'], data['solvent']['dDeltaF']))
    logger.info("DeltaG complex      : %16.3f +- %.3f kT" % (data['complex']['DeltaF'], data['complex']['dDeltaF']))
    logger.info("DeltaG restraint    : %16.3f          kT" % DeltaF_restraints)
    logger.info("")

    # Compute binding enthalpy
    DeltaH = data['solvent']['DeltaH'] - DeltaF_restraints - data['complex']['DeltaH']
    dDeltaH = np.sqrt(data['solvent']['dDeltaH']**2 + data['complex']['dDeltaH']**2)
    logger.info("Binding enthalpy    : %16.3f +- %.3f kT (%16.3f +- %.3f kcal/mol)" % (DeltaH, dDeltaH, DeltaH * kT / units.kilocalories_per_mole, dDeltaH * kT / units.kilocalories_per_mole))

