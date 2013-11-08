import numpy as np

import logging
logger = logging.getLogger(__name__)

def show_mixing_statistics(ncfile, cutoff=0.05, nequil=0):
    """
    Print summary of mixing statistics.

    ARGUMENTS

    ncfile (netCDF4.Dataset) - NetCDF file
    
    OPTIONAL ARGUMENTS

    cutoff (float) - only transition probabilities above 'cutoff' will be printed (default: 0.05)
    nequil (int) - if specified, only samples nequil:end will be used in analysis (default: 0)
    
    """
    
    # Get dimensions.
    niterations = ncfile.variables['states'].shape[0]
    nstates = ncfile.variables['states'].shape[1]

    # Compute statistics of transitions.
    Nij = np.zeros([nstates,nstates], np.float64)
    for iteration in range(nequil, niterations-1):
        for ireplica in range(nstates):
            istate = ncfile.variables['states'][iteration,ireplica]
            jstate = ncfile.variables['states'][iteration+1,ireplica]
            Nij[istate,jstate] += 0.5
            Nij[jstate,istate] += 0.5
    Tij = np.zeros([nstates,nstates], np.float64)
    for istate in range(nstates):
        Tij[istate,:] = Nij[istate,:] / Nij[istate,:].sum()

    # Print observed transition probabilities.
    print "Cumulative symmetrized state mixing transition matrix:"
    print "%6s" % "",
    for jstate in range(nstates):
        print "%6d" % jstate,
    print ""
    for istate in range(nstates):
        print "%-6d" % istate,
        for jstate in range(nstates):
            P = Tij[istate,jstate]
            if (P >= cutoff):
                print "%6.3f" % P,
            else:
                print "%6s" % "",
        print ""

    # Estimate second eigenvalue and equilibration time.
    mu = np.linalg.eigvals(Tij)
    mu = -np.sort(-mu) # sort in descending order
    if (mu[1] >= 1):
        logger.info("Perron eigenvalue is unity; Markov chain is decomposable.")
    else:
        logger.info("Perron eigenvalue is %9.5f; state equilibration timescale is ~ %.1f iterations" % (mu[1], 1.0 / (1.0 - mu[1])))
        
    return
