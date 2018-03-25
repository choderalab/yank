from simtk import openmm, unit
import numpy as np
from openmmtools.constants import kB

# ==============================================================================
# FIRE minimizer
#
# TODO: Move this to openmmtools once it is sufficiently stable
# ==============================================================================

class FIREMinimizationIntegrator(openmm.CustomIntegrator):
    """Fast Internal Relaxation Engine (FIRE) minimization.

    Notes
    -----
    This integrator is taken verbatim from Peter Eastman's example appearing in the CustomIntegrator header file documentation.

    References
    ----------
    Erik Bitzek, Pekka Koskinen, Franz Gaehler, Michael Moseler, and Peter Gumbsch.
    Structural Relaxation Made Simple. PRL 97:170201, 2006.
    http://dx.doi.org/10.1103/PhysRevLett.97.170201

    Examples
    --------

    >>> from openmmtools import testsystems
    >>> t = testsystems.AlanineDipeptideVacuum()
    >>> system, positions = t.system, t.positions

    Create a FIRE integrator with default parameters and minimize for 100 steps

    >>> integrator = FIREMinimizationIntegrator()
    >>> context = openmm.Context(system, integrator)
    >>> context.setPositions(positions)
    >>> integrator.step(100)

    """

    def __init__(self, timestep=1.0 * unit.femtoseconds, tolerance=None, alpha=0.1, dt_max=10.0 * unit.femtoseconds, f_inc=1.1, f_dec=0.5, f_alpha=0.99, N_min=5):
        """Construct a Fast Internal Relaxation Engine (FIRE) minimization integrator.
        Parameters
        ----------
        timestep : unit.Quantity compatible with femtoseconds, optional, default = 1*femtoseconds
            The integration timestep.
        tolerance : unit.Quantity compatible with kilojoules_per_mole/nanometer, optional, default = None
            Minimization will be terminated when RMS force reaches this tolerance.
        alpha : float, optional default = 0.1
            Velocity relaxation parameter, alpha \in (0,1).
        dt_max : unit.Quantity compatible with femtoseconds, optional, default = 10*femtoseconds
            Maximum allowed timestep.
        f_inc : float, optional, default = 1.1
            Timestep increment multiplicative factor.
        f_dec : float, optional, default = 0.5
            Timestep decrement multiplicative factor.
        f_alpha : float, optional, default = 0.99
            alpha multiplicative relaxation parameter
        N_min : int, optional, default = 5
            Limit on number of timesteps P is negative before decrementing timestep.
        Notes
        -----
        Velocities should be set to zero before using this integrator.
        """

        # Check input ranges.
        if not ((alpha > 0.0) and (alpha < 1.0)):
            raise Exception("alpha must be in the interval (0,1); specified alpha = %f" % alpha)

        if tolerance is None:
            tolerance = 0 * unit.kilojoules_per_mole / unit.nanometers

        super(FIREMinimizationIntegrator, self).__init__(timestep)

        # Use high-precision constraints
        self.setConstraintTolerance(1.0e-8)

        self.addGlobalVariable("alpha", alpha)  # alpha
        self.addGlobalVariable("P", 0)  # P
        self.addGlobalVariable("N_neg", 0.0)
        self.addGlobalVariable("fmag", 0)  # |f|
        self.addGlobalVariable("fmax", 0)  # max|f_i|
        self.addGlobalVariable("ndof", 0)  # number of degrees of freedom
        self.addGlobalVariable("ftol", tolerance.value_in_unit_system(unit.md_unit_system))  # convergence tolerance
        self.addGlobalVariable("vmag", 0)  # |v|
        self.addGlobalVariable("converged", 0) # 1 if convergence threshold reached, 0 otherwise
        self.addPerDofVariable("x0", 0)
        self.addPerDofVariable("v0", 0)
        self.addPerDofVariable("x1", 0)
        self.addGlobalVariable("E0", 0) # old energy associated with x0
        self.addGlobalVariable("dE", 0)
        self.addGlobalVariable("restart", 0)
        self.addGlobalVariable("delta_t", timestep.value_in_unit_system(unit.md_unit_system))

        # Assess convergence
        # TODO: Can we more closely match the OpenMM criterion here?
        self.beginIfBlock('converged < 1')

        # Compute fmag = |f|
        #self.addComputeGlobal('fmag', '0.0')
        self.addComputeSum('fmag', 'f*f')
        self.addComputeGlobal('fmag', 'sqrt(fmag)')

        # Compute ndof
        self.addComputeSum('ndof', '1')

        self.addComputeSum('converged', 'step(ftol - fmag/ndof)')
        self.endBlock()

        # Enclose everything in a block that checks if we have already converged.
        self.beginIfBlock('converged < 1')

        # Store old positions and energy
        self.addComputePerDof('x0', 'x')
        self.addComputePerDof('v0', 'v')
        self.addComputeGlobal('E0', 'energy')

        # MD: Take a velocity Verlet step.
        self.addComputePerDof("v", "v+0.5*delta_t*f/m")
        self.addComputePerDof("x", "x+delta_t*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*delta_t*f/m+(x-x1)/delta_t")
        self.addConstrainVelocities()

        self.addComputeGlobal('dE', 'energy - E0')

        # Compute fmag = |f|
        #self.addComputeGlobal('fmag', '0.0')
        self.addComputeSum('fmag', 'f*f')
        self.addComputeGlobal('fmag', 'sqrt(fmag)')
        # Compute vmag = |v|
        #self.addComputeGlobal('vmag', '0.0')
        self.addComputeSum('vmag', 'v*v')
        self.addComputeGlobal('vmag', 'sqrt(vmag)')

        # F1: Compute P = F.v
        self.addComputeSum('P', 'f*v')

        # F2: set v = (1-alpha) v + alpha \hat{F}.|v|
        # Update velocities.
        # TODO: This must be corrected to be atomwise redirection of v magnitude along f
        self.addComputePerDof('v', '(1-alpha)*v + alpha*(f/fmag)*vmag')

        # Back up if the energy went up, protecing against NaNs
        self.addComputeGlobal('restart', '1')
        self.beginIfBlock('dE < 0')
        self.addComputeGlobal('restart', '0')
        self.endBlock()
        self.beginIfBlock('restart > 0')
        self.addComputePerDof('v', 'v0')
        self.addComputePerDof('x', 'x0')
        self.addComputeGlobal('P', '-1')
        self.endBlock()

        # If dt goes to zero, signal we've converged!
        dt_min = 1.0e-5 * timestep
        self.beginIfBlock('delta_t <= %f' % dt_min.value_in_unit_system(unit.md_unit_system))
        self.addComputeGlobal('converged', '1')
        self.endBlock()

        # F3: If P > 0 and the number of steps since P was negative > N_min,
        # Increase timestep dt = min(dt*f_inc, dt_max) and decrease alpha = alpha*f_alpha
        self.beginIfBlock('P > 0')
        # Update count of number of steps since P was negative.
        self.addComputeGlobal('N_neg', 'N_neg + 1')
        # If we have enough steps since P was negative, scale up timestep.
        self.beginIfBlock('N_neg > %d' % N_min)
        self.addComputeGlobal('delta_t', 'min(delta_t*%f, %f)' % (f_inc, dt_max.value_in_unit_system(unit.md_unit_system))) # TODO: Automatically convert dt_max to md units
        self.addComputeGlobal('alpha', 'alpha * %f' % f_alpha)
        self.endBlock()
        self.endBlock()

        # F4: If P < 0, decrease the timestep dt = dt*f_dec, freeze the system v=0,
        # and set alpha = alpha_start
        self.beginIfBlock('P < 0')
        self.addComputeGlobal('N_neg', '0.0')
        self.addComputeGlobal('delta_t', 'delta_t*%f' % f_dec)
        self.addComputePerDof('v', '0.0')
        self.addComputeGlobal('alpha', '%f' % alpha)
        self.endBlock()

        # Close block that checks for convergence.
        self.endBlock()
