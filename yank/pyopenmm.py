#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Pure Python implementations of some OpenMM objects.

For now, we provide pure Python implementations of System, Force, and some derived classes.
In addition, these objects can be constructed from their equivalent Swig proxies, and an
asSwig() method is provided to construct and return a Swig proxy for each class.

COPYRIGHT

@author John D. Chodera <jchodera@gmail.com>

EXAMPLES

First, some useful imports:

>>> import simtk.openmm as openmm # Swig wrappers for OpenMM objects
>>> import pyopenmm # pure Python OpenMM objects (containers, really)
>>> import simtk.pyopenmm.extras.testsystems as testsystems # some test systems to play with
>>> import simtk.unit as units # unit system

We can easily create pure Python versions of objects created with simtk.openmm by passing
them to the System constructor:

>>> [system, coordinates] = testsystems.LennardJonesFluid(mm=openmm) # create a Swig wrapped object
>>> pyopenmm_system = pyopenmm.System(system) # creates pure Python forms of System and all Force objects

The Python implementation provides a complete implementation of the Swig OpenMM API, which
follows from the C++ API:

>>> (system.getNumParticles(), pyopenmm_system.getNumParticles())
(216, 216)
>>> (system.getForce(0).getParticleParameters(0)[0], pyopenmm_system.getForce(0).getParticleParameters(0)[0])
(Quantity(value=0.0, unit=elementary charge), Quantity(value=0.0, unit=elementary charge))
    
But the pure Python pyopenmm implementation is a 'thick' wrapper.  In addition to the Swig
OpenMM API, it provides a more 'pythonic' way to access information within objects:

>>> pyopenmm_system.nparticles
216
>>> pyopenmm_system.forces[0].particles[0].charge
Quantity(value=0.0, unit=elementary charge)

Each class also provides an asSwig() method for creating a Swig-wrapped object from the pure Python object:

>>> swig_force = pyopenmm_system.forces[0].asSwig()
>>> swig_system = pyopenmm_system.asSwig() # all Force objects are also converted to Swig

We don't need to do this explicitly, however, because we also provide a new Context implementation
which automatically creates Swig-wrapped objects only when necessary:

>>> integrator = pyopenmm.VerletIntegrator(1.0 * units.femtosecond) # all simtk.openmm symbols are imported first; integrator is actually a Swig-wrapped object from simtk.openmm
>>> context = pyopenmm.Context(pyopenmm_system, integrator) # create a context from a pure Python System object, automatically converting to Swig-wrapped object as needed

Pure Python objects can be deep-copied:

>>> import copy
>>> system_copy = copy.deepcopy(pyopenmm_system)

and pickled:

>>> import pickle
>>> import tempfile # for creating a temporary file for the purpose of this example
>>> file = tempfile.NamedTemporaryFile()
>>> pickle.dump(pyopenmm_system, file)

The operator '+' has been overloaded to allow System objects to be concatenated, provided it is
sensible to do so (i.e. they have the same Force objects in the same order, with the same settings):

>>> import simtk.pyopenmm.amber.amber_file_parser as amber # for reading AMBER files
>>> import os.path
>>> path = os.path.join(os.getenv('PYOPENMM_SOURCE_DIR'), 'test', 'additional-tests', 'systems', 'T4-lysozyme-L99A-implicit')
>>> receptor_system = amber.readAmberSystem(os.path.join(path, 'receptor.prmtop'), mm=pyopenmm) # create System from AMBER prmtop
>>> receptor_coordinates = amber.readAmberCoordinates(os.path.join(path, 'receptor.crd')) # read AMBER coordinates
>>> ligand_system = amber.readAmberSystem(os.path.join(path, 'ligand.prmtop'), mm=pyopenmm) # create System from AMBER prmtop
>>> ligand_coordinates = amber.readAmberCoordinates(os.path.join(path, 'ligand.crd')) # read AMBER coordinates
>>> complex_system = receptor_system + ligand_system # concatenate atoms in systems
>>> complex_coordinates = numpy.concatenate([receptor_coordinates, ligand_coordinates])

TODO

* Rework Pythonic extensions to use OpenMM API, so they can be used to extend OpenMM Swig implementation?
* Debug small discrepancies in explicit solvent energies.
* Fix System _copyDataUsingInterface and addForce methods.
* Debug tiny differences in energies when system is copied repeatedly.
* Use decorators @accepts, etc. everywhere.
* Update docstrings.
* Add more doctests for Force classes.
* Add support for CustomBondForce
* Interoperability with OpenMM App topology classes?
* Change *Info structures to encapsulated classes since we don't have to worry about network transport of these objects anymore.
* return values: Many of the OpenMM interface methods return an integer index of the particle, force, or
parameter added, which doesn't make much sense in the Pythonic way of doing things, and causes
emission of the index to the terminal output unless the argument is eaten by assignment. Can
we modify this behavior without breaking anything?
* pickling: Pickling only works if all classes are defined at top-level for the module.
We can move classes like NonbondedForce.ParticleInfo out of NonbondedForce, calling it something
like NonbondedForceParticleInfo, to allow pickling to work.  Alternatively, we can turn these
ParticleInfo classes into tuples.
* Add support for enumerating and manipulating molecules in System based on constraints and bonds.
* Add support for identifying and manipulating waters or other solvents?
* Add 'validate' method to System to make sure all masses, constraints, and Force objects have 
  consistent particle numbers and entries within bounds.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import re
import copy
import numpy 

import simtk.unit as units
import simtk.openmm as openmm 

# Import everything that we don't override directly from pyopenmm.
from simtk.openmm import *

#=============================================================================================
# EXCEPTIONS
#=============================================================================================

class UnitsException(Exception):
    """
    Exception denoting that an argument has the incorrect units.

    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class ValueException(Exception):
    """
    Exception denoting that an argument has the incorrect value.

    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

#=============================================================================================
# DECORATORS
#=============================================================================================

#TODO: Do we need to use 'from functools import wraps' to help us here?

def accepts(*types):
    """
    Decorator for class methods that should accept only specified types.

    EXAMPLE

    @accepts(float, int)
    def function(a, b):
        return b*a
    
    """
    def check_accepts(f):
        nargs = (f.func_code.co_argcount - 1) # exclude self
        assert len(types) == nargs, "incorrect number of args supplied in @accepts decorator for class method %s" % (f.func_name)        
        def new_f(*args, **kwds):
            for (a, t) in zip(args[1:], types):
                if a is not None:
                    assert isinstance(a, t), "arg %r does not match %s" % (a,t)
            return f(*args, **kwds)
        new_f.func_name = f.func_name # copy function name
        new_f.func_doc = f.func_doc # copy docstring        
        return new_f
    return check_accepts

def accepts_compatible_units(*units):
    """
    Decorator for class methods that should accept only arguments compatible with specified units.

    Each argument of the function will be matched with an argument of @acceptunits.
    Those arguments of the function that correspond @acceptunits which are not None
    will be checked to ensure they are compatible with the specified units.
    
    EXAMPLE

    @acceptsunits(units.meter, None, units.kilocalories_per_mole)
    def function(a, b, c): pass
    function(1.0 * units.angstrom, 3, 1.0 * units.kilojoules_per_mole)
    
    """
    def check_units(f):
        nargs = (f.func_code.co_argcount - 1) # exclude self
        assert len(units) == nargs, "incorrect number of units supplied in @accepts_compatible_units decorator for class method %s" % (f.func_name)
        def new_f(*args, **kwds):
            for (a, u) in zip(args[1:], units):
                if u is not None:                    
                    assert (a.unit).is_compatible(u), "arg %r does not have units compatible with %s" % (a,u)
            return f(*args, **kwds)
        new_f.func_name = f.func_name # copy function name
        new_f.func_doc = f.func_doc # copy docstring
        return new_f
    return check_units

def returns(rtype):
    """
    Decorator for functions that should only return specific types.

    EXAMPLE

    @returns(int)
    def function(): return 7    
            
    """
    
    def check_returns(f):
        def new_f(*args, **kwds):
            result = f(*args, **kwds)
            assert isinstance(result, rtype), "return value %r does not match %s" % (result,rtype)
            return result
        new_f.func_name = f.func_name # copy function name
        new_f.func_doc = f.func_doc # copy docstring
        return new_f
    return check_returns

#=============================================================================================
# System
#=============================================================================================

class System(object):
    """
    This class represents a molecular system.  The definition of a System involves
    four elements:
    
    <ol>
    <li>The set of particles in the system</li>
    <li>The forces acting on them</li>
    <li>Pairs of particles whose separation should be constrained to a fixed value</li>
    <li>For periodic systems, the dimensions of the periodic box</li>
    </ol>
    
    The particles and constraints are defined directly by the System object, while
    forces are defined by objects that extend the Force class.  After creating a
    System, call addParticle() once for each particle, addConstraint() for each constraint,
    and addForce() for each Force.

    Create a System object.

    >>> system = System()

    Add a particle.

    >>> mass = 12.0 * units.amu
    >>> system.addParticle(mass)
    0

    Add a NonbondedForce.

    >>> nonbondedForce = NonbondedForce()
    >>> system.addForce(nonbondedForce)
    0

    Create a system from a Swig proxy.

    >>> import simtk.pyopenmm.extras.testsystems as testsystems
    >>> import pyopenmm
    >>> [system, coordinates] = testsystems.LennardJonesFluid(mm=pyopenmm)
    >>> proxy_system = system.asSwig()
    >>> system_from_proxy = System(proxy_system)

    Create a deep copy.
    
    >>> import copy
    >>> system_copy = copy.deepcopy(system_from_proxy)

    Create a Swig proxy.

    >>> proxy_system_from_proxy = system_from_proxy.asSwig()

    Construct from a pyre Python class.

    >>> system_from_system = System(system_from_proxy)
        
    """

    PYOPENMM_API_EXTENSIONS = True # signal that members of this class have pyopenmm API extensions

    def __init__(self, system=None):
        """
        Create a new System.

        If an openmm.System object is specified, it will be queried to construct the class.

        """

        # Set defaults.
        self.masses      = list() # masses[i] is a Quantity denoting the mass of particle i
        self.constraints = list() # constraints[i] is the ith ConstraintInfo entry
        self.forces      = list() # forces[i] is the ith force term
        self.periodicBoxVectors = [ units.Quantity((2.,0.,0.), units.nanometer), units.Quantity((0.,2.,0.), units.nanometer), units.Quantity((0.,0.,2.), units.nanometer) ] # periodic box vectors (only for periodic systems)
        # TODO: Store periodicBoxVectors as units.Quantity(numpy.array([3,3], numpy.float64), units.nanometer)?

        # Populate the system from a provided Swig proxy or Python system, if given.
        if system is not None:
            self._copyDataUsingInterface(self, system)
    
        return

    @classmethod
    def _copyDataUsingInterface(cls, dest, src):
        """
        Use the public interface to populate 'dest' from 'src'.
        
        """
        dest.__init__() # TODO: Do we need this?
        for index in range(src.getNumParticles()):
            mass = src.getParticleMass(index)            
            dest.addParticle(mass)
        for index in range(src.getNumConstraints()):
            args = src.getConstraintParameters(index)
            dest.addConstraint(*args)
        for index in range(src.getNumForces()):
            force = src.getForce(index)
            if not hasattr(dest, 'PYOPENMM_API_EXTENSIONS'):
                # We're copying a Python to a Swig System object.  Make a Swig copy.
                force = force.asSwig()
                dest.addForce(force)
            else:
                import pyopenmm
                force_name = type(force).__name__ # get name
                #force = globals()[force_name](force) # create pure Python force object
                ForceSubclass = getattr(pyopenmm, force_name) # get corresponding pyopenmm function
                force = ForceSubclass(force=force) # construct pure Python version
                dest.addForce(force)

        box_vectors = src.getDefaultPeriodicBoxVectors()
        # TODO: Regularize how box vectors are returned to math OpenMM interface (using Vec3?).
        #print box_vectors
        #box_vectors = [box_vectors[0].in_units_of(units.nanometers), box_vectors[1].in_units_of(units.nanometers), box_vectors[2].in_units_of(units.nanometers)] # DEBUG
        dest.setDefaultPeriodicBoxVectors(*box_vectors)
        
        return
        
    def asSwig(self):
        """
        Create a Swig proxy for this system object.

        """
        # Create an empty swig proxy object.
        system = openmm.System()
        # Copy data using interface.
        self._copyDataUsingInterface(system, self)

        return system

    def getNumParticles(self):
        """
        Get the number of particles in this System.
        
        """
        return len(self.masses)

    @accepts_compatible_units(units.amu)
    def addParticle(self, mass):
        """
        Add a particle to the System.

        @param mass   the mass of the particle (in atomic mass units)
        @return the index of the particle that was added

        """

        self.masses.append(mass);
        return len(self.masses)-1;

    def getParticleMass(self, index):
        """
        Get the mass (in atomic mass units) of a particle.
    
        @param index the index of the particle for which to get the mass

        """        
        return self.masses[index]

    @accepts_compatible_units(None, units.amu)
    def setParticleMass(self, index, mass):
        """
        Set the mass (in atomic mass units) of a particle.

        @param index  the index of the particle for which to set the mass
        @param mass   the mass of the particle

        """
        masses[index] = mass
        return        

    def getNumConstraints(self):
        """
        Get the number of distance constraints in this System.

        """
        return len(self.constraints)

    @accepts_compatible_units(None, None, units.nanometer)
    def addConstraint(self, particle1, particle2, distance):
        """
        Add a constraint to the System.
        
        @param particle1 the index of the first particle involved in the constraint
        @param particle2 the index of the second particle involved in the constraint
        @param distance  the required distance between the two particles, measured in nm
        @return the index of the constraint that was added

        """
        if particle1 not in range(self.getNumParticles()):
            raise ValueError("particle1 must be in range(0, getNumParticles())")
        if particle2 not in range(self.getNumParticles()):
            raise ValueError("particle1 must be in range(0, getNumParticles())")        
        constraint = self.ConstraintInfo(particle1, particle2, distance)
        self.constraints.append(constraint)

        return

    def getConstraintParameters(self, index):
        """
        Get the parameters defining a distance constraint.
        
        @param index     the index of the constraint for which to get parameters
        @return a tuple of (particle1, particle2, distance) for the given constraint index

        """
        constraint = self.constraints[index]
        return (constraint.particle1, constraint.particle2, constraint.distance)

    @accepts_compatible_units(None, None, None, units.nanometer)
    def setConstraintParameters(self, index, particle1, particle2, distance):
        """
        Set the parameters defining a distance constraint.
        
        @param index     the index of the constraint for which to set parameters
        @param particle1 the index of the first particle involved in the constraint
        @param particle2 the index of the second particle involved in the constraint
        @param distance  the required distance between the two particles, measured in nm

        """
        if particle1 not in range(self.getNumParticles()):
            raise ValueError("particle1 must be in range(0, getNumParticles())")
        if particle2 not in range(self.getNumParticles()):
            raise ValueError("particle1 must be in range(0, getNumParticles())")
        constraint = self.ConstraintInfo(particle1,particle2,distance)
        self.constraints[index] = constraint

        return

    def addForce(self, force):
        """
        Add a Force to the System.

        @param force   the Force object to be added
        @return        the index within the System of the Force that was added

        NOTES

        If a Swig object is specified, a pure Python deep copy will be constructed.
        If a Python object is specified, the actual object will be added, not a deep copy.

        """

        # If not pure Python, make a Python copy.
        #if not hasattr(force, 'PYOPENMM_API_EXTENSIONS'):
        #    import pyopenmm
        #    # TODO: Fix this magic to make sure we're looking up the pyopenmm force.
        #    force_name = type(force).__name__ # get name
        #    #force = globals()[force_name](force) # create pure Python force object
        #    ForceSubclass = getattr(pyopenmm, force_name) # get corresponding pyopenmm function
        #    force = ForceSubclass(force) # construct pure Python version

        # Append the force.
        self.forces.append(force)
        return len(self.forces)-1

    def getNumForces(self):
        """
        Get the number of Force objects that have been added to the System.

        """
        return len(self.forces)

    def getForce(self, index):
        """
        Get a const reference to one of the Forces in this System.

        @param index  the index of the Force to get

        """
        return self.forces[index]

    def getDefaultPeriodicBoxVectors(self):
        """
        Get the default values of the vectors defining the axes of the periodic box (measured in nm).  Any newly
        created Context will have its box vectors set to these.  They will affect
        any Force added to the System that uses periodic boundary conditions.

        Currently, only rectangular boxes are supported.  This means that a, b, and c must be aligned with the
        x, y, and z axes respectively.  Future releases may support arbitrary triclinic boxes.
        
        @returns a      the vector defining the first edge of the periodic box
        @returns b      the vector defining the second edge of the periodic box
        @returns c      the vector defining the third edge of the periodic box
     
        """        
        return self.periodicBoxVectors

    def setDefaultPeriodicBoxVectors(self, a, b, c):
        """
        Set the default values of the vectors defining the axes of the periodic box (measured in nm).  Any newly
        created Context will have its box vectors set to these.  They will affect
        any Force added to the System that uses periodic boundary conditions.

        Currently, only rectangular boxes are supported.  This means that a, b, and c must be aligned with the
        x, y, and z axes respectively.  Future releases may support arbitrary triclinic boxes.
        
        @param a      the vector defining the first edge of the periodic box
        @param b      the vector defining the second edge of the periodic box
        @param c      the vector defining the third edge of the periodic box

        """
        # TODO: Argument checking.
        self.periodicBoxVectors = [a,b,c]
    
    #==========================================================================
    # CONTAINERS
    #==========================================================================

    class ConstraintInfo(object):
        """
        Distance constraint information for particles in System object.

        """

        @accepts_compatible_units(None, None, units.nanometers)
        def __init__(self, particle1, particle2, distance):
            self.particle1 = particle1
            self.particle2 = particle2
            self.distance = distance
            return

    #==========================================================================
    # PYTHONIC EXTENSIONS
    #==========================================================================    
    
    @property    
    def nparticles(self):
        """
        The number of particles.

        """
        return len(self.masses)

    @property
    def nforces(self):
        """
        The number of force objects in the system.

        """
        return len(self.forces)
        
    @property
    def nconstraints(self):
        """
        The number of interparticle distance constraints defined.

        """
        return len(self.constraints)

    @property    
    def is_periodic(self):
        """
        True if system is periodic (contains a force with 'nonbondedMethod' set to 'CutoffPeriodic', 'Ewald', or 'PME').  False otherwise.

        """
        for force in self.forces:
            if hasattr(force, 'nonbondedMethod'):
                # Make a list of potential periodic methods.
                periodic_methods = list()
                for method in ['CutoffPeriodic', 'Ewald', 'PME']:
                    if hasattr(force, method): periodic_methods.append(getattr(force, method))
                if force.nonbondedMethod in periodic_methods: return True
            
        return False

    def __str__(self):
        """
        Return an 'informal' human-readable string representation of the System object.

        """
        
        r = ""
        r += "System object containing %d particles\n" % self.nparticles

        # Show particles.
        r += "Particle masses:\n"
        r += "%8s %24s\n" % ("particle", "mass")
        for index in range(self.getNumParticles()):
            mass = self.getParticleMass(index)
            r += "%8d %24s\n" % (index, str(mass))
        r += "\n"        

        # Show constraints.
        r += "Constraints:\n"
        r += "%8s %8s %16s\n" % ("particle1", "particle2", "distance")
        for index in range(self.getNumConstraints()):
            (particle1, particle2, distance) = self.getConstraintParameters(index)
            r += "%8d %8d %s\n" % (particle1, particle2, str(distance))
        r += "\n"
        
        # Show forces.
        r += "Forces:\n"
        for force in self.forces:
            r += str(force)
            r += "\n"
            
        return r

    def __add__(self, other):
        """
        Binary concatenation of two systems.

        The atoms of the second system appear, in order, after the atoms of the first system
        in the new combined system.

        USAGE

        combined_system = system1 + system2

        NOTES

        Both systems must have identical ordering of Force terms.
        Any non-particle settings from the first System override those of the second, if they differ.

        EXAMPLES

        Concatenate two systems in vacuum.

        >>> import simtk.pyopenmm.extras.testsystems as testsystems
        >>> [system1, coordinates1] = testsystems.LennardJonesFluid()
        >>> system1 = System(system1) # convert to pure Python system
        >>> [system2, coordinates2] = testsystems.LennardJonesFluid()
        >>> system2 = System(system2) # convert to pure Python system        
        >>> combined_system = system1 + system2

        Check number of particles in each system.

        >>> print system1.nparticles
        216
        >>> print combined_system.nparticles
        432

        Make sure the appended Force terms are independent of the source system.
        >>> print "%.1f" % (combined_system.forces[0].particles[216].sigma / units.angstroms)
        3.4
        >>> system2.forces[0].particles[0].sigma *= -1
        >>> print "%.1f" % (combined_system.forces[0].particles[216].sigma / units.angstroms)
        3.4

        """        
        system = System(self)
        system += other
        return system
        
    def __iadd__(self, other):
        """
        Append specified system.

        USAGE

        system += additional_system

        NOTES

        Both systems must have identical ordering of Force terms.
        Any non-particle settings from the first System override those of the second, if they differ.
        
        EXAMPLES

        Append atoms from a second system to the first.
    
        >>> import simtk.pyopenmm.extras.testsystems as testsystems
        >>> [system1, coordinates1] = testsystems.LennardJonesFluid()
        >>> system1 = System(system1) # convert to pure Python system
        >>> [system2, coordinates2] = testsystems.LennardJonesFluid()
        >>> system2 = System(system2) # convert to pure Python system        
        >>> system1 += system2
        
        Check the number of particles in each system.
        
        >>> print system2.nparticles
        216
        >>> print system1.nparticles
        432
        
        Make sure the appended Force terms are independent of the source system.
        >>> print "%.1f" % (system1.forces[0].particles[216].sigma / units.angstroms)
        3.4
        >>> system2.forces[0].particles[0].sigma *= -1
        >>> print "%.1f" % (system1.forces[0].particles[216].sigma / units.angstroms)
        3.4
        
        """
        # Make sure other system is Pythonic instance by making a deep copy.
        other = System(other)

        # Check to make sure both systems are compatible.
        if not isinstance(other, System):
            raise ValueError("both arguments must be System objects")
        if (self.nforces != other.nforces):
            raise ValueError("both System objects must have identical number of Force classes")
        for (force1,force2) in zip(self.forces, other.forces):
            if type(force1) != type(force2):
                raise ValueError("both System objects must have identical ordering of Force classes")

        # Combine systems.
        offset = self.nparticles
        self.masses += other.masses
        for constraint in other.constraints:            
            self.addConstraint(constraint.particle1+offset, constraint.particle2+offset, constraint.distance)
        for (force1, force2) in zip(self.forces, other.forces):
            force1._appendForce(force2, offset)
                
        return self
      
    def removeForce(self, force):
        """
        Remove the specified force from the System.

        ARGUMENTS

        force (Force) - the Force object to be removed

        EXAMPLES

        >>> system = System()        
        >>> nonbonded_force = NonbondedForce()
        >>> bond_force = HarmonicBondForce()
        >>> system.addForce(nonbonded_force)
        0
        >>> system.addForce(bond_force)
        1
        >>> system.removeForce(nonbonded_force)
        >>> system.removeForce(bond_force)

        """
        self.forces.remove(force)
        return

#=============================================================================================
# Force base class
#=============================================================================================

class Force(object):
    """
    Force objects apply forces to the particles in a System, or alter their behavior in other
    ways.  This is an abstract class.  Subclasses define particular forces.
    
    More specifically, a Force object can do any or all of the following:
    
    * Add a contribution to the force on each particle
    * Add a contribution to the potential energy of the System
    * Modify the positions and velocities of particles at the start of each time step
    * Define parameters which are stored in the Context and can be modified by the user
    * Change the values of parameters defined by other Force objects at the start of each time step
    
    """

    PYOPENMM_API_EXTENSIONS = True

    pass

#=============================================================================================
# NonbondedForce
#=============================================================================================

class NonbondedForce(Force):
    """
    This class implements nonbonded interactions between particles, including a Coulomb force to represent
    electrostatics and a Lennard-Jones force to represent van der Waals interactions.  It optionally supports
    periodic boundary conditions and cutoffs for long range interactions.  Lennard-Jones interactions are
    calculated with the Lorentz-Bertelot combining rule: it uses the arithmetic mean of the sigmas and the
    geometric mean of the epsilons for the two interacting particles.
    
    To use this class, create a NonbondedForce object, then call addParticle() once for each particle in the
    System to define its parameters.  The number of particles for which you define nonbonded parameters must
    be exactly equal to the number of particles in the System, or else an exception will be thrown when you
    try to create a Context.  After a particle has been added, you can modify its force field parameters
    by calling setParticleParameters().
    
    NonbondedForce also lets you specify "exceptions", particular pairs of particles whose interactions should be
    computed based on different parameters than those defined for the individual particles.  This can be used to
    completely exclude certain interactions from the force calculation, or to alter how they interact with each other.
    
    Many molecular force fields omit Coulomb and Lennard-Jones interactions between particles separated by one
    or two bonds, while using modified parameters for those separated by three bonds (known as "1-4 interactions").
    This class provides a convenience method for this case called createExceptionsFromBonds().  You pass to it
    a list of bonds and the scale factors to use for 1-4 interactions.  It identifies all pairs of particles which
    are separated by 1, 2, or 3 bonds, then automatically creates exceptions for them.

    EXAMPLES

    Create a force object.

    >>> force = NonbondedForce()

    Add a particle.

    >>> charge = 1.0 * units.elementary_charge 
    >>> sigma = 1.0 * units.angstrom
    >>> epsilon = 0.001 * units.kilocalories_per_mole
    >>> force.addParticle(charge, sigma, epsilon)
    0

    Set the nonbonded method.

    >>> nonbondedMethod = NonbondedForce.NoCutoff
    >>> force.setNonbondedMethod(nonbondedMethod)

    Return a Swig proxy.
    
    >>> force_proxy = force.asSwig()

    Create a new force from proxy.

    >>> force_from_proxy = NonbondedForce(force_proxy)

    Create a deep copy.

    >>> import copy
    >>> force_copy = copy.deepcopy(force)

    """

    # TODO: Can these be made static?
    NoCutoff = openmm.NonbondedForce.NoCutoff #: No cutoff is applied to nonbonded interactions.  The full set of N^2 interactions is computed exactly. This necessarily means that periodic boundary conditions cannot be used.  This is the default.    
    CutoffNonPeriodic = openmm.NonbondedForce.CutoffNonPeriodic #: Interactions beyond the cutoff distance are ignored.  Coulomb interactions closer than the cutoff distance are modified using the reaction field method.
    CutoffPeriodic = openmm.NonbondedForce.CutoffPeriodic #: Periodic boundary conditions are used, so that each particle interacts only with the nearest periodic copy of each other particle.  Interactions beyond the cutoff distance are ignored.  Coulomb interactions closer than the cutoff distance are modified using the reaction field method.
    Ewald = openmm.NonbondedForce.Ewald #: Periodic boundary conditions are used, and Ewald summation is used to compute the interaction of each particle with all periodic copies of every other particle.
    PME = openmm.NonbondedForce.PME #: Periodic boundary conditions are used, and Particle-Mesh Ewald (PME) summation is used to compute the interaction of each particle with all periodic copies of every other particle.

    def __init__(self, force=None):
        """
        Create a NonbondedForce.
        
        """
        # Initialize with defaults.
        self.particles = list() # particles[i] is the ParticleInfo object for particle i
        self.exceptions = list() # exceptions[i] is the ith ExceptionInfo entry
        self.exceptionMap = dict()
        self.nonbondedMethod = NonbondedForce.NoCutoff # the method of computing nonbonded interactions to use
        self.cutoffDistance = units.Quantity(1.0, units.nanometer) # cutoff used for cutoff-based nonbondedMethod choices, in units of distance
        self.rfDielectric = units.Quantity(78.3, units.dimensionless) # reaction-field dielectric
        self.ewaldErrorTol = 5.0e-4 # relative Ewald error tolerance
        self.useDispersionCorrection = True
    
        # Populate data structures from swig object, if specified
        if force is not None:        
            self._copyDataUsingInterface(self, force)
            
        return

    def _copyDataUsingInterface(self, dest, src):
        """
        Use the public interface to populate 'dest' from 'src'.

        """
        
        dest.__init__()
        dest.setNonbondedMethod( src.getNonbondedMethod() )
        dest.setCutoffDistance( src.getCutoffDistance() )
        dest.setReactionFieldDielectric( src.getReactionFieldDielectric() )
        dest.setEwaldErrorTolerance( src.getEwaldErrorTolerance() )
        dest.setUseDispersionCorrection( src.getUseDispersionCorrection() )
        for index in range(src.getNumParticles()):
            args = src.getParticleParameters(index)
            dest.addParticle(*args)
        for index in range(src.getNumExceptions()):
            args = src.getExceptionParameters(index)
            dest.addException(*args)
        
        return
        
    def asSwig(self):
        """
        Construct a corresponding Swig object.

        """
        force = openmm.NonbondedForce()
        self._copyDataUsingInterface(force, self)
        return force
        
    def getNumParticles(self):
        """
        Get the number of particles for which force field parameters have been defined.

        """        
        return len(self.particles)

    def getNumExceptions(self):
        """
        Get the number of special interactions that should be calculated differently from other interactions.

        """
        return len(self.exceptions)

    def getNonbondedMethod(self):
        """
        Get the method used for handling long range nonbonded interactions.

        """
        return self.nonbondedMethod

    def setNonbondedMethod(self, nonbondedMethod):
        """
        Set the method used for handling long range nonbonded interactions.

        """
        # TODO: Argument checking.
        self.nonbondedMethod = nonbondedMethod
        return

    def getCutoffDistance(self):
        """
        Get the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
        is NoCutoff, this value will have no effect.

        """
        return self.cutoffDistance

    @accepts_compatible_units(units.nanometer)
    def setCutoffDistance(self, cutoffDistance):
        """
        Set the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
        is NoCutoff, this value will have no effect.
        
        """
        self.cutoffDistance = cutoffDistance                
        return

    def getReactionFieldDielectric(self):
        """
        Get the dielectric constant to use for the solvent in the reaction field approximation.

        """
        return self.rfDielectric

    def setReactionFieldDielectric(self, reactionFieldDielectric):
        """
        Set the dielectric constant to use for the solvent in the reaction field approximation.

        @param reactionFieldDielectric     the reaction field dielectric (unitless Quantity)

        """
        self.rfDielectric = reactionFieldDielectric
        return
    
    def getEwaldErrorTolerance(self):
        """
        Get the error tolerance for Ewald summation.  This corresponds to the fractional error in the forces
        which is acceptable.  This value is used to select the reciprocal space cutoff and separation
        parameter so that the average error level will be less than the tolerance.  There is not a
        rigorous guarantee that all forces on all atoms will be less than the tolerance, however.
        
        """
        return self.ewaldErrorTol

    @accepts(float)
    def setEwaldErrorTolerance(self, tol):
        """
        Set the error tolerance for Ewald summation.  This corresponds to the fractional error in the forces
        which is acceptable.  This value is used to select the reciprocal space cutoff and separation
        parameter so that the average error level will be less than the tolerance.  There is not a
        rigorous guarantee that all forces on all atoms will be less than the tolerance, however.

        """
        self.ewaldErrorTol = tol
        return 

    @accepts_compatible_units(units.elementary_charge, units.nanometers, units.kilojoules_per_mole)
    def addParticle(self, charge, sigma, epsilon):
        """
        Add the nonbonded force parameters for a particle.  This should be called once for each particle
        in the System.  When it is called for the i'th time, it specifies the parameters for the i'th particle.
        For calculating the Lennard-Jones interaction between two particles, the arithmetic mean of the sigmas
        and the geometric mean of the epsilons for the two interacting particles is used (the Lorentz-Bertelot
        combining rule).
        
        @param charge    the charge of the particle, measured in units of the proton charge
        @param sigma     the sigma parameter of the Lennard-Jones potential (corresponding to the van der Waals radius of the particle), measured in nm
        @param epsilon   the epsilon parameter of the Lennard-Jones potential (corresponding to the well depth of the van der Waals interaction), measured in kJ/mol
        @return the index of the particle that was added

        """
        particle = NonbondedForceParticleInfo(charge, sigma, epsilon)
        self.particles.append(particle)
        return (len(self.particles) - 1)
        
    def getParticleParameters(self, index):
        """
        Set the nonbonded force parameters for a particle.  When calculating the Lennard-Jones interaction between two particles,
        it uses the arithmetic mean of the sigmas and the geometric mean of the epsilons for the two interacting particles
        (the Lorentz-Bertelot combining rule).
    
        @param index     the index of the particle for which to set parameters
        @param charge    the charge of the particle, measured in units of the proton charge
        @param sigma     the sigma parameter of the Lennard-Jones potential (corresponding to the van der Waals radius of the particle), measured in nm
        @param epsilon   the epsilon parameter of the Lennard-Jones potential (corresponding to the well depth of the van der Waals interaction), measured in kJ/mol

        """
        particle = self.particles[index]
        return (particle.charge, particle.sigma, particle.epsilon)

    @accepts_compatible_units(None, units.elementary_charge, units.nanometers, units.kilojoules_per_mole)
    def setParticleParameters(self, index, charge, sigma, epsilon):
        """
        Set the nonbonded force parameters for a particle.  When calculating the Lennard-Jones interaction between two particles,
        it uses the arithmetic mean of the sigmas and the geometric mean of the epsilons for the two interacting particles
        (the Lorentz-Bertelot combining rule).

        @param index     the index of the particle for which to set parameters
        @param charge    the charge of the particle, measured in units of the proton charge
        @param sigma     the sigma parameter of the Lennard-Jones potential (corresponding to the van der Waals radius of the particle), measured in nm
        @param epsilon   the epsilon parameter of the Lennard-Jones potential (corresponding to the well depth of the van der Waals interaction), measured in kJ/mol

        """
        particle = NonbondedForceParticleInfo(charge, sigma, epsilon)
        self.particles[index] = particle
        return    

    @accepts_compatible_units(None, None, units.elementary_charge**2, units.nanometers, units.kilojoules_per_mole, None)
    def addException(self, particle1, particle2, chargeProd, sigma, epsilon, replace=False):
        """
        Add an interaction to the list of exceptions that should be calculated differently from other interactions.
        If chargeProd and epsilon are both equal to 0, this will cause the interaction to be completely omitted from
        force and energy calculations.
    
        In many cases, you can use createExceptionsFromBonds() rather than adding each exception explicitly.
        
        @param particle1  the index of the first particle involved in the interaction
        @param particle2  the index of the second particle involved in the interaction
        @param chargeProd the scaled product of the atomic charges (i.e. the strength of the Coulomb interaction), measured in units of the proton charge squared
        @param sigma      the sigma parameter of the Lennard-Jones potential (corresponding to the van der Waals radius of the particle), measured in nm
        @param epsilon    the epsilon parameter of the Lennard-Jones potential (corresponding to the well depth of the van der Waals interaction), measured in kJ/mol
        @param replace    determines the behavior if there is already an exception for the same two particles.  If true, the existing one is replaced.  If false,
                          an exception is thrown.
        @return the index of the exception that was added

        """

        # TODO: Eliminate exceptionMap so that exceptions can be directly manipulated by user in a more pythonic manner without ill consequence?

        if particle1 not in range(self.getNumParticles()):
            raise ValueError("particle1 must be in range(0, getNumParticles())")
        if particle2 not in range(self.getNumParticles()):
            raise ValueError("particle1 must be in range(0, getNumParticles())")

        # Ensure particle1 < particle2.
        if (particle2 < particle1):
            (particle1, particle2) = (particle2, particle1)
         
        # Create exception entry.
        exception = NonbondedForceExceptionInfo(particle1, particle2, chargeProd, sigma, epsilon)

        # Store it.
        try:
            # Only one set of exceptions are allowed per pair of particles.
            index = self.exceptionMap[(particle1,particle2)]
            if replace:
                self.exceptions[index] = exception
            else:
                raise ValueError("Exception already exists for particle pair (%d,%d); set replace=True optional argument to allow replacement." % (particle1,particle2))
        except:
            # Add the exception and note the pair of particles added.
            index = len(self.exceptions)
            self.exceptions.append(exception)
            self.exceptionMap[(particle1,particle2)] = index

        return (len(self.exceptions) - 1)
     
    def getExceptionParameters(self, index):
        """
        Get the force field parameters for an interaction that should be calculated differently from others.
        
        @param index      the index of the interaction for which to get parameters
        @param particle1  the index of the first particle involved in the interaction
        @param particle2  the index of the second particle involved in the interaction
        @param chargeProd the scaled product of the atomic charges (i.e. the strength of the Coulomb interaction), measured in units of the proton charge squared
        @param sigma      the sigma parameter of the Lennard-Jones potential (corresponding to the van der Waals radius of the particle), measured in nm
        @param epsilon    the epsilon parameter of the Lennard-Jones potential (corresponding to the well depth of the van der Waals interaction), measured in kJ/mol
        """
        exception = self.exceptions[index]
        return (exception.particle1, exception.particle2, exception.chargeProd, exception.sigma, exception.epsilon)
    
    def setExceptionParameters(self, index, particle1, particle2, chargeProd, sigma, epsilon):
        """
        Set the force field parameters for an interaction that should be calculated differently from others.
        If chargeProd and epsilon are both equal to 0, this will cause the interaction to be completely omitted from
        force and energy calculations.
        
        @param index      the index of the interaction for which to get parameters
        @param particle1  the index of the first particle involved in the interaction
        @param particle2  the index of the second particle involved in the interaction
        @param chargeProd the scaled product of the atomic charges (i.e. the strength of the Coulomb interaction), measured in units of the proton charge squared
        @param sigma      the sigma parameter of the Lennard-Jones potential (corresponding to the van der Waals radius of the particle), measured in nm
        @param epsilon    the epsilon parameter of the Lennard-Jones potential (corresponding to the well depth of the van der Waals interaction), measured in kJ/mol
        """
        nparticles = self.getNumParticles()
        if particle1 not in range(0,nparticles):
            raise ValueError("particle1 must be in range(0, getNumParticles())")
        if particle2 not in range(0,nparticles):
            raise ValueError("particle1 must be in range(0, getNumParticles())")
         
        exception = NonbondedForceExceptionInfo(particle1, particle2, chargeProd, sigma, epsilon)
        self.exceptions[index] = exception
        
        return
    
    def createExceptionsFromBonds(self, bonds, coulomb14Scale, lj14Scale):
        """    
        Identify exceptions based on the molecular topology.  Particles which are separated by one or two bonds are set
        to not interact at all, while pairs of particles separated by three bonds (known as "1-4 interactions") have
        their Coulomb and Lennard-Jones interactions reduced by a fixed factor.
        
        @param bonds           the set of bonds based on which to construct exceptions.  Each element specifies the indices of
                               two particles that are bonded to each other.
        @param coulomb14Scale  pairs of particles separated by three bonds will have the strength of their Coulomb interaction
                               multiplied by this factor
        @param lj14Scale       pairs of particles separated by three bonds will have the strength of their Lennard-Jones interaction
                               multiplied by this factor
        """

        # TODO: Create a pure Python implementation of this.

        # Create a Swig object.
        swig_self = self.asSwig()
        # Create exceptions from bonds in Swig object.
        swig_self.createExceptionsFromBonds(bonds, coulomb14Scale, lj14Scale)
        # Reconstitute Python object.
        self._copyDataUsingInterface(self, swig_self)

        return

    def getUseDispersionCorrection(self):
        """
        Get whether to add a contribution to the energy that approximately represents the effect of Lennard-Jones
        interactions beyond the cutoff distance.  The energy depends on the volume of the periodic box, and is only
        applicable when periodic boundary conditions are used.  When running simulations at constant pressure, adding
        this contribution can improve the quality of results.

        @returns     True if the analytical dispersion correction will be used; otherwise False
        """
        return self.useDispersionCorrection

    def setUseDispersionCorrection(self, useCorrection):
        """
        Set whether to add a contribution to the energy that approximately represents the effect of Lennard-Jones
        interactions beyond the cutoff distance.  The energy depends on the volume of the periodic box, and is only
        applicable when periodic boundary conditions are used.  When running simulations at constant pressure, adding
        this contribution can improve the quality of results.

        @param useCorrection   if True, analytical dispersion correction will be used
        """
        self.useDispersionCorrection = useCorrection
        return

    #==========================================================================
    # PYTHONIC EXTENSIONS
    #==========================================================================    
    
    @property
    def nparticles(self):
        return len(self.particles)

    @property
    def nexceptions(self):
        return len(self.exceptions)        

    def __str__(self):
        """
        Return an 'informal' human-readable string representation of the System object.

        """
        
        r = ""

        # Show settings.
        r += "nonbondedMethod: %s\n" % str(self.nonbondedMethod)
        r += "cutoffDistance: %s\n" % str(self.cutoffDistance)
        r += "rfDielectric: %s\n" % str(self.rfDielectric)
        r += "ewaldErrorTolerance: %s\n" % str(self.ewaldErrorTol)

        # Show particles.
        if (self.nparticles > 0):
            r += "Particles:\n"
            r += "%8s %24s %24s %24s %24s\n" % ("particle", "charge", "sigma", "epsilon", "lambda")
            for (index, particle) in enumerate(self.particles):
                r += "%8d %24s %24s %24s %24f\n" % (index, str(particle.charge), str(particle.sigma), str(particle.epsilon), particle.lambda_)
            r += "\n"        

        # Show exceptions
        if (self.nexceptions > 0):
            r += "Exceptions:\n"
            r += "%8s %8s %24s %24s %24s\n" % ("particle1", "particle2", "chargeProd", "sigma", "epsilon")
            for exception in self.exceptions:
                r += "%8d %8d %24s %24s %24s\n" % (exception.particle1, exception.particle2, str(exception.chargeProd), str(exception.sigma), str(exception.epsilon))
            r += "\n"

        return r
        
    def _appendForce(self, force, offset):
        """
        Append atoms defined in another force of the same type.

        @param force      the force to be appended
        @param offset     integral offset for atom number

        EXAMPLES

        Create two force objects and append the second to the first.
    
        >>> force1 = NonbondedForce()
        >>> force2 = NonbondedForce()        
        >>> charge = 1.0 * units.elementary_charge 
        >>> sigma = 1.0 * units.angstrom
        >>> epsilon = 0.001 * units.kilocalories_per_mole
        >>> force1.addParticle(charge, sigma, epsilon)
        0
        >>> force2.addParticle(charge, sigma, epsilon)
        0
        >>> offset = 1
        >>> force1._appendForce(force2, offset)

        """
        # Check to make sure both forces are compatible.
        if type(self) != type(force):
            raise ValueError("other force object must be of identical Force subclass")

        # Check to make sure both forces have same settings.
        # TODO: This checking is probably more paranoid than we need.
        if (self.nonbondedMethod != force.nonbondedMethod):
            raise ValueError("other force has incompatible nonbondedMethod")
        if (self.cutoffDistance != force.cutoffDistance):
            raise ValueError("other force has incompatible cutoffDistance")
        if (self.rfDielectric != force.rfDielectric):
            raise ValueError("other force has incompatible rfDielectric")
        if (self.ewaldErrorTol != force.ewaldErrorTol):
            raise ValueError("other force has incompatible ewaldErrorTol")        
        if (self.useDispersionCorrection != force.useDispersionCorrection):            
            raise ValueError("other force has incompatible useDispersionCorrection")        

        # Combine systems.
        for particle in force.particles:
            self.addParticle(particle.charge, particle.sigma, particle.epsilon)
        for exception in force.exceptions:
            self.addException(exception.particle1+offset, exception.particle2+offset, exception.chargeProd, exception.sigma, exception.epsilon)

        return

#==========================================================================
# CONTAINER CLASSES
#==========================================================================
    
class NonbondedForceParticleInfo(object):
    @accepts_compatible_units(units.elementary_charge, units.nanometers, units.kilojoules_per_mole, None)
    def __init__(self, charge, sigma, epsilon, lambda_ = 1.0):
        """
        """
        self.charge = charge
        self.sigma = sigma
        self.epsilon = epsilon
        self.lambda_ = lambda_
        
        return
    
class NonbondedForceExceptionInfo(object):
    @accepts_compatible_units(None, None, units.elementary_charge**2, units.nanometers, units.kilojoules_per_mole)
    def __init__(self, particle1, particle2, chargeProd, sigma, epsilon):
        """
        """
        self.particle1 = particle1
        self.particle2 = particle2
        self.chargeProd = chargeProd
        self.sigma = sigma
        self.epsilon = epsilon
        
        return


#=============================================================================================
# HarmonicBondForce
#=============================================================================================

class HarmonicBondForce(Force):
    """
    This class implements an interaction between pairs of particles that varies harmonically with the distance
    between them.  To use it, create a HarmonicBondForce object then call addBond() once for each bond.  After
    a bond has been added, you can modify its force field parameters by calling setBondParameters().

    EXAMPLE

    Create and populate a HarmonicBondForce object.
    
    >>> bondforce = HarmonicBondForce()
    >>> bondforce.addBond(0, 1, 1.0*units.angstrom, 1.0*units.kilocalories_per_mole/units.angstrom**2)
    >>> bondforce.addBond(0, 2, 1.0*units.angstrom, 1.0*units.kilocalories_per_mole/units.angstrom**2)    
    >>> bondforce.addBond(0, 3, 1.0*units.angstrom, 1.0*units.kilocalories_per_mole/units.angstrom**2)
    
    Create a Swig object.

    >>> bondforce_proxy = bondforce.asSwig()

    Create a deep copy.

    >>> bondforce_copy = copy.deepcopy(bondforce)

    Append a set of bonds.

    >>> bondforce_copy._appendForce(bondforce, 3)
    
    """

    def __init__(self, force=None):
        """
        Create a HarmonicBondForce.

        """
        # Initialize defaults.
        self.bonds = list()

        # Populate data structures from swig object, if specified
        if force is not None:
            self._copyDataUsingInterface(self, force)
            
        return

    @classmethod
    def _copyDataUsingInterface(cls, dest, src):
        """
        Use the public interface to populate 'dest' from 'src'.
        
        """
        dest.__init__()        
        for index in range(src.getNumBonds()):            
            args = src.getBondParameters(index)
            dest.addBond(*args)
        
        return
        
    def asSwig(self):
        """
        Construct a Swig object.
        
        """
        force = openmm.HarmonicBondForce()
        self._copyDataUsingInterface(force, self)
        return force
        
    def getNumBonds(self):
        """
        Get the number of harmonic bond stretch terms in the potential function

        """
        return len(self.bonds)

    @accepts_compatible_units(None, None, units.nanometers, units.kilojoules_per_mole / units.nanometers**2)
    def addBond(self, particle1, particle2, length, k):
        """
        Add a bond term to the force field.
        *
        @param particle1 the index of the first particle connected by the bond
        @param particle2 the index of the second particle connected by the bond
        @param length    the equilibrium length of the bond
        @param k         the harmonic force constant for the bond
        @return the index of the bond that was added

        """
        bond = HarmonicBondForceBondInfo(particle1, particle2, length, k)
        self.bonds.append(bond)        
        return

    def getBondParameters(self, index):
        """
        Get the force field parameters for a bond term.
        
        @param index     the index of the bond for which to get parameters
        @returns particle1 the index of the first particle connected by the bond
        @returns particle2 the index of the second particle connected by the bond
        @returns length    the equilibrium length of the bond
        @returns k         the harmonic force constant for the bond

        """
        # TODO: Return deep copy?
        bond = self.bonds[index]        
        return (bond.particle1, bond.particle2, bond.length, bond.k)

    @accepts_compatible_units(None, None, None, units.nanometers, units.kilojoules_per_mole / units.nanometers**2)
    def setBondParameters(self, index, particle1, particle2, length, k):
        """
        Set the force field parameters for a bond term.
        
        @param index     the index of the bond for which to set parameters
        @param particle1 the index of the first particle connected by the bond
        @param particle2 the index of the second particle connected by the bond
        @param length    the equilibrium length of the bond
        @param k         the harmonic force constant for the bond
        """

        bond = HarmonicBondForceBondInfo(particle1, particle2, length, k)
        self.bonds[index] = bond
        return

    #==========================================================================
    # PYTHONIC EXTENSIONS
    #==========================================================================    
    
    @property
    def nbonds(self):
        return len(self.bonds)

    def __str__(self):
        """
        Return an 'informal' human-readable string representation of the System object.

        """
        
        r = ""

        # Show bonds.
        if (self.nbonds > 0):
            r += "Bonds:\n"
            r += "%8s %10s %10s %24s %24s\n" % ("index", "particle1", "particle2", "length", "k")
            for (index, bond) in enumerate(self.bonds):
                r += "%8d %10d %10d %24s %24s\n" % (index, bond.particle1, bond.particle2, str(bond.length), str(bond.k))
            r += "\n"        
            r += "\n"

        return r

    def _appendForce(self, force, offset):
        """
        Append atoms defined in another force of the same type.

        @param force      the force to be appended
        @param offset     integral offset for atom number

        """
        # Check to make sure both forces are compatible.
        if type(self) != type(force):
            raise ValueError("other force object must be of identical Force subclass")

        # Combine systems.
        for bond in force.bonds:
            self.addBond(bond.particle1+offset, bond.particle2+offset, bond.length, bond.k)
                
        return

class HarmonicBondForceBondInfo(object):
    """
    Information about a harmonic bond.
    """
    @accepts_compatible_units(None, None, units.nanometers, units.kilojoules_per_mole / units.nanometers**2)
    def __init__(self, particle1, particle2, length, k):
        # Store data.
        self.particle1 = particle1
        self.particle2 = particle2
        self.length = length
        self.k = k
        return        

#=============================================================================================
# HarmonicAngleForce
#=============================================================================================

class HarmonicAngleForce(Force):
    """
    This class implements an interaction between groups of three particles that varies harmonically with the angle
    between them.  To use it, create a HarmonicAngleForce object then call addAngle() once for each angle.  After
    an angle has been added, you can modify its force field parameters by calling setAngleParameters().

    EXAMPLE

    Create and populate a HarmonicAngleForce object.

    >>> angle = 120.0 * units.degree
    >>> k = 0.01 * units.kilocalories_per_mole / units.degree**2
    >>> angleforce = HarmonicAngleForce()
    >>> angleforce.addAngle(0, 1, 2, angle, k)
    
    Create a Swig object.

    >>> angleforce_proxy = angleforce.asSwig()

    Create a deep copy.

    >>> angleforce_copy = copy.deepcopy(angleforce)

    Append a set of angles.

    >>> angleforce_copy._appendForce(angleforce, 3)
    
    """

    def __init__(self, force=None):
        """
        Create a HarmonicAngleForce.

        """
        # Initialize defaults.
        self.angles = list()

        # Populate data structures from swig object, if specified
        if force is not None:
            self._copyDataUsingInterface(self, force)

        return

    @classmethod
    def _copyDataUsingInterface(cls, dest, src):
        """
        Use the public interface to populate 'dest' from 'src'.
        
        """
        dest.__init__()        
        for index in range(src.getNumAngles()):            
            args = src.getAngleParameters(index)
            dest.addAngle(*args)
        
        return
        
    def asSwig(self):
        """
        Construct a Swig object.
        
        """
        force = openmm.HarmonicAngleForce()
        self._copyDataUsingInterface(force, self)
        return force
        
    def getNumAngles(self):
        """
        Get the number of harmonic angle stretch terms in the potential function

        """
        return len(self.angles)

    @accepts_compatible_units(None, None, None, units.radians, units.kilojoules_per_mole / units.radians**2)    
    def addAngle(self, particle1, particle2, particle3, angle, k):
        """
        Add an angle term to the force field.

        @param particle1 the index of the first particle forming the angle
        @param particle2 the index of the second particle forming the angle
        @param particle3 the index of the third particle forming the angle
        @param angle     the equilibrium angle
        @param k         the harmonic force constant for the angle
        @return the index of the angle that was added

        """
        angle = HarmonicAngleForceAngleInfo(particle1, particle2, particle3, angle, k)
        self.angles.append(angle)        
        return

    def getAngleParameters(self, index):
        """
        Get the force field parameters for an angle term.
        
        @param index     the index of the angle for which to get parameters
        @returns particle1 the index of the first particle forming the angle
        @returns particle2 the index of the second particle forming the angle
        @returns particle3 the index of the third particle forming the angle
        @returns angle     the equilibrium angle
        @returns k         the harmonic force constant for the angle

        """
        angle = self.angles[index]        
        return (angle.particle1, angle.particle2, angle.particle3, angle.angle, angle.k)

    @accepts_compatible_units(None, None, None, None, units.radians, units.kilojoules_per_mole / units.radians**2)
    def setAngleParameters(self, index, particle1, particle2, particle3, angle, k):
        """
        Set the force field parameters for an angle term.
        
        @param index     the index of the angle for which to set parameters
        @param particle1 the index of the first particle forming the angle
        @param particle2 the index of the second particle forming the angle
        @param particle3 the index of the third particle forming the angle
        @param angle     the equilibrium angle
        @param k         the harmonic force constant for the angle
        """

        angle = HarmonicAngleForceAngleInfo(particle1, particle2, particle3, angle, k)
        self.angles[index] = angle
        return

    #==========================================================================
    # PYTHONIC EXTENSIONS
    #==========================================================================    
    
    @property
    def nangles(self):
        return len(self.angles)

    def __str__(self):
        """
        Return an 'informal' human-readable string representation of the System object.

        """
        
        r = ""

        # Show angles.
        if (self.nangles > 0):
            r += "Angles:\n"
            r += "%8s %10s %10s %10s %24s %24s\n" % ("index", "particle1", "particle2", "particle3", "length", "k")
            for (index, angle) in enumerate(self.angles):
                r += "%8d %10d %10d %10d %24s %24s\n" % (index, angle.particle1, angle.particle2, angle.particle3, str(angle.angle), str(angle.k))
            r += "\n"        
            r += "\n"

        return r

    def _appendForce(self, force, offset):
        """
        Append atoms defined in another force of the same type.

        @param force      the force to be appended
        @param offset     integral offset for atom number

        """
        # Check to make sure both forces are compatible.
        if type(self) != type(force):
            raise ValueError("other force object must be of identical Force subclass")

        # Combine systems.
        for angle in force.angles:
            self.addAngle(angle.particle1+offset, angle.particle2+offset, angle.particle3+offset, angle.angle, angle.k)

        return

class HarmonicAngleForceAngleInfo(object):
    """
    Information about a harmonic angle.
    """
    @accepts_compatible_units(None, None, None, units.radians, units.kilojoules_per_mole / units.radians**2)
    def __init__(self, particle1, particle2, particle3, angle, k):
        # Store data.
        self.particle1 = particle1
        self.particle2 = particle2
        self.particle3 = particle3
        self.angle = angle
        self.k = k
        return
        
#=============================================================================================
# PeriodicTorsionForce
#=============================================================================================

class PeriodicTorsionForce(Force):
    """
    This class implements an interaction between groups of four particles that varies periodically with the torsion angle
    between them.  To use it, create a PeriodicTorsionForce object then call addTorsion() once for each torsion.  After
    a torsion has been added, you can modify its force field parameters by calling setTorsionParameters().

    EXAMPLE

    Create and populate a PeriodicTorsionForce object.

    >>> periodicity = 3
    >>> phase = 30 * units.degrees
    >>> k = 1.0 * units.kilocalories_per_mole
    >>> torsionforce = PeriodicTorsionForce()
    >>> torsionforce.addTorsion(0, 1, 2, 3, periodicity, phase, k)
    
    Create a Swig object.

    >>> torsionforce_proxy = torsionforce.asSwig()

    Create a deep copy.

    >>> torsionforce_copy = copy.deepcopy(torsionforce)

    Append a set of angles.
    
    >>> torsionforce_copy._appendForce(torsionforce, 4)
    
    """

    def __init__(self, force=None):
        """
        Create a PeriodicTorsionForce.

        """
        # Initialize defaults.
        self.torsions = list()

        # Populate data structures from swig object, if specified
        if force is not None:
            self._copyDataUsingInterface(self, force)
                    
        return

    @classmethod
    def _copyDataUsingInterface(cls, dest, src):
        """
        Use the public interface to populate 'dest' from 'src'.
        
        """
        dest.__init__()        
        for index in range(src.getNumTorsions()):            
            args = src.getTorsionParameters(index)
            dest.addTorsion(*args)
        
        return
        
    def asSwig(self):
        """
        Construct a Swig object.
        
        """
        force = openmm.PeriodicTorsionForce()
        self._copyDataUsingInterface(force, self)
        return force
        
    def getNumTorsions(self):
        """
        Get the number of periodic torsion terms in the potential function

        """
        return len(self.torsions)

    @accepts_compatible_units(None, None, None, None, None, units.radians, units.kilojoules_per_mole)
    def addTorsion(self, particle1, particle2, particle3, particle4, periodicity, phase, k):
        """
        Add a periodic torsion term to the force field.

        @param particle1    the index of the first particle forming the torsion
        @param particle2    the index of the second particle forming the torsion
        @param particle3    the index of the third particle forming the torsion
        @param particle3    the index of the fourth particle forming the torsion
        @param periodicity  the periodicity of the torsion
        @param phase        the phase offset of the torsion
        @param k            the force constant for the torsion
        @return the index of the torsion that was added

        """
        torsion = PeriodicTorsionForcePeriodicTorsionInfo(particle1, particle2, particle3, particle4, periodicity, phase, k)
        self.torsions.append(torsion) 
        return

    def getTorsionParameters(self, index):
        """
        Get the force field parameters for a periodic torsion term.
        
        @param index        the index of the torsion for which to get parameters
        @returns particle1    the index of the first particle forming the torsion
        @returns particle2    the index of the second particle forming the torsion
        @returns particle3    the index of the third particle forming the torsion
        @returns particle3    the index of the fourth particle forming the torsion
        @returns periodicity  the periodicity of the torsion
        @returns phase        the phase offset of the torsion
        @returns k            the force constant for the torsion

        """
        torsion = self.torsions[index]
        return (torsion.particle1, torsion.particle2, torsion.particle3, torsion.particle4, torsion.periodicity, torsion.phase, torsion.k)

    @accepts_compatible_units(None, None, None, None, None, None, units.radians, units.kilojoules_per_mole)
    def setTorsionParameters(self, index, particle1, particle2, particle3, particle4, periodicity, phase, k):
        """
        Set the force field parameters for a periodic torsion term.
        
        @param index        the index of the torsion for which to set parameters
        @param particle1    the index of the first particle forming the torsion
        @param particle2    the index of the second particle forming the torsion
        @param particle3    the index of the third particle forming the torsion
        @param particle3    the index of the fourth particle forming the torsion
        @param periodicity  the periodicity of the torsion
        @param phase        the phase offset of the torsion
        @param k            the force constant for the torsion

        """
        torsion = PeriodicTorsionForcePeriodicTorsionInfo(particle1, particle2, particle3, particle4, periodicity, phase, k)
        self.torsions[index] = torsion
        return

    #==========================================================================
    # PYTHONIC EXTENSIONS
    #==========================================================================    
    
    @property
    def ntorsions(self):
        return len(self.torsions)

    def __str__(self):
        """
        Return an 'informal' human-readable string representation of the System object.

        """
        
        r = ""

        # Show torsions.
        if (self.ntorsions > 0):
            r += "Torsions:\n"
            r += "%8s %10s %10s %10s %10s %10s %24s %24s\n" % ("index", "particle1", "particle2", "particle3", "particle4", "periodicity", "phase", "k")
            for (index, torsion) in enumerate(self.torsions):
                r += "%8d %10d %10d %10d %10d %10d %24s %24s\n" % (index, torsion.particle1, torsion.particle2, torsion.particle3, torsion.particle4, torsion.periodicity, str(torsion.phase), str(torsion.k))
            r += "\n"        
            r += "\n"

        return r

    def _appendForce(self, force, offset):
        """
        Append atoms defined in another force of the same type.

        @param force      the force to be appended
        @param offset     integral offset for atom number

        """
        # Check to make sure both forces are compatible.
        if type(self) != type(force):
            raise ValueError("other force object must be of identical Force subclass")

        # Combine systems.
        for torsion in force.torsions:
            self.addTorsion(torsion.particle1+offset, torsion.particle2+offset, torsion.particle3+offset, torsion.particle4+offset, torsion.periodicity, torsion.phase, torsion.k)
                
        return

class PeriodicTorsionForcePeriodicTorsionInfo(object):
    """
    Information about a periodic torsion.
    """
    @accepts_compatible_units(None, None, None, None, None, units.radians, units.kilojoules_per_mole)
    def __init__(self, particle1, particle2, particle3, particle4, periodicity, phase, k):
        self.particle1 = particle1
        self.particle2 = particle2
        self.particle3 = particle3
        self.particle4 = particle4
        self.periodicity = periodicity
        self.phase = phase
        self.k = k
        return        

#=============================================================================================
# GBSAOBCForce
#=============================================================================================

class GBSAOBCForce(Force):
    """
    This class implements an implicit solvation force using the GBSA-OBC model.

    To use this class, create a GBSAOBCForce object, then call addParticle() once for each particle in the
    System to define its parameters.  The number of particles for which you define GBSA parameters must
    be exactly equal to the number of particles in the System, or else an exception will be thrown when you
    try to create a Context.  After a particle has been added, you can modify its force field parameters
    by calling setParticleParameters().

    EXAMPLE

    Create and populate a GBSAOBCForce object.

    >>> charge = 1.0 * units.elementary_charge
    >>> radius = 1.5 * units.angstrom
    >>> scalingFactor = 1.0
    >>> gbsaforce = GBSAOBCForce()
    >>> gbsaforce.addParticle(charge, radius, scalingFactor)
    0
    >>> gbsaforce.addParticle(charge, radius, scalingFactor)
    1
    
    Create a Swig object.

    >>> gbsaforce_proxy = gbsaforce.asSwig()

    Create a deep copy.

    >>> gbsaforce_copy = copy.deepcopy(gbsaforce)

    Append a set of particles.
    
    >>> gbsaforce_copy._appendForce(gbsaforce, 2)
    
    """

    # TODO: Can these be made static?
    NoCutoff = openmm.GBSAOBCForce.NoCutoff #: No cutoff is applied to nonbonded interactions.  The full set of N^2 interactions is computed exactly. This necessarily means that periodic boundary conditions cannot be used.  This is the default.
    CutoffNonPeriodic = openmm.GBSAOBCForce.CutoffNonPeriodic #: Interactions beyond the cutoff distance are ignored.
    CutoffPeriodic = openmm.GBSAOBCForce.CutoffPeriodic #: Periodic boundary conditions are used, so that each particle interacts only with the nearest periodic copy of each other particle.  Interactions beyond the cutoff distance are ignored.

    def __init__(self, force=None):
        """
        Create a GBSAOBCForce.

        """

        # Initialize with defaults.
        self.particles = list()
        self.nonbondedMethod = GBSAOBCForce.NoCutoff
        self.cutoffDistance = units.Quantity(1.0, units.nanometer)        
        self.solventDielectric = units.Quantity(78.3, units.dimensionless)
        self.soluteDielectric = units.Quantity(1.0, units.dimensionless)        
        
        # Populate data structures from swig object, if specified
        if force is not None:
            self._copyDataUsingInterface(self, force)

        return

    @classmethod
    def _copyDataUsingInterface(cls, dest, src):
        """
        Use the public interface to populate 'dest' from 'src'.
        
        """
        dest.__init__()
        dest.setNonbondedMethod( src.getNonbondedMethod() )
        dest.setCutoffDistance( src.getCutoffDistance() ) 
        dest.setSoluteDielectric( src.getSoluteDielectric() )
        dest.setSolventDielectric( src.getSolventDielectric() )        
        for index in range(src.getNumParticles()):
            args = src.getParticleParameters(index)
            dest.addParticle(*args)

        return
        
    def asSwig(self):
        """
        Construct a Swig object.
        
        """
        force = openmm.GBSAOBCForce()
        self._copyDataUsingInterface(force, self)
        return force
        
    def getNumParticles(self):
        """
        Get the number of particles in the system.

        """
        return len(self.particles)
    
    def addParticle(self, charge, radius, scalingFactor):
        """
        Add the GBSA parameters for a particle.  This should be called once for each particle
        in the System.  When it is called for the i'th time, it specifies the parameters for the i'th particle.
     
        @param charge         the charge of the particle
        @param radius         the GBSA radius of the particle
        @param scalingFactor  the OBC scaling factor for the particle
        @return the index of the particle that was added

        """
        particle = GBSAOBCForceParticleInfo(charge, radius, scalingFactor)
        self.particles.append(particle)
        return (len(self.particles) - 1)
        
    def getParticleParameters(self, index):
        """
        Get the force field parameters for a particle.
        
        @param index          the index of the particle for which to get parameters
        @returns charge         the charge of the particle
        @returns radius         the GBSA radius of the particle
        @returns scalingFactor  the OBC scaling factor for the particle

        """
        particle = self.particles[index]
        return (particle.charge, particle.radius, particle.scalingFactor)

    def setParticleParameters(self, index, charge, radius, scalingFactor):
        """
        Set the force field parameters for a particle.
        
        @param index          the index of the particle for which to set parameters
        @param charge         the charge of the particle
        @param radius         the GBSA radius of the particle
        @param scalingFactor  the OBC scaling factor for the particle
        
        """        
        particle = GBSAOBCForceParticleInfo(charge, radius, scalingFactor)
        self.particles[index] = particle
        return

    def getSolventDielectric(self):
        """
        Get the dielectric constant for the solvent.
        
        """
        return self.solventDielectric

    @accepts_compatible_units(None)
    def setSolventDielectric(self, solventDielectric):
        """
        Set the dielectric constant for the solvent.
        
        """
        self.solventDielectric = solventDielectric
        return

    def getSoluteDielectric(self):
        """
        Get the dielectric constant for the solute.
        
        """
        return self.soluteDielectric

    @accepts_compatible_units(None)
    def setSoluteDielectric(self, soluteDielectric):
        """
        Set the dielectric constant for the solute.
        
        """
        self.soluteDielectric = soluteDielectric
        return

    def getNonbondedMethod(self):
        """
        Get the method used for handling long range nonbonded interactions.

        """
        return self.nonbondedMethod

    def setNonbondedMethod(self, nonbondedMethod):
        """
        Set the method used for handling long range nonbonded interactions.

        """
        # TODO: Argument checking.
        self.nonbondedMethod = nonbondedMethod            
        return

    def getCutoffDistance(self):
        """
        Get the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
        is NoCutoff, this value will have no effect.

        """
        return self.cutoffDistance

    @accepts_compatible_units(units.nanometer)
    def setCutoffDistance(self, cutoffDistance):
        """
        Set the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
        is NoCutoff, this value will have no effect.
        
        """
        self.cutoffDistance = cutoffDistance                
        return

    #==========================================================================
    # PYTHONIC EXTENSIONS
    #==========================================================================    
    
    @property
    def nparticles(self):
        return len(self.particles)

    def __str__(self):
        """
        Return an 'informal' human-readable string representation of the System object.

        """
        
        r = ""

        # Show settings.
        r += "nonbondedMethod: %s\n" % str(self.nonbondedMethod)
        r += "cutoffDistance: %s\n" % str(self.cutoffDistance)
        r += "solventDielectric: %s\n" % str(self.solventDielectric)        
        r += "soluteDielectric: %s\n" % str(self.soluteDielectric)

        # Show particles.
        if (self.nparticles > 0):
            r += "Particles:\n"
            r += "%8s %24s %24s %24s %24s\n" % ("particle", "charge", "radius", "scalingFactor", "lambda")
            for (index, particle) in enumerate(self.particles):
                r += "%8d %24s %24s %24s %24f\n" % (index, str(particle.charge), str(particle.radius), str(particle.scalingFactor), 1.0)
            r += "\n"        

        return r
        
    def _appendForce(self, force, offset):
        """
        Append atoms defined in another force of the same type.

        @param force      the force to be appended
        @param offset     integral offset for atom number

        EXAMPLES
        Create two force objects and append the second to the first.
    
        >>> force1 = GBSAOBCForce()
        >>> force2 = GBSAOBCForce()        
        >>> charge = 1.0 * units.elementary_charge 
        >>> radius = 1.0 * units.angstrom
        >>> scalingFactor = 1.0
        >>> force1.addParticle(charge, radius, scalingFactor)
        0
        >>> force2.addParticle(charge, radius, scalingFactor)
        0
        >>> offset = 1
        >>> force1._appendForce(force2, offset)

        """
        # Check to make sure both forces are compatible.
        if type(self) != type(force):
            raise ValueError("other force object must be of identical Force subclass")

        # Check to make sure both forces have same settings.
        # TODO: This checking is probably more paranoid than we need.  Can this be relaxed?
        if (self.nonbondedMethod != force.nonbondedMethod):
            raise ValueError("other force has incompatible nonbondedMethod")
        if (self.cutoffDistance != force.cutoffDistance):
            raise ValueError("other force has incompatible cutoffDistance")
        if (self.solventDielectric != force.solventDielectric):
            raise ValueError("other force has incompatible solventDielectric")
        if (self.soluteDielectric != force.soluteDielectric):
            raise ValueError("other force has incompatible soluteDielectric")

        # Combine systems.
        for particle in force.particles:
            self.addParticle(particle.charge, particle.radius, particle.scalingFactor)
                
        return

class GBSAOBCForceParticleInfo(object):
    @accepts_compatible_units(units.elementary_charge, units.nanometers, None)
    def __init__(self, charge, radius, scalingFactor):
        """
        """
        self.charge = charge
        self.radius = radius
        self.scalingFactor = scalingFactor
        return

#=============================================================================================
# CustomNonbondedForce
#=============================================================================================

class CustomNonbondedForce(Force):
    """
    This class implements nonbonded interactions between particles.  Unlike NonbondedForce, the functional form
    of the interaction is completely customizable, and may involve arbitrary algebraic expressions and tabulated
    functions.  It may depend on the distance between particles, as well as on arbitrary global and
    per-particle parameters.  It also optionally supports periodic boundary conditions and cutoffs for long range interactions.

    To use this class, create a CustomNonbondedForce object, passing an algebraic expression to the constructor
    that defines the interaction energy between each pair of particles.  The expression may depend on r, the distance
    between the particles, as well as on any parameters you choose.  Then call addPerParticleParameter() to define per-particle
    parameters, and addGlobalParameter() to define global parameters.  The values of per-particle parameters are specified as
    part of the system definition, while values of global parameters may be modified during a simulation by calling Context::setParameter().
    
    Next, call addParticle() once for each particle in the System to set the values of its per-particle parameters.
    The number of particles for which you set parameters must be exactly equal to the number of particles in the
    System, or else an exception will be thrown when you try to create a Context.  After a particle has been added,
    you can modify its parameters by calling setParticleParameters().

    CustomNonbondedForce also lets you specify 'exclusions', particular pairs of particles whose interactions should be
    omitted from force and energy calculations.  This is most often used for particles that are bonded to each other.

    As an example, the following code creates a CustomNonbondedForce that implements a 12-6 Lennard-Jones potential:

    >>> force = CustomNonbondedForce('4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=0.5*(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2)')

    This force depends on two parameters: sigma and epsilon.  The following code defines these parameters, and
    specifies combining rules for them which correspond to the standard Lorentz-Bertelot combining rules:

    >>> force.addPerParticleParameter('sigma')
    0
    >>> force.addPerParticleParameter('epsilon')
    1

    Expressions may involve the operators + (add), - (subtract), * (multiply), / (divide), and ^ (power), and the following
    functions: sqrt, exp, log, sin, cos, sec, csc, tan, cot, asin, acos, atan, sinh, cosh, tanh.  All trigonometric functions
    are defined in radians, and log is the natural logarithm.  The names of per-particle parameters have the suffix '1' or '2'
    appended to them to indicate the values for the two interacting particles.  As seen in the above example, the expression may
    also involve intermediate quantities that are defined following the main expression, using ';' as a separator.

    In addition, you can call addFunction() to define a new function based on tabulated values.  You specify a vector of
    values, and an interpolating or approximating spline is created from them.  That function can then appear in the expression.

    EXAMPLES

    TESTS

    Add particles.
    
    >>> sigma = 1.0 * units.angstrom
    >>> epsilon = 0.001 * units.kilocalories_per_mole
    >>> force.addParticle([sigma, epsilon])
    0
    >>> force.addParticle([sigma, epsilon])
    1
    
    Create a Swig object.

    >>> force_proxy = force.asSwig()

    Create a deep copy.

    >>> force_copy = copy.deepcopy(force)

    Append a set of particles.
    
    >>> force_copy._appendForce(force, 2)    

    """

    # TODO: Can these be made static?
    NoCutoff = openmm.CustomNonbondedForce.NoCutoff #: No cutoff is applied to nonbonded interactions.  The full set of N^2 interactions is computed exactly. This necessarily means that periodic boundary conditions cannot be used.  This is the default.
    CutoffNonPeriodic = openmm.CustomNonbondedForce.CutoffNonPeriodic #: Interactions beyond the cutoff distance are ignored.
    CutoffPeriodic = openmm.CustomNonbondedForce.CutoffPeriodic #: Periodic boundary conditions are used, so that each particle interacts only with the nearest periodic copy of each other particle.  Interactions beyond the cutoff distance are ignored.

    def __init__(self, energy=None, parameter_units=None, force=None):
        """
        Create a CustomNonbondedForce.

        @param energy    an algebraic expression giving the interaction energy between two particles as a function of r, the distance between them, as well as any global and per-particle parameters
        
        """

        # Initialize with defaults.
        self.energyExpression = energy #: Energy expression
        self.cutoffDistance = units.Quantity(1.0, units.nanometer)
        self.nonbondedMethod = CustomNonbondedForce.NoCutoff

        # Ensure that either the energy or the force object is specified.
        if (energy is None) and (force is None):
            raise ValueException("An energy function expression must be specified.")
                
        self.parameters = list() #: parameters[i] is the ParticleInfo object for particle i
        self.globalParameters = list() #: globalParameters[i] is the GlobalParameterInfo object for particle i
        self.particles = list()
        self.exclusions = list()
        self.functions = list()            

        # Populate data structures from swig object, if specified
        if force is not None:        
            self._copyDataUsingInterface(self, force)

        # If parameters are specified, check that the expression is correct.
        if (parameter_units is not None):
            self.checkUnits(energy, parameter_units)
            
        return

    def _checkUnits(self, energy_expression, parameter_units):
        """
        Check to ensure that the specified energy_expression and parameter_units have compatible dimensions.

        @param energy_expression    a string specifying the energy expression for Lepton
        @param parameter_units      a dict specifying the units for each parameter (e.g. parameter_units['epsilon'] = 'kilocalories_per_mole')

        """

        # TODO

        return

    @classmethod
    def _copyDataUsingInterface(cls, dest, src):
        """
        Use the public interface to populate 'dest' from 'src'.

        """
        dest.__init__( src.getEnergyFunction() )
        dest.setNonbondedMethod( src.getNonbondedMethod() )
        dest.setCutoffDistance( src.getCutoffDistance() )
        for index in range(src.getNumPerParticleParameters()):
            name = src.getPerParticleParameterName(index)
            dest.addPerParticleParameter(name)            
        for index in range(src.getNumGlobalParameters()):
            name = src.getGlobalParameterName(index)
            defaultValue = src.getGlobalParameterDefaultValue(index)
            dest.addGlobalParameter(name, defaultValue)
        for index in range(src.getNumFunctions()):
            args = src.getFunctionParameters(index)
            dest.addFunction(*args)
        for index in range(src.getNumParticles()):
            parameters = src.getParticleParameters(index)
            dest.addParticle(parameters)
        for index in range(src.getNumExclusions()):
            (particle1, particle2) = src.getExclusionParticles(index)
            dest.addExclusion(particle1, particle2)
        
        return
        
    def asSwig(self):
        """
        Construct a Swig object.
        """
        force = openmm.CustomNonbondedForce(self.energyExpression)
        self._copyDataUsingInterface(force, self)
        return force
        
    def getNumParticles(self):
        """
        Get the number of particles for which force field parameters have been defined.

        """        
        return len(self.particles)

    def getNumExclusions(self):
        """
        Get the number of particle pairs whose interactions should be excluded. 

        """
        return len(self.exclusions)

    def getNumPerParticleParameters(self):
        """
        Get the number of per-particle parameters that the interaction depends on.

        """
        return len(self.parameters)

    def getNumGlobalParameters(self):
        """
        Get the number of global parameters that the interaction depends on.

        """
        return len(self.globalParameters)

    def getNumFunctions(self):
        """
        Get the number of tabulated functions that have been defined.

        """
        return len(self.functions)

    def getEnergyFunction(self):
        """
        Get the number of special interactions that should be calculated differently from other interactions.

        """
        return self.energyExpression

    def setEnergyFunction(self, energy):
        """
        Set the algebraic expression that gives the interaction energy between two particles.

        @params energy  an algebraic expression giving the interaction energy between two particles as a function of r, the distance between them, as well as any global and per-particle parameters 
        
        """
        self.energyExpression = energy
        return

    def getNonbondedMethod(self):
        """
        Get the method used for handling long range nonbonded interactions.

        """
        return self.nonbondedMethod

    def setNonbondedMethod(self, nonbondedMethod):
        """
        Set the method used for handling long range nonbonded interactions.

        """
        # TODO: Argument checking.
        self.nonbondedMethod = nonbondedMethod            
        return

    def getCutoffDistance(self):
        """
        Get the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
        is NoCutoff, this value will have no effect.

        """
        return self.cutoffDistance

    @accepts_compatible_units(units.nanometers)
    def setCutoffDistance(self, cutoffDistance):
        """
        Set the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
        is NoCutoff, this value will have no effect.
        
        """
        self.cutoffDistance = cutoffDistance        
        return

    def addPerParticleParameter(self, name):
        """
        Add a new per-particle parameter that the interaction may depend on.

        @param name     the name of the parameter
        @return the index of the parameter that was added  
        
        """
        self.parameters.append(name)
        return (len(self.parameters) - 1)

    def getPerParticleParameterName(self, index):
        """
        Get the name of a per-particle parameter.

        @param index     the index of the parameter for which to get the name
        @return the parameter name
        """
        return self.parameters[index]

    def setPerParticleParameterName(self, index, name):
        """
        Set the name of a per-particle parameter.

        @param index          the index of the parameter for which to set the name
        @param name           the name of the parameter
          
        """
        self.parameters[index] = name
        return

    def addGlobalParameter(self, name, defaultValue):
        """
        Add a new global parameter that the interaction may depend on.

        @param name             the name of the parameter
        @param defaultValue     the default value of the parameter
        @return the index of the parameter that was added

        """
        parameter = CustomNonbondedForceGlobalParameterInfo(name, defaultValue)
        self.globalParameters.append(parameter)
        return (len(self.globalParameters) - 1)

    def getGlobalParameterName(self, index):
        """
        Get the name of a global parameter.

        @param index     the index of the parameter for which to get the name
        @return the parameter name
        
        """
        return self.globalParameters[index].name

    def setGlobalParameterName(self, index, name):
        """
        Set the name of a global parameter.
        
        @param index          the index of the parameter for which to set the name
        @param name           the name of the parameter

        """
        self.globalParameters[index].name = name
        return

    def getGlobalParameterDefaultValue(self, index):
        """
        Get the default value of a global parameter.

        @param index     the index of the parameter for which to get the default value
        @return the parameter default value

        """
        return self.globalParameters[index].defaultValue

    def setGlobalParameterDefaultValue(self, index, defaultValue):
        """
        Set the default value of a global parameter.
        
        @param index          the index of the parameter for which to set the default value
        @param name           the default value of the parameter

        """
        self.globalParameters[index].defaultValue = defaultValue
        return

    def addParticle(self, parameters):
        """
        Add the nonbonded force parameters for a particle.  This should be called once for each particle
        in the System.  When it is called for the i'th time, it specifies the parameters for the i'th particle.

        @param parameters    the list of parameters for the new particle
        @return the index of the particle that was added
                         
        """
        # Unpack until we have just a list or a tuple of parameters.
        #while (len(parameters) == 1) and type(parameters[0]) in (list, tuple):
        #    parameters = parameters[0]
        particle = CustomNonbondedForceParticleInfo(parameters)
        self.particles.append(particle)
        return (len(self.particles) - 1)
        
    def getParticleParameters(self, index):
        """
        Get the nonbonded force parameters for a particle.
        
        @param index       the index of the particle for which to get parameters
        @param parameters  the list of parameters for the specified particle

        """
        return self.particles[index].parameters

    def setParticleParameters(self, index, parameters):
        """
        Set the nonbonded force parameters for a particle.

        @param index       the index of the particle for which to set parameters
        @param parameters  the list of parameters for the specified particle 

        """
        # Unpack until we have just a list or a tuple of parameters.
        #while (len(parameters) == 1) and type(parameters[0]) in (list, tuple):
        #    parameters = parameters[0]
        particle = CustomNonbondedForceParticleInfo(parameters)
        self.particles[index] = particle
        return    

    def addExclusion(self, particle1, particle2):
        """
        Add a particle pair to the list of interactions that should be excluded.
        
        @param particle1  the index of the first particle in the pair
        @param particle2  the index of the second particle in the pair
        @return the index of the exclusion that was added

        """
        exclusion = CustomNonbondedForceExclusionInfo(particle1, particle2)
        self.exclusions.append(exclusion)
        return (len(self.exclusions) - 1)

    def getExclusionParticles(self, index):
        """
        Get the particles in a pair whose interaction should be excluded.

        @param index      the index of the exclusion for which to get particle indices
        @return particle1  the index of the first particle in the pair
        @return particle2  the index of the second particle in the pair

        """
        exclusion = self.exclusions[index]
        return (exclusion.particle1, exclusion.particle2)
                         
    def setExclusionParticles(self, index, particle1, particle2):
        """
        Set the particles in a pair whose interaction should be excluded.

        @param index      the index of the exclusion for which to set particle indices
        @param particle1  the index of the first particle in the pair
        @param particle2  the index of the second particle in the pair

        """
        self.exclusions[index] = CustomNonbondedForceExclusionInfo(particle1, particle2)
        return

    def addFunction(self, name, values, min, max, interpolating):
        """
        Add a tabulated function that may appear in the energy expression.

        @param name           the name of the function as it appears in expressions
        @param values         the tabulated values of the function f(x) at uniformly spaced values of x between min and max.
                              The function is assumed to be zero for x &lt; min or x &gt; max.
        @param min            the value of the independent variable corresponding to the first element of values
        @param max            the value of the independent variable corresponding to the last element of values
        @param interpolating  if true, an interpolating (Catmull-Rom) spline will be used to represent the function.
                              If false, an approximating spline (B-spline) will be used.
        @return the index of the function that was added

        """
        function = CustomNonbondedForceFunctionInfo(name, values, min, max, interpolating)
        self.functions.append(function)
        return (len(self.functions) - 1)

    def getFunctionParameters(self, index):
        """
        Get the parameters for a tabulated function that may appear in the energy expression.
        
        @param index          the index of the function for which to get parameters
        @return name          the name of the function as it appears in expressions
        @return values        the tabulated values of the function f(x) at uniformly spaced values of x between min and max.
                              The function is assumed to be zero for x &lt; min or x &gt; max.
        @return min           the value of the independent variable corresponding to the first element of values
        @return max           the value of the independent variable corresponding to the last element of values
        @return interpolating if true, an interpolating (Catmull-Rom) spline will be used to represent the function.
                              If false, an approximating spline (B-spline) will be used.
                              
        """        
        function = self.functions[index]
        return (function.name, function.values, function.min, function.max, function.interpolating)
        
    def setFunctionParameters(self, index, name, values, min, max, interpolating):
        """
        Set the parameters for a tabulated function that may appear in algebraic expressions.
        
        @param index          the index of the function for which to set parameters
        @param name           the name of the function as it appears in expressions
        @param values         the tabulated values of the function f(x) at uniformly spaced values of x between min and max.
                              The function is assumed to be zero for x &lt; min or x &gt; max.
        @param min            the value of the independent variable corresponding to the first element of values
        @param max            the value of the independent variable corresponding to the last element of values
        @param interpolating  if true, an interpolating (Catmull-Rom) spline will be used to represent the function.
        
        """
        function = CustomNonbondedForceFunctionInfo(name, values, min, max, interpolating)
        self.functions[index] = function
        return
    #==========================================================================
    # PYTHONIC EXTENSIONS
    #==========================================================================    
    
    @property
    def nparameters(self):
        return len(self.parameters)

    @property
    def nglobalparameters(self):
        return len(self.globalParameters)

    @property
    def nparticles(self):
        return len(self.particles)

    @property
    def nexclusions(self):
        return len(self.exclusions)

    @property
    def nfunctions(self):
        return len(self.functions)

    def __str__(self):
        """
        Return an 'informal' human-readable string representation of the CustomNonbondedForce object.

        """
        
        r = ""

        # Show settings.
        r += "energyExpression: %s\n" % str(self.energyExpression)

        # TODO: Show nonbondedmethod, cutoff, particles, functions, exceptions, parameters, etc.

        return r
        
    def _appendForce(self, force, offset):
        """
        Append atoms defined in another force of the same type.

        @param force      the force to be appended
        @param offset     integral offset for atom number

        EXAMPLES

        Create two force objects and append the second to the first.

        >>> energy = 'k*r^2'
        >>> force1 = CustomNonbondedForce(energy)
        >>> force1.addGlobalParameter('k', 1.0 * units.kilocalories_per_mole / units.nanometers)
        0
        >>> force1.addParticle([])
        0
        >>> force2 = CustomNonbondedForce(energy)
        >>> force2.addGlobalParameter('k', 1.0 * units.kilocalories_per_mole / units.nanometers)
        0
        >>> force2.addParticle([])
        0
        >>> offset = 1
        >>> force1._appendForce(force2, offset)

        """

        # TODO: Allow coercion of Swig force objects into Python force objects.
        
        # Check to make sure both forces are compatible.
        if type(self) != type(force):
            raise ValueError("other force object must be of identical Force subclass")

        # Check to make sure force objects are compatible.
        if (self.energyExpression != force.energyExpression):
            raise ValueError("other force has incompatible energyExpression")        
        if (self.nonbondedMethod != force.nonbondedMethod):
            raise ValueError("other force has incompatible nonbondedMethod")
        if (self.cutoffDistance != force.cutoffDistance):
            raise ValueError("other force has incompatible cutoffDistance")
        if (self.nfunctions != force.nfunctions):
            raise ValueError("other force has incompatible functions")
        for (function1, function2) in zip(self.functions, force.functions):
            if (function1 != function2):
                raise ValueError("functions are incompatible: (%s) and (%s)" % (function1, function2))
        # TODO: More checking?

        # Combine systems.
        for particle in force.particles:
            self.particles.append(particle)
        for exclusion in force.exclusions:
            offset_exclusion = CustomNonbondedForceExclusionInfo(exclusion.particle1 + offset, exclusion.particle2 + offset)
            self.exclusions.append(offset_exclusion)
                
        return
        
class CustomNonbondedForceParticleInfo(object):
    def __init__(self, parameters):
        self.parameters = parameters
        return

class CustomNonbondedForcePerParticleParameterInfo(object):
    def __init__(self, name):
        self.name = name
        return
        
class CustomNonbondedForceGlobalParameterInfo(object):
    def __init__(self, name, defaultValue):
        self.name = name
        self.defaultValue = defaultValue
        return

class CustomNonbondedForceExclusionInfo(object):
    def __init__(self, particle1, particle2):
        self.particle1 = particle1
        self.particle2 = particle2
        return

class CustomNonbondedForceFunctionInfo(object):
    def __init__(self, name, values, min, max, interpolating):
        self.name = name
        # TODO: Check to ensure 'values' is numpy array or list.
        self.values = values
        self.min = min
        self.max = max
        self.interpolating = interpolating
        return

    def __eq__(self, other):
        if ((self.name != other.name) or
            (self.values != other.values) or
            (self.min != other.min) or
            (self.max != other.max) or
            (self.interpolating != other.interpolating)): 
            return false
        return true
    
#=============================================================================================
# CustomExternalForce
#=============================================================================================

class CustomExternalForce(Force):
    """
    This class implements an 'external' force on particles.  The force may be applied to any subset of the particles
    in the System.  The force on each particle is specified by an arbitrary algebraic expression, which may depend
    on the current position of the particle as well as on arbitrary global and per-particle parameters.
    
    To use this class, create a CustomExternalForce object, passing an algebraic expression to the constructor
    that defines the potential energy of each affected particle.  The expression may depend on the particle's x, y, and
    z coordinates, as well as on any parameters you choose.  Then call addPerParticleParameter() to define per-particle
    parameters, and addGlobalParameter() to define global parameters.  The values of per-particle parameters are specified as
    part of the system definition, while values of global parameters may be modified during a simulation by calling Context::setParameter().
    Finally, call addParticle() once for each particle that should be affected by the force.  After a particle has been added,
    you can modify its parameters by calling setParticleParameters().
    
    As an example, the following code creates a CustomExternalForce that attracts each particle to a target position (x0, y0, z0)
    via a harmonic potential:
    
    >>> force = CustomExternalForce('k*((x-x0)^2+(y-y0)^2+(z-z0)^2')

    This force depends on four parameters: the spring constant k and equilibrium coordinates x0, y0, and z0.  The following code defines these parameters:

    >>> force.addGlobalParameter('k', 1.0)
    0
    >>> force.addPerParticleParameter('x0')
    0
    >>> force.addPerParticleParameter('y0')
    1
    >>> force.addPerParticleParameter('z0')
    2

    Expressions may involve the operators + (add), - (subtract), * (multiply), / (divide), and ^ (power), and the following
    functions: sqrt, exp, log, sin, cos, sec, csc, tan, cot, asin, acos, atan, sinh, cosh, tanh, step.  All trigonometric functions
    are defined in radians, and log is the natural logarithm.  step(x) = 0 if x is less than 0, 1 otherwise.

    """
    def __init__(self, energy=None, force=None):
        """
        Create a CustomExternalForce.

        @param energy    an algebraic expression giving the potential energy of each particle as a function
                         of its x, y, and z coordinates

        """
        self.energyExpression = energy #: The algebraic expression giving the energy for each particle
        self.parameters = list() #: parameters[i] is the ith per-particle PerParticleParameterInfo object
        self.globalParameters = list() #: globalParameters[i] is the ith GlobalParameterInfo object
        self.particles = list() #: particles[i] is the ith ParticleInfo object containing per-particle parameters for particle i

        # Ensure that either the energy or the force object is specified.
        if (energy is None) and (force is None):
            raise ValueException("An energy function expression must be specified.")

        # Populate data structures from swig object, if specified
        if force is not None:        
            self._copyDataUsingInterface(self, force)

        return


    @classmethod
    def _copyDataUsingInterface(cls, dest, src):
        """
        Use the public interface to populate 'dest' from 'src'.

        """
        dest.__init__( src.getEnergyFunction() )
        for index in range(src.getNumPerParticleParameters()):
            name = src.getPerParticleParameterName(index)
            dest.addPerParticleParameter(name)            
        for index in range(src.getNumGlobalParameters()):
            name = src.getGlobalParameterName(index)
            defaultValue = src.getGlobalParameterDefaultValue(index)
            dest.addGlobalParameter(name, defaultValue)
        for index in range(src.getNumParticles()):
            (particle, parameters) = src.getParticleParameters(index)
            dest.addParticle(particle, parameters)

        return
        
    def asSwig(self):
        """
        Construct a Swig object.
        """
        force = openmm.CustomExternalForce(self.energyExpression)
        self._copyDataUsingInterface(force, self)
        return force
        
    def getNumParticles(self):
        """
        Get the number of particles for which force field parameters have been defined.

        @returns the number of particles for which force field parameters have been defined
        
        """        
        return len(self.particles)

    def getNumPerParticleParameters(self):
        """
        Get the number of per-particle parameters that the force depends on.

        @returns the number of per-particle parameters that have been specified

        """
        return len(self.parameters)

    def getNumGlobalParameters(self):
        """
        Get the number of global parameters that the force depends on.

        @returns the number of global parameters that have been specified

        """

        return len(self.globalParameters)

    def getEnergyFunction(self):
        """
        Get the algebraic expression that gives the potential energy of each particle

        @returns the algebraic energy expression

        """
        return self.energyExpression

    def setEnergyFunction(self, energy):
        """
        Set the algebraic expression that gives the potential energy of each particle

        @param energy   an algebraic expression specifying the potential energy of each particle

        """
        # TODO: Argument checking with Lepton.
        self.energyExpression = energy

    def addPerParticleParameter(self, name):
        """
        Add a new per-particle parameter that the force may depend on.

        @param name             the name of the parameter   
        @return the index of the parameter that was added

        """
        parameter = CustomExternalForcePerParticleParameterInfo(name)
        self.parameters.append(parameter)
        return (len(self.parameters) - 1)

    def getPerParticleParameterName(self, index):
        """
        Get the name of a per-particle parameter.
        
        @param index     the index of the parameter for which to get the name
        @return the parameter name
        
        """
        return self.parameters[index].name

    def setPerParticleParameterName(self, index, name):
        """
        Set the name of a per-particle parameter.

        @param index          the index of the parameter for which to set the name
        @param name           the name of the parameter

        """
        self.parameters[index].name = name

    def addGlobalParameter(self, name, defaultValue):
        """
        Add a new global parameter that the force may depend on.

        @param name             the name of the parameter
        @param defaultValue     the default value of the parameter
        @return the index of the parameter that was added

        """
        globalParameter = CustomExternalForceGlobalParameterInfo(name, defaultValue)
        self.globalParameters.append(globalParameter)
        return (len(self.globalParameters) - 1)

    def getGlobalParameterName(self, index):
        """
        Get the name of a global parameter.

        @param index     the index of the parameter for which to get the name
        @return the parameter name

        """
        return self.globalParameters[index].name

    def setGlobalParameterName(self, index, name):
        """
        Set the name of a global parameter.

        @param index          the index of the parameter for which to set the name
        @param name           the name of the parameter

        """
        self.globalParameters[index].name = name

    def getGlobalParameterDefaultValue(self, index):
        """
        Get the default value of a global parameter.

        @param index     the index of the parameter for which to get the default value
        @return the parameter default value

        """
        return self.globalParameters[index].defaultValue
    
    def setGlobalParameterDefaultValue(self, index, defaultValue):
        """
        Set the default value of a global parameter.
        
        @param index          the index of the parameter for which to set the default value
        @param name           the default value of the parameter

        """
        self.globalParameters[index].defaultValue = defaultValue

    def addParticle(self, particle, parameters):
        """
        Add a particle term to the force field.

        @param particle     the index of the particle this term is applied to
        @param parameters   the list of parameters for the new force term
        @return the index of the particle term that was added

        """
        # Unpack until we have just a list or a tuple of parameters.
        #while (len(parameters) == 1) and type(parameters[0]) in (list, tuple):
        #    parameters = parameters[0]        
        particle = CustomExternalForceParticleInfo(particle, parameters)
        self.particles.append(particle)
        return (len(self.particles) - 1)
    
    def getParticleParameters(self, index):
        """
        Get the force field parameters for a force field term.
        
        @param index         the index of the particle term for which to get parameters
        @returns particle      the index of the particle this term is applied to
        @returns parameters    the list of parameters for the force field term

        """
        particleinfo = self.particles[index]
        return (particleinfo.particle, particleinfo.parameters)

    def setParticleParameters(self, index, particle, parameters):
        """
        Set the force field parameters for a force field term.

        @param index         the index of the particle term for which to set parameters
        @param particle      the index of the particle this term is applied to
        @param parameters    the list of parameters for the force field term

        """
        # Unpack until we have just a list or a tuple of parameters.
        #while (len(parameters) == 1) and type(parameters[0]) in (list, tuple):
        #    parameters = parameters[0]
        self.particles[index] = self.ParticleInfo(particle, parameters)    

    #==========================================================================
    # PYTHONIC EXTENSIONS
    #==========================================================================    
    
    @property
    def nparameters(self):
        return len(self.parameters)

    @property
    def nglobalparameters(self):
        return len(self.globalParameters)

    @property
    def nparticles(self):
        return len(self.particles)

    def __str__(self):
        """
        Return an 'informal' human-readable string representation of the CustomNonbondedForce object.

        """
        
        r = ""

        # Show settings.
        r += "energyExpression: %s\n" % str(self.energyExpression)

        for (index, particleinfo) in enumerate(self.particles):
            r += "%8d %8d %s\n" % (index, particleinfo.particle, str(particleinfo.parameters))

        # TODO

        return r
        
    def _appendForce(self, force, offset):
        """
        Append atoms defined in another force of the same type.

        @param force      the force to be appended
        @param offset     integral offset for atom number

        EXAMPLES

        Create two force objects and append the second to the first.

        >>> energy = 'Ez*q*z'
        >>> force1 = CustomExternalForce(energy)
        >>> force1.addGlobalParameter('Ez', 1.0 * units.kilocalories_per_mole / units.nanometers / units.elementary_charge)
        0
        >>> force1.addPerParticleParameter('q')
        0
        >>> force1.addParticle(0, 1.0 * units.elementary_charge)
        0
        >>> force2 = CustomExternalForce(energy)
        >>> force2.addGlobalParameter('Ez', 1.0 * units.kilocalories_per_mole / units.nanometers / units.elementary_charge)
        0
        >>> force2.addPerParticleParameter('q')
        0
        >>> force2.addParticle(0, 1.0 * units.elementary_charge)
        0
        >>> offset = 1
        >>> force1._appendForce(force2, offset)

        """

        # TODO: Allow coercion of Swig force objects into Python force objects.
        
        # Check to make sure both forces are compatible.
        if type(self) != type(force):
            raise ValueError("other force object must be of identical Force subclass")

        # Check to make sure force objects are compatible.
        if (self.energyExpression != force.energyExpression):
            raise ValueError("other force has incompatible energyExpression")        
        # TODO: More checking?

        # Combine systems.
        for particle in force.particles:
            self.addParticle(particle.particle+offset, particle.parameters)

        return
    
class CustomExternalForceParticleInfo(object):
    def __init__(self, particle, parameters):
        self.particle = particle
        self.parameters = parameters
        return

class CustomExternalForcePerParticleParameterInfo(object):
    def __init__(self, name):
        self.name = name
        return
        
class CustomExternalForceGlobalParameterInfo(object):
    def __init__(self, name, defaultValue):
        self.name = name
        self.defaultValue = defaultValue
        return    

#=============================================================================================
# CustomGBForce
#=============================================================================================

class CustomGBForce(Force):
    """
    This class implements complex, multiple stage nonbonded interactions between particles.  It is designed primarily
    for implementing Generalized Born implicit solvation models, although it is not strictly limited to that purpose.
    The interaction is specified as a series of computations, each defined by an arbitrary algebraic expression.
    It also allows tabulated functions to be defined and used with the computations.  It optionally supports periodic boundary
    conditions and cutoffs for long range interactions.

    The computation consists of calculating some number of per-particle <i>computed values</i>, followed by one or more
    <i>energy terms</i>.  A computed value is a scalar value that is computed for each particle in the system.  It may
    depend on an arbitrary set of global and per-particle parameters, and well as on other computed values that have
    been calculated before it.  Once all computed values have been calculated, the energy terms and their derivatives
    are evaluated to determine the system energy and particle forces.  The energy terms may depend on global parameters,
    per-particle parameters, and per-particle computed values.

    When specifying a computed value or energy term, you provide an algebraic expression to evaluate and a <i>computation type</i>
    describing how the expression is to be evaluated.  There are two main types of computations:

    <ul>
    <li><b>Single Particle</b>: The expression is evaluated once for each particle in the System.  In the case of a computed
    value, this means the value for a particle depends only on other properties of that particle (its position, parameters, and other
    computed values).  In the case of an energy term, it means each particle makes an independent contribution to the System
    energy.</li>
    <li><b>Particle Pairs</b>: The expression is evaluated for every pair of particles in the system.  In the case of a computed
    value, the value for a particular particle is calculated by pairing it with every other particle in the system, evaluating
    the expression for each pair, and summing them.  For an energy term, each particle pair makes an independent contribution to
    the System energy.  (Note that energy terms are assumed to be symmetric with respect to the two interacting particles, and
    therefore are evaluated only once per pair.  In contrast, expressions for computed values need not be symmetric and therefore are calculated
    twice for each pair: once when calculating the value for the first particle, and again when calculating the value for the
    second particle.)</li>
    </ul>

    Be aware that, although this class is extremely general in the computations it can define, particular Platforms may only support
    more restricted types of computations.  In particular, all currently existing Platforms require that the first computed value
    <i>must</i> be a particle pair computation, and all computed values after the first <i>must</i> be single particle computations.
    This is sufficient for most Generalized Born models, but might not permit some other types of calculations to be implemented.

    This is a complicated class to use, and an example may help to clarify it.  The following code implements the OBC variant
    of the GB/SA solvation model, using the ACE approximation to estimate surface area:
    
    >>> # OBC GBSA reference force
    >>> obc = GBSAOBCForce() 
    >>> # Create an equivalent OBC GBSA implementation as a CustomGBForce object.
    >>> custom = CustomGBForce()
    >>> custom.addPerParticleParameter("q")
    0
    >>> custom.addPerParticleParameter("radius")
    1
    >>> custom.addPerParticleParameter("scale")
    2
    >>> custom.addGlobalParameter("solventDielectric", obc.getSolventDielectric())
    0
    >>> custom.addGlobalParameter("soluteDielectric", obc.getSoluteDielectric())
    1
    >>> custom.addComputedValue("I", "step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(1/U^2-1/L^2)*(r-sr2*sr2/r)+0.5*log(L/U)/r+C);"
    ...                              "U=r+sr2;"
    ...                              "C=2*(1/or1-1/L)*step(sr2-r-or1);"
    ...                              "L=max(or1, D);"
    ...                              "D=abs(r-sr2);"
    ...                              "sr2 = scale2*or2;"
    ...                              "or1 = radius1-0.009; or2 = radius2-0.009", CustomGBForce.ParticlePairNoExclusions)
    0
    >>> custom.addComputedValue("B", "1/(1/or-tanh(1*psi-0.8*psi^2+4.85*psi^3)/radius);"
    ...                              "psi=I*or; or=radius-0.009", CustomGBForce.SingleParticle)
    1
    >>> custom.addEnergyTerm("28.3919551*(radius+0.14)^2*(radius/B)^6-0.5*138.935456*(1/soluteDielectric-1/solventDielectric)*q^2/B",
    ...                      CustomGBForce.SingleParticle)
    0
    >>> custom.addEnergyTerm("-138.935456*(1/soluteDielectric-1/solventDielectric)*q1*q2/f;"
    ...                      "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))", CustomGBForce.ParticlePair)
    1
    
    It begins by defining three per-particle parameters (charge, atomic radius, and scale factor) and two global parameters
    (the dielectric constants for the solute and solvent).  It then defines a computed value "I" of type ParticlePair.  The
    expression for evaluating it is a complicated function of the distance between each pair of particles (r), their atomic
    radii (radius1 and radius2), and their scale factors (scale1 and scale2).  Very roughly speaking, it is a measure of the
    distance between each particle and other nearby particles.

    Next a computation is defined for the Born Radius (B).  It is computed independently for each particle, and is a function of
    that particle's atomic radius and the intermediate value I defined above.

    Finally, two energy terms are defined.  The first one is computed for each particle and represents the surface area term,
    as well as the self interaction part of the polarization energy.  The second term is calculated for each pair of particles,
    and represents the screening of electrostatic interactions by the solvent.

    After defining the force as shown above, you should then call addParticle() once for each particle in the System to set the
    values of its per-particle parameters (q, radius, and scale).  The number of particles for which you set parameters must be
    exactly equal to the number of particles in the System, or else an exception will be thrown when you try to create a Context.
    After a particle has been added, you can modify its parameters by calling setParticleParameters().

    CustomNonbondedForce also lets you specify "exclusions", particular pairs of particles whose interactions should be
    omitted from calculations.  This is most often used for particles that are bonded to each other.  Even if you specify exclusions,
    however, you can use the computation type ParticlePairNoExclusions to indicate that exclusions should not be applied to a
    particular piece of the computation.

    Expressions may involve the operators + (add), - (subtract),    (multiply), / (divide), and ^ (power), and the following
    functions: sqrt, exp, log, sin, cos, sec, csc, tan, cot, asin, acos, atan, sinh, cosh, tanh, erf, erfc, min, max, abs, step.  All trigonometric functions
    are defined in radians, and log is the natural logarithm.  step(x) = 0 if x is less than 0, 1 otherwise.  In expressions for
    particle pair calculations, the names of per-particle parameters and computed values
    have the suffix "1" or "2" appended to them to indicate the values for the two interacting particles.  As seen in the above example,
    an expression may also involve intermediate quantities that are defined following the main expression, using ";" as a separator.

    In addition, you can call addFunction() to define a new function based on tabulated values.  You specify a vector of
    values, and a natural spline is created from them.  That function can then appear in expressions.

    """

    # This is an enumeration of the different methods that may be used for handling long range nonbonded forces.
    # NonbondedMethod 

    # No cutoff is applied to nonbonded interactions.  The full set of N^2 interactions is computed exactly.
    # This necessarily means that periodic boundary conditions cannot be used.  This is the default.
    NoCutoff = openmm.CustomGBForce.NoCutoff
    
    # Interactions beyond the cutoff distance are ignored.
    CutoffNonPeriodic = openmm.CustomGBForce.CutoffNonPeriodic

    # Periodic boundary conditions are used, so that each particle interacts only with the nearest periodic copy of
    # each other particle.  Interactions beyond the cutoff distance are ignored.
    CutoffPeriodic = openmm.CustomGBForce.CutoffPeriodic

    # This is an enumeration of the different ways in which a computed value or energy term can be calculated.
    # ComputationType

    # The value is computed independently for each particle, based only on the parameters and computed values for that particle.
    SingleParticle = openmm.CustomGBForce.SingleParticle

    # The value is computed as a sum over all pairs of particles, except those which have been added as exclusions.
    ParticlePair = openmm.CustomGBForce.ParticlePair

    # The value is computed as a sum over all pairs of particles.  Unlike ParticlePair, the list of exclusions is ignored
    # and all pairs are included in the sum, even those marked as exclusions.
    ParticlePairNoExclusions = openmm.CustomGBForce.ParticlePairNoExclusions

    def __init__(self, force=None):
        """
        Create a CustomGBForce.

        """
        self.nonbondedMethod = CustomGBForce.NoCutoff
        self.cutoffDistance = 1.0 * units.nanometer
        self.parameters = list()
        self.globalParameters = list()
        self.particles = list()
        self.exclusions = list()
        self.functions = list()
        self.computedValues = list()
        self.energyTerms = list()
        
        # Populate data structures from swig object, if specified
        if force is not None:        
            self._copyDataUsingInterface(self, force)
            
        return
    
    @classmethod
    def _copyDataUsingInterface(cls, dest, src):
        """
        Use the public interface to populate 'dest' from 'src'.

        """
        dest.__init__() # TODO: Do we need this?

        dest.setNonbondedMethod( src.getNonbondedMethod() )
        dest.setCutoffDistance( src.getCutoffDistance() )

        for index in range(src.getNumPerParticleParameters()):
            name = src.getPerParticleParameterName(index)
            dest.addPerParticleParameter(name)            
        for index in range(src.getNumGlobalParameters()):
            name = src.getGlobalParameterName(index)
            defaultValue = src.getGlobalParameterDefaultValue(index)
            dest.addGlobalParameter(name, defaultValue)
        for index in range(src.getNumParticles()):
            parameters = src.getParticleParameters(index)
            dest.addParticle(parameters)
        for index in range(src.getNumComputedValues()):
            (name, expression, type) = src.getComputedValueParameters(index)
            dest.addComputedValue(name, expression, type)
        for index in range(src.getNumEnergyTerms()):
            (expression, type) = src.getEnergyTermParameters(index)
            dest.addEnergyTerm(expression, type)
        for index in range(src.getNumExclusions()):
            (particle1, particle2) = src.getExclusionParticles(index)
            dest.addExclusion(particle1, particle2)
        for index in range(src.getNumFunctions()):
            (name, values, min, max) = src.getFunctionParameters(index)
            dest.addFunction(name, values, min, max)

        return
        
    def asSwig(self):
        """
        Construct a Swig object.
        """
        force = openmm.CustomGBForce()
        self._copyDataUsingInterface(force, self)
        return force

    def getNumParticles(self):
        """
        Get the number of particles for which force field parameters have been defined.
        
        RETURNS
        
        nparticles (int) - the number of particles

        """
        return len(self.particles)


    def getNumExclusions(self):
        """
        Get the number of particle pairs whose interactions should be excluded.

        RETURNS

        nexclusions (int) - the number of exclusions

        """
        return len(self.exclusions)

    def getNumPerParticleParameters(self):
        """
        Get the number of per-particle parameters that the interaction depends on.
        
        RETURNS 

        nparameters (int) - the number of per-particle parameters

        """
        return len(self.parameters)

    def getNumGlobalParameters(self):
        """
        Get the number of global parameters that the interaction depends on.
        
        RETURNS
        
        nglobals (int) - the number of global parameters

        """
        return len(self.globalParameters)

    def getNumFunctions(self):
        """
        Get the number of tabulated functions that have been defined.
        
        RETURNS
        
        nfunctions (int) - the number of tabulated functions

        """        
        return len(self.functions)

    def getNumComputedValues(self):
        """
        Get the number of per-particle computed values the interaction depends on.

        RETURNS

        nvalues (int) - the number of computed values
        
        """
        return len(self.computedValues)

    def getNumEnergyTerms(self):
        """
        Get the number of terms in the energy computation.

        RETURNS

        nterms (int) - the number of terms in energy computation

        """        
        return len(self.energyTerms)
    
    def getNonbondedMethod(self):
        """
        Get the method used for handling long range nonbonded interactions.

        RETURNS
        
        method - the nonbonded method

        """
        return self.nonbondedMethod

    def setNonbondedMethod(self, method):
        """
        Set the method used for handling long range nonbonded interactions.

        ARGUMENTS

        method - the nonbonded method (one of CustomGBForce.NoCutoff, CustomGBForce.CutoffNonPeriodic, CustomGBForce.CutoffPeriodic)

        """
        self.nonbondedMethod = method

    def getCutoffDistance(self):
        """
        Get the cutoff distance being used for nonbonded interactions.  If the NonbondedMethod in use
        is NoCutoff, this value will have no effect.
        
        distance (simtk.unit.Quantity compatible with simtk.unit.nanometer) - the cutoff distance
        """
        return self.cutoffDistance

    @accepts_compatible_units(units.nanometer)
    def setCutoffDistance(self, distance):
        """
        Set the cutoff distance being used for nonbonded interactions.  If the NonbondedMethod in use
        is NoCutoff, this value will have no effect.
        
        ARGUMENTS
        
        distance (simtk.unit.Quantity compatible with simtk.unit.nanometer) - the cutoff distance
        
        """
        self.cutoffDistance = distance

    def addPerParticleParameter(self, name):
        """
        Add a new per-particle parameter that the interaction may depend on.
        
        ARGUMENTS

        name (string) - the name of the parameter

        RETURNS
        
        index (int) - the index of the parameter that was added

        """
        self.parameters.append(name)
        return len(self.parameters) - 1

    def getPerParticleParameterName(self, index):
        """
        Get the name of a per-particle parameter.
     
        @param index     the index of the parameter for which to get the name
        @return the parameter name

        """
        return self.parameters[index]

    def setPerParticleParameterName(self, index, name):
        """
        Set the name of a per-particle parameter.
     
        @param index          the index of the parameter for which to set the name
        @param name           the name of the parameter

        """
        self.parameters[index] = name

    def addGlobalParameter(self, name, defaultValue):
        """
        Add a new global parameter that the interaction may depend on.
     
        @param name             the name of the parameter
        @param defaultValue     the default value of the parameter
        @return the index of the parameter that was added
        
        """
        parameter = CustomGBForceGlobalParameterInfo(name, defaultValue)
        self.globalParameters.append(parameter)
        return len(self.globalParameters) - 1

    def getGlobalParameterName(self, index):
        """
        Get the name of a global parameter.
     
        @param index     the index of the parameter for which to get the name
        @return the parameter name

        """
        return self.globalParameters[index].name

    def setGlobalParameterName(self, index, name):
        """
        Set the name of a global parameter.
        
        @param index          the index of the parameter for which to set the name
        @param name           the name of the parameter
        
        """
        self.globalParameters[index].name = name
        
    def getGlobalParameterDefaultValue(self, index):
        """
        Get the default value of a global parameter.
     
        @param index     the index of the parameter for which to get the default value
        @return the parameter default value
        
        """
        return self.globalParameters[index].defaultValue

    def setGlobalParameterDefaultValue(self, index, defaultValue):
        """
        Set the default value of a global parameter.
     
        @param index          the index of the parameter for which to set the default value
        @param name           the default value of the parameter
        
        """
        self.globalParameters[index] = defaultValue

    def addParticle(self, parameters):
        """
        Add the nonbonded force parameters for a particle.  This should be called once for each particle
        in the System.  When it is called for the i'th time, it specifies the parameters for the i'th particle.

        @param parameters    the list of parameters for the new particle
        @return the index of the particle that was added

        """        
        self.particles.append(CustomGBForceParticleInfo(parameters))
        return len(self.particles) - 1
        
    def getParticleParameters(self, index):
        """
        Get the nonbonded force parameters for a particle.
     
        @param index       the index of the particle for which to get parameters
        @returns parameters  the list of parameters for the specified particle
        
        """
        return self.particles[index].parameters
    
    def setParticleParameters(self, index, parameters):
        """
        Set the nonbonded force parameters for a particle.
     
        @param index       the index of the particle for which to set parameters
        @param parameters  the list of parameters for the specified particle
        
        """
        self.particles[index] = CustomGBForceParticleInfo(parameters)

    def addComputedValue(self, name, expression, type):
        """
        Add a computed value to calculate for each particle.
     
        @param name        the name of the value
        @param expression  an algebraic expression to evaluate when calculating the computed value.  If the
                           ComputationType is SingleParticle, the expression is evaluated independently
                           for each particle, and may depend on its x, y, and z coordinates, as well as the per-particle
                           parameters and previous computed values for that particle.  If the ComputationType is ParticlePair
                           or ParticlePairNoExclusions, the expression is evaluated once for every other
                           particle in the system and summed to get the final value.  In the latter case,
                           the expression may depend on the distance r between the two particles, and on
                           the per-particle parameters and previous computed values for each of them.
                           Append "1" to a variable name to indicate the parameter for the particle whose
                           value is being calculated, and "2" to indicate the particle it is interacting with.
        @param type        the method to use for computing this value

        """
        self.computedValues.append(CustomGBForceComputationInfo(name, expression, type));
        return len(self.computedValues) - 1

    def getComputedValueParameters(self, index):
        """
        Get the properties of a computed value.
        
        @param index       the index of the computed value for which to get parameters
        @param name        the name of the value
        @param expression  an algebraic expression to evaluate when calculating the computed value.  If the
                           ComputationType is SingleParticle, the expression is evaluated independently
                           for each particle, and may depend on its x, y, and z coordinates, as well as the per-particle
                           parameters and previous computed values for that particle.  If the ComputationType is ParticlePair
                           or ParticlePairNoExclusions, the expression is evaluated once for every other
                           particle in the system and summed to get the final value.  In the latter case,
                           the expression may depend on the distance r between the two particles, and on
                           the per-particle parameters and previous computed values for each of them.
                           Append "1" to a variable name to indicate the parameter for the particle whose
                           value is being calculated, and "2" to indicate the particle it is interacting with.
        @param type        the method to use for computing this value
        """
        info = self.computedValues[index]
        return (info.name, info.expression, info.type)
    
    def setComputedValueParameters(self, index, name, expression, type):
        """
        Set the properties of a computed value.

        @param index       the index of the computed value for which to set parameters
        @param name        the name of the value
        @param expression  an algebraic expression to evaluate when calculating the computed value.  If the
                           ComputationType is SingleParticle, the expression is evaluated independently
                           for each particle, and may depend on its x, y, and z coordinates, as well as the per-particle
                           parameters and previous computed values for that particle.  If the ComputationType is ParticlePair
                           or ParticlePairNoExclusions, the expression is evaluated once for every other
                           particle in the system and summed to get the final value.  In the latter case,
                           the expression may depend on the distance r between the two particles, and on
                           the per-particle parameters and previous computed values for each of them.
                           Append "1" to a variable name to indicate the parameter for the particle whose
                           value is being calculated, and "2" to indicate the particle it is interacting with.
        @param type        the method to use for computing this value

        """
        self.computedValues[index] = CustomGBForceComputationInfo(name, expression, type)

    def addEnergyTerm(self, expression, type):
        """
        * Add a term to the energy computation.
        *
        * @param expression  an algebraic expression to evaluate when calculating the energy.  If the
        *                    ComputationType is SingleParticle, the expression is evaluated once
        *                    for each particle, and may depend on its x, y, and z coordinates, as well as the per-particle
        *                    parameters and computed values for that particle.  If the ComputationType is ParticlePair or
        *                    ParticlePairNoExclusions, the expression is evaluated once for every pair of
        *                    particles in the system.  In the latter case,
        *                    the expression may depend on the distance r between the two particles, and on
        *                    the per-particle parameters and computed values for each of them.
        *                    Append "1" to a variable name to indicate the parameter for the first particle
        *                    in the pair and "2" to indicate the second particle in the pair.
        * @param type        the method to use for computing this value
        
        """
        self.energyTerms.append(CustomGBForceComputationInfo("", expression, type))
        return len(self.energyTerms) - 1

    def getEnergyTermParameters(self, index):
        """
        * Get the properties of a term to the energy computation.
        *
        * @param index       the index of the term for which to get parameters
        * @param expression  an algebraic expression to evaluate when calculating the energy.  If the
        *                    ComputationType is SingleParticle, the expression is evaluated once
        *                    for each particle, and may depend on its x, y, and z coordinates, as well as the per-particle
        *                    parameters and computed values for that particle.  If the ComputationType is ParticlePair or
        *                    ParticlePairNoExclusions, the expression is evaluated once for every pair of
        *                    particles in the system.  In the latter case,
        *                    the expression may depend on the distance r between the two particles, and on
        *                    the per-particle parameters and computed values for each of them.
        *                    Append "1" to a variable name to indicate the parameter for the first particle
        *                    in the pair and "2" to indicate the second particle in the pair.
        * @param type        the method to use for computing this value
        
        """
        info = self.energyTerms[index]
        return (info.expression, info.type)

    def setEnergyTermParameters(self, index, expression, type):
        """
        * Set the properties of a term to the energy computation.
        *
        * @param index       the index of the term for which to set parameters
        * @param expression  an algebraic expression to evaluate when calculating the energy.  If the
        *                    ComputationType is SingleParticle, the expression is evaluated once
        *                    for each particle, and may depend on its x, y, and z coordinates, as well as the per-particle
        *                    parameters and computed values for that particle.  If the ComputationType is ParticlePair or
        *                    ParticlePairNoExclusions, the expression is evaluated once for every pair of
        *                    particles in the system.  In the latter case,
        *                    the expression may depend on the distance r between the two particles, and on
        *                    the per-particle parameters and computed values for each of them.
        *                    Append "1" to a variable name to indicate the parameter for the first particle
        *                    in the pair and "2" to indicate the second particle in the pair.
        * @param type        the method to use for computing this value
        
        """
        self.energyTerms[index] = CustomGBForceComputationInfo("", expression, type)
        
    def addExclusion(self, particle1, particle2):
        """
        * Add a particle pair to the list of interactions that should be excluded.
        *
        * @param particle1  the index of the first particle in the pair
        * @param particle2  the index of the second particle in the pair
        * @return the index of the exclusion that was added
        """
        self.exclusions.append(CustomGBForceExclusionInfo(particle1, particle2))
        return len(self.exclusions) - 1

    def getExclusionParticles(self, index):
        """
        * Get the particles in a pair whose interaction should be excluded.
        *
        * @param index      the index of the exclusion for which to get particle indices
        * @param particle1  the index of the first particle in the pair
        * @param particle2  the index of the second particle in the pair
        """
        exclusion = self.exclusions[index]
        return (exclusion.particle1, exclusion.particle2)

    def setExclusionParticles(self, index, particle1, particle2):
        """
        * Set the particles in a pair whose interaction should be excluded.
        *
        * @param index      the index of the exclusion for which to set particle indices
        * @param particle1  the index of the first particle in the pair
        * @param particle2  the index of the second particle in the pair
        """
        self.exclusions[index] = CustomGBForceExclusionInfo(particle1, particle2)

    def addFunction(self, name, values, min, max):
        """
        * Add a tabulated function that may appear in the energy expression.
        *
        * @param name           the name of the function as it appears in expressions
        * @param values         the tabulated values of the function f(x) at uniformly spaced values of x between min and max.
        *                       The function is assumed to be zero for x &lt; min or x &gt; max.
        * @param min            the value of the independent variable corresponding to the first element of values
        * @param max            the value of the independent variable corresponding to the last element of values
        * @return the index of the function that was added
        """
        # TODO: Suport numpy arrays and lists for 'values'.
        if (max <= min):
            raise Exception("CustomGBForce: max <= min for a tabulated function.");
        if (len(values) < 2):
            raise Exception("CustomGBForce: a tabulated function must have at least two points");
        self.functions.append(CustomGBForceFunctionInfo(name, values, min, max))
        return len(self.functions)-1;
    
    def getFunctionParameters(self, index):
        """
        * Get the parameters for a tabulated function that may appear in the energy expression.
        *
        * @param index          the index of the function for which to get parameters
        * @param name           the name of the function as it appears in expressions
        * @param values         the tabulated values of the function f(x) at uniformly spaced values of x between min and max.
        *                       The function is assumed to be zero for x &lt; min or x &gt; max.
        * @param min            the value of the independent variable corresponding to the first element of values
        * @param max            the value of the independent variable corresponding to the last element of values
        """
        info = self.functions[index]
        return (info.name, info.values, info.min, info.max)

    def setFunctionParameters(self, index, name, values, min, max):
        """
        * Set the parameters for a tabulated function that may appear in algebraic expressions.
        *
        * @param index          the index of the function for which to set parameters
        * @param name           the name of the function as it appears in expressions
        * @param values         the tabulated values of the function f(x) at uniformly spaced values of x between min and max.
        *                       The function is assumed to be zero for x &lt; min or x &gt; max.
        * @param min            the value of the independent variable corresponding to the first element of values
        * @param max            the value of the independent variable corresponding to the last element of values
        """
        # TODO: Suport numpy arrays and lists for 'values'.
        if (max <= min):
            raise Exception("CustomGBForce: max <= min for a tabulated function.");
        if (len(values) < 2):
            raise Exception("CustomGBForce: a tabulated function must have at least two points");
        self.functions.append(CustomGBForceFunctionInfo(name, values, min, max))
        self.functions[index] = CustomGBForceFunctionInfo(name, values, min, max)

    #==========================================================================
    # PYTHONIC EXTENSIONS
    #==========================================================================    
    
    @property
    def nparameters(self):
        return len(self.parameters)

    @property
    def nglobalparameters(self):
        return len(self.globalParameters)

    @property
    def nparticles(self):
        return len(self.particles)

    @property
    def nfunctions(self):
        return len(self.funtions)

    @property
    def nterms(self):
        return len(self.energyTerms)

    def __str__(self):
        """
        Return an 'informal' human-readable string representation of the CustomNonbondedForce object.

        """
        
        r = ""

        # TODO

        return r
        
    def _appendForce(self, force, offset):
        """
        Append atoms defined in another force of the same type.

        @param force      the force to be appended
        @param offset     integral offset for atom number

        EXAMPLES

        TODO

        """

        # TODO: Allow coercion of Swig force objects into Python force objects.
        
        # Check to make sure both forces are compatible.
        if type(self) != type(force):
            raise ValueError("other force object must be of identical Force subclass")

        # TODO: Check that all other options are the same.

        # Combine systems.
        for particle in force.particles:
            self.addParticle(particle.parameters)
        for exclusion in force.exclusions:
            self.addExclusion(exclusion.particle1, exclusion.particle2)

        return
        
class CustomGBForceParticleInfo(object):
    def __init__(self, parameters):
        self.parameters = parameters
        return

class CustomGBForcePerParticleParameterInfo(object):
    def __init__(self, name):
        self.name = name
        return
        
class CustomGBForceGlobalParameterInfo(object):
    def __init__(self, name, defaultValue):
        self.name = name
        self.defaultValue = defaultValue
        return

class CustomGBForceExclusionInfo(object):
    def __init__(self, particle1, particle2):
        self.particle1 = particle1
        self.particle2 = particle2
        return

class CustomGBForceFunctionInfo(object):
    def __init__(self, name, values, min, max):
        self.name = name
        # TODO: Check to ensure 'values' is numpy array or list.
        self.values = values
        self.min = min
        self.max = max
        return

    def __eq__(self, other):
        if ((self.name != other.name) or
            (self.values != other.values) or
            (self.min != other.min) or
            (self.max != other.max)):
            return false
        return true

class CustomGBForceComputationInfo(object):
    def __init__(self, name, expression, type):
        self.name = name
        self.expression = expression
        # TODO: Check to ensure 'type' is acceptable type.
        self.type = type
        return

        self.energyExpression = energy #: The algebraic expression giving the energy for each particle
        self.parameters = list() #: parameters[i] is the ith per-particle PerParticleParameterInfo object
        self.globalParameters = list() #: globalParameters[i] is the ith GlobalParameterInfo object
        self.particles = list() #: particles[i] is the ith ParticleInfo object containing per-particle parameters for particle i
        return

#=============================================================================================
# Context
#=============================================================================================

class Context(openmm.Context):
    """

    EXAMPLE

    >>> # Create a pure Python System object
    >>> import pyopenmm
    >>> import simtk.pyopenmm.extras.testsystems as testsystems
    >>> [system, coordinates] = testsystems.LennardJonesFluid(mm=pyopenmm)
    >>> # Create an integrator.
    >>> collision_rate = 90.0 / units.picosecond
    >>> timestep = 1.0 * units.femtosecond
    >>> temperature = 298.0 * units.kelvin
    >>> integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    >>> # Create a context
    >>> context = Context(system, integrator)
    >>> # Evaluate energy
    >>> state = context.getState(getEnergy=True)
    >>> # Destroy context
    >>> del context
    
    """
    def __init__(self, system, integrator, platform=None):
        """
        Construct a new Context in which to run a simulation, explicitly specifying what Platform should be used to perform calculations.
        
        @param system -- the System which will be simulated (either pymm.System or openmm.System)
        @param integrator -- the Integrator which will be used to simulate the System
        @param platform -- the Platform to use for calculations

        """

        self.proxy_system = system
        # Convert to Swig object only if necessary.
        if 'asSwig' in dir(system):
            self.proxy_system = system.asSwig()

        if platform is not None:
            openmm.Context.__init__(self, self.proxy_system, integrator, platform)
        else:
            openmm.Context.__init__(self, self.proxy_system, integrator)
        return

#=============================================================================================
# Global definitions
#=============================================================================================

IMPLICIT_SOLVATION_FORCES = [GBSAOBCForce, GBVIForce, CustomGBForce] # Force classes that are implicit solvation classes

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod() # DEBUG

    # Reconstitute all systems in testsystems as pure Python systems.
    import simtk.openmm as openmm
    import pyopenmm
    import simtk.pyopenmm.extras.testsystems as testsystems
    test_systems = [ (name, getattr(testsystems, name)) for name in dir(testsystems) if callable(getattr(testsystems, name)) ]
    #test_systems = [ ('WaterBox', getattr(testsystems, 'WaterBox')) ]
    #test_systems = [ ('CustomGBForceSystem', getattr(testsystems, 'CustomGBForceSystem')) ]
    #test_systems = [ ('LennardJonesCluster', getattr(testsystems, 'LennardJonesCluster')) ]
    for (name, test_system) in test_systems:
        print "Constructing system '%s'..." % name
        # Construct openmm and pyopenmm systems.

        timestep = 1.0 * units.femtosecond

        # Create the system using openmm interface.
        [openmm_system, coordinates] = test_system(mm=openmm)
        openmm_integrator = openmm.VerletIntegrator(timestep)        
        openmm_context   = openmm.Context(openmm_system, openmm_integrator)
        openmm_context.setPositions(coordinates)
        openmm_state = openmm_context.getState(getEnergy=True)
        openmm_energy = openmm_state.getPotentialEnergy()
        print "openmm construction:          %s (%s)" % (str(openmm_energy), openmm_context.getPlatform().getName())
        del openmm_context, openmm_integrator

        # Create the system using pyopenmm interface.
        [pyopenmm_system, coordinates] = test_system(mm=pyopenmm)    
        pyopenmm_integrator = openmm.VerletIntegrator(timestep)        
        pyopenmm_context = pyopenmm.Context(pyopenmm_system, pyopenmm_integrator)
        pyopenmm_context.setPositions(coordinates)        
        pyopenmm_state = pyopenmm_context.getState(getEnergy=True)
        pyopenmm_energy = pyopenmm_state.getPotentialEnergy()
        print "pyopenmm construction:        %s (%s)" % (str(pyopenmm_energy), pyopenmm_context.getPlatform().getName())
        del pyopenmm_context, pyopenmm_integrator

        # Create openmm Swig version from pyopenmm-constructed system.
        openmm_pyopenmm_system = pyopenmm_system.asSwig()
        openmm_integrator = openmm.VerletIntegrator(timestep)        
        openmm_context   = openmm.Context(openmm_pyopenmm_system, openmm_integrator)
        openmm_context.setPositions(coordinates)
        openmm_state = openmm_context.getState(getEnergy=True)
        openmm_energy = openmm_state.getPotentialEnergy()
        print "asSwig:                       %s (%s)" % (str(openmm_energy), openmm_context.getPlatform().getName())
        del openmm_context, openmm_integrator

        # Create pyopenmm version from openmm-constructed system using copy construction.
        pyopenmm_openmm_system = pyopenmm.System(openmm_system) # openmm -> pyopenmm
        pyopenmm_integrator = openmm.VerletIntegrator(timestep)        
        pyopenmm_context = pyopenmm.Context(pyopenmm_openmm_system, pyopenmm_integrator)
        pyopenmm_context.setPositions(coordinates)        
        pyopenmm_state = pyopenmm_context.getState(getEnergy=True)
        pyopenmm_energy = pyopenmm_state.getPotentialEnergy()        
        print "copy constructor:             %s (%s)" % (str(pyopenmm_energy), pyopenmm_context.getPlatform().getName())
        del pyopenmm_context, pyopenmm_integrator
        
        # Create pyopenmm version from openmm Swig version from pyopenmm-constructed system.
        system = pyopenmm.System(pyopenmm_system.asSwig())
        integrator = openmm.VerletIntegrator(timestep)        
        context = pyopenmm.Context(system, integrator)
        context.setPositions(coordinates)
        state = context.getState(getEnergy=True)
        energy = state.getPotentialEnergy()
        print "pyopenmm from asSwig :        %s (%s)" % (str(energy), context.getPlatform().getName())
        del context, integrator, system

        # Create swig version from pyopenmm version from openmm-constructed system using copy construction.
        system = pyopenmm.System(openmm_system).asSwig()
        integrator = openmm.VerletIntegrator(timestep)        
        context = pyopenmm.Context(system, integrator)
        context.setPositions(coordinates)
        state = context.getState(getEnergy=True)
        energy = state.getPotentialEnergy()
        print "asSwig from copy cons:        %s (%s)" % (str(energy), context.getPlatform().getName())
        del context, integrator, system
        
        # DEBUG
        # Append system.
        appended_system = pyopenmm_system + pyopenmm_system
        nparticles = pyopenmm_system.nparticles
        unit = coordinates.unit
        coordinates = units.Quantity(numpy.array(coordinates / unit), unit)
        appended_coordinates = units.Quantity(numpy.zeros([nparticles*2, 3]), unit)
        appended_coordinates[0:nparticles,:] = coordinates[:,:]
        appended_coordinates[nparticles:2*nparticles,:] = coordinates[:,:] + 1000.0 * units.angstroms
        pyopenmm_integrator = openmm.VerletIntegrator(timestep)        
        pyopenmm_context = pyopenmm.Context(appended_system, pyopenmm_integrator)
        pyopenmm_context.setPositions(appended_coordinates)        
        pyopenmm_state = pyopenmm_context.getState(getEnergy=True)
        pyopenmm_energy = pyopenmm_state.getPotentialEnergy()
        print "appended coordinates:        %s (%s)" % (str(pyopenmm_energy / 2.0), pyopenmm_context.getPlatform().getName())
        del pyopenmm_context, pyopenmm_integrator
        
        
        
