from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

gro = GromacsGroFile('complex.gro')
top = GromacsTopFile('complex.top', unitCellDimensions=gro.getUnitCellDimensions(), includeDir='./top')
system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)

integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 2.0*femtoseconds)
simulation = Simulation(top.topology, system, integrator)

simulation.context.setPositions(gro.positions)

simulation.reporters.append(StateDataReporter('data_min.csv', 10, step=True, potentialEnergy=True, temperature=True, speed=True))
simulation.reporters.append(DCDReporter('min1.dcd', 100))

print('Minimizing...')
simulation.minimizeEnergy(maxIterations=100)

simulation.reporters.append(DCDReporter('trajectory.dcd', 100))
simulation.reporters.append(StateDataReporter('data.csv', 100, step=True, potentialEnergy=True, temperature=True, speed=True))

simulation.context.setVelocitiesToTemperature(300*kelvin)
print('Equilibrating...')
simulation.step(1000)


