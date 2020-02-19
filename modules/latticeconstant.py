import numpy as np
import sys, copy, time
from MAUtil import *
from MACalc import *

start = time.time()

a = 3.65 # approximate lattice constant
b = a / 2
surface = 'Cu'
env = 'spacom'

atoms = Atoms(surface,
              cell=[(0, b, b), (b, 0, b), (b, b, 0)],
              pbc=1)

kpoints = getkpts(atoms)
nb = getnbands(atoms, 2) # default value is 0.5

tagdict = getdefaultvasptags('RPBE')
vasptags = Vasp(
    xc = tagdict['xc'],
    pp = tagdict['pp'],
    ncore = tagdict['ncore'],
	encut = tagdict['xc'],
    nsw = tagdict['nsw'],
    kpts = kpoints,
    ibrion = tagdict['ibrion'],
    isif = tagdict['isif'],
    ediffg = tagdict['ediffg'],
    isym = tagdict['isym'],
    lreal = tagdict['lreal'],
    lcharg = tagdict['lcharg'],
    lwave = tagdict['lwave'],
    ivdw = tagdict['ivdw'],
    lasph = tagdict['lasph'],
    )

latticeconstant = getLC(atoms, vasptags, env)
f = open('result.txt', 'w')
f.write('{0}\n'.format(str(latticeconstant)))
f.close()

print((time.time() - start)/60, 'min')