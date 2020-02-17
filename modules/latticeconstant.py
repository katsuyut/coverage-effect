import numpy as np
import sys
from MAUtil import *
from MACalc import *

a = 3.65 # approximate lattice constant
b = a / 2
surface = 'Cu'
env = 'spacom'

atoms = Atoms(surface,
              cell=[(0, b, b), (b, 0, b), (b, b, 0)],
              pbc=1)

kpoints = getkpts(atoms)
nb = getnbands(atoms, 2) # default value is 0.5

vaspset = Vasp(
                xc = 'PBE',
                gga = 'RP',
                ncore = 4,
                encut = 350,
                nsw = 200,
                kpts = kpoints,
                ibrion = 2,
                isif = 0,
                ediffg = -3.00e-02,
                isym = 0,
                symprec = 1e-10,
                lreal = 'Auto',
                lcharg = False,
                lwave = False,
                )


latticeconstant = getLC(atoms, vaspset, env)
f = open('result.txt', 'w')
f.write('{0}\n'.format(str(latticeconstant)))
f.close()
