import numpy as np
import sys
import copy
from MAUtil import *
from MACalc import *

env = 'local'
# env = 'spacom'

d = 1.13

dist = np.linspace(1.5/2**0.5, 5/2**0.5, 5)

for r in dist:
    atoms = Atoms('COCO',
                  positions=[(0., 0., 0.), (0., 0., d), (r, r, 0), (r, r, d)],
                  cell=(10., 10., 10.))

    ### Set vasp ###    
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

    ### Get energy ###
    atoms.set_calculator(vaspset)
    e_atoms = atoms.get_potential_energy()

    print('{0}, {1}'.format(r ,e_atoms))
    f = open('result.txt', 'a')
    f.write('{0}, {1}'.format(r ,e_atoms))
    f.close()
