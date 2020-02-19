import numpy as np
import sys, copy, time
from MAUtil import *
from MACalc import *

start = time.time()

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

    tagdict = getdefaultvasptags('RPBE')
    vapstags = Vasp(
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

    ### Get energy ###
    atoms.set_calculator(vapstags)
    e_atoms = atoms.get_potential_energy()

    print('{0}, {1}'.format(r ,e_atoms))
    f = open('result.txt', 'a')
    f.write('{0}, {1}'.format(r ,e_atoms))
    f.close()

print((time.time() - start)/60)