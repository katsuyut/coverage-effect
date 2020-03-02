import numpy as np
import sys
import copy
from MAUtil import *
from MACalc import *

# env = 'local'
env = 'spacom'

for d in [1.14, 1.15]:
    for xc in ['RPBE', 'RPBE-D2', 'vdW-DF', 'optB88-vdW', 'vdW-DF2', 'BEEF-vdW']:
        for nsw in [0, 200]
        dist = np.linspace(1.5/2**0.5, 5/2**0.5, 20)
        for r in dist:
            atoms = Atoms('COCO',
                          positions=[(0., 0., 0.), (0., 0., d),
                                     (r, r, 0), (r, r, d)],
                          cell=(10., 10., 10.))

            ### Set vasp ###
            kpoints = get_kpts(atoms)
#                 nb = getnbands(atoms, 2) # default value is 0.5
            tagdict = get_default_vasp_tags(xc)
            tagdict['kpts'] = kpoints
            tagdict['nsw'] = nsw
            vasptags = set_vasp_tags(tagdict)

            ### Get energy ###
            atoms.set_calculator(vasptags)
            e_atoms = atoms.get_potential_energy()
            print('{0}, {1}'.format(r, e_atoms))
            filename = 'result-' + xc + '-' + \
                str(d) + '-' + str(nsw) + '.txt'
            with open(filename, 'a') as f:
                f.write('{0}, {1}\n'.format(r, e_atoms))