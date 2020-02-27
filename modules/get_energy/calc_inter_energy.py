import numpy as np
import sys, copy, time
from MAUtil import *
from MACalc import *

start = time.time()

# env = 'local'
env = 'spacom'
name = sys.argv[1]

### Set coefficients ###
fixlayer = 3


### Set structure ###
atoms = init_query(name, env)
cell = atoms.cell

if 'no' not in name:
    sys.exit()

del atoms[[atom.index for atom in atoms if atom.tag!=0]]

### Set vasp ###    
kpoints = get_kpts(atoms)    
nb = get_nbands(atoms, 2) # default value is 0.5

tagdict = get_default_vasp_tags('RPBE')
tagdict['kpts'] = kpoints
tagdict['nsw'] = 0
vasptags = set_vasp_tags(tagdict)
        
### Get energy ###
e_atoms = get_energy(atoms, name[0:-5]+'__', vasptags, env)
        
print('{0}, {1}'.format(name[0:-5]+'__'+name[-5:] ,e_atoms))
f = open('result.txt', 'a')
f.write('{0}, {1}'.format(name[0:-5]+'__'+name[-5:] ,e_atoms))
f.close()

print((time.time() - start)/60, 'min')