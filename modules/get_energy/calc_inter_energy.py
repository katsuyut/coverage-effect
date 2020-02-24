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

del atoms[[atom.index for atom in atoms if atom.tag!=0]]

### Set vasp ###    
kpoints = getkpts(atoms)    
nb = getnbands(atoms, 2) # default value is 0.5

tagdict = getdefaultvasptags('RPBE')
tagdict['kpts'] = kpoints
vasptags = setvasptags(tagdict)
        
### Get energy ###
e_atoms = getenergy(atoms, name[0:-5]+'__', vasptags, env)
        
print('{0}, {1}'.format(name ,e_atoms))
f = open('result.txt', 'a')
f.write('{0}, {1}'.format(name ,e_atoms))
f.close()

print((time.time() - start)/60, 'min')