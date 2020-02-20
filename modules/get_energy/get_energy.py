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

if 3 in set(atoms.get_tags()): # set constraint only on surface calc
    constraint = FixAtoms(mask=[atom.tag >= fixlayer for atom in atoms])
    atoms.set_constraint(constraint)
    
    
### Set vasp ###    
kpoints = getkpts(atoms)    
nb = getnbands(atoms, 2) # default value is 0.5

tagdict = getdefaultvasptags('RPBE')
tagdict['kpts'] = kpoints
vasptags = setvasptags(tagdict)

### Get energy ###
if query(name, env) != 'No file':
    atoms = query(name, env)
    print(atoms.get_potential_energy())
    e_atoms = 'Already in directory'
else:
    atoms.set_calculator(vasptags)
    e_atoms = getenergy(atoms, name[0:-5], vasptags, env)

print('{0}, {1}'.format(name ,e_atoms))
f = open('result.txt', 'a')
f.write('{0}, {1}'.format(name ,e_atoms))
f.close()

print((time.time() - start)/60, 'min')