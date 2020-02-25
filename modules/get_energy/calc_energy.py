import numpy as np
import sys, copy, time
from MAUtil import *
from MACalc import *
from custodian.custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler, UnconvergedErrorHandler, MaxForceErrorHandler

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
kpoints = get_kpts(atoms)    
nb = get_nbands(atoms, 2) # default value is 0.5

tagdict = get_default_vasp_tags('RPBE')
tagdict['kpts'] = kpoints
vasptags = set_vasp_tags(tagdict)

### Get energy ###
if query(name, env) != 'No file':
    atoms = query(name, env)
    print(atoms.get_potential_energy())
    e_atoms = 'Already in directory'
else:
    atoms.set_calculator(vasptags)
    e_atoms = get_energy(atoms, name[0:-5], vasptags, env)


### use custodian and if error is found restart ###
handlers = [VaspErrorHandler(), UnconvergedErrorHandler(), MaxForceErrorHandler()]

flag = False
for handler in handlers:
    if handler.check():
        flag = True
        handler.correct()

if flag:
    vasptags = Vasp(restart=True)
    atoms = vasptags.get_atoms()
    atoms.set_calculator(vasptags)
    e_atoms = get_energy(atoms, name[0:-5], vasptags, env)

print('{0}, {1}'.format(name ,e_atoms))
f = open('result.txt', 'a')
f.write('{0}, {1}'.format(name ,e_atoms))
f.close()

print((time.time() - start)/60, 'min')