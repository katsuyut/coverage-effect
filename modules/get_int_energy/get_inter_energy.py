import numpy as np
import sys
import copy
from MAUtil import *
from MACalc import *

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
e_atoms = getenergy(atoms, name[0:-5]+'__', vapstags, env)
        
print('{0}, {1}'.format(name ,e_atoms))
f = open('result.txt', 'a')
f.write('{0}, {1}'.format(name ,e_atoms))
f.close()
