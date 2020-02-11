import numpy as np
import sys
import copy
from ktms import *
from MACalc import *

# env = 'local'
env = 'spacom'
name = sys.argv[1]

### Set coefficients ###
fixlayer = 3



### Set structure ###
atoms = query(name, env)
cell = atoms.cell
    
if 3 in set(atoms.get_tags()): # set constraint only on surface calc
    constraint = FixAtoms(mask=[atom.tag >= fixlayer for atom in atoms])
    atoms.set_constraint(constraint)
    
    
### Set vasp ###    
kpoints = getkpts(atoms)    
nb = getnbands(atoms, 2) # default value is 0.5

if 'O' in atoms.symbols:
	vaspset = Vasp(                     
        	      xc = 'PBE',
                      gga = 'RP',
             	      ncore = 4,
	     	      encut = 350,
	              nsw = 0,
 	              kpts = kpoints,
              	      ibrion = 2,
              	      isif = 0,
              	      ediffg = -3.00e-02,
              	      isym = 0,
              	      lreal = 'Auto',
              	      lcharg = False,
        	      lwave = False,
	              lorbit = 0,
              	      rwigs = [1.5, 0.8, 0.8],
             	      )
else:
	vaspset = Vasp(
                       xc = 'PBE',
                       gga = 'RP',
                       ncore = 4,
                       encut = 350,
                       nsw = 0,
                       kpts = kpoints,
                       ibrion = 2,
                       isif = 0,
                       ediffg = -3.00e-02,
                       isym = 0,
                       lreal = 'Auto',
                       lcharg = False,
                       lwave = False,
                       lorbit = 0,
                       rwigs = [1.5],
                       )

### Get energy ###
atoms.set_calculator(vaspset)
e_atoms = atoms.get_potential_energy()
        
print('{0}, {1}'.format(name ,e_atoms))
