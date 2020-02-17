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
atoms = init_query(name, env)
cell = atoms.cell

del atoms[[atom.index for atom in atoms if atom.tag!=0]]

### Set vasp ###    
kpoints = getkpts(atoms)    
nb = getnbands(atoms, 2) # default value is 0.5

vaspset = Vasp(                              
            xc = 'PBE',                           
            gga = 'RP',                           
            ncore = 4,                            
	    encut = 350,                          
            nsw = 200, # bond length to 1.148                            
            kpts = kpoints,                       
            ibrion = 2,                           
            isif = 0,                             
            ediffg = -3.00e-02,                   
            isym = 0,                             
            lreal = 'Auto',                       
            lcharg = False,                       
            lwave = False,                        
            )                                     

    
### Get energy ###
e_atoms = getenergy(atoms, name[0:-5]+'__', vaspset, env)
        
print('{0}, {1}'.format(name ,e_atoms))
f = open('result.txt', 'a')
f.write('{0}, {1}'.format(name ,e_atoms))
f.close()
