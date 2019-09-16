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
    
if 3 in set(atoms.get_tags()): # set constraint only on surface calc
    constraint = FixAtoms(mask=[atom.tag >= fixlayer for atom in atoms])
    atoms.set_constraint(constraint)
    
    
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
            lreal = 'Auto',                       
            lcharg = False,                       
            lwave = False,                        
            )                                     

    
### Get energy ###
print(query(name, env)[1])
if type(query(name, env)[1]) != float:
    e_atoms = getenergy(atoms, name[0:-5], vaspset, env)
        
print('{0}, {1}'.format(name ,e_atoms))
f = open('result.txt', 'a')
f.write('{0}, {1}'.format(name ,e_atoms))
f.close()
