import numpy as np
import sys, copy, time
from MAUtil import *
from MACalc import *

start = time.time()

env='local'

a = 2.91
b = 4.12

def getatoms(a, b):    
    x = a/2
    y = b/2
    z = b/2

    pos = np.array([[0,y,0],[0,0,z],[x,0,0],[x,y,z]])

    atoms = Atoms(symbols = 'Ga2Ni2',
                       positions= pos,
                       cell = ktms.np.array([[x*2,0,0],
                                             [0,y*2,0],
                                             [0,0,z*2]]),
                       pbc=1
                       )
    return atoms

kpoints = getkpts(atoms)
nb = getnbands(atoms, 2) # default value is 0.5
tagdict = getdefaultvasptags('RPBE')
tagdict['kpts'] = kpoints
vasptags = setvasptags(tagdict)


energies = []

testrange = np.linspace(0.95, 1.05, 5)

for i in a*testrange:
    for j in b*testrange:
        atoms = getatoms(i,j)
        
        if env == 'local':
            atoms.set_calculator(EMT())

            dyn = QuasiNewton(atoms)
            dyn.run(fmax=0.05)
        elif env == 'spacom':
            atoms.set_calculator(vasptags)

        try:
            atoms.get_potential_energy()
            ene = atoms.get_potential_energy()
            energies.append(i,j,ene)

        except:
            print('Error while calculating bulk energy!')
        
for i in range(len(energylist)):
    print(energylist[i])

print((time.time() - start)/60, 'min')