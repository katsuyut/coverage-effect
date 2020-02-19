import numpy as np
import sys
from MAUtil import *
from MACalc import *

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
    )                                 


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
            atoms.set_calculator(vapstags)

        try:
            atoms.get_potential_energy()
            ene = atoms.get_potential_energy()
            energies.append(i,j,ene)

        except:
            print('Error while calculating bulk energy!')
        
for i in range(len(energylist)):
    print(energylist[i])
