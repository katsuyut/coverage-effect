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
                symprec = 1e-10,
                lreal = 'Auto',
                lcharg = False,
                lwave = False,
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
            atoms.set_calculator(calc)

        try:
            atoms.get_potential_energy()
            ene = atoms.get_potential_energy()
            energies.append(i,j,ene)

        except:
            print('Error while calculating bulk energy!')
        
for i in range(len(energylist)):
    print(energylist[i])
