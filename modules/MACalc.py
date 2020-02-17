import numpy as np
import sys, os, random, itertools, warnings, math, copy
from ase import Atoms, Atom
from ase.calculators.emt import EMT
from ase.calculators.vasp import Vasp, Vasp2
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.eos import EquationOfState
from ase.io import read, write
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.optimize import QuasiNewton


def getnbands(atoms, f=0.5):
    file = open('/home/katsuyut/module/zval.txt', 'r')
    combs = file.read().split('\n')
    electrondict = {}
    for comb in combs:
        kye = comb.split('\t')[0]
        val = comb.split('\t')[1]
        electrondict[kye] = float(val)
        
    species = set(atoms.symbols)

    speciesdict = {}
    for i in species:
        bools = (i == atoms.symbols)
        speciesdict[i] = list(bools).count(True)
    
    keys = speciesdict.keys()
    vals = speciesdict.values()    
    
    nelectrons = 0
    for key in keys:
        nelectrons += speciesdict[key]*electrondict[key]

    nbands = int(nelectrons/2 + len(atoms)*f)
    
    return nbands


def getkpts(atoms):
    cell = atoms.get_cell()
    x = np.sqrt(np.square(cell[0][0]) + np.square(cell[0][1]) + np.square(cell[0][2]))
    y = np.sqrt(np.square(cell[1][0]) + np.square(cell[1][1]) + np.square(cell[1][2]))
    z = np.sqrt(np.square(cell[2][0]) + np.square(cell[2][1]) + np.square(cell[2][2]))
    kpts1 = int(30/x)
    kpts2 = int(30/y)
    kpts3 = int(30/z)
    
    return [kpts1, kpts2, kpts3]
                    

def getenergy(atoms, name, vaspset, env):
    calc = vaspset
    
    trajpath = '/home/katsuyut/database/' + str(name) + '.traj'
    trajpath_all = '/home/katsuyut/database/' + str(name) + '_all.traj'

    if env == 'local':
        atoms.set_calculator(EMT())
        
        dyn = QuasiNewton(atoms, trajectory=trajpath)
        dyn.run(fmax=0.05)
    elif env == 'spacom':
        atoms.set_calculator(calc)
        
    try:
        e_atoms = atoms.get_potential_energy()
        atoms.write(trajpath)
    except:
        print('Error while calculating {0} energy!'.format(name))
        return None

    if env == 'spacom':
        atomslist = []
        for atoms in read('vasprun.xml', ':'):
            catoms = atoms.copy()
            catoms = catoms[calc.resort]
            catoms.set_calculator(SPC(catoms,
                                      energy=atoms.get_potential_energy(),
                                      forces=atoms.get_forces()[calc.resort]))
            atomslist += [catoms]

        # Write a traj file for the optimization
        tj = TrajectoryWriter(trajpath_all, 'a')
        for atoms in atomslist:
            print('writing trajectory file!')
            print(atoms)
            tj.write(atoms)
        tj.close()

    return e_atoms


def getLC(atoms, vaspset, env='spacom'):
    volumes = []
    energies = []
    cells = []

    cell = atoms.get_cell()
    
    for x in np.linspace(0.95, 1.05, 5):
        atoms.set_cell(cell * x, scale_atoms=True)

        calc = vaspset
        
        if env == 'local':
            atoms.set_calculator(EMT())

            dyn = QuasiNewton(atoms)
            dyn.run(fmax=0.05)
        elif env == 'spacom':
            atoms.set_calculator(calc)

        try:
            atoms.get_potential_energy()
            volumes.append(atoms.get_volume())
            energies.append(atoms.get_potential_energy())
            cells.append(atoms.get_cell())

        except:
            print('Error while calculating bulk energy!')
            
    eos = EquationOfState(volumes, energies)
    v0, e0, B = eos.fit()
    latticeconstant = (v0/2.0)**(1.0/3.0)*2.0

    return latticeconstant