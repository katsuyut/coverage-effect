import numpy as np
import sys, os, random, itertools, warnings, math, copy
from ase import Atoms, Atom
from ase.calculators.emt import EMT
from ase.calculators.vasp import Vasp, Vasp2
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.constraints import FixAtoms
from ase.eos import EquationOfState
from ase.io import read, write
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.optimize import QuasiNewton

databasepath = '/home/katsuyut/research/coverage-effect/database/'
initpath = '/home/katsuyut/research/coverage-effect/init/'
zvalpath = '/home/katsuyut/research/coverage-effect/zval.txt'


def getnbands(atoms, f=0.5):
    file = open(zvalpath, 'r')
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
    c = 30
    cell = atoms.get_cell()
    kpts = []

    for i in range(3):
        l = np.sqrt(np.square(cell[i][0]) + np.square(cell[i][1]) + np.square(cell[i][2]))
        if int(c/l)==0:
            kpts.append(1)
        else:
            kpts.append(int(c/l))

    return kpts


def getdefaultvasptags(xc = 'RPBE'):
    """
    Default is same as used in GASpy (xc=RPBE)
    If xc is specified, a different set of tags is returned
    Available xcs are :RPBE, RPBE-D2, vdW-DF(revPBE-DF), optB88-vdW, vdW-DF2 (rPW86-vdw), BEEF-vdw

    reference
    https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html
    """
    if xc == 'RPBE':
        tagdict = {
            'xc' : 'RPBE',
            'pp' : 'PBE',
            'ncore' : 4,
            'encut' : 350,
            'nsw' : 200,
            # 'kpts' : None,
            'ibrion' : 2,
            'isif' : 0,
            'ediffg' : -3.00e-02,
            'isym' : 0,
            'symprec' : 1.00e-10,
            'lreal' : 'Auto',
            'lcharg' : False,
            'lwave' : False,
            'ivdw' : 0,
            'lasph' : False,
        }
    elif xc == 'RPBE-D2':
        tagdict = {
            'xc' : 'RPBE-D2',
            'pp' : 'PBE',
            'ncore' : 4,
            'encut' : 350,
            'nsw' : 200,
            # 'kpts' : None,
            'ibrion' : 2,
            'isif' : 0,
            'ediffg' : -3.00e-02,
            'isym' : 0,
            'symprec' : 1.00e-10,
            'lreal' : 'Auto',
            'lcharg' : False,
            'lwave' : False,
            'ivdw' : 1,
            'lasph' : False,
        }
    elif xc == 'vdW-DF':
        tagdict = {
            'xc' : 'vdW-DF',
            'pp' : 'PBE',
            'ncore' : 4,
            'encut' : 350,
            'nsw' : 200,
            # 'kpts' : None,
            'ibrion' : 2,
            'isif' : 0,
            'ediffg' : -3.00e-02,
            'isym' : 0,
            'symprec' : 1.00e-10,
            'lreal' : 'Auto',
            'lcharg' : False,
            'lwave' : False,
            'ivdw' : 0,
            'lasph' : True,
        }
    elif xc == 'optB88-vdW':
        tagdict = {
            'xc' : 'optB88-vdW',
            'pp' : 'PBE',
            'ncore' : 4,
            'encut' : 350,
            'nsw' : 200,
            # 'kpts' : None,
            'ibrion' : 2,
            'isif' : 0,
            'ediffg' : -3.00e-02,
            'isym' : 0,
            'symprec' : 1.00e-10,
            'lreal' : 'Auto',
            'lcharg' : False,
            'lwave' : False,
            'ivdw' : 0,
            'lasph' : True,
        }
    elif xc == 'vdW-DF2':
        tagdict = {
            'xc' : 'vdW-DF2',
            'pp' : 'PBE',
            'ncore' : 4,
            'encut' : 350,
            'nsw' : 200,
            # 'kpts' : None,
            'ibrion' : 2,
            'isif' : 0,
            'ediffg' : -3.00e-02,
            'isym' : 0,
            'symprec' : 1.00e-10,
            'lreal' : 'Auto',
            'lcharg' : False,
            'lwave' : False,
            'ivdw' : 0,
            'lasph' : True,
        }    
    elif xc == 'BEEF-vdW':
        tagdict = {
            'xc' : 'BEEF-vdW',
            'pp' : 'PBE',
            'ncore' : 4,
            'encut' : 350,
            'nsw' : 200,
            # 'kpts' : None,
            'ibrion' : 2,
            'isif' : 0,
            'ediffg' : -3.00e-02,
            'isym' : 0,
            'symprec' : 1.00e-10,
            'lreal' : 'Auto',
            'lcharg' : False,
            'lwave' : False,
            'ivdw' : 0,
            'lasph' : True,
        }
    else:
        print('No default tags set found')

    return tagdict


def getenergy(atoms, name, vasptags, env):
    trajpath = databasepath + str(name) + '.traj'
    trajpath_all = databasepath + str(name) + '_all.traj'

    if env == 'local':
        atoms.set_calculator(EMT())
        
        dyn = QuasiNewton(atoms, trajectory=trajpath)
        dyn.run(fmax=0.05)
    elif env == 'spacom':
        atoms.set_calculator(vasptags)
        
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