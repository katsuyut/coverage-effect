import numpy as np
import sys
import os
import random
import itertools
import warnings
import math
import copy
import re
from ase import Atoms, Atom
from ase.calculators.emt import EMT
from ase.calculators.vasp import Vasp, Vasp2
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.constraints import FixAtoms
from ase.eos import EquationOfState
from ase.optimize import QuasiNewton
from ase.io import read, write
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.build import bulk, add_adsorbate, rotate
from ase.build import fcc100, fcc111, fcc110, fcc211, bcc100, bcc111, bcc110, hcp0001
from MAUtil import *


databasepath = os.environ['DATABASEPATH']
initpath = os.environ['INITPATH']
mppath = os.environ['MPPATH']
zvalpath = os.environ['ZVALPATH']


def get_nbands(atoms, f=0.5):
    with open(zvalpath, 'r') as f:
        combs = f.read().split('\n')
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


def get_kpts(atoms):
    c = 30
    cell = atoms.get_cell()
    kpts = []

    for i in range(3):
        l = np.sqrt(np.square(cell[i][0]) +
                    np.square(cell[i][1]) + np.square(cell[i][2]))
        if int(c/l) == 0:
            kpts.append(1)
        else:
            kpts.append(int(c/l))

    return kpts


def get_default_vasp_tags(xc='RPBE'):
    """
    Default is same as used in GASpy (xc=RPBE)
    If xc is specified, a different set of tags is returned
    Available xcs are :RPBE, RPBE-D2, vdW-DF(revPBE-DF), optB88-vdW, vdW-DF2(rPW86-vdw), BEEF-vdw

    reference
    https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html
    """
    commontags = {
        'pp': 'PBE',
        'ncore': 4,
        'encut': 350,
        'nsw': 200,
        # 'kpts' : None,
        'ibrion': 2,  # 2 is good for calcs with bad initial positions
        'isif': 0,
        'ediffg': -3.00e-02,
        'isym': 0,
        'symprec': 1.00e-10,
        'lreal': 'Auto',
        'lcharg': False,
        'lwave': False,
    }

    if xc == 'RPBE':
        tagdict = {
            **commontags,
            'xc': 'RPBE',
            'ivdw': 0,
            'vdw_s6': 0.75,
            'lasph': False,
        }
    elif xc == 'RPBE-D2':
        tagdict = {
            'xc': 'RPBE',
            **commontags,
            'ivdw': 1,
            'vdw_s6': 0.75,
            'lasph': False,
        }
    elif xc == 'vdW-DF':
        tagdict = {
            'xc': 'vdW-DF',
            **commontags,
            'ivdw': 0,
            'vdw_s6': 0.75,
            'lasph': True,
        }
    elif xc == 'optB88-vdW':
        tagdict = {
            'xc': 'optB88-vdW',
            **commontags,
            'ivdw': 0,
            'vdw_s6': 0.75,
            'lasph': True,
        }
    elif xc == 'vdW-DF2':
        tagdict = {
            'xc': 'vdW-DF2',
            **commontags,
            'ivdw': 0,
            'vdw_s6': 0.75,
            'lasph': True,
        }
    elif xc == 'BEEF-vdW':
        tagdict = {
            'xc': 'BEEF-vdW',
            **commontags,
            'ivdw': 0,
            'vdw_s6': 0.75,
            'lasph': True,
        }
    else:
        print('No default tags set found')

    return tagdict


def set_vasp_tags(tagdict):
    vasptags = Vasp(
        xc=tagdict['xc'],
        pp=tagdict['pp'],
        ncore=tagdict['ncore'],
        encut=tagdict['encut'],
        nsw=tagdict['nsw'],
        kpts=tagdict['kpts'],
        ibrion=tagdict['ibrion'],
        isif=tagdict['isif'],
        ediffg=tagdict['ediffg'],
        isym=tagdict['isym'],
        lreal=tagdict['lreal'],
        lcharg=tagdict['lcharg'],
        lwave=tagdict['lwave'],
        ivdw=tagdict['ivdw'],
        vdw_s6=tagdict['vdw_s6'],
        lasph=tagdict['lasph'],
    )

    return vasptags


def get_energy(atoms, name, vasptags, env):
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
            catoms = catoms[vasptags.resort]
            catoms.set_calculator(SPC(catoms,
                                      energy=atoms.get_potential_energy(),
                                      forces=atoms.get_forces()[vasptags.resort]))
            atomslist += [catoms]

        # Write a traj file for the optimization
        tj = TrajectoryWriter(trajpath_all, 'a')
        for atoms in atomslist:
            print('writing trajectory file!')
            print(atoms)
            tj.write(atoms)
        tj.close()

    return e_atoms


def get_equiblium_bulk(mpname, xc='RPBE', env='spacom'):
    '''
    Now can only deal with cubic, hexagonal and trigonal systems
    OK:cubic trigonal tetragonal (This does not optimize angle for trigonal)
    NG:others (orthohombic hexagona triclinic monoclinic)
    '''
    bulk, formula, crystal_system = mp_query(mpname)
    eps = 0.03

    cell = bulk.get_cell()
    v = bulk.get_volume()

    kpoints = get_kpts(bulk)
    tagdict = get_default_vasp_tags(xc)
    tagdict['kpts'] = kpoints
    vasptags = set_vasp_tags(tagdict)

    volumes = []
    energies = []

    # cubic and trigonal system
    if crystal_system == 'cubic' or crystal_system == 'trigonal':
        for x in np.linspace(1-eps, 1+eps, 5):
            bulk.set_cell(cell * x, scale_atoms=True)

            if env == 'local':
                bulk.set_calculator(EMT())

                dyn = QuasiNewton(bulk)
                dyn.run(fmax=0.05)
            elif env == 'spacom':
                bulk.set_calculator(vasptags)

            try:
                energies.append(bulk.get_potential_energy())
                volumes.append(bulk.get_volume())

            except:
                print('Error while calculating bulk energy!')

        eos = EquationOfState(volumes, energies)
        v0, e0, B = eos.fit()
        ratio = (v0/v)**(1/3)
        newcell = cell * ratio
        a = newcell[0][0] * 2**0.5

        with open('result.txt', 'a') as f:
            f.write('{0}, {1}, {2}\n'.format(
                formula, xc, str(a)))

        bulk.set_cell(newcell, scale_atoms=True)
        trajpath = initpath + formula + '_' + xc + '.traj'
        bulk.write(trajpath)

    # hexagonal system
    elif crystal_system == 'hexagonal':
        a0 = cell[0][0]
        c0 = cell[2][2]
        a = []
        c = []
        for x in np.linspace(1-eps, 1+eps, 3):
            for y in np.linspace(1-eps, 1+eps, 3):
                calccell = copy.deepcopy(cell)
                calccell[0][0] = a0 * x
                calccell[1][0] = a0 * x * np.cos(np.pi*2/3)
                calccell[1][1] = a0 * x * np.sin(np.pi*2/3)
                calccell[2][2] = c0 * y
                a.append(calccell[0][0])
                c.append(calccell[2][2])

                bulk.set_cell(calccell, scale_atoms=True)

                if env == 'local':
                    bulk.set_calculator(EMT())

                    dyn = QuasiNewton(bulk)
                    dyn.run(fmax=0.05)
                elif env == 'spacom':
                    bulk.set_calculator(vasptags)

                try:
                    energies.append(bulk.get_potential_energy())
                    volumes.append(bulk.get_volume())

                except:
                    print('Error while calculating bulk energy!')

        a = np.array(a)
        c = np.array(c)
        functions = np.array([a**0, a, c, a**2, a * c, c**2])
        p = np.linalg.lstsq(functions.T, energies, rcond=-1)[0]

        p0 = p[0]
        p1 = p[1:3]
        p2 = np.array([(2 * p[3], p[4]),
                       (p[4], 2 * p[5])])
        a, c = np.linalg.solve(p2.T, -p1)

        with open('result.txt', 'a') as f:
            f.write('{0}, {1}, {2}, {3}\n'.format(
                formula, xc, str(a), str(c)))

        newcell = copy.deepcopy(cell)
        newcell[0][0] = a
        newcell[1][0] = a * np.cos(np.pi*2/3)
        newcell[1][1] = a * np.sin(np.pi*2/3)
        newcell[2][2] = c
        bulk.set_cell(newcell, scale_atoms=True)
        trajpath = initpath + formula + '_' + xc + '.traj'
        bulk.write(trajpath)

def custodian_correct_alternative():
    '''
    This is alternative function to correct of custodian errorhander.
    Custodian errorhandler's correct does not make INCAR file with EDIFFG for 
    some reason and calculations does not converge correctly because of that.
    This only change ibrion = 2 to ibrion = 1.
    '''
    contents = []
    with open('INCAR','r') as f:
        data = 1
        while data:
            data = f.readline()
            if 'IBRION = 2' in data:
                data = ' IBRION = 1\n'
            contents.append(data)

    with open('INCAR','w') as f:
        data = ''.join(contents)
        f.write(data)
