########### used until 6/1/2019 ##########

import numpy as np
import sys, os, random, itertools, warnings, math
import matplotlib.pyplot as plt
from ase import Atoms, Atom
from ase.build import fcc111, fcc100, add_adsorbate, rotate
from ase.calculators.emt import EMT
from ase.calculators.vasp import Vasp, Vasp2
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.constraints import FixAtoms
from ase.eos import EquationOfState
from ase.geometry import find_mic
from ase.io import read, write
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.lattice.cubic import FaceCenteredCubic
from ase.optimize import QuasiNewton
from ase.visualize import view
from scipy.spatial.qhull import QhullError
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder

def getnbands(atoms, f):
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

        
def query(name, env='spacom'):
    path = '/home/katsuyut/database/' + name
    try:
        traj = Trajectory(path)
        atoms = read(path)
        if env == 'local':
            view(traj)
        energy = atoms.get_potential_energy()
        return atoms, energy
    except IOError as e:
        return 'No file'
    except:
        return 'No energy'


def init_query(name, env='spacom'):
    path = '/home/katsuyut/init/' + name
    try:
        atoms = read(path)
        if env == 'local':
            view(atoms)
        return atoms
    except IOError as e:
        return 'No file'


def getallene():
    files = os.listdir('/home/katsuyut/database/')

    for filename in files:
        if '.traj' in filename:
            if not 'all' in filename:
                path = '/home/katsuyut/database/' + filename
                atoms = read(path)
                try:
                    energy = atoms.get_potential_energy()
                    print('{0}, {1}'.format(filename, energy))
                except:
                    print('No energy')
                

def getmoleculeenergy(atoms, name, env):
    nb = getnbands(atoms, 2)
    calcmol = Vasp(
            xc = 'PBE',
            gga = 'RP',
            ncore = 4,
            encut = 500,
            nsw = 200,
            ibrion = 2,
            isif = 2,
            ismear = 1,
            sigma = 0.4,
            nbands = nb,
            lcharg = False,
            lwave = False,
            )
    
    trajpath = '/home/katsuyut/database/' + str(name) + '.traj'
#    trajpath_ = '/home/katsuyut/database/' + str(name) + '_.traj'
    trajpath_all = '/home/katsuyut/database/' + str(name) + '_all.traj'

    if env == 'local':
        atoms.set_calculator(EMT())
        
        dyn = QuasiNewton(atoms, trajectory=trajpath)
        dyn.run(fmax=0.05)
    elif env == 'spacom':
        atoms.set_calculator(calcmol)
    
    try:
        e_eqmol = atoms.get_potential_energy()
        atoms.write(trajpath)
    except:
        print('Error while calculating molecue energy!')    

    if env == 'spacom':
        atomslist = []
        for atoms in read('vasprun.xml', ':'):
            catoms = atoms.copy()
            catoms = catoms[calcmol.resort]
            catoms.set_calculator(SPC(catoms,
                                      energy=atoms.get_potential_energy(),
                                      forces=atoms.get_forces()[calcmol.resort]))
            atomslist += [catoms]

#        # Get the final trajectory
#        finalimage = atoms

        # Write a traj file for the optimization
        tj = TrajectoryWriter(trajpath_all, 'a')
        for atoms in atomslist:
            print('writing trajectory file!')
            print(atoms)
            tj.write(atoms)
        tj.close()

#        # Write the final structure
#        finalimage.write(trajpath_)
    
    return e_eqmol
                    

def getslabenergy(atoms, name, cell, depth, env):
    cell=atoms.get_cell()
    x=np.sqrt(np.square(cell[0][0])+np.square(cell[0][1])+np.square(cell[0][2]))
    y=np.sqrt(np.square(cell[1][0])+np.square(cell[1][1])+np.square(cell[1][2]))
    z=np.sqrt(np.square(cell[2][0])+np.square(cell[2][1])+np.square(cell[2][2]))
    kpts1=int(30/x)
    kpts2=int(30/y)
    kpts3=int(30/z)
    
    nb = getnbands(atoms, 2)

    calcslab = Vasp(
                xc = 'PBE',
                gga = 'RP',
                ncore = 4,
                encut = 500,
                nsw = 200,
                kpts = [kpts1, kpts2, kpts3],
                ibrion = 2,
                isif = 2,
                ismear = 1,
                sigma = 0.4,
                nbands = nb,
                lcharg = False,
                lwave = False,
                )
    
    trajpath = '/home/katsuyut/database/' + str(name) + '.traj'
#    trajpath_ = '/home/katsuyut/database/' + str(name) + '_.traj'
    trajpath_all = '/home/katsuyut/database/' + str(name) + '_all.traj'

    if env == 'local':
        atoms.set_calculator(EMT())
        
        dyn = QuasiNewton(atoms, trajectory=trajpath)
        dyn.run(fmax=0.05)
    elif env == 'spacom':
        atoms.set_calculator(calcslab)

    constraint = FixAtoms(mask=[atom.tag >= depth for atom in atoms])
    atoms.set_constraint(constraint)
        
    
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
            catoms = catoms[calcslab.resort]
            catoms.set_calculator(SPC(catoms,
                                      energy=atoms.get_potential_energy(),
                                      forces=atoms.get_forces()[calcslab.resort]))
            atomslist += [catoms]

#        # Get the final trajectory
#        finalimage = atoms

        # Write a traj file for the optimization
        tj = TrajectoryWriter(trajpath_all, 'a')
        for atoms in atomslist:
            print('writing trajectory file!')
            print(atoms)
            tj.write(atoms)
        tj.close()

#        # Write the final structure
#        finalimage.write(trajpath_)
    
    return e_atoms


def getfccLC(surface, init, env='spacom'):
    a = init # approximate lattice constant
    b = a / 2
    volumes = []
    energies = []
    cells = []
    
    atoms = Atoms(surface, 
                  cell=[(0, b, b), (b, 0, b), (b, b, 0)],
                  pbc=1)
    cell = atoms.get_cell()
    
    for x in np.linspace(0.95, 1.05, 5):
        atoms.set_cell(cell * x, scale_atoms=True)

        x=np.sqrt(np.square(cell[0][0])+np.square(cell[0][1])+np.square(cell[0][2]))
        y=np.sqrt(np.square(cell[1][0])+np.square(cell[1][1])+np.square(cell[1][2]))
        z=np.sqrt(np.square(cell[2][0])+np.square(cell[2][1])+np.square(cell[2][2]))
        kpts1=int(30/x)
        kpts2=int(30/y)
        kpts3=int(30/z)

        nb = getnbands(atoms, 2)

        calc = Vasp(
                    xc = 'PBE',
                    gga = 'RP',
                    ncore = 4,
                    encut = 500,
                    nsw = 200,
                    kpts = [kpts1, kpts2, kpts3],
                    ibrion = 2,
                    isif = 2,
                    ismear = 1,
                    sigma = 0.4,
                    nbands = nb,
                    lcharg = False,
                    lwave = False,
                    )
        
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
            print('Error while calculating molecue energy!')
            
    eos = EquationOfState(volumes, energies)
    v0, e0, B = eos.fit()
    latticeconstant = (v0/2.0)**(1.0/3.0)*2.0

    return latticeconstant


def getadsites(atoms, symm_reduce):
    '''
    Given surface, return sites dictionary.
    '''
    if not symm_reduce:
        struct = AseAtomsAdaptor.get_structure(atoms)
        sites_dict = AdsorbateSiteFinder(struct).find_adsorption_sites(put_inside=True, symm_reduce=0)
        return sites_dict
    
    elif symm_reduce:
        struct = AseAtomsAdaptor.get_structure(atoms)
        sites_dict = AdsorbateSiteFinder(struct).find_adsorption_sites(put_inside=True)
        return sites_dict
        

def getalladsitecomb(sites):
    '''
    Given sites dictionary, return all possible combinations of positions.
    '''
    positions = []
    
    allsites = np.append(sites['ontop'], sites['bridge'], axis=0)
    try:
        allsites = np.append(allsites, sites['hollow'], axis=0)
    except:
        print('No hollow')
        
    for i in range(len(allsites)):
        for j in itertools.combinations(allsites, i):
            positions.append(i)
    
    return positions


def getadsitecomb(sites, num):
    '''
    Given sites dictionary and adsorbed atom number, return all possible combinations of positions.
    '''
    positions = []
    
    allsites = np.append(sites['ontop'], sites['bridge'], axis=0)
    try:
        allsites = np.append(allsites, sites['hollow'], axis=0)
    except:
        print('No hollow')
        
    for i in itertools.combinations(allsites, num):
        positions.append(i)
    
    return positions


def checksitetype(comb, sites):
    '''
    Given one combination of adsorption sites, return number of type of the sites.
    '''
    ontop = 0
    bridge = 0
    hollow = 0
    
    for i in range(len(comb)):
        ncomb = np.round(comb[i], 4)

        o = (ncomb == np.round(sites['ontop'], 4))
        for j in o:
            if ([True, True, True] == j).all():
                ontop += 1

        b = (ncomb == np.round(sites['bridge'], 4))
        for j in b:
            if ([True, True, True] == j).all():
                bridge += 1

        if sites['hollow'] != []:
            h = (ncomb == np.round(sites['hollow'], 4))
            for j in h:
                if ([True, True, True] == j).all():
                    hollow += 1

    return [ontop, bridge, hollow]


def getmindist(comb, sites, cell):
    '''
    Given one combination of adsorption sites, return minimum distance.
    '''
    compcomb = []
    for i in range(len(comb)):
        compcomb.append(comb[i])
        compcomb.append(comb[i] + cell[0])
        compcomb.append(comb[i] + cell[1])
        compcomb.append(comb[i] + cell[0] + cell[1])
    
    distlis = []
    for i in range(len(compcomb)):
        for j in range(i+1, len(compcomb)):
            dist = ((compcomb[i][0] - compcomb[j][0])**2 
                    + (compcomb[i][1] - compcomb[j][1])**2)**0.5
            distlis.append(dist)
        mindist = np.min(distlis)
    
    return mindist


def getmindistlist(combs, sites, cell):
    '''
    combs: array of positions, or double array of positions
    '''
    mindistlis  = []
    for i in range(len(combs)):
        if type(combs[0][0][0]) == np.float64:
            mindist = getmindist(combs[i], sites, cell)
            mindistlis.append(mindist)
        
        else:
            for j in range(len(combs[i])):
                if combs[i][j] != 0:
                    mindist = getmindist(combs[i][j], sites, cell)
                    mindistlis.append(mindist)

    return mindistlis
    

def getcalccomb(sites, num, cell):
    '''
    calccomb[i][j]
    i: unique comb
    j: 0:1.0-1.5, 1:1.5-2.0, 2:2.5-3.0, 3:3.0-3.5, 4:3.5-4.0, 5:4.0-4.5, 6:4.5-5.0 
    '''
    positions = getadsitecomb(sites, num)
    listoftypes = []
    results = []

    for i in range(len(positions)):
        listoftypes.append(checksitetype(positions[i], sites))
    uniquecomb = [list(x) for x in set(tuple(x) for x in listoftypes)]

    mindistlist = getmindistlist(positions, sites, cell)
    maxmindist = int(np.ceil(max(mindistlist)/0.5)-3)
    calccombs = [[0 for i in range(maxmindist + 1)] for i in range(len(uniquecomb))]
    
    for i in range(len(positions)):
        comb = checksitetype(positions[i], sites)
        dist = getmindist(positions[i], sites, cell) 
        
        ind0 = uniquecomb.index(comb)
        ind1 = int(np.ceil(dist/0.5)-3)

        if dist > 1:
            if calccombs[ind0][ind1] == 0:
                calccombs[ind0][ind1] = positions[i]
                results.append([ind0, ind1, dist])

    return calccombs, uniquecomb, results


def getcalcatoms(atoms, molecule, h, calccomb):
    if type(calccomb) != int:
        for i in range(len(calccomb)):
            add_adsorbate(atoms, molecule, h, calccomb[i][:2])