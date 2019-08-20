import numpy as np
import sys, os, random, itertools, warnings, math, copy
import matplotlib.pyplot as plt
from ase import Atoms, Atom
from ase.build import fcc100, fcc111, fcc110, bcc100, bcc111, bcc110, add_adsorbate, rotate
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
from pymatgen.util.coord import in_coord_list_pbc
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


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


def getmindist(comb, cell):
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


def getmindistlist(combs, cell):
    '''
    combs: array of positions, or double array of positions
    '''
    mindistlis  = []
    for i in range(len(combs)):
        if type(combs[0][0][0]) == np.float64:
            mindist = getmindist(combs[i], cell)
            mindistlis.append(mindist)
        
        else:
            for j in range(len(combs[i])):
                if combs[i][j] != 0:
                    mindist = getmindist(combs[i][j], cell)
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

    mindistlist = getmindistlist(positions, cell)
    maxmindist = int(np.ceil(max(mindistlist)/0.5)-3)
    calccombs = [[0 for i in range(maxmindist + 1)] for i in range(len(uniquecomb))]
    
    for i in range(len(positions)):
        comb = checksitetype(positions[i], sites)
        dist = getmindist(positions[i], cell) 
        
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

            
def getNiGa(a):
    b = a * 2**0.5
    x = a/2
    y = b/2
    z = b/2

    pos = np.array([[0,y,0],[0,0,z],[x,0,0],[x,y,z]])

    atoms = Atoms(symbols = 'Ga2Ni2', # Ga2Ni2
                       positions= pos,
                       cell = np.array([[x*2,0,0],
                                        [0,y*2,0],
                                        [0,0,z*2]]),
                       pbc=1
                       )
    
    return atoms


def settag(atoms):
    poslis = list(set(atoms.get_positions()[:,2]))
    poslis.sort()

    for i in range(len(atoms)):
        for j in range(len(poslis)):
            if atoms[i].position[2] == poslis[j]:
                atoms[i].tag = len(poslis) - j
    return atoms