import numpy as np
import sys
import os
import random
import itertools
import warnings
import math
import copy
from ase import Atoms, Atom
from ase.build import fcc100, fcc111, fcc110, bcc100, bcc111, bcc110, add_adsorbate, rotate
from ase.constraints import FixAtoms
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from MAUtil import *


def getNiGa(a):
    b = a * 2**0.5
    x = a/2
    y = b/2
    z = b/2

    pos = np.array([[0, y, 0], [0, 0, z], [x, 0, 0], [x, y, z]])

    atoms = Atoms(symbols='Ga2Ni2',  # Ga2Ni2
                  positions=pos,
                  cell=np.array([[x*2, 0, 0],
                                 [0, y*2, 0],
                                 [0, 0, z*2]]),
                  pbc=1
                  )

    return atoms


def getCoPt3(a):
    x = a/2
    y = a/2
    z = a/2

    pos = np.array([[0, 0, 0], [x, y, 0], [0, y, z], [x, 0, z]])

    atoms = Atoms(symbols='CoPt3',  # Ga2Ni2
                  positions=pos,
                  cell=np.array([[x*2, 0, 0],
                                 [0, y*2, 0],
                                 [0, 0, z*2]]),
                  pbc=1
                  )

    return atoms


def get_all_elements(atoms):
    atoms
    elements = []
    for ele in atoms.symbols:
        elements.append(ele)

    return elements


def remove_molecule(atoms, molecule):
    cpatoms = copy.deepcopy(atoms)
    poslis = []
    for i in reversed(range(len(cpatoms))):
        if cpatoms[i].symbol == molecule[0]:
            poslis.append(cpatoms[i].position)

        for j in molecule:
            if cpatoms[i].symbol == j:
                cpatoms.pop(i)
                break
    return cpatoms, np.array(poslis)


def create_site_group(baresurface):
    redbareadsites = getadsites(baresurface, True)
    redbareadsites = np.array(redbareadsites['all'])

    allbareadsites = getadsites(baresurface, False)
    allbareadsites = np.array(allbareadsites['all'])

    group = [[list(item)] for item in redbareadsites]

    for i in range(len(allbareadsites)):
        flag = 0
        for j in range(len(redbareadsites)):
            if np.allclose(allbareadsites[i], redbareadsites[j]):
                flag = 1
        if flag == 1:
            continue

        for j in range(len(redbareadsites)):
            if check_if_same(baresurface, np.array([allbareadsites[i]]), np.array([np.array([redbareadsites[j]])])):
                group[j].append(list(allbareadsites[i]))
                break

    return group


def getadsites(atoms, symm_reduce):
    '''
    Given surface, return sites dictionary.
    '''
    if not symm_reduce:
        struct = AseAtomsAdaptor.get_structure(atoms)
        sites_dict = AdsorbateSiteFinder(struct).find_adsorption_sites(
            put_inside=True, symm_reduce=0)
        return sites_dict

    elif symm_reduce:
        struct = AseAtomsAdaptor.get_structure(atoms)
        sites_dict = AdsorbateSiteFinder(
            struct).find_adsorption_sites(put_inside=True)
        return sites_dict


def check_if_same(baresurface, sites, sitesset):
    '''
    Check if in sitesset(3d-array, sets of sites) has same configuration of symmetrical operated sites(2d-array, a sets of sites).
    Return True if found.
    '''
    struct = AseAtomsAdaptor.get_structure(baresurface)
    surf_sg = SpacegroupAnalyzer(struct, 0.01)
    symm_ops = surf_sg.get_symmetry_operations()

    frsitesset = []
    for item in sitesset:  # need this part to avoid giving empty item to the function
        frsitesset.append(struct.lattice.get_fractional_coords(item))

    for op in symm_ops:
        froperatedsites = []
        for i in range(len(sites)):
            frsites = struct.lattice.get_fractional_coords(sites[i])
            froperatedsites.append(list(op.operate(frsites)))
        froperatedsites.sort()
        modfropsites = modpos(froperatedsites)

        for used in frsitesset:
            used = [list(item) for item in used]
            if len(modfropsites) == len(used):
                if np.allclose(sorted(modfropsites), sorted(used), atol=0.01):
                    #                     print('Symmetrically same structure found!')
                    return True


def modpos(sites):
    """
    sites must be 2D array
    """
    modsites = []
    for i in range(len(sites)):
        modsite = []
        for j in range(len(sites[i]) - 1):
            if sites[i][j] < 0:
                sites[i][j] += np.ceil(abs(sites[i][j]))
            elif sites[i][j] >= 1:
                sites[i][j] -= np.floor(sites[i][j])

            if abs(sites[i][j]) < 0.01 or abs(sites[i][j] - 1) < 0.01:
                sites[i][j] = 0
            modsite.append(sites[i][j])
        modsite.append(sites[i][2])
        modsites.append(modsite)
    return modsites


def getuniqueatoms(atoms, bareatoms, adsites, maxmole, mindist, initadsites, group, molecule):
    '''
    Given surface Atom object and all possible attaching site positions, create all possible unique attached Atom object.
    Can exclude by the maximum number of molecules or minimum distance. 

    return
    allatoms   :all possible Atom object [[[x,y,z]], [[a,b,c]], [[x,y,z],[a,b,c]]]
    allused    :attached sites for each Atom object
    mindistlis :minimum distance of molecule for each Atom object
    numdict    :dictionary, key=number of attached molecule, value=number of object created
    '''
    height = 1.85
    allatoms = []
    allused = []
    molenum = []
    usedadsites = []
    count = 0

    allused = copy.deepcopy(initadsites)
    allused = list(initadsites)

    def recursive(ratoms, rsites, rused, molecules, tmpused, count):
        molecules += 1

        if molecules > maxmole:
            return None

        for i in range(len(rsites)):
            nextatoms = copy.deepcopy(ratoms)
            nextused = copy.deepcopy(rused)
            if molecules == 1:
                tmpused = copy.deepcopy(allused)
#                 print('Used initialized!')

            add_adsorbate(nextatoms, molecule, height, rsites[i][:2])
            nextused.append(rsites[i])

            dist = getmindist(nextused, bareatoms.cell)
            if dist < mindist:
                print('Distance {0:.2f} is below {1}'.format(dist, mindist))
                continue

            sameflag = check_if_same(bareatoms, nextused, tmpused)
            if sameflag:
                continue

            allatoms.append(nextatoms)
            allused.append(nextused)
            tmpused.append(nextused)
            molenum.append(molecules)
            struct = AseAtomsAdaptor.get_structure(nextatoms)

            # print('{0}-------{1}-------'.format(molecules, nextatoms.symbols))
            if molecules == initmol+1:
                count += 1
                print('progress: {}/{}'.format(count, len(rsites)))

            try:
                nextsites = AdsorbateSiteFinder(struct).symm_reduce(adsites)
                indexlist = []
                for j in range(len(nextused)):
                    for k in range(len(nextsites)):
                        if np.allclose(nextused[j], nextsites[k]):
                            indexlist.append(k)

                for j in range(len(usedadsites)):
                    for k in range(len(nextsites)):
                        if np.allclose(usedadsites[j], nextsites[k]):
                            indexlist.append(k)

                indexlist.sort(reverse=True)
                for j in range(len(indexlist)):
                    nextsites.pop(indexlist[j])

                recursive(nextatoms, nextsites, nextused,
                          molecules, tmpused, count)

            except:
                print('Error!!')

            if molecules == 1:
                for j in range(len(group)):
                    if rsites[i] in group[j]:
                        for k in range(len(group[j])):
                            usedadsites.append(group[j][k])

    redadsites_ = getadsites(atoms, True)['all']
    redadsites = [list(i) for i in redadsites_]
    initmol = len(initadsites)

    recursive(atoms, redadsites, list(initadsites), initmol, allused, count)
    mindistlis = getmindistlist(allused, bareatoms.cell)

    numdict = {}
    for site in allused:
        if str(len(site)) not in numdict.keys():
            numdict[str(len(site))] = 1
        else:
            numdict[str(len(site))] += 1

    return allatoms, allused, mindistlis, numdict, molenum


def getmindist(comb, cell):
    '''
    Given one combination of adsorption sites(2d-array), return minimum distance.
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
    mindistlis = []
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


class make_adsorbed_surface():
    """
    Assuming that atoms and adsorbates are in init folder
    Name is a filename of trajectory file withought extention
    """

    def __init__(self, surfacename, adsorbatename, env='spacom'):
        self.surfacename = surfacename
        self.adsorbatename = adsorbatename
        self.initatoms = init_query(surfacename+'.traj', env)
        self.adsorbate = init_query(adsorbatename+'.traj', env)

    def make_surface(self, maxmole, mindist):
        adseles = get_all_elements(self.adsorbate)
        baresurface, initposlis = removemolecule(self.initatoms, adseles)
        # barestruct = AseAtomsAdaptor.get_structure(baresurface)
        bareadsites = getadsites(baresurface, False)
        allbareadsites = np.array(bareadsites['all'])
        sites0 = [list(i) for i in allbareadsites]

# group = creategroup(bareatoms, sites0)

# rused = []
# for i in reversed(range(len(sites0))):
#     for j in range(len(poslis)):
#         if np.allclose(sites0[i][:2], poslis[j][:2]):
#             tmp = sites0.pop(i)
#             rused.append(tmp)
# baresites, rused


# def getalladsitecomb(sites):
#     '''
#     Given sites dictionary, return all possible combinations of positions.
#     '''
#     positions = []

#     allsites = np.append(sites['ontop'], sites['bridge'], axis=0)
#     try:
#         allsites = np.append(allsites, sites['hollow'], axis=0)
#     except:
#         print('No hollow')

#     for i in range(len(allsites)):
#         for j in itertools.combinations(allsites, i):
#             positions.append(i)

#     return positions


# def getadsitecomb(sites, num):
#     '''
#     Given sites dictionary and adsorbed atom number, return all possible combinations of positions.
#     '''
#     positions = []

#     allsites = np.append(sites['ontop'], sites['bridge'], axis=0)
#     try:
#         allsites = np.append(allsites, sites['hollow'], axis=0)
#     except:
#         print('No hollow')

#     for i in itertools.combinations(allsites, num):
#         positions.append(i)

#     return positions


# def checksitetype(comb, sites):
#     '''
#     Given one combination of adsorption sites, return number of type of the sites.
#     '''
#     ontop = 0
#     bridge = 0
#     hollow = 0

#     for i in range(len(comb)):
#         ncomb = np.round(comb[i], 4)

#         o = (ncomb == np.round(sites['ontop'], 4))
#         for j in o:
#             if ([True, True, True] == j).all():
#                 ontop += 1

#         b = (ncomb == np.round(sites['bridge'], 4))
#         for j in b:
#             if ([True, True, True] == j).all():
#                 bridge += 1

#         if sites['hollow'] != []:
#             h = (ncomb == np.round(sites['hollow'], 4))
#             for j in h:
#                 if ([True, True, True] == j).all():
#                     hollow += 1

#     return [ontop, bridge, hollow]


# def getcalccomb(sites, num, cell):
#     '''
#     calccomb[i][j]
#     i: unique comb
#     j: 0:1.0-1.5, 1:1.5-2.0, 2:2.5-3.0, 3:3.0-3.5, 4:3.5-4.0, 5:4.0-4.5, 6:4.5-5.0 
#     '''
#     positions = getadsitecomb(sites, num)
#     listoftypes = []
#     results = []

#     for i in range(len(positions)):
#         listoftypes.append(checksitetype(positions[i], sites))
#     uniquecomb = [list(x) for x in set(tuple(x) for x in listoftypes)]

#     mindistlist = getmindistlist(positions, cell)
#     maxmindist = int(np.ceil(max(mindistlist)/0.5)-3)
#     calccombs = [[0 for i in range(maxmindist + 1)]
#                  for i in range(len(uniquecomb))]

#     for i in range(len(positions)):
#         comb = checksitetype(positions[i], sites)
#         dist = getmindist(positions[i], cell)

#         ind0 = uniquecomb.index(comb)
#         ind1 = int(np.ceil(dist/0.5)-3)

#         if dist > 1:
#             if calccombs[ind0][ind1] == 0:
#                 calccombs[ind0][ind1] = positions[i]
#                 results.append([ind0, ind1, dist])

#     return calccombs, uniquecomb, results


# def getcalcatoms(atoms, molecule, h, calccomb):
#     if type(calccomb) != int:
#         for i in range(len(calccomb)):
#             add_adsorbate(atoms, molecule, h, calccomb[i][:2])