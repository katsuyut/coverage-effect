import numpy as np
import sys
import os
import random
import itertools
import warnings
import math
import copy
import time
from ase import Atoms, Atom
from ase.build import fcc100, fcc111, fcc110, bcc100, bcc111, bcc110, add_adsorbate, rotate
from ase.constraints import FixAtoms
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from MAUtil import *

databasepath = '/home/katsuyut/research/coverage-effect/database/'
initpath = '/home/katsuyut/research/coverage-effect/init/'


def get_all_elements(atoms):
    atoms
    elements = []
    for ele in atoms.symbols:
        elements.append(ele)

    return elements


def remove_adsorbate(atoms, molecule):
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
    redbareadsites = get_adsorption_sites(baresurface, True)
    redbareadsites = np.array(redbareadsites['all'])

    allbareadsites = get_adsorption_sites(baresurface, False)
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


def get_adsorption_sites(atoms, symm_reduce):
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
    surf_sg = SpacegroupAnalyzer(struct, 0.1)
    symm_ops = surf_sg.get_symmetry_operations()

    modfrsitesset = []
    for item in sitesset:  # need this part to avoid giving empty item to the function
        frsites = struct.lattice.get_fractional_coords(item)
        modfrsites = modify_possitions(frsites)
        modfrsitesset.append(modfrsites)


    for op in symm_ops:
        froperatedsites = []
        for i in range(len(sites)):
            frsites = struct.lattice.get_fractional_coords(sites[i])
            froperatedsites.append(list(op.operate(frsites)))
        modfropsites = modify_possitions(froperatedsites)
        
        for used in modfrsitesset:
            used = [list(item) for item in used]
            if len(modfropsites) == len(used):
                if np.allclose(sorted(modfropsites), sorted(used), atol=0.01):
                    # print('Symmetrically same structure found!')
                    return True


def modify_possitions(sites):
    """
    sites must be 2D array
    sites are expressed in fractional coordinates
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
            modsite.append(round(sites[i][j], 3))
        modsite.append(sites[i][2])
        modsites.append(modsite)
    return modsites


def get_unique_surface(atoms, bareatoms, adsites, maxmole, mindist, initadsites, group, adsorbate):
    '''
    Given surface Atom object and all possible attaching site positions, create all possible unique attached Atom object.
    Can exclude by the maximum number of molecules or minimum distance. 

    return
    allatoms   :all possible Atom object [[[x,y,z]], [[a,b,c]], [[x,y,z],[a,b,c]]]
    allused    :attached sites for each Atom object
    mindistlis :minimum distance of molecule for each Atom object
    numdict    :dictionary, key=number of attached molecule, value=number of object created
    '''
    start = time.time()

    height = 1.85
    allatoms = []
    allused = []
    molenum = []
    usedadsites = []
    count = 0

    allused = copy.deepcopy(initadsites)
    allused = list(initadsites)

    def recursive(ratoms, rsites, rused, molnum, tmpused, count):
        molnum += 1

        if molnum > maxmole:
            return None

        for i in range(len(rsites)):
            nextatoms = copy.deepcopy(ratoms)
            nextused = copy.deepcopy(rused)
            if molnum == 1:
                tmpused = copy.deepcopy(allused)
                # print('Used initialized!')

            add_adsorbate(nextatoms, adsorbate, height, rsites[i][:2])
            nextused.append(list(rsites[i]))

            dist = get_minimum_distance(nextused, bareatoms.cell)
            if dist < mindist:
                # print('Distance {0:.2f} is below {1}'.format(dist, mindist))
                continue

            # When next sites candidates is symmetrycally same as in the sites previously used, skip
            if check_if_same(bareatoms, nextused, tmpused):
                continue

            allatoms.append(nextatoms)
            allused.append(nextused)
            tmpused.append(nextused)
            molenum.append(molnum)

            # print('{0}-------{1}-------'.format(molnum, nextatoms.symbols)) # keep it for debugging
            if molnum == initmol+1:
                print('progress: {}/{}, {:.2f} min'.format(count,
                                                           len(rsites), (time.time() - start)/60))
                count += 1

            struct = AseAtomsAdaptor.get_structure(nextatoms)
            nextsites = AdsorbateSiteFinder(struct).symm_reduce(adsites)
            indexlist = []
            
            for j in range(len(nextused)):
                for k in range(len(nextsites)):
                    if np.allclose(nextused[j], nextsites[k], atol=0.01):
                        indexlist.append(k)

            for j in range(len(usedadsites)):
                for k in range(len(nextsites)):
                    if np.allclose(usedadsites[j], nextsites[k], atol=0.01):
                        indexlist.append(k)

            indexlist.sort(reverse=True)
            for j in range(len(indexlist)):
                nextsites.pop(indexlist[j])

            recursive(nextatoms, nextsites, nextused,
                      molnum, tmpused, count)


            if molnum == 1:
                for j in range(len(group)):
                    if rsites[i] in group[j]:
                        for k in range(len(group[j])):
                            usedadsites.append(group[j][k])

    redadsites_ = get_adsorption_sites(atoms, True)['all']
    redadsites = [list(i) for i in redadsites_]
    initmol = len(initadsites)

    recursive(atoms, redadsites, list(initadsites), initmol, allused, count)
    mindistlis = get_minimum_distance_list(allused, bareatoms.cell)

    numdict = {}
    for site in allused:
        if str(len(site)) not in numdict.keys():
            numdict[str(len(site))] = 1
        else:
            numdict[str(len(site))] += 1

    print('\nadsorbates : # of structures, {}'.format(numdict))
    print('total structures: {}'.format(len(allatoms)))

    return allatoms, mindistlis, numdict, molenum


def get_minimum_distance(comb, cell):
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


def get_minimum_distance_list(combs, cell):
    '''
    combs: array of positions, or double array of positions
    '''
    mindistlis = []
    for i in range(len(combs)):
        if type(combs[0][0][0]) == np.float64:
            mindist = get_minimum_distance(combs[i], cell)
            mindistlis.append(mindist)

        else:
            for j in range(len(combs[i])):
                if combs[i][j] != 0:
                    mindist = get_minimum_distance(combs[i][j], cell)
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
        self.initatoms = init_query(surfacename +'.traj', env='spacom')
        self.adsorbate = query(adsorbatename+'.traj', env='spacom')

    def make_surface(self, maxmole, mindist):
        adseles = get_all_elements(self.adsorbate)
        baresurface, initadsites = remove_adsorbate(self.initatoms, adseles)
        bareadsites = get_adsorption_sites(baresurface, False)

        allbareadsites = np.array(bareadsites['all'])
        group = create_site_group(baresurface)

        ind = []
        for i in range(len(allbareadsites)):
            for j in range(len(initadsites)):
                if np.allclose(allbareadsites[i][:2], initadsites[j][:2]):
                    ind.append(i)

        adsites = np.delete(allbareadsites, ind, 0)
        inuse = allbareadsites[ind]

        allatoms, mindistlis, numdict, molenum \
            = get_unique_surface(self.initatoms, baresurface, adsites, maxmole, mindist, initadsites, group, self.adsorbate)

        self.allatoms = allatoms
        self.mindistlis = mindistlis
        self.numdict = numdict
        self.molenum = molenum

    def write_trajectory(self):
        for i in range(len(self.allatoms)):
            outname = self.surfacename + str('_no') + str('{0:03d}'.format(i+1)) + '_CO_n' + str(
                self.molenum[i]) + str('_d') + str(int(np.ceil(self.mindistlis[i]/0.5)-3)) + '.traj'
            print(outname)
            outpath = initpath + str(outname)
            self.allatoms[i].write(outpath)