import numpy as np
import sys, os, random, itertools, warnings, math, copy
from ase import Atoms, Atom
from ase.build import fcc100, fcc111, fcc110, bcc100, bcc111, bcc110, add_adsorbate, rotate
from ase.constraints import FixAtoms
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
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


def getCoPt3(a):
    x = a/2
    y = a/2
    z = a/2

    pos = np.array([[0,0,0],[x,y,0],[0,y,z],[x,0,z]])

    atoms = Atoms(symbols = 'CoPt3', # Ga2Ni2
                       positions= pos,
                       cell = np.array([[x*2,0,0],
                                        [0,y*2,0],
                                        [0,0,z*2]]),
                       pbc=1
                       )
    
    return atoms


def make_surface_from_bulk(atoms, unitlength, height):
    atoms.pbc[2] = False

    atoms = atoms.repeat([unitlength, unitlength, height])
    atoms.center(vacuum=10, axis=2)

    atoms = settag(atoms)
    return atoms


def settag(atoms):
    poslis = list(set(atoms.get_positions()[:,2]))
    poslis.sort()

    for i in range(len(atoms)):
        for j in range(len(poslis)):
            if atoms[i].position[2] == poslis[j]:
                atoms[i].tag = len(poslis) - j
    return atoms


def modpos(sites):
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


def checkifsame(bareatoms, sites, allused):
    '''
    Check if in allused has same configuration of symmetrical operated sites.
    Return True if found.
    '''
    struct = AseAtomsAdaptor.get_structure(bareatoms)
    surf_sg = SpacegroupAnalyzer(struct, 0.01)
    symm_ops = surf_sg.get_symmetry_operations()

    cpallused = copy.deepcopy(allused)
    frcpallused = []
    
    for i in range(len(cpallused)):
        tmp = []
        frused = struct.lattice.get_fractional_coords(cpallused[i])
        for j in range(len(frused)):
            tmp.append(list(frused[j]))
        frcpallused.append(modpos(tmp))

    for op in symm_ops:
        froperatedsites = []
        for i in range(len(sites)):
            frsites = struct.lattice.get_fractional_coords(sites[i])
            froperatedsites.append(list(op.operate(frsites)))
        froperatedsites.sort()
        modfropsites = modpos(froperatedsites)
        
        for used in frcpallused:
            if len(modfropsites) == len(used):
                if np.allclose(sorted(modfropsites), sorted(used), atol=0.01):
                    print('Symmetrically same structure found!')
                    return True
                
                
def getuniqueatoms(atoms, bareatoms, sites, maxmole, mindist, rused, group, molecule):
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
    usedsites = []
    
    for i in range(len(rused)):
        tmp = []
        tmp.append(rused[i])
        allused.append(tmp)

    initallused = copy.deepcopy(allused)
    
    def recursive(ratoms, rsites, rused, molecules, tmpused):
        molecules += 1
        if molecules > maxmole:
            return None
        
        for i in range(len(rsites)):
            nextatoms = copy.deepcopy(ratoms)
            nextused = copy.deepcopy(rused)
            if molecules == 1:
                tmpused = copy.deepcopy(initallused)
                print('Used initialized!')
            
            add_adsorbate(nextatoms, molecule, height, rsites[i][:2])
            nextused.append(list(rsites[i]))
            
            dist = getmindist(nextused, bareatoms.cell)
            if dist < mindist:
                print('Distance {0:.2f} is below {1}'.format(dist, mindist))
                continue

            sameflag = checkifsame(bareatoms, nextused, tmpused)
            if sameflag:
                continue
            
            allatoms.append(nextatoms)
            allused.append(nextused)
            tmpused.append(nextused)
            molenum.append(molecules)
            struct = AseAtomsAdaptor.get_structure(nextatoms)
            
            print('{0}-------{1}-------'.format(molecules, nextatoms.symbols))
            
            try:
                nextsites = AdsorbateSiteFinder(struct).symm_reduce(sites)
                indexlist = []
                for j in range(len(nextused)):
                    for k in range(len(nextsites)):
                        if np.allclose(nextused[j], nextsites[k]):
                            indexlist.append(k)

                for j in range(len(usedsites)):
                    for k in range(len(nextsites)):
                        if np.allclose(usedsites[j], nextsites[k]):
                            indexlist.append(k)

                indexlist.sort(reverse = True) 
                for j in range(len(indexlist)):
                    nextsites.pop(indexlist[j])
                    
                recursive(nextatoms, nextsites, nextused, molecules, tmpused)
            
            except:
                print('Error!!')
            
            if molecules == 1:
                for j in range(len(group)):
                    if rsites[i] in group[j]:
                        for k in range(len(group[j])):
                            usedsites.append(group[j][k])

    redsites_ = getadsites(atoms, True)['all']
    redsites = [list(i) for i in redsites_]

    recursive(atoms, redsites, rused, len(rused), initallused)
    mindistlis = getmindistlist(allused, bareatoms.cell)
    
    numdict = {}
    for site in allused:
        if str(len(site)) not in numdict.keys():
            numdict[str(len(site))] = 1
        else:
            numdict[str(len(site))] += 1
    
    return allatoms, allused, mindistlis, numdict, molenum


def removemolecule(atoms, molecule):
    cpatoms = copy.deepcopy(atoms)
    poslis = []
    for i in reversed(range(len(cpatoms))):
        if cpatoms[i].symbol == molecule[0]:
            poslis.append(cpatoms[i].position)

        for j in molecule:
            if cpatoms[i].symbol == j:
                cpatoms.pop(i)
#                 print(cpatoms.symbols)
                break
    return cpatoms, poslis


def creategroup(bareatoms, sites):
    barestruct = AseAtomsAdaptor.get_structure(bareatoms)
    baresites = getadsites(bareatoms, True)
    redsites_ = baresites['all']
    redsites = [list(i) for i in redsites_]

    unique = []
    for i in range(len(redsites)):
        tmp = []
        tmp.append(redsites[i])
        unique.append(tmp)
    
    group = copy.deepcopy(unique)
    num = 0
    for i in range(len(sites)):
        tmp = []
        tmp.append(sites[i])

        flag = 0
        for j in range(len(unique)):
            if np.allclose(tmp, unique[j]):
                flag = 1
        if flag == 1:
            continue

        for j in range(len(unique)):
            tmp2 = []
            tmp2.append(unique[j])
            if checkifsame(bareatoms, tmp, tmp2):
#                 print(j, 'same config found!')
                group[j].append(sites[i])
                break

    return group