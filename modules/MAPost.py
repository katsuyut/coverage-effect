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
from ase.build import fcc100, fcc111, fcc110, bcc100, bcc111, bcc110, add_adsorbate, rotate
from ase.io import read, write
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.visualize import view
from scipy.spatial.qhull import QhullError
from pymatgen.analysis.local_env import VoronoiNN
from MAUtil import *
from MAInit import *
from pymongo import MongoClient

databasepath = '/home/katsuyut/research/coverage-effect/database/'
initpath = '/home/katsuyut/research/coverage-effect/init/'


def get_maximum_movement(name):
    '''
    Using adjust_positions from MAInit to avoid the pbc problem
    '''
    name = name.split('.traj')[0]
    name = name + '_all.traj'
    path = databasepath + name
    traj = Trajectory(path)

    initpos = adjust_possitions(traj[-1].positions)
    postpos = adjust_possitions(traj[0].positions)
    diff = abs(np.array(initpos) - np.array(postpos))
    maxdiff = np.max(diff)

    return maxdiff


def get_adsorbates_position_info(file, flag=0):
    '''
    if flag == 0 then calc both init and relaxed
    else calc only init
    '''
    iatoms = init_query(file, 'spacom')
    bareatoms, iposlis = remove_adsorbate(iatoms, ['C', 'O'])
    group = create_site_group(bareatoms)
    cell = bareatoms.cell

    def assign_group(groups, poslis):
        for i in range(len(poslis)):
            mindist = 10000
            assign = None
            for j in range(len(group)):
                for k in range(len(group[j])):
                    # dist = np.linalg.norm(poslis[i][:2] - group[j][k][:2])
                    dist = min(np.linalg.norm(poslis[i][:2] - group[j][k][:2]),
                               np.linalg.norm(
                                   (poslis[i] + cell[0])[:2] - group[j][k][:2]),
                               np.linalg.norm(
                                   (poslis[i] - cell[0])[:2] - group[j][k][:2]),
                               np.linalg.norm(
                                   (poslis[i] + cell[1])[:2] - group[j][k][:2]),
                               np.linalg.norm(
                                   (poslis[i] - cell[1])[:2] - group[j][k][:2]),
                               np.linalg.norm(
                                   (poslis[i] + cell[0] + cell[1])[:2] - group[j][k][:2]),
                               np.linalg.norm(
                                   (poslis[i] + cell[0] - cell[1])[:2] - group[j][k][:2]),
                               np.linalg.norm(
                                   (poslis[i] - cell[0] + cell[1])[:2] - group[j][k][:2]),
                               np.linalg.norm(
                                   (poslis[i] - cell[0] - cell[1])[:2] - group[j][k][:2])
                               )
                    if dist < mindist:
                        mindist = dist
                        assign = j
            groups.append(assign)
        return groups

    igroups = []
    igroups = assign_group(igroups, iposlis)

    if flag == 0:
        ratoms = query(file, 'spacom')
        bareatoms, rposlis = remove_adsorbate(ratoms, ['C', 'O'])
        rgroups = []
        rgroups = assign_group(rgroups, rposlis)

        return igroups, iposlis, rgroups, rposlis

    return igroups, iposlis


def __get_coordination_string_mod(nn_info):
    '''
    This helper function takes the output of the `VoronoiNN.get_nn_info` method
    and gives you a standardized coordination string.
    Arg:
        nn_info     The output of the
                    `pymatgen.analysis.local_env.VoronoiNN.get_nn_info` method.
    Returns:
        coordination    A string indicating the coordination of the site
                        you fed implicitly through the argument, e.g., 'Cu-Cu-Cu'
    '''
    coordinated_atoms = [neighbor_info['site'].species_string
                         for neighbor_info in nn_info
                         if neighbor_info['site'].species_string != 'Kr']
    coordination = '-'.join(sorted(coordinated_atoms))

    coordinated_indexes = [neighbor_info['site_index']
                           for neighbor_info in nn_info
                           if neighbor_info['site'].species_string != 'Kr']

    return coordination, coordinated_indexes


def get_coordination_matrix(atoms, expression=2):
    '''
    This function will fingerprint a slab+adsorbate atoms object for you.
    It only  with multiple adsorbates.
    Arg:
        atoms   `ase.Atoms` object to fingerprint. The slab atoms must be
                tagged with non-zero integers and adsorbate atoms must be
                tagged with zero. This function also assumes that the
                first atom in each adsorbate is the binding atom (e.g.,
                of all atoms with tag==1, the first atom is the binding;
                the same goes for tag==2 and tag==3 etc.).
    Returns:
        fingerprint A dictionary whose keys are:
                        coordination            A string indicating the
                                                first shell of
                                                coordinated atoms
                        neighborcoord           A list of strings
                                                indicating the coordination
                                                of each of the atoms in
                                                the first shell of
                                                coordinated atoms
                        nextnearestcoordination A string identifying the
                                                coordination of the
                                                adsorbate when using a
                                                loose tolerance for
                                                identifying "neighbors"
    '''
    # Replace the adsorbate[s] with a single Krypton atom at the first binding
    # site. We need the Krypton there so that pymatgen can find its
    # coordination.
    atoms, binding_positions = remove_adsorbate(atoms, 'CO')
    nads = len(binding_positions)
    for i in reversed(range(nads)):
        atoms += Atoms('Kr', positions=[binding_positions[i]])
    b_mat = np.zeros([len(atoms), len(atoms)])
    Krypton_indexes = []
    for atom in atoms:
        if atom.symbol == 'Kr':
            Krypton_indexes.append(atom.index)
    struct = AseAtomsAdaptor.get_structure(atoms)

    try:
        for atom in atoms:
            # We have a standard and a loose Voronoi neighbor finder for various
            # purposes
            vnn = VoronoiNN(allow_pathological=True, tol=0.6,
                            cutoff=10)  # originally tol=0.8
            vnn_loose = VoronoiNN(allow_pathological=True, tol=0.2, cutoff=10)

            # Find the coordination
            if atom.symbol == 'Kr':
                nn_info = vnn.get_nn_info(struct, n=atom.index)
                coordination, cindexes = __get_coordination_string_mod(nn_info)
            else:
                nn_info = vnn_loose.get_nn_info(struct, n=atom.index)
                coordination, cindexes = __get_coordination_string_mod(nn_info)

            for cindex in cindexes:
                if expression == 1:
                    b_mat[atom.index][cindex] = 1
                    b_mat[cindex][atom.index] = 1
                elif expression == 2:
                    b_mat[atom.index][cindex] = 1/len(cindexes)
                    b_mat[cindex][atom.index] = 1/len(cindexes)

        return b_mat, nads

    # If we get some QHull or ValueError, then just assume that the adsorbate desorbed
    except (QhullError, ValueError):
        return None


def get_repeated_atoms(atoms, repeat):
    cpatoms = copy.deepcopy(atoms)
    for i in reversed(range(len(cpatoms))):
        if cpatoms[i].tag == 3 or cpatoms[i].tag == 4:
            cpatoms.pop(i)
    cpatoms = cpatoms.repeat([repeat, repeat, 1])
    return cpatoms


def get_adsorbates_correlation(b_mat, nads, maximumdistance=3):
    # You need to repead atoms
    if maximumdistance != 3 and maximumdistance != 4:
        print('Maximumdistance must be 3 or 4.')
        return None
    elif maximumdistance == 3:
        repeat = 3
        terminate = 4
    else:
        repeat = 5
        terminate = 5

    newb_mat = copy.deepcopy(b_mat)
    done = np.zeros([nads//repeat**2])
    results = []
    i = 2
    mask = np.ones(np.shape(newb_mat[nads//repeat**2*(math.floor(
        repeat**2/2.0)):nads//repeat**2*(math.ceil(repeat**2/2.0)), -nads:]))
    while True:
        newb_mat = newb_mat @ b_mat
        # extract related adsorbate and non-diagonal terms
        newb_matCO = (newb_mat - np.diag(np.diag(newb_mat)))[-nads:, -nads:]
        newb_matCO = newb_matCO[nads//repeat**2*(math.floor(repeat**2/2.0)):nads//repeat**2*(
            math.ceil(repeat**2/2.0))]  # examinimg only center one is sufficient

        masked = newb_matCO * mask
        mask = (newb_matCO == 0)
        nnearestbonding = np.sum(masked)
        results.append([i, nnearestbonding])

        i += 1
        if i == terminate:
            break
    return np.array(results)  # [distance, # of nearest adsorbate]


class make_database():
    """
    Create json(python dictionary) object for adsorbate adsorbed surface
    This requires specific filename convention
    ex) Pd_111_u2_RPBE_no015_CO_n1_d9.traj
        (element)_(face)_(u + unitlength)_(xc used in vasp)_(number)_(adsorbate)_
        (n + number of adsorbate)_(d + minimum distance of each adsorbates).traj
    """

    def __init__(self, filename):
        self.filename = filename
        res = re.match(
            '(.*)_(.*)_u(.*)_(.*)_(.*)_(.*)_n(.*)_(.*)(.traj)', filename)
        if not res:
            print('This file is not for this class.')
            raise NameError('This file is not for this class.')
        self.numads = int(res.group(7))
        self.iatoms = init_query(filename)
        self.ratoms = query(filename)
        client = MongoClient('localhost', 27017)
        db = client.adsE_database
        self.collection = db.adsE_collection
        print('-----------------------------------------------------------')

    def make_json(self):
        dic = {}

        res = re.match(
            '(.*)_(.*)_u(.*)_(.*)_(.*)_(.*)_n(.*)_(.*)(.traj)', self.filename)
        ele = res.group(1)
        face = res.group(2)
        unit = int(res.group(3))
        xc = res.group(4)
        ads = res.group(6)
        numads = int(res.group(7))

        res = re.match('(.*)_no.*', self.filename)
        barefile = res.group(1) + '.traj'
        bareatoms = query(barefile, 'spacom')

        adsfile = ads + '_' + xc + '.traj'
        COatoms = query(adsfile, 'scpacom')

        ### calc area ###
        x = self.ratoms.cell[0][0]
        y = self.ratoms.cell[1][1]
        area = x*y

        ### calc surface atom number ###
        tot = 0
        for atom in bareatoms:
            if atom.tag == 1:
                tot += 1

        ### get energies ###
        ene = self.ratoms.get_potential_energy()
        bareene = bareatoms.get_potential_energy()
        COene = COatoms.get_potential_energy()
        totaladsene = ene - (COene*numads + bareene)

        ### get adsorbate position info ###
        igroups, iposlis, rgroups, rposlis = get_adsorbates_position_info(
            self.filename, 0)

        ### get COint info ###
        intfile = self.filename[:-5] + '__.traj'
        initatoms = query(intfile, 'spacom')
        intene = initatoms.get_potential_energy() - numads*COene

        ### get COlength info ###
        COlengthlis = []
        for i in range(len(self.ratoms)):
            if self.ratoms[i].symbol == 'C':
                Cpos = self.ratoms[i].position
                Opos = self.ratoms[i+1].position
                COlengthlis.append(np.linalg.norm(Cpos-Opos))

        ### judge valid or not ###
        converged = np.max(np.abs(self.ratoms.get_forces())) < 0.03
        not_moved = get_maximum_movement(self.filename) < 1.5
        kept_sites = igroups == rgroups
        if converged and not_moved and kept_sites:
            isvalid = True
        else:
            isvalid = False

        dic['name'] = self.filename
        dic['isvalid'] = 'yes' if isvalid else 'no'
        dic['element'] = ele
        dic['face'] = face
        dic['unitlength'] = unit
        dic['xc'] = xc
        dic['adsorbate'] = ads
        dic['numberofads'] = numads
        dic['coverage'] = numads/tot
        dic['surfatomnum'] = tot
        dic['E'] = ene
        dic['bareE'] = bareene
        dic['E_CO'] = COene
        dic['totaladsE'] = totaladsene
        dic['aveadsE/suratom'] = totaladsene/tot
        dic['aveadsE/ads'] = totaladsene/numads
        dic['E_int_space'] = intene
        dic['E_each_ads'] = None
        dic['area'] = area
        dic['density'] = numads/area
        dic['igroups'] = igroups
#         dic['iposlis'] = [list(item) for item in iposlis]
        dic['rgroups'] = rgroups
#         dic['rposlis'] = [list(item) for item in rposlis]
        dic['COlengthlis'] = COlengthlis
        dic['converged'] = 'yes' if converged else 'no'
        dic['not_moved'] = 'yes' if not_moved else 'no'
        dic['kept_sites'] = 'yes' if kept_sites else 'no'
        dic['minimum_distance'] = None
        dic['ads_dist2'] = None
        dic['ads_dist3'] = None

#         for key in dic.keys():
#             print('{}: {}'.format(key, dic[key]))
        print(self.filename)

        return dic

    def add_to_database(self):
        if self.check_database():
            print('Already in database')
            return None
        post_data = self.make_json()
        result = self.collection.insert_one(post_data)
        print('One post: {0}\n'.format(result.inserted_id))

    def get_energy_for_each_adsorbates(self):
        data = list(self.collection.find({'name': self.filename}))
        if len(data) != 1:
            print('More than two data found!')
            return None
        else:
            data = data[0]

        E_each_ads = []  # adsE + slabE

        for site in data['rgroups']:
            refdata = self.collection.find_one({'element': data['element'], 'face': data['face'], 'unitlength': 2,
                                                'xc': data['xc'], 'adsorbate': data['adsorbate'],
                                                'rgroups': [site]})
            if refdata == None:
                print(
                    'Reference data is invalid. Each adsorption energy cannnot be calculated.')
                return None
            else:
                E_each_ads.append(refdata['totaladsE'])

        return E_each_ads

    def update_database_Eeach(self):
        """
        Assuming more than two adsorbates are on the surface.
        """
        if self.numads <= 1:
            print('This function is for surface with more than 2 adsorbates.')
            return None
        E_each_ads = self.get_energy_for_each_adsorbates()
        self.collection.find_one_and_update(
            {'name': self.filename}, {'$set': {'E_each_ads': E_each_ads}})
        print(self.filename, ' E_each_ads updated.')

    def update_database_adsorbates_correlation(self, maximumdistance=3):
        """
        Assuming more than two adsorbates are on the surface.
        """
        if self.numads <= 1:
            print('This function is for surface with more than 2 adsorbates.')
            return None

        if maximumdistance == 3:
            repeat = 3
            rratoms = get_repeated_atoms(self.ratoms, repeat)
        else:
            print('Currently maximum distance is 3.')

        b_mat, nads = get_coordination_matrix(rratoms, expression=2)
        correlation = get_adsorbates_correlation(b_mat, nads, repeat)
        if correlation[0][1] != 0:
            mindist = 2
        elif correlation[1][1] != 0:
            mindist = 3
        else:
            mindist = None

        self.collection.find_one_and_update(
            {'name': self.filename}, {'$set': {'minimum_distance': mindist}})
        self.collection.find_one_and_update(
            {'name': self.filename}, {'$set': {'ads_dist2': correlation[0][1]}})
        self.collection.find_one_and_update(
            {'name': self.filename}, {'$set': {'ads_dist3': correlation[1][1]}})
        print(self.filename, 'Adsorbate correlation updated.')

    def check_database(self):
        data = list(self.collection.find({'name': self.filename}))
        if not data:
            print('Not in database.')
            return None
        elif len(data) > 1:
            print('More than two data found!')
            return data[0]
        else:
            return data[0]

    def delete_from_database(self):
        self.collection.delete_many({'name': self.filename})
        print(self.filename, 'deleted')
