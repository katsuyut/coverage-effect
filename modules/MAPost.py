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
from ase.io import read, write
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.visualize import view
from scipy.spatial.qhull import QhullError
from pymatgen.analysis.local_env import VoronoiNN
from MAUtil import *
from MAInit import *

databasepath = '/home/katsuyut/research/coverage-effect/database/'
initpath = '/home/katsuyut/research/coverage-effect/init/'


def get_maximum_movement(name):
    name = name.split('.traj')[0]
    name = name + '_all.traj'
    path = databasepath + name
    traj = Trajectory(path)

    diff = abs(traj[-1].positions - traj[0].positions)
    maxdiff = np.max(diff)

    return maxdiff


def get_adsorbates_position_info(file, flag=0):
    '''
    if flag == 0 then calc both init and relaxed
    else calc only init
    '''
    iatoms = init_query(file, 'spacom')
    bareatoms, iposlis = removemolecule(iatoms, ['C', 'O'])
    barestruct = AseAtomsAdaptor.get_structure(bareatoms)
    baresites = getadsites(bareatoms, False)
    sites0_ = baresites['all']
    sites0 = [list(i) for i in sites0_]
    group = creategroup(bareatoms, sites0)
    cell = bareatoms.cell

    def assign_group(groups, poslis):
        for i in range(len(poslis)):
            mindist = 10000
            assign = None
            for j in range(len(group)):
                for k in range(len(group[j])):
                    #                     dist = np.linalg.norm(poslis[i][:2] - group[j][k][:2])
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
        bareatoms, rposlis = removemolecule(ratoms, ['C', 'O'])
        rgroups = []
        rgroups = assign_group(rgroups, rposlis)

        return igroups, iposlis, rgroups, rposlis

    return igroups, iposlis


def fingerprint_adslab_multiads(atoms):
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
    atoms, binding_positions = removemolecule(atoms, 'CO')
    for i in reversed(range(len(binding_positions))):
        atoms += Atoms('Kr', positions=[binding_positions[i]])
    Krypton_indexes = []
    for atom in atoms:
        if atom.symbol == 'Kr':
            Krypton_indexes.append(atom.index)
    struct = AseAtomsAdaptor.get_structure(atoms)
    results = []
    try:
        for Krypton_index in Krypton_indexes:
            # We have a standard and a loose Voronoi neighbor finder for various
            # purposes
            vnn = VoronoiNN(allow_pathological=True, tol=0.6,
                            cutoff=10)  # originally tol=0.8
            vnn_loose = VoronoiNN(allow_pathological=True, tol=0.2, cutoff=10)

            # Find the coordination
            nn_info = vnn.get_nn_info(struct, n=Krypton_index)
            coordination, cindex = __get_coordination_string_mod(nn_info)

            # Find the neighborcoord
            neighborcoord = []
            for neighbor_info in nn_info:
                # Need this to exclude just neighbor adsorbate to recognize as bonding atom
                if neighbor_info['site'].species_string != 'Kr':
                    # Get the coordination of this neighbor atom, e.g., 'Cu-Cu'
                    neighbor_index = neighbor_info['site_index']
                    neighbor_nn_info = vnn_loose.get_nn_info(
                        struct, n=neighbor_index)
                    neighbor_coord, cc_indexes = __get_coordination_string_mod(
                        neighbor_nn_info)
                    # Prefix the coordination of this neighbor atom with the identity
                    # of the neighber, e.g. 'Cu:Cu-Cu'
                    neighbor_element = neighbor_info['site'].species_string
                    neighbor_coord_labeled = neighbor_element + ':' + neighbor_coord
                    neighborcoord.append(neighbor_coord_labeled)

            # Find the nextnearestcoordination
            nn_info_loose = vnn_loose.get_nn_info(struct, n=Krypton_index)
            nextnearestcoordination = __get_coordination_string_mod(
                nn_info_loose)

            results.append({'adsorbate_index': Krypton_index,
                            'coordination': [coordination, cindex],
                            'neighborcoord': [neighborcoord, cc_indexes],
                            'nextnearestcoordination': nextnearestcoordination})
        return results

    # If we get some QHull or ValueError, then just assume that the adsorbate desorbed
    except (QhullError, ValueError, UnboundLocalError):
        results.append({'coordination': '',
                        'neighborcoord': '',
                        'nextnearestcoordination': ''})
        return results


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


def get_bonding_matrix(atoms):
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
    atoms, binding_positions = removemolecule(atoms, 'CO')
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
                b_mat[atom.index][cindex] = 1
                b_mat[cindex][atom.index] = 1

        return b_mat, nads

    # If we get some QHull or ValueError, then just assume that the adsorbate desorbed
    except (QhullError, ValueError):
        return None


def get_modified_bonding_matrix(atoms):
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
    atoms, binding_positions = removemolecule(atoms, 'CO')
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


def get_old_number_matrix(b_mat, nads, repeat):
    #     b_matlis = []
    #     sumb_matlis = []
    newb_mat = copy.deepcopy(b_mat)
#     sumb_mat = 0
    done = np.zeros([nads//repeat**2])
    results = []
    i = 2
    while True:
        newb_mat = newb_mat @ b_mat
#         b_matlis.append(newb_mat)
#         sumb_mat = sumb_mat + newb_mat
#         sumb_matlis.append(sumb_mat)
        # extract related adsorbate and non-diagonal terms
        newb_matCO = (newb_mat - np.diag(np.diag(newb_mat)))[-nads:, -nads:]
        # examinimg only center one is sufficient
        newb_matCO = newb_matCO[nads//repeat**2*4:nads//repeat**2*5]

        for j in range(nads//repeat**2):
            if done[j] == 0:
                if not (newb_matCO[j] == 0).all():
                    done[j] = 1
                    nnearestbonding = np.max(
                        np.sum(newb_matCO[j:nads:nads//repeat**2, :], axis=1))
#                     print('adsorbate {} has {} nearest adsorbate at {}th neighbor'.format(j,nnearestbonding,i))
#                     print(newb_matCO[j:nads:nads//repeat**2,:])
                    results.append([j, i, nnearestbonding])

#         print('')
        i += 1
        if (done == 1).all() or i == 6:
            break
    # [adsorbate_index, distance, # of nearest adsorbate]
    return np.array(results)


def get_number_matrix(b_mat, nads, repeat):
    '''
    If you want to set maximum distance to 4, you need to set repeat=5.
    If you want to set maximum distance to 3, you can set repeat=3.    
    '''
    if repeat != 3 and repeat != 5:
        print('Repeat should be 3 or 5. If you want to set maximum distance to 4, set 5 for repeat. If you want to set maximum distance to 3, set 3 for repeat.')
        return None
    elif repeat == 3:
        terminate = 4
    else:
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

    #     print('orig', newb_matCO)
    #     print(mask)
        masked = newb_matCO * mask
#         print('mod',masked)
        mask = (newb_matCO == 0)
        nnearestbonding = np.sum(masked)
        results.append([i, nnearestbonding])

        i += 1
    #     if (done == 1).all() or i==6:
    #         break
        if i == terminate:
            break
    return np.array(results)  # [distance, # of nearest adsorbate]
