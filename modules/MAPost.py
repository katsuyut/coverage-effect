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
