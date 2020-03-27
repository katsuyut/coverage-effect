import numpy as np
import sys
import os
import random
import itertools
import warnings
import math
import copy
import re
import pickle
import matplotlib.pyplot as plt
from ase import Atoms, Atom
from ase.io import read, write
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.visualize import view
from pymatgen.ext.matproj import MPRester

databasepath = os.environ['DATABASEPATH']
initpath = os.environ['INITPATH']
mppath = os.environ['MPPATH']

def query(name, env='spacom'):
    path = databasepath + name
    try:
        traj = Trajectory(path)
        atoms = read(path)
        if env == 'local':
            view(traj)
        return atoms
    except IOError as e:
        print('No file named {} in database'.format(name))
        return None


def init_query(name, env='spacom'):
    path = initpath + name
    try:
        atoms = read(path)
        if env == 'local':
            view(atoms)
        return atoms
    except IOError as e:
        print('No file named {} in init'.format(name))
        return None


def cif_query(name, env='spacom'):
    path = mppath + name
    try:
        atoms = read(path)
        if env == 'local':
            view(atoms)
        return atoms
    except IOError as e:
        print('No file named {} in mp'.format(name))
        return None

def mp_query(name, env='spacom'):
    path = mppath + name
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        formula = data['pretty_formula']
        crystal_system = data['spacegroup']['crystal_system']

        atoms = read(path[:-2]+'.cif')
        
        if env == 'local':
            view(atoms)
        return atoms, formula, crystal_system
    except IOError as e:
        print('No file named {} in mp'.format(name))
        return None

def get_all_energy():
    files = os.listdir(databasepath)

    for filename in files:
        if '.traj' in filename:
            if not 'all' in filename:
                path = databasepath + filename
                atoms = read(path)
                try:
                    energy = atoms.get_potential_energy()
                    print('{0}, {1}'.format(filename, energy))
                except:
                    print('No energy')


def request_mp(mpid):
    '''
    Request cif data to mateials project. Cif data is saved in cif folder is not exists.
    You need materials project api_key as MAPIKEY in your environment varible

    return response[0]
    '''
    with MPRester(api_key=os.environ['MAPIKEY']) as m:
        data = m.get_data(mpid)

    # cifdata = data[0]['cif']
    formula = data[0]['pretty_formula']
    path = mppath + mpid + '_' + formula + '.b'

    if os.path.exists(path):
        print('Already in mppath')
    else:
        with open(path, 'wb') as f:
            pickle.dump(data[0] , f)
    
        path = mppath + mpid + '_' + formula + '.cif'
        with open(path, 'w') as f:
            f.write(data[0]['cif'])
            print('Added to mppath')

    crystal_system = data[0]['spacegroup']['crystal_system']

    print('material: {0}'.format(formula))
    print('crystal system: {0}'.format(crystal_system))

    return data[0]


def assign_group(group, poslis, cell):
    groups = []
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
