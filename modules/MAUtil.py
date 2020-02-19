import numpy as np
import sys, os, random, itertools, warnings, math, copy
import matplotlib.pyplot as plt
from ase import Atoms, Atom
from ase.io import read, write
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.visualize import view

databasepath = '/home/katsuyut/research/coverage-effect/database/'
initpath = '/home/katsuyut/research/coverage-effect/init/'

def query(name, env='spacom'):
    path = databasepath + name
    try:
        traj = Trajectory(path)
        atoms = read(path)
        if env == 'local':
            view(traj)
        return atoms
    except IOError as e:
        return 'No file'
    

def init_query(name, env='spacom'):
    path = initpath + name
    try:
        atoms = read(path)
        if env == 'local':
            view(atoms)
        return atoms
    except IOError as e:
        return 'No file'


def getallene():
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