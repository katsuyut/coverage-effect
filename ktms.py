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


def query(name, env='spacom'):
    path = '/home/katsuyut/database/' + name
    try:
        traj = Trajectory(path)
        atoms = read(path)
        if env == 'local':
            view(traj)
        return atoms
    except IOError as e:
        return 'No file'
    

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
