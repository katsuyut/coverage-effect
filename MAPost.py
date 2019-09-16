import numpy as np
import sys, os, random, itertools, warnings, math, copy
from ase import Atoms, Atom
from ase.build import fcc100, fcc111, fcc110, bcc100, bcc111, bcc110, add_adsorbate, rotate
from ase.io import read, write
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.visualize import view
from ktms import *
from MAInit import *


def getmaxdiff(name):
    name = name.split('.traj')[0]
    name = name + '_all.traj'
    path = '/home/katsuyut/database/' + name
    traj = Trajectory(path)
    
    diff = abs(traj[-1].positions - traj[0].positions)
    maxdiff = np.max(diff)

    return maxdiff


def getadsposinfo(file, flag=0):
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
    
    def assigngroup(groups, poslis):
        for i in range(len(poslis)):
            mindist = 10000
            assign = None
            for j in range(len(group)):
                for k in range(len(group[j])):
#                     dist = np.linalg.norm(poslis[i][:2] - group[j][k][:2])
                    dist = min(np.linalg.norm(poslis[i][:2] - group[j][k][:2]),
                               np.linalg.norm((poslis[i] + cell[0])[:2] - group[j][k][:2]),
                               np.linalg.norm((poslis[i] - cell[0])[:2] - group[j][k][:2]),
                               np.linalg.norm((poslis[i] + cell[1])[:2] - group[j][k][:2]),
                               np.linalg.norm((poslis[i] - cell[1])[:2] - group[j][k][:2]),
                               np.linalg.norm((poslis[i] + cell[0] + cell[1])[:2] - group[j][k][:2]),
                               np.linalg.norm((poslis[i] + cell[0] - cell[1])[:2] - group[j][k][:2]),
                               np.linalg.norm((poslis[i] - cell[0] + cell[1])[:2] - group[j][k][:2]),
                               np.linalg.norm((poslis[i] - cell[0] - cell[1])[:2] - group[j][k][:2])
                              )
                    if dist < mindist:
                        mindist = dist
                        assign = j
            groups.append(assign)
        return groups

    igroups = []
    igroups = assigngroup(igroups, iposlis)
    
    if flag == 0:
        ratoms = query(file, 'spacom')
        bareatoms, rposlis = removemolecule(ratoms, ['C', 'O'])
        rgroups = []
        rgroups = assigngroup(rgroups, rposlis)
                
        return igroups, iposlis, rgroups, rposlis
    
    return igroups, iposlis