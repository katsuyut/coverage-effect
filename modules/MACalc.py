import numpy as np
import sys, os, random, itertools, warnings, math, copy
from ase import Atoms, Atom
from ase.calculators.emt import EMT
from ase.calculators.vasp import Vasp, Vasp2
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.constraints import FixAtoms
from ase.eos import EquationOfState
from ase.io import read, write
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.optimize import QuasiNewton
from ase.build import bulk

databasepath = '/home/katsuyut/research/coverage-effect/database/'
initpath = '/home/katsuyut/research/coverage-effect/init/'
zvalpath = '/home/katsuyut/research/coverage-effect/zval.txt'


def getnbands(atoms, f=0.5):
    file = open(zvalpath, 'r')
    combs = file.read().split('\n')
    electrondict = {}
    for comb in combs:
        kye = comb.split('\t')[0]
        val = comb.split('\t')[1]
        electrondict[kye] = float(val)
        
    species = set(atoms.symbols)

    speciesdict = {}
    for i in species:
        bools = (i == atoms.symbols)
        speciesdict[i] = list(bools).count(True)
    
    keys = speciesdict.keys()
    vals = speciesdict.values()    
    
    nelectrons = 0
    for key in keys:
        nelectrons += speciesdict[key]*electrondict[key]

    nbands = int(nelectrons/2 + len(atoms)*f)
    
    return nbands


def getkpts(atoms):
    c = 30
    cell = atoms.get_cell()
    kpts = []

    for i in range(3):
        l = np.sqrt(np.square(cell[i][0]) + np.square(cell[i][1]) + np.square(cell[i][2]))
        if int(c/l)==0:
            kpts.append(1)
        else:
            kpts.append(int(c/l))

    return kpts


def getdefaultvasptags(xc = 'RPBE'):
    """
    Default is same as used in GASpy (xc=RPBE)
    If xc is specified, a different set of tags is returned
    Available xcs are :RPBE, RPBE-D2, vdW-DF(revPBE-DF), optB88-vdW, vdW-DF2 (rPW86-vdw), BEEF-vdw

    reference
    https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html
    """
    if xc == 'RPBE':
        tagdict = {
            'xc' : 'RPBE',
            'pp' : 'PBE',
            'ncore' : 4,
            'encut' : 350,
            'nsw' : 200,
            # 'kpts' : None,
            'ibrion' : 2,
            'isif' : 0,
            'ediffg' : -3.00e-02,
            'isym' : 0,
            'symprec' : 1.00e-10,
            'lreal' : 'Auto',
            'lcharg' : False,
            'lwave' : False,
            'ivdw' : 0,
            'lasph' : False,
        }
    elif xc == 'RPBE-D2':
        tagdict = {
            'xc' : 'RPBE',
            'pp' : 'PBE',
            'ncore' : 4,
            'encut' : 350,
            'nsw' : 200,
            # 'kpts' : None,
            'ibrion' : 2,
            'isif' : 0,
            'ediffg' : -3.00e-02,
            'isym' : 0,
            'symprec' : 1.00e-10,
            'lreal' : 'Auto',
            'lcharg' : False,
            'lwave' : False,
            'ivdw' : 1,
            'lasph' : False,
        }
    elif xc == 'vdW-DF':
        tagdict = {
            'xc' : 'vdW-DF',
            'pp' : 'PBE',
            'ncore' : 4,
            'encut' : 350,
            'nsw' : 200,
            # 'kpts' : None,
            'ibrion' : 2,
            'isif' : 0,
            'ediffg' : -3.00e-02,
            'isym' : 0,
            'symprec' : 1.00e-10,
            'lreal' : 'Auto',
            'lcharg' : False,
            'lwave' : False,
            'ivdw' : 0,
            'lasph' : True,
        }
    elif xc == 'optB88-vdW':
        tagdict = {
            'xc' : 'optB88-vdW',
            'pp' : 'PBE',
            'ncore' : 4,
            'encut' : 350,
            'nsw' : 200,
            # 'kpts' : None,
            'ibrion' : 2,
            'isif' : 0,
            'ediffg' : -3.00e-02,
            'isym' : 0,
            'symprec' : 1.00e-10,
            'lreal' : 'Auto',
            'lcharg' : False,
            'lwave' : False,
            'ivdw' : 0,
            'lasph' : True,
        }
    elif xc == 'vdW-DF2':
        tagdict = {
            'xc' : 'vdW-DF2',
            'pp' : 'PBE',
            'ncore' : 4,
            'encut' : 350,
            'nsw' : 200,
            # 'kpts' : None,
            'ibrion' : 2,
            'isif' : 0,
            'ediffg' : -3.00e-02,
            'isym' : 0,
            'symprec' : 1.00e-10,
            'lreal' : 'Auto',
            'lcharg' : False,
            'lwave' : False,
            'ivdw' : 0,
            'lasph' : True,
        }    
    elif xc == 'BEEF-vdW':
        tagdict = {
            'xc' : 'BEEF-vdW',
            'pp' : 'PBE',
            'ncore' : 4,
            'encut' : 350,
            'nsw' : 200,
            # 'kpts' : None,
            'ibrion' : 2,
            'isif' : 0,
            'ediffg' : -3.00e-02,
            'isym' : 0,
            'symprec' : 1.00e-10,
            'lreal' : 'Auto',
            'lcharg' : False,
            'lwave' : False,
            'ivdw' : 0,
            'lasph' : True,
        }
    else:
        print('No default tags set found')

    return tagdict


def getenergy(atoms, name, vasptags, env):
    trajpath = databasepath + str(name) + '.traj'
    trajpath_all = databasepath + str(name) + '_all.traj'

    if env == 'local':
        atoms.set_calculator(EMT())
        
        dyn = QuasiNewton(atoms, trajectory=trajpath)
        dyn.run(fmax=0.05)
    elif env == 'spacom':
        atoms.set_calculator(vasptags)
        
    try:
        e_atoms = atoms.get_potential_energy()
        atoms.write(trajpath)
    except:
        print('Error while calculating {0} energy!'.format(name))
        return None

    if env == 'spacom':
        atomslist = []
        for atoms in read('vasprun.xml', ':'):
            catoms = atoms.copy()
            catoms = catoms[vasptags.resort]
            catoms.set_calculator(SPC(catoms,
                                      energy=atoms.get_potential_energy(),
                                      forces=atoms.get_forces()[vasptags.resort]))
            atomslist += [catoms]

        # Write a traj file for the optimization
        tj = TrajectoryWriter(trajpath_all, 'a')
        for atoms in atomslist:
            print('writing trajectory file!')
            print(atoms)
            tj.write(atoms)
        tj.close()

    return e_atoms


class make_surface():
    def __init__(self, ele):
        # https://periodictable.com/Properties/A/LatticeConstants.html
        self.ele = ele
        defprops = {
            'Cu' : ['fcc', 3.6149, 0],
            # 'Cu' : ['hcp', 3.6149, 5],  # for test
            'Pt' : ['fcc', 3.9242, 0],
            'Ag' : ['fcc', 4.0853, 0],
            'Pd' : ['fcc', 3.8907, 0],
            'Au' : ['fcc', 4.0782, 0],
            'Ni' : ['fcc', 3.5240, 0],
            'Al' : ['fcc', 4.0495, 0],
            'Rh' : ['fcc', 3.8034, 0],
            'Ru' : ['hcp', 2.7059, 4.2815],
            'Zn' : ['hcp', 2.6649, 4.9468],
            }
        self.defprops = defprops
        self.structure = defprops[ele][0]
        self.a0 = defprops[ele][1]
        self.c0 = defprops[ele][2]
        self.eps = 0.03
        self.a = None
        self.c = None

    def createbulk(self, a0, c0):
        if self.ele not in self.defprops.keys():
            print('This materials is not available. Add to props.')
            return None
        
        if self.defprops[self.ele][0] == 'fcc':
            atom = bulk(self.ele, self.defprops[self.ele][0], a = a0)
        elif self.defprops[self.ele][0] == 'hcp':
            atom = bulk(self.ele, self.defprops[self.ele][0], a = a0, c = c0)
        else:
            print('Only fcc and hcp is available')
            return None

        return atom

    def calcLC(self, xc, env='spacom'):
        volumes = []
        energies = []
        cells = []

        atom = self.createbulk(self.a0, self.c0)
        filename = self.ele + '.traj'
        traj = Trajectory(filename, 'w')
        
        tagdict = getdefaultvasptags(xc)
        kpoints = getkpts(atom)
        vasptags = Vasp(
            xc = tagdict['xc'],
            pp = tagdict['pp'],
            ncore = tagdict['ncore'],
            encut = tagdict['encut'],
            nsw = tagdict['nsw'],
            kpts = kpoints,
            ibrion = tagdict['ibrion'],
            isif = tagdict['isif'],
            ediffg = tagdict['ediffg'],
            isym = tagdict['isym'],
            lreal = tagdict['lreal'],
            lcharg = tagdict['lcharg'],
            lwave = tagdict['lwave'],
            ivdw = tagdict['ivdw'],
            lasph = tagdict['lasph'],
            )


        ### fcc ###
        if self.structure == 'fcc':
            cell = atom.get_cell()
        
            for x in np.linspace(1-self.eps, 1+self.eps, 5):
                atom.set_cell(cell * x, scale_atoms=True)
            
                if env == 'local':
                    atom.set_calculator(EMT())

                    dyn = QuasiNewton(atom)
                    dyn.run(fmax=0.05)
                elif env == 'spacom':
                    atom.set_calculator(vasptags)

                try:
                    atom.get_potential_energy()
                    volumes.append(atom.get_volume())
                    energies.append(atom.get_potential_energy())
                    cells.append(atom.get_cell())

                except:
                    print('Error while calculating bulk energy!')
                
            eos = EquationOfState(volumes, energies)
            v0, e0, B = eos.fit()
            a = (v0/2.0)**(1.0/3.0)*2.0

            f = open('result.txt', 'a')
            f.write('{0}, {1}, {2}\n'.format(self.ele, xc,str(a)))
            f.close()

            self.a = a

            return a

        ### hcp ###
        elif self.structure == 'hcp':
            for a in self.a0 * np.linspace(1-self.eps, 1+self.eps, 5):
                for c in self.c0 * np.linspace(1-self.eps, 1+self.eps, 5):
                    atom = self.createbulk(a, c)

                    if env == 'local':
                        atom.set_calculator(EMT())
                        dyn = QuasiNewton(atom)
                        dyn.run(fmax=0.05)

                    elif env == 'spacom':
                        atom.set_calculator(vasptags)
                    
                    atom.get_potential_energy()
                    traj.write(atom)

            filenameat = filename + '@:'
            configs = read(filenameat) 
            energies = [config.get_potential_energy() for config in configs]
            a = np.array([config.cell[0, 0] for config in configs])
            c = np.array([config.cell[2, 2] for config in configs])

            functions = np.array([a**0, a, c, a**2, a * c, c**2])
            p = np.linalg.lstsq(functions.T, energies, rcond=-1)[0]

            p0 = p[0]
            p1 = p[1:3]
            p2 = np.array([(2 * p[3], p[4]),
                        (p[4], 2 * p[5])])
            a, c = np.linalg.solve(p2.T, -p1)

            f = open('result.txt', 'a')
            f.write('{0}, {1}, {2}, {3}\n'.format(self.ele, xc, str(a),str(c)))
            f.close()

            self.a = a
            self.c = c

            return a, c

    def make_surface_from_bulk(self, unitlength, height):
        atom = self.createbulk(self.a, self.c)
        atom.pbc[2] = False

        atoms = atom.repeat([unitlength, unitlength, height])
        atoms.center(vacuum=10, axis=2)

        atoms = self.settag(atoms)
        return atoms


    def settag(self, atoms):
        poslis = list(set(atoms.get_positions()[:,2]))
        poslis.sort()

        for i in range(len(atoms)):
            for j in range(len(poslis)):
                if atoms[i].position[2] == poslis[j]:
                    atoms[i].tag = len(poslis) - j
        return atoms