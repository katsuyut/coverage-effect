import numpy as np
import sys
import copy
import time
from MAUtil import *
from MACalc import *
from custodian.custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler, UnconvergedErrorHandler

start = time.time()

# env = 'local'
env = 'spacom'
name = sys.argv[1]
res = re.match('(.*)_(.*)_u(.*)_(.*)_(.*)_(.*)_n(.*)_(.*)(.traj)', name)
if res:
    xc = res.group(4)
else:
    res = re.match('(.*)_(.*)_u(.*)_(.*)(.traj)', name)
    if res:
        xc = res.group(4)
    else:
        res = re.match('(.*)_(.*)(.traj)', name)
        xc = res.group(2)

### Set coefficients ###
fixlayer = 3


### Set structure ###
atoms = init_query(name, env)
cell = atoms.cell

if 3 in set(atoms.get_tags()):  # set constraint only on surface calc
    constraint = FixAtoms(mask=[atom.tag >= fixlayer for atom in atoms])
    atoms.set_constraint(constraint)

### Set vasp ###
kpoints = get_kpts(atoms)
# nb = get_nbands(atoms, 2)  # default value is 0.5

tagdict = get_default_vasp_tags(xc)
tagdict['kpts'] = kpoints
vasptags = set_vasp_tags(tagdict)

### Get energy ###
if query(name, env) != None:
    atoms = query(name, env)
    print(atoms.get_potential_energy())
    e_atoms = 'Already in directory'
    sys.exit()
else:
    e_atoms = get_energy(atoms, name[0:-5], vasptags, env)


### use custodian and if error is found restart ###
handlers = [VaspErrorHandler(), UnconvergedErrorHandler()]

flag = False
for handler in handlers:
    if handler.check():
        flag = True
        # handler.correct()
        # custodian_correct_alternative()

        with open('error_custodian.log', 'a') as f:
            f.write(type(handler).__name__ + ': ')
            if type(handler).__name__ == 'VaspErrorHandler':
                f.write(str(handler.errors))

if flag:
    vasptags = Vasp(restart=True, ediffg= -3.00e-02, ibrion=1)
    atoms = vasptags.get_atoms()
    e_atoms = get_energy(atoms, name[0:-5], vasptags, env)


print('{0}, {1}'.format(name, e_atoms))
with open('result.txt', 'a') as f:
    f.write('{0}, {1}\n'.format(name, e_atoms))
    f.write('{0} min'.format((time.time() - start)/60))

print((time.time() - start)/60, 'min')
